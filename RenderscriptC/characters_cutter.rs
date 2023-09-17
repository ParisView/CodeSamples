

#pragma version(1)

#pragma rs java_package_name(com.code.sample.readmrz)


// structs and functions
typedef struct GroupDetectionResult {
    int groupDetectedFlag;
    int newIndex;
    int characterStart;
    int characterEnd;
} GroupDetectionResult;
static GroupDetectionResult detectNextCharacterGroup();
static void cleanUpDetectedCharacters();

// allocations
rs_allocation fbAllocation;
rs_allocation segmenterNormalizedInputAllocation;
rs_allocation segmenterPredictionAllocation;
rs_allocation detectedCharactersAllocation;

// feedback input parameters
int fbAllocationNDetectedCharactersColumn;

// segmenter parameter for characterGroupDetectionKernel
int segmenterPredictionToCutoutShift;

// characterGroupDetectionKernel input parameters
int characterPresentFlagRow;
int characterStartRow;
int characterEndRow;
int characterLowestPointRow;
int characterHighestPointRow;
// value of character present flag denoting that character is present
int characterPresentFlag;
// values of segmenter predictions
int predictionCharacterStart;
int predictionCharacterMiddle;
int predictionCharacterEnd;
int predictionSpace;
int endOfPrediction;
// group detection parameters
int maxAllowedErrorsInGroup;
int maxNextPredictionsToCheck;
int maxEndSearchRounds;
int minGroupWidth;

// characterHeightDetectionKernel input parameters
int characterCutoutPadding;


void RS_KERNEL characterGroupDetectionKernel(int in, uint32_t x, uint32_t y) {
    int numberOfDetectedCharacterGroups = 0;
    int predictionIndex = 0;
    int predictionsLength = rsAllocationGetDimX(segmenterPredictionAllocation);
    int maxCharacters = rsAllocationGetDimX(detectedCharactersAllocation);

    // zero all character present flags in detectedCharactersAllocation
    for (int i = 0; i < maxCharacters; ++i) {
        rsSetElementAt_int(detectedCharactersAllocation, 0, i, characterPresentFlagRow);
    };

    // detect chracter groups in segmenterPredictionAllocation
    while (predictionIndex < predictionsLength && numberOfDetectedCharacterGroups < maxCharacters) {
        GroupDetectionResult detectionResult = detectNextCharacterGroup(
            &segmenterPredictionAllocation, predictionIndex, predictionsLength);
        predictionIndex = detectionResult.newIndex;
        // if character group is detected, write its start and end to detectedCharactersAllocation
        if (detectionResult.groupDetectedFlag == characterPresentFlag) {
            rsSetElementAt_int(detectedCharactersAllocation, characterPresentFlag,
                numberOfDetectedCharacterGroups, characterPresentFlagRow);

            // add shift to character start and end indices, that comes from layers of convolution
            // without padding that result in segmenterPredictionAllocation being shorter than
            // text strip cutout
            rsSetElementAt_int(detectedCharactersAllocation, detectionResult.characterStart +
                segmenterPredictionToCutoutShift, numberOfDetectedCharacterGroups, characterStartRow);
            rsSetElementAt_int(detectedCharactersAllocation, detectionResult.characterEnd +
                segmenterPredictionToCutoutShift, numberOfDetectedCharacterGroups, characterEndRow);
            ++numberOfDetectedCharacterGroups;
        };
    };

    // clean up detected characters: remove end characters, that stand too far from main text string,
    // unite too narrow characters and split too wide characters
    cleanUpDetectedCharacters(&detectedCharactersAllocation, &numberOfDetectedCharacterGroups, maxCharacters);

    // write number of detected characters to fbAllocation
    rsSetElementAt_int(fbAllocation, numberOfDetectedCharacterGroups, fbAllocationNDetectedCharactersColumn, 0);
}

void RS_KERNEL characterHeightDetectionKernel(int in, uint32_t x, uint32_t y) {
    // set cutout strip start (included)
    int cutoutStart = rsGetElementAt_int(detectedCharactersAllocation, x, characterStartRow) -
        characterCutoutPadding;
    if (cutoutStart < 0) {cutoutStart = 0;};
    // set cutout strip end (excluded)
    int cutoutEnd = rsGetElementAt_int(detectedCharactersAllocation, x, characterEndRow) + 1 +
        characterCutoutPadding;
    int cutoutEndLimit = rsAllocationGetDimX(segmenterNormalizedInputAllocation);
    if (cutoutEnd > cutoutEndLimit) {cutoutEnd = cutoutEndLimit;};

    // calculate cutout strip minimums along x axis for each y position
    int cutoutHeight = rsAllocationGetDimY(segmenterNormalizedInputAllocation);
    float cutoutStripMinimums[cutoutHeight];
    float cutoutStripMinimumsMean = 0;
    for (int i = 0; i < cutoutHeight; ++i) {
        float minValue = rsGetElementAt_float(segmenterNormalizedInputAllocation, cutoutStart, i);
        for (int j = cutoutStart + 1; j < cutoutEnd; ++j) {
            float nextValue = rsGetElementAt_float(segmenterNormalizedInputAllocation, j, i);
            if (minValue > nextValue) {minValue = nextValue;};
        };
        cutoutStripMinimums[i] = minValue;
        cutoutStripMinimumsMean += minValue;
    };
    // calculate mean value of minimums
    cutoutStripMinimumsMean /= (float) cutoutHeight;
    // subtract mean value from array of minimums
    for (int i = 0; i < cutoutHeight; ++i) {
        cutoutStripMinimums[i] -= cutoutStripMinimumsMean;
    };

    // calculate correlations of cutout strip minimums array and thresthold functions
    //  for highest and lowest points with threshold position at 0
    float lowestPointCorrelation = cutoutStripMinimums[0];
    float highestPointCorrelation = cutoutStripMinimums[0];
    for (int i = 1; i < cutoutHeight; ++i) {
        lowestPointCorrelation -= cutoutStripMinimums[i];
        highestPointCorrelation += cutoutStripMinimums[i];
    };

    // initialize maximums of correlations and position indices of their corresponding thresholds
    float maxLowestPointCorrelation = lowestPointCorrelation;
    int maxLowestPointCorrelationIndex = 0;
    float maxHighestPointCorrelation = highestPointCorrelation;
    int maxHighestPointCorrelationIndex = 0;

    // calculate correlations for the rest of threshold positions and get indices of correlation maximums
    for (int i = 1; i < cutoutHeight; ++i) {
        lowestPointCorrelation += 2 * cutoutStripMinimums[i];
        highestPointCorrelation -= 2 * cutoutStripMinimums[i - 1];
        if (maxLowestPointCorrelation < lowestPointCorrelation) {
            maxLowestPointCorrelation = lowestPointCorrelation;
            maxLowestPointCorrelationIndex = i;
        };
        if (maxHighestPointCorrelation < highestPointCorrelation) {
            maxHighestPointCorrelation = highestPointCorrelation;
            maxHighestPointCorrelationIndex = i;
        };
    };

    // if calculated highest point position is lower than calculated lowest point position
    if (maxHighestPointCorrelationIndex <= maxLowestPointCorrelationIndex) {
        // if lowest point is closer to middle then it is the correct one, therefor the
        // highest point must be set to the strip height
        int middle = cutoutHeight / 2;
        if (
            abs(maxLowestPointCorrelationIndex - middle) <= abs(
                maxHighestPointCorrelationIndex - middle)
        ) {
            maxHighestPointCorrelationIndex = cutoutHeight - 1;
        } else {
        // if highest point is closer to middle then it is the correct one, therefor the
        // lowest point must be set to 0
            maxLowestPointCorrelationIndex = 0;
        };
    };

    // include padding in character start and end positions in detected characters
    rsSetElementAt_int(detectedCharactersAllocation, cutoutStart, x, characterStartRow);
    rsSetElementAt_int(detectedCharactersAllocation, cutoutEnd, x, characterEndRow);

    // include padding in lowest point position
    maxLowestPointCorrelationIndex -= characterCutoutPadding;
    if (maxLowestPointCorrelationIndex < 0) {maxLowestPointCorrelationIndex = 0;};
    // include padding in highest point position
    maxHighestPointCorrelationIndex += 1 + characterCutoutPadding;
    if (maxHighestPointCorrelationIndex > cutoutHeight) {maxHighestPointCorrelationIndex = cutoutHeight;};

    // write highest and lowest point positions to detected characters
    rsSetElementAt_int(detectedCharactersAllocation, maxLowestPointCorrelationIndex,
        x, characterLowestPointRow);
    rsSetElementAt_int(detectedCharactersAllocation, maxHighestPointCorrelationIndex,
        x, characterHighestPointRow);
}


static int noExtremumsMean(int *array, int arrayLen, int nExcludedExtremums) {
    int sum = 0;
    int minArray[nExcludedExtremums];
    int maxArray[nExcludedExtremums];

    // initialize min value array and max value array with first elements of input array
    // and calculate sum of first elements of input array
    for (int j = 0; j < nExcludedExtremums; ++j) {
        minArray[j] = array[j];
        maxArray[j] = array[j];
        sum += array[j];
    };

    // add to sum the rest of the elements of the input array
    for (int i = nExcludedExtremums; i < arrayLen; ++i) {
        sum += array[i];

        // calculate indices of maximum in min value array and minimum in max value array
        int maxMinValue = minArray[0]; // maximum in min value array
        int maxMinValueIndex = 0;
        int minMaxValue = maxArray[0]; // minimum in max value array
        int minMaxValueIndex = 0;
        for (int j = 1; j < nExcludedExtremums; ++j) {
            if (maxMinValue < minArray[j]) {
                maxMinValue = minArray[j];
                maxMinValueIndex = j;
            };
            if (minMaxValue > maxArray[j]) {
                minMaxValue = maxArray[j];
                minMaxValueIndex = j;
            };
        };

        // save extremums with highest and lowest values to exclude them from sum later
        if (maxMinValue > array[i]) { minArray[maxMinValueIndex] = array[i]; };
        if (minMaxValue < array[i]) { maxArray[minMaxValueIndex] = array[i]; };
    };

    // subtract extremums from sum
    for (int j = 0; j < nExcludedExtremums; ++j) {
        sum -= minArray[j] + maxArray[j];
    };

    // calculate and return mean value
    return sum / (arrayLen - 2 * nExcludedExtremums);
};

static int cleanUpEndCharacters(
    rs_allocation *pDetectedCharactersAllocation,
    int *pNDetectedCharacters,
    int meanValueCalculationLength
) {
    // get starts and ends of detected characters at the beginning of the text strip
    int start1 = rsGetElementAt_int(*pDetectedCharactersAllocation, 1, characterStartRow);
    int start2 = rsGetElementAt_int(*pDetectedCharactersAllocation, 2, characterStartRow);
    int end0 = rsGetElementAt_int(*pDetectedCharactersAllocation, 0, characterEndRow);
    int end1 = rsGetElementAt_int(*pDetectedCharactersAllocation, 1, characterEndRow);
    int end2 = rsGetElementAt_int(*pDetectedCharactersAllocation, 2, characterEndRow);

    // calculate distances between centers of detected characters starting from character with
    // index (2) and ending with character with index (2 + meanValueCalculationLength)
    int centers[meanValueCalculationLength + 1];
    int distancesBetweenCenters[meanValueCalculationLength];
    centers[0] = (start2 + end2) / 2;
    for (int i = 0; i < meanValueCalculationLength; ++i) {
        centers[i + 1] = (rsGetElementAt_int(*pDetectedCharactersAllocation, i + 3, characterStartRow) +
            rsGetElementAt_int(*pDetectedCharactersAllocation, i + 3, characterEndRow)) / 2;
        distancesBetweenCenters[i] = centers[i + 1] - centers[i];
    };

    // calculate mean distance between centers of detected characters with "no extremums mean"
    int meanDistanceAtStart = noExtremumsMean(&distancesBetweenCenters[0], meanValueCalculationLength, 1);

    // get starts and ends of detected characters at the end of the text strip
    int startMinus1 = rsGetElementAt_int(*pDetectedCharactersAllocation,
        *pNDetectedCharacters - 1, characterStartRow);
    int startMinus2 = rsGetElementAt_int(*pDetectedCharactersAllocation,
        *pNDetectedCharacters - 2, characterStartRow);
    int startMinus3 = rsGetElementAt_int(*pDetectedCharactersAllocation,
        *pNDetectedCharacters - 3, characterStartRow);
    int endMinus2 = rsGetElementAt_int(*pDetectedCharactersAllocation,
        *pNDetectedCharacters - 2, characterEndRow);
    int endMinus3 = rsGetElementAt_int(*pDetectedCharactersAllocation,
        *pNDetectedCharacters - 3, characterEndRow);

    // calculate distances between centers of detected characters starting from character with
    // index (number of detected characters - 3 - meanValueCalculationLength) and ending with
    // character with index (number of detected characters - 3)
    centers[0] = (startMinus3 + endMinus3) / 2;
    for (int i = 0; i < meanValueCalculationLength; ++i) {
        centers[i + 1] = (rsGetElementAt_int(*pDetectedCharactersAllocation,
            *pNDetectedCharacters - 4 - i, characterStartRow) +
            rsGetElementAt_int(*pDetectedCharactersAllocation,
            *pNDetectedCharacters - 4 - i, characterEndRow)) / 2;
        distancesBetweenCenters[i] = centers[i] - centers[i + 1];
    };

    // calculate mean distance between centers of detected characters with "no extremums mean"
    int meanDistanceAtEnd = noExtremumsMean(&distancesBetweenCenters[0], meanValueCalculationLength, 1);

    int nDetectedCharactersDecrease = 0;
    // clean up characters at the beginning of the text strip
    if (start2 - end1 - 1 > meanDistanceAtStart) {
        // delete two first characters that stand too far from the others
        rsSetElementAt_int(*pDetectedCharactersAllocation, 0, 0, characterPresentFlagRow);
        rsSetElementAt_int(*pDetectedCharactersAllocation, 0, 1, characterPresentFlagRow);
        nDetectedCharactersDecrease += 2;
    } else {
        if (start1 - end0 - 1 > meanDistanceAtStart) {
            // delete first character that stands too far from the others
            rsSetElementAt_int(*pDetectedCharactersAllocation, 0, 0, characterPresentFlagRow);
            nDetectedCharactersDecrease += 1;
        };
    };
    int nCharactersRemovedAtStart = nDetectedCharactersDecrease;

    // clean up characters at the end of the text strip
    if (startMinus2 - endMinus3 - 1 > meanDistanceAtEnd) {
        // delete two last characters that stand too far from the others
        rsSetElementAt_int(*pDetectedCharactersAllocation, 0, *pNDetectedCharacters - 1, characterPresentFlagRow);
        rsSetElementAt_int(*pDetectedCharactersAllocation, 0, *pNDetectedCharacters - 2, characterPresentFlagRow);
        nDetectedCharactersDecrease += 2;
    } else {
        if (startMinus1 - endMinus2 - 1 > meanDistanceAtEnd) {
            // delete last character that stands too far from the others
            rsSetElementAt_int(*pDetectedCharactersAllocation, 0, *pNDetectedCharacters - 1, characterPresentFlagRow);
            nDetectedCharactersDecrease += 1;
        };
    };

    // update number of detected characters
    *pNDetectedCharacters -= nDetectedCharactersDecrease;

    // return number of characters removed at the beginning of the text strip
    return nCharactersRemovedAtStart;
};

static void cleanUpTooWideOrNarrowCharacters(
    rs_allocation *pDetectedCharactersAllocation,
    int firstCharacterIndex,
    int *pNDetectedCharacters,
    int meanValueCalculationLength,
    int maxCharacters
) {
    float tooWideCharacterFactor = 1.2;
    float tooNarrowCharacterFactor = 1.5;
    int doudleMeanValueCalculationLength = 2 * meanValueCalculationLength;

    // max number of characters added during split of one too wide character
    int maxAddedCharactersInSplit = 5;

    // create arrays for character starts, character ends, character centers and character widths
    int starts[*pNDetectedCharacters];
    int ends[*pNDetectedCharacters];
    int centers[*pNDetectedCharacters];
    int widths[*pNDetectedCharacters];
    for (int i = 0; i < *pNDetectedCharacters; ++i) {
        int start = rsGetElementAt_int(*pDetectedCharactersAllocation, i + firstCharacterIndex, characterStartRow);
        int end = rsGetElementAt_int(*pDetectedCharactersAllocation, i + firstCharacterIndex, characterEndRow);
        starts[i] = start;
        ends[i] = end;
        centers[i] = (start + end) / 2;
        widths[i] = end - start + 1;
    };
    // calculate distances between centers of detected characters
    int distancesBetweenCenters[*pNDetectedCharacters - 1];
    for (int i = 0; i < *pNDetectedCharacters - 1; ++i) {
        distancesBetweenCenters[i] = centers[i + 1] - centers[i];
    };

    // create arrays for united too narrow characters and added characters during split of too wide characters
    int unitedCharacters[*pNDetectedCharacters]; // 0 or 1 flag, 1 - if this character should
                                                       //  be united with next character

    // in addedCharacters[i][j][0] = 0 or 1 flag, 1 - if this character is added to detected characters
    //                         [1] = character start
    //                         [2] = character end
    int3 addedCharacters[*pNDetectedCharacters][maxAddedCharactersInSplit];
    for (int i = 0; i < *pNDetectedCharacters; ++i) {
        unitedCharacters[i] = 0;
        for (int j = 0; j < maxAddedCharactersInSplit; ++j) {
            addedCharacters[i][j][0] = 0;
        };
    };

    // calculate mean distance between centers for the first meanValueCalculationLength characters
    int meanDistance = noExtremumsMean(&distancesBetweenCenters[0], doudleMeanValueCalculationLength, 3);

    // for each detected character
    for (int i = 0; i < *pNDetectedCharacters; ++i) {
        // recalculate mean distance between centers for characters in the middle of text strip
        if (meanValueCalculationLength < i && i < *pNDetectedCharacters - meanValueCalculationLength) {
            meanDistance = noExtremumsMean(&distancesBetweenCenters[i - meanValueCalculationLength],
                doudleMeanValueCalculationLength, 3);
        };

        // if character is too wide, it should be split into several, so some characters should be added
        if (((float) widths[i]) > ((float) meanDistance) * tooWideCharacterFactor) {
            // determine number of characters to add
            int nCharactersToAdd = widths[i] / meanDistance;
            if (nCharactersToAdd > maxAddedCharactersInSplit) {nCharactersToAdd = maxAddedCharactersInSplit;};
            // calculate width of added characters and padding
            int recoveredWidth = widths[i] / (nCharactersToAdd + 1);
            int padding;
            if (i < *pNDetectedCharacters - 1) {
                padding = (starts[i + 1] - ends[i]) / 2;
            } else {
                padding = (starts[i] - ends[i - 1]) / 2;
            };

            // calculate new start of the original character
            int originalCharacterNewStart = ends[i] + padding - recoveredWidth;

            // fill added characters array
            int previousStart = originalCharacterNewStart;
            for (int j = nCharactersToAdd - 1; j >= 0; --j) {
                // set flag denoting that this character should be added to detected characters
                addedCharacters[i][j][0] = 1;
                // set character end
                addedCharacters[i][j][2] = previousStart - 1;
                // move previousStart to next position
                previousStart -= recoveredWidth;
                // set character start
                addedCharacters[i][j][1] = previousStart;
            };

            // set first added character start to original character start
            addedCharacters[i][0][1] = starts[i];
            // set original character start to new value
            starts[i] = originalCharacterNewStart;

        } else {
            // if character is too narrow. it should be united with the next one
            if (
                i < *pNDetectedCharacters - 1 &&
                ((float) distancesBetweenCenters[i]) * tooNarrowCharacterFactor < ((float) meanDistance)
            ) {
                unitedCharacters[i] = 1;
            };
        };
    };

    // reconstruct detected characters
    int arraysIndex = 0;
    int detectedCharactersIndex = 0;
    while (arraysIndex < *pNDetectedCharacters) {
        // if character is split into several
        if (addedCharacters[arraysIndex][0][0] == 1) {
            for (
                int i = 0;
                i < maxAddedCharactersInSplit && addedCharacters[arraysIndex][i][0] == 1;
                ++i
            ) {
                // for character with detectedCharactersIndex write its data from added characters
                if (detectedCharactersIndex < maxCharacters) {
                    rsSetElementAt_int(*pDetectedCharactersAllocation, characterPresentFlag,
                        detectedCharactersIndex, characterPresentFlagRow);
                    rsSetElementAt_int(*pDetectedCharactersAllocation, addedCharacters[arraysIndex][i][1],
                        detectedCharactersIndex, characterStartRow);
                    rsSetElementAt_int(*pDetectedCharactersAllocation, addedCharacters[arraysIndex][i][2],
                        detectedCharactersIndex, characterEndRow);
                    ++detectedCharactersIndex;
                };
            };
        } else {
            // if character is not split into several and is united with the next one
            if (unitedCharacters[arraysIndex] == 1) {
                starts[arraysIndex + 1] = starts[arraysIndex];
                ++arraysIndex;
            };
        };

        // for character with detectedCharactersIndex write its data from arrays cell with arraysIndex
        if (detectedCharactersIndex < maxCharacters) {
            rsSetElementAt_int(*pDetectedCharactersAllocation, characterPresentFlag,
                detectedCharactersIndex, characterPresentFlagRow);
            rsSetElementAt_int(*pDetectedCharactersAllocation, starts[arraysIndex],
                detectedCharactersIndex, characterStartRow);
            rsSetElementAt_int(*pDetectedCharactersAllocation, ends[arraysIndex],
                detectedCharactersIndex, characterEndRow);
            ++detectedCharactersIndex;
        };

        // proceed to the next character
        ++arraysIndex;
    };

    // reset character present flags in the rest of the cells of the detected characters
    for (int i = detectedCharactersIndex; i < maxCharacters; ++i) {
        rsSetElementAt_int(*pDetectedCharactersAllocation, 0, i, characterPresentFlagRow);
    };

    // update number of detected characters
    *pNDetectedCharacters = detectedCharactersIndex;
};

static void cleanUpDetectedCharacters(
    rs_allocation *pDetectedCharactersAllocation,
    int *pNDetectedCharacters,
    int maxCharacters
) {
    // number of characters on which mean distance between their centers is calculated for
    // end characters. for too wide and too narrow characters this number is doubled
    int meanValueCalculationLength = 5;

    // if number of detected characters is high enough, perform the clean up
    int minDetectedCharacters = 2 * meanValueCalculationLength + 4;
    if (*pNDetectedCharacters > minDetectedCharacters) {
        // clean up characters at the beginning and at the end of the text strip
        int nCharactersRemovedAtStart = cleanUpEndCharacters(pDetectedCharactersAllocation,
            pNDetectedCharacters, meanValueCalculationLength);
        // clean up too wide and too narrow characters
        cleanUpTooWideOrNarrowCharacters(pDetectedCharactersAllocation, nCharactersRemovedAtStart, pNDetectedCharacters,
            meanValueCalculationLength, maxCharacters);
    };
};

static int readPrediction(
    rs_allocation *pPredictionAllocation,
    int predictionIndex,
    int predictionsLength
) {
    int out = endOfPrediction;
    if (predictionIndex < predictionsLength) {
        out = rsGetElementAt_int(*pPredictionAllocation, predictionIndex, 0);
    };
    return out;
};

static int sortMultipleEncounters(
    rs_allocation *pPredictionAllocation,
    int predictionIndex,
    int predictionsLength,
    int predictionValue
) {
    int indexIncrement = 0;
    while ((predictionIndex + indexIncrement < predictionsLength) && (rsGetElementAt_int(
        *pPredictionAllocation, predictionIndex + indexIncrement, 0) == predictionValue)
    ) {
        ++indexIncrement;
    };
    return indexIncrement;
};

static int2 findNext(
    rs_allocation *pPredictionAllocation,
    int predictionIndex,
    int predictionsLength,
    int predictionValue
) {
    // Searching for the specified prediction value in the next
    // maxNextPredictionsToCheck positions of the prediction allocation.
    // Index increment is returned as the position at which the specified value was
    // found counting from the position of the predictionIndex.
    int indexIncrement = 1;
    int continueSearch = 1;
    int2 foundSpecifiedValue;
    foundSpecifiedValue[0] = 0;

    while (continueSearch == 1) {
        // if index is within allocation length
        if (predictionIndex + indexIncrement < predictionsLength) {
            // if found the specified value
            if (rsGetElementAt_int(*pPredictionAllocation, predictionIndex + indexIncrement, 0) ==
                predictionValue) {
                continueSearch = 0;
                // set flag, that specified value is found
                foundSpecifiedValue[0] = 1;
                // write index increment to output
                foundSpecifiedValue[1] = indexIncrement;
            } else { // did not find the specified value
                // if checked less than maxNextPredictionsToCheck
                if (indexIncrement < maxNextPredictionsToCheck - 1) {
                    ++indexIncrement;
                } else {
                    continueSearch = 0;
                };
            };
        } else { // if index is out of allocation length stop searching
            continueSearch = 0;
        };
    };
    return foundSpecifiedValue;
};

static void findGroupEnd(
    rs_allocation *pPredictionAllocation,
    int predictionIndex,
    int predictionsLength,
    int errorCount,
    GroupDetectionResult *pDetectionResult
) {
    int endSearchRound = 0;
    // while no more than maxEndSearchRounds passed
    while (endSearchRound < maxEndSearchRounds) {
        // reading next prediction
        int nextPrediction = readPrediction(pPredictionAllocation,
            predictionIndex + endSearchRound, predictionsLength);

        // if prediction is character end, then character group is detected
        if (nextPrediction == predictionCharacterEnd) {
            // handle double end
            if (readPrediction(pPredictionAllocation, predictionIndex + 1 + endSearchRound,
                predictionsLength) == predictionCharacterEnd) {
                ++predictionIndex;
            };
            // write character end position
            (*pDetectionResult).characterEnd = predictionIndex + endSearchRound;
            (*pDetectionResult).newIndex = predictionIndex + endSearchRound + 1;
            if ((*pDetectionResult).characterEnd - (*pDetectionResult).characterStart + 1 >= minGroupWidth) {
                (*pDetectionResult).groupDetectedFlag = characterPresentFlag;
            };
            // stop search
            endSearchRound = maxEndSearchRounds;
        } else { // if prediction is not character end, then increment error counter
            ++errorCount;

            // if too many errors for this group search iteration
            if (errorCount > maxAllowedErrorsInGroup) {

                if (endSearchRound > 0) { // consider that character group has been detected
                    // write character end position
                    (*pDetectionResult).characterEnd = predictionIndex - 1;
                    (*pDetectionResult).newIndex = predictionIndex;
                    if ((*pDetectionResult).characterEnd - (*pDetectionResult).characterStart + 1 >= minGroupWidth) {
                        (*pDetectionResult).groupDetectedFlag = characterPresentFlag;
                    };
                    // stop search
                    endSearchRound = maxEndSearchRounds;

                } else { // try to fing character group from next starting position
                    ++(*pDetectionResult).newIndex;
                    // stop search
                    endSearchRound = maxEndSearchRounds;
                };
            } else { // try to find character end one more time
                ++endSearchRound;
            };
        };
    };
};

static GroupDetectionResult detectNextCharacterGroup(
    rs_allocation *pPredictionAllocation,
    int predictionIndex,
    int predictionsLength
) {
    // reading next prediction
    int nextPrediction = readPrediction(pPredictionAllocation, predictionIndex, predictionsLength);
    // if prediction is space, then go through all consecutive spaces
    if (nextPrediction == predictionSpace) {
        ++predictionIndex;
        predictionIndex += sortMultipleEncounters(pPredictionAllocation, predictionIndex,
            predictionsLength, predictionSpace);
        nextPrediction = readPrediction(pPredictionAllocation, predictionIndex, predictionsLength);
    };
    // if prediction is character end, then go through all consecutive character ends
    if (nextPrediction == predictionCharacterEnd) {
        ++predictionIndex;
        predictionIndex += sortMultipleEncounters(pPredictionAllocation, predictionIndex,
            predictionsLength, predictionCharacterEnd);
        nextPrediction = readPrediction(pPredictionAllocation, predictionIndex, predictionsLength);
    };

    // creating output object
    GroupDetectionResult detectionResult;
    detectionResult.groupDetectedFlag = 0;
    detectionResult.newIndex = predictionIndex;

    // if prediction is character start, then try to find character group
    if (nextPrediction == predictionCharacterStart) {
        // write character start position
        detectionResult.characterStart = predictionIndex;
        // search for character middle in the next predictions
        int2 charcterMiddleFound = findNext(pPredictionAllocation, predictionIndex,
            predictionsLength, predictionCharacterMiddle);

        // if character middle is found
        if (charcterMiddleFound[0] == 1) {
            // move index to the point of the found middle
            predictionIndex += charcterMiddleFound[1];
            // initialize counter of errors per detected group
            int errorCount = charcterMiddleFound[1] - 1;
            // go through all consecutive character middles
            predictionIndex += sortMultipleEncounters(pPredictionAllocation, predictionIndex,
                predictionsLength, predictionCharacterMiddle);
            // find group end, and, if found, write character detected flag in detection result
            findGroupEnd(pPredictionAllocation, predictionIndex, predictionsLength,
                errorCount, &detectionResult);

        } else { // if character middle is not found, then increment new index and return result
            ++detectionResult.newIndex;
        };
    } else {
        // if prediction is character middle, then try to find character group
        if (nextPrediction == predictionCharacterMiddle) {
            // write character start position
            detectionResult.characterStart = predictionIndex;
            // initialize counter of errors per detected group
            int errorCount = 1;
            // go through all consecutive character middles
            predictionIndex += sortMultipleEncounters(pPredictionAllocation, predictionIndex,
                predictionsLength, predictionCharacterMiddle);
            // find group end, and, if found, write character detected flag in detection result
            findGroupEnd(pPredictionAllocation, predictionIndex, predictionsLength,
                errorCount, &detectionResult);
        };
    };

    return detectionResult;
};

