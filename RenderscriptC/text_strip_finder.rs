

#pragma version(1)

#pragma rs java_package_name(com.code.sample.readmrz)


// structs and functions
static int calculateLocalDeviation();
static int calculateStripCorrelation(
    rs_allocation *inputAllocation,
    rs_allocation *correlationCoreAllocation,
    int xCoordinate,
    int yStart,
    int yLength,
    int stripHalfWidth,
    float maxAmplitudeToMeanRatio
);
static void setElementWithAllocationBordesCheck_uchar4(
    rs_allocation *allocation,
    uchar4 value,
    int xCoordinate,
    int yCoordinate
);
static void drawRectangleWithAllocationBordesCheck_uchar4(
    rs_allocation *allocation,
    uchar4 value,
    int xStart,
    int xEnd,
    int yStart,
    int yEnd
);

// allocations
rs_allocation outAllocation;
rs_allocation grayAllocation;
rs_allocation reducedGrayAllocation;
rs_allocation reducedColumnSumAllocation;
rs_allocation reducedLocalMeanAllocation;
rs_allocation reducedLocalDeviationAllocation;
rs_allocation textSearchCorrelationFunctionAllocation;
rs_allocation lineDetectionAllocation;
rs_allocation fbAllocation;
rs_allocation endDetectionCorrelationFunctionAllocation;

// reductionKernel input parameters
int meanPullKernelWidth;
int numberOfMeanPullSumElements;

// reducedLocalMeanColumnSumKernel, reducedLocalMeanRowSumKernel and reducedLocalDeviationKernel input parameters
int reducedAllocationWidth;
int reducedAllocationHeight;
int reducedKernelWidth;
int reducedKernelHalfWidth;
int numberOfReducedKernelSumElements;
int localDeviationLimit;

// textSearchCorrelationKernel input parameters
int correlationFunctionTextTopEdge;
int textStripTopEdge;
int textStripBottomEdge;
int lineDetectionAllocationTextTopEdgeYRowNumber;
int textSearchCorrelationKernelLength;
int textSearchCorrelationKernelXStep;
int textSearchCorrelationKernelStripHalfWidth = 15;
float correlationAmplitudePeakToMeanRatio;

// lineDetectionKernel input parameters
int lineDetectionKernelMaxAngleCtg; // cotangent = delta width / delta height
int lineDetectionKernelMaxDeltaAngleCtg; // cotangent = delta width / delta height
int textSearchCorrelationKernelNumberOfRuns;
int calculatedMeanAngleXBase;
int lineDetectionAllocationIncludeFlagRowNumber;
int lineDetectionAllocationTempBufferRowNumber;
int markerIncludedInLineFlag;
int minPointsForDetectedLine;
// connecting variables, that carry calculated values from one kernel to another
int detectedLineX = 0;
int detectedLineY = 0;
int detectedLineAngleX = 100; // line tg = detectedLineAngleY / detectedLineAngleX
int detectedLineAngleY = 0;

// feedback input parameters
int fbAllocationLineDetectionFlagColumn;
int fbAllocationLineDetectedFlag;
int fbAllocationNDetectedCharactersColumn;
int fbAllocationCharacterClassColumn;

// lineMarkerDrawKernel input parameters
int yuvToGrayKernelShiftX;
int yuvToGrayKernelShiftY;
int yuvToOutKernelShiftX;
int yuvToOutKernelShiftY;
int colorOfMarkerIncludedInLine[4];
int colorOfMarkerNotIncludedInLine[4];
int lineMarkerHalfWidth;
int lineMarkerDashThickness;
int lineMarkerHeight;
int lineMarkerVerticalDashXShift;

// endDetectionKernel input parameters
int endDetectionCorrelationFunctionLength;
int endDetectionCorrelationStripWidth;
int endDetectionTextEdgeColumn;
int endDetectionCorrelationStartColumn;
int endDetectionCorrelationEndColumn;
int endDetectionAmplitudeBeforeTextEdgeColumn;
int endDetectionAmplitudeAfterTextEdgeColumn;
int endDetectionDirectionSignColumn;
int vicinityHalfWidth;
int localMaxAreaMeanHalfWidth;
int localMaxCalculationPadding;
// connecting variables, that carry calculated values from one kernel to another
int2 detectedLineEndsX;
int2 detectedLineEndsY;

// endMarkerDrawKernel input parameters
int endMarkerDashThickness;


void RS_KERNEL reductionKernel(int in, uint32_t x, uint32_t y) {
    // calculate mean value
    int meanValue = 0;
    for (int i = x * meanPullKernelWidth; i < (x + 1) * meanPullKernelWidth; ++i) {
        for (int j = y * meanPullKernelWidth; j < (y + 1) * meanPullKernelWidth; ++j) {
            meanValue += rsGetElementAt_int(grayAllocation, i, j);
        };
    };
    meanValue /= numberOfMeanPullSumElements;

    // write mean value to reducedGrayAllocation
    rsSetElementAt_int(reducedGrayAllocation, meanValue, x, y);
}

void RS_KERNEL reducedLocalMeanColumnSumKernel(int in, uint32_t x, uint32_t y) {
    int columnSum = 0;
    int heightIndex;

    // calculate and write first column sum element on reducedGrayAllocation
    for (heightIndex = 0; heightIndex < reducedKernelWidth; ++heightIndex) {
        columnSum += rsGetElementAt_int(reducedGrayAllocation, x, heightIndex);
    };
    rsSetElementAt_int(reducedColumnSumAllocation, columnSum, x, reducedKernelHalfWidth);

    // calculate and write the rest of the elements of the column
    for (heightIndex = reducedKernelWidth; heightIndex < reducedAllocationHeight; ++heightIndex) {
        columnSum -= rsGetElementAt_int(reducedGrayAllocation, x, heightIndex - reducedKernelWidth);
        columnSum += rsGetElementAt_int(reducedGrayAllocation, x, heightIndex);
        rsSetElementAt_int(reducedColumnSumAllocation, columnSum, x,
            heightIndex - reducedKernelHalfWidth);
    }
}

void RS_KERNEL reducedLocalMeanRowSumKernel(int in, uint32_t x, uint32_t y) {
    int rowSum = 0;
    int widthIndex;
    int localMeanValue;

    // calculate first row sum element on reducedColumnSumAllocation
    for (widthIndex = 0; widthIndex < reducedKernelWidth; ++widthIndex) {
        rowSum += rsGetElementAt_int(reducedColumnSumAllocation, widthIndex, y);
    };
    // calculate the first element of the row of local mean
    localMeanValue = rowSum / numberOfReducedKernelSumElements;

    // write first element of the row to reducedLocalMeanAllocation
    rsSetElementAt_int(reducedLocalMeanAllocation, localMeanValue, reducedKernelHalfWidth, y);

    // calculate the rest of row sum elements on reducedColumnSumAllocation
    for (widthIndex = reducedKernelWidth; widthIndex < reducedAllocationWidth; ++widthIndex) {
        rowSum += rsGetElementAt_int(reducedColumnSumAllocation, widthIndex, y);
        rowSum -= rsGetElementAt_int(reducedColumnSumAllocation,
            widthIndex - reducedKernelWidth, y);

        // calculate local mean
        localMeanValue = rowSum / numberOfReducedKernelSumElements;

        // write local mean to reducedLocalMeanAllocation
        rsSetElementAt_int(reducedLocalMeanAllocation, localMeanValue,
            widthIndex - reducedKernelHalfWidth, y);
    };
}

void RS_KERNEL reducedLocalDeviationKernel(int in, uint32_t x, uint32_t y) {
    // calculate local deviation
    int localDeviation = calculateLocalDeviation(
        &reducedGrayAllocation,
        x,
        y,
        reducedKernelHalfWidth,
        reducedKernelWidth,
        numberOfReducedKernelSumElements,
        in
    );

    // limit localMeanDeviation by localDeviationLimit to remove unwanted peaks on document page edges.
    // experimentally found that the value of input containing text is rarely more than
    // localDeviationLimit, and simultaneously if document is on a dark background, its edges produce
    // very large peaks of local deviation (especially when calculated with reduced kernels), that
    // potencially can reach full pixel amplitude.
    // these peaks produce correlation maximums, that are mach higher than maximums from input
    // containing text
    if (localDeviation > localDeviationLimit) {localDeviation = localDeviationLimit;};

    // write local deviation to reducedLocalDeviationAllocation
    rsSetElementAt_int(reducedLocalDeviationAllocation, localDeviation, x, y);
}

void RS_KERNEL textSearchCorrelationKernel(int in, uint32_t x, uint32_t y) {
    int correlation = 0;
    int maxCorrelation = 0;
    int maxCorrelationStartY = 0;
    int xCoordinate = textSearchCorrelationKernelXStep * (x + 1);

    // calculate correlation for all kernel positions along allocation height
    for (int i = 0; i < reducedAllocationHeight - textSearchCorrelationKernelLength; ++i) {
        correlation = calculateStripCorrelation(
            &reducedLocalDeviationAllocation,
            &textSearchCorrelationFunctionAllocation,
            xCoordinate,
            i,
            textSearchCorrelationKernelLength,
            textSearchCorrelationKernelStripHalfWidth,
            correlationAmplitudePeakToMeanRatio
        );

        if (maxCorrelation < correlation) {
            maxCorrelation = correlation;
            maxCorrelationStartY = i;
        };
    };

    // write text edge Y coordinate to lineDetectionAllocation
    rsSetElementAt_int(
        lineDetectionAllocation,
        maxCorrelationStartY + correlationFunctionTextTopEdge,
        x,
        lineDetectionAllocationTextTopEdgeYRowNumber
    );
}

void RS_KERNEL lineDetectionKernel(int in, uint32_t x, uint32_t y) {
    int pointWithNumberOfPointsWithSameDirection = 0;
    int numberOfPointsWithSameDirection = 1;
    int meanAngleYForMaxPointsWithSameDirection = 1;
    int meanAngleXForMaxPointsWithSameDirection = lineDetectionKernelMaxAngleCtg;
    int calculatedMeanAngleY = 1;
    int calculatedNumberOfPointsWithSameDirection;
    int point_i_Y = 0;
    int point_j_Y = 0;
    int point_i_to_point_j_angleX = 100;
    int point_i_to_point_j_angleY = 0;
    int point_i_to_point_k_angleX = 100;
    int point_i_to_point_k_angleY = 0;

    // calculating and writing to lineDetectionAllocation numerator (angleX in number of steps of
    // textSearchCorrelationKernelXStep) and denominator (angleY) of line angle tg (tg = angleY / angleX)
    // from each correlation point (point i) to all other correlation points (point j)
    for (int i = 0; i < textSearchCorrelationKernelNumberOfRuns - 1; ++i) {
        point_i_Y = rsGetElementAt_int(
            lineDetectionAllocation,
            i,
            lineDetectionAllocationTextTopEdgeYRowNumber
        );
        for (int j = i + 1; j < textSearchCorrelationKernelNumberOfRuns; ++j) {
            point_j_Y = rsGetElementAt_int(
                lineDetectionAllocation,
                j,
                lineDetectionAllocationTextTopEdgeYRowNumber
            );

            // line angle tg = (point2Y - point1Y) / ((j - i) * textSearchCorrelationKernelXStep)
            // tg numerator angleY
            rsSetElementAt_int(lineDetectionAllocation, point_j_Y - point_i_Y, i, j);

            // tg denominator angleX in steps of textSearchCorrelationKernelXStep
            rsSetElementAt_int(lineDetectionAllocation, (j - i),
                textSearchCorrelationKernelNumberOfRuns - 1 - i, j - i);
        };
    };

    // reset flag row in lineDetectionAllocation
    for (int flagIndex = 0; flagIndex < textSearchCorrelationKernelNumberOfRuns; ++flagIndex) {
        rsSetElementAt_int(lineDetectionAllocation, 0, flagIndex, lineDetectionAllocationIncludeFlagRowNumber);
    };

    // determining correlation point with maximum number of other correlation points with
    // the same tg (tangents of other points should deviate
    // less than 1/lineDetectionKernelMaxDeltaAngleCtg and all tangents should be
    // less than 1/lineDetectionKernelMaxAngleCtg)
    for (int i = 0; i < textSearchCorrelationKernelNumberOfRuns; ++i) {

        // mark point i as included in line
        rsSetElementAt_int(lineDetectionAllocation, markerIncludedInLineFlag, i,
            lineDetectionAllocationTempBufferRowNumber);

        for (int j = 1; j < textSearchCorrelationKernelNumberOfRuns; ++j) {

            int bufferIndex;
            if (j > i) {
                point_i_to_point_j_angleY = rsGetElementAt_int(lineDetectionAllocation, i, j);
                point_i_to_point_j_angleX = rsGetElementAt_int(lineDetectionAllocation,
                    textSearchCorrelationKernelNumberOfRuns - 1 - i, j - i);
                bufferIndex = j;
            } else {
                point_i_to_point_j_angleY = rsGetElementAt_int(lineDetectionAllocation, j - 1, i);
                point_i_to_point_j_angleX = rsGetElementAt_int(lineDetectionAllocation,
                    textSearchCorrelationKernelNumberOfRuns - 1 - (j - 1), i - (j - 1));
                bufferIndex = j - 1;
            };

            calculatedMeanAngleY = point_i_to_point_j_angleY * (
                calculatedMeanAngleXBase / point_i_to_point_j_angleX);
            calculatedNumberOfPointsWithSameDirection = 1;

            // check if point i to point j angle is less than max allowed angle
            // (angle ctg = x / y > lineDetectionKernelMaxAngleCtg)
            if (abs(point_i_to_point_j_angleX * textSearchCorrelationKernelXStep) >
                abs(lineDetectionKernelMaxAngleCtg * point_i_to_point_j_angleY)) {

                // mark point as included in line
                rsSetElementAt_int(lineDetectionAllocation, markerIncludedInLineFlag, bufferIndex,
                    lineDetectionAllocationTempBufferRowNumber);

                for (int k = 1; k < textSearchCorrelationKernelNumberOfRuns; ++k) {
                    if (k != j) {
                        if (k > i) {
                            point_i_to_point_k_angleY = rsGetElementAt_int(lineDetectionAllocation, i, k);
                            point_i_to_point_k_angleX = rsGetElementAt_int(lineDetectionAllocation,
                                textSearchCorrelationKernelNumberOfRuns - 1 - i, k - i);
                            bufferIndex = k;
                        } else {
                            point_i_to_point_k_angleY = rsGetElementAt_int(lineDetectionAllocation, k - 1, i);
                            point_i_to_point_k_angleX = rsGetElementAt_int(lineDetectionAllocation,
                                textSearchCorrelationKernelNumberOfRuns - 1 - (k - 1), i - (k - 1));
                            bufferIndex = k - 1;
                        };

                        // mark initially point as not included in line
                        rsSetElementAt_int(lineDetectionAllocation, 0, bufferIndex,
                            lineDetectionAllocationTempBufferRowNumber);

                        // check if point i to point k angle is less than max allowed angle
                        // (angle ctg = x / y > lineDetectionKernelMaxAngleCtg)
                        if (abs(point_i_to_point_k_angleX * textSearchCorrelationKernelXStep) >
                            abs(lineDetectionKernelMaxAngleCtg * point_i_to_point_k_angleY)
                        ) {
                            // check if difference between two angles is less than max allowed difference
                            if (abs((point_i_to_point_j_angleY * point_i_to_point_k_angleX -
                                point_i_to_point_k_angleY * point_i_to_point_j_angleX) *
                                lineDetectionKernelMaxDeltaAngleCtg) < abs(
                                point_i_to_point_j_angleX * point_i_to_point_k_angleX *
                                textSearchCorrelationKernelXStep)
                            ) {
                                calculatedNumberOfPointsWithSameDirection++;
                                calculatedMeanAngleY += point_i_to_point_k_angleY * (
                                    calculatedMeanAngleXBase / point_i_to_point_k_angleX);

                                // mark point as included in line
                                rsSetElementAt_int(lineDetectionAllocation, markerIncludedInLineFlag, bufferIndex,
                                    lineDetectionAllocationTempBufferRowNumber);
                            };
                        };
                    };
                };
                calculatedMeanAngleY /= calculatedNumberOfPointsWithSameDirection;
            };

            if (calculatedNumberOfPointsWithSameDirection > numberOfPointsWithSameDirection) {
                numberOfPointsWithSameDirection = calculatedNumberOfPointsWithSameDirection;
                meanAngleYForMaxPointsWithSameDirection = calculatedMeanAngleY;
                meanAngleXForMaxPointsWithSameDirection = calculatedMeanAngleXBase * textSearchCorrelationKernelXStep;
                pointWithNumberOfPointsWithSameDirection = i;

                // write buffer row to flag row (row of flags showing if point is included in detected line)
                for (int flagIndex = 0; flagIndex < textSearchCorrelationKernelNumberOfRuns; ++flagIndex) {
                    int bufferValue = rsGetElementAt_int(lineDetectionAllocation, flagIndex,
                        lineDetectionAllocationTempBufferRowNumber);
                    rsSetElementAt_int(lineDetectionAllocation, bufferValue, flagIndex,
                        lineDetectionAllocationIncludeFlagRowNumber);
                };
            };
        };
    };

    // write results
    detectedLineY = rsGetElementAt_int(lineDetectionAllocation,
        pointWithNumberOfPointsWithSameDirection,
        lineDetectionAllocationTextTopEdgeYRowNumber
    );
    detectedLineX = (pointWithNumberOfPointsWithSameDirection + 1) * textSearchCorrelationKernelXStep;

    detectedLineAngleX = meanAngleXForMaxPointsWithSameDirection;
    detectedLineAngleY = meanAngleYForMaxPointsWithSameDirection;

    // if line was detected then write corresponding flag to fbAllocation, or write 0 to that flag
    if (numberOfPointsWithSameDirection < minPointsForDetectedLine) {
        rsSetElementAt_int(fbAllocation, 0, fbAllocationLineDetectionFlagColumn, 0);
    } else {
        rsSetElementAt_int(fbAllocation, fbAllocationLineDetectedFlag, fbAllocationLineDetectionFlagColumn, 0);
    };
}

void RS_KERNEL lineMarkerDrawKernel(int in, uint32_t x, uint32_t y) {
    uchar4 out;
    int xHorizontalDashStart = (textSearchCorrelationKernelXStep * (x + 1) * meanPullKernelWidth
        ) - lineMarkerHalfWidth + yuvToGrayKernelShiftX - yuvToOutKernelShiftX;
    int yHorizontalDashStart = in * meanPullKernelWidth + yuvToGrayKernelShiftY - yuvToOutKernelShiftY;
    int xVerticalDashStart = xHorizontalDashStart + lineMarkerVerticalDashXShift;
    int yVerticalDashStart = yHorizontalDashStart - lineMarkerDashThickness;

    // determine if marker is included in detected line and set its color accordingly
    if (
        rsGetElementAt_int(fbAllocation, fbAllocationLineDetectionFlagColumn, 0
            ) == fbAllocationLineDetectedFlag &&
        rsGetElementAt_int(lineDetectionAllocation, x, lineDetectionAllocationIncludeFlagRowNumber
            ) == markerIncludedInLineFlag
    ) {
        out[0] = clamp(colorOfMarkerIncludedInLine[0], 0, 255);
        out[1] = clamp(colorOfMarkerIncludedInLine[1], 0, 255);
        out[2] = clamp(colorOfMarkerIncludedInLine[2], 0, 255);
        out[3] = clamp(colorOfMarkerIncludedInLine[3], 0, 255);
    } else {
        out[0] = clamp(colorOfMarkerNotIncludedInLine[0], 0, 255);
        out[1] = clamp(colorOfMarkerNotIncludedInLine[1], 0, 255);
        out[2] = clamp(colorOfMarkerNotIncludedInLine[2], 0, 255);
        out[3] = clamp(colorOfMarkerNotIncludedInLine[3], 0, 255);
    };

    // draw top edge marker on outAllocation
    // draw horizontal dash
    drawRectangleWithAllocationBordesCheck_uchar4(
        &outAllocation,
        out,
        xHorizontalDashStart,
        xHorizontalDashStart + 2 * lineMarkerHalfWidth + 1,
        yHorizontalDashStart - lineMarkerDashThickness,
        yHorizontalDashStart
    );
    // draw vertical dash
    drawRectangleWithAllocationBordesCheck_uchar4(
        &outAllocation,
        out,
        xHorizontalDashStart + lineMarkerVerticalDashXShift,
        xHorizontalDashStart + lineMarkerVerticalDashXShift + lineMarkerDashThickness,
        yHorizontalDashStart - lineMarkerHeight,
        yHorizontalDashStart - lineMarkerDashThickness
    );

    // draw bottom edge marker on outAllocation
    yHorizontalDashStart += textStripBottomEdge - textStripTopEdge;
    yVerticalDashStart = yHorizontalDashStart + lineMarkerDashThickness;
    // draw horizontal dash
    drawRectangleWithAllocationBordesCheck_uchar4(
        &outAllocation,
        out,
        xHorizontalDashStart,
        xHorizontalDashStart + 2 * lineMarkerHalfWidth + 1,
        yHorizontalDashStart,
        yHorizontalDashStart + lineMarkerDashThickness
    );
    // draw vertical dash
    drawRectangleWithAllocationBordesCheck_uchar4(
        &outAllocation,
        out,
        xHorizontalDashStart + lineMarkerVerticalDashXShift,
        xHorizontalDashStart + lineMarkerVerticalDashXShift + lineMarkerDashThickness,
        yHorizontalDashStart + lineMarkerDashThickness,
        yHorizontalDashStart + lineMarkerHeight
    );
}

void RS_KERNEL endDetectionKernel(int in, uint32_t x, uint32_t y) {
    int deviationColumnSum;
    int areaCorrelationSum = 0;
    int meanValueSum = 0;
    int meanValue;

    int textEdge = rsGetElementAt_int(endDetectionCorrelationFunctionAllocation,
        endDetectionTextEdgeColumn, y);
    int xTopStart = rsGetElementAt_int(endDetectionCorrelationFunctionAllocation,
        endDetectionCorrelationStartColumn, y);
    int xTopEnd = rsGetElementAt_int(endDetectionCorrelationFunctionAllocation,
        endDetectionCorrelationEndColumn, y);
    int amplitudeBeforeTextEdge = rsGetElementAt_int(endDetectionCorrelationFunctionAllocation,
        endDetectionAmplitudeBeforeTextEdgeColumn, y);
    int amplitudeAfterTextEdge = rsGetElementAt_int(endDetectionCorrelationFunctionAllocation,
        endDetectionAmplitudeAfterTextEdgeColumn, y);
    int directionSign = rsGetElementAt_int(endDetectionCorrelationFunctionAllocation,
        endDetectionDirectionSignColumn, y);

    int maxCorrelation = 0;
    int maxCorrelationStartX = xTopStart;

    int yTopCoordinate;
    int xCoordinate;
    int yCoordinate;

    // calculate deviation column sums for the first x position of the correlation function
    for (int i = xTopStart; i < xTopStart + endDetectionCorrelationFunctionLength; ++i) {
        deviationColumnSum = 0;
        yTopCoordinate = detectedLineY + ((i - detectedLineX) * detectedLineAngleY) / detectedLineAngleX;

        // calculate deviation column sum
        for (int yDelta = 0; yDelta < endDetectionCorrelationStripWidth; ++yDelta) {
            yCoordinate = yTopCoordinate + yDelta;
            xCoordinate = i - (yDelta * detectedLineAngleY) / detectedLineAngleX;

            // determine if coordinates of the point are within required borders
            if (yCoordinate >= reducedKernelHalfWidth &&
                yCoordinate < (reducedAllocationHeight - reducedKernelHalfWidth) &&
                xCoordinate >= reducedKernelHalfWidth &&
                xCoordinate < reducedAllocationWidth - reducedKernelHalfWidth
            ) {
                // add local deviation to deviation column sum
                deviationColumnSum += rsGetElementAt_int(reducedLocalDeviationAllocation,
                    xCoordinate, yCoordinate);
            };
        };

        // write deviation column sum to reducedColumnSumAllocation
        rsSetElementAt_int(reducedColumnSumAllocation, deviationColumnSum, i, 0);

        // calculate mean value sum
        meanValueSum += deviationColumnSum;
    };

    // calculate mean value
    meanValue = meanValueSum / endDetectionCorrelationFunctionLength;

    // calculate correlation for the first x position of the correlation function
    for (int i = xTopStart; i < xTopStart + endDetectionCorrelationFunctionLength; ++i) {
        // calculate correlation and add it to the areaCorrelationSum
        int amplitude;
        if (i - xTopStart < textEdge) {
            amplitude = amplitudeBeforeTextEdge;
        } else {
            amplitude = amplitudeAfterTextEdge;
        };
        areaCorrelationSum += (rsGetElementAt_int(reducedColumnSumAllocation, i, 0) -
            meanValue) * amplitude;
    };

    // write area correlation sum to reducedGrayAllocation
    rsSetElementAt_int(reducedGrayAllocation, areaCorrelationSum, xTopStart, 0);

    // init max correlation (max correlation index is initialized at the begining of the kernel)
    maxCorrelation = areaCorrelationSum;

    // calculate correlation for the rest of x positions of the kernel
    for (int i = xTopStart + endDetectionCorrelationFunctionLength;
        i < xTopEnd;
        ++i
    ) {
        deviationColumnSum = 0;
        yTopCoordinate = detectedLineY + ((i - detectedLineX) * detectedLineAngleY) / detectedLineAngleX;

        // calculate deviation column sum
        for (int yDelta = 0; yDelta < endDetectionCorrelationStripWidth; ++yDelta) {
            yCoordinate = yTopCoordinate + yDelta;
            xCoordinate = i - (yDelta * detectedLineAngleY) / detectedLineAngleX;

            // determine if coordinates of the point are within required borders
            if (yCoordinate >= reducedKernelHalfWidth &&
                yCoordinate < (reducedAllocationHeight - reducedKernelHalfWidth) &&
                xCoordinate >= reducedKernelHalfWidth &&
                xCoordinate < reducedAllocationWidth - reducedKernelHalfWidth
            ) {
                // add local deviation to deviation column sum
                deviationColumnSum += rsGetElementAt_int(reducedLocalDeviationAllocation,
                    xCoordinate, yCoordinate);
            };
        };

        // write deviation column sum to reducedColumnSumAllocation
        rsSetElementAt_int(reducedColumnSumAllocation, deviationColumnSum, i, 0);

        // read deviation column sums right before stert and right before text edge
        int deviationColumnSumBeforeStart = rsGetElementAt_int(reducedColumnSumAllocation,
            i - endDetectionCorrelationFunctionLength, 0);
        int deviationColumnSumBeforeTextEdge = rsGetElementAt_int(reducedColumnSumAllocation,
            i - endDetectionCorrelationFunctionLength + textEdge, 0);

        // recalculate correlation sum with previous mean value at start, end and text edge
        areaCorrelationSum -= (deviationColumnSumBeforeStart - meanValue) * amplitudeBeforeTextEdge;
        areaCorrelationSum += (deviationColumnSumBeforeTextEdge - meanValue) * (
            amplitudeBeforeTextEdge - amplitudeAfterTextEdge);
        areaCorrelationSum += (deviationColumnSum - meanValue) * amplitudeAfterTextEdge;

        // recalculate mean value
        meanValueSum += deviationColumnSum;
        meanValueSum -= deviationColumnSumBeforeStart;
        int nextMeanValue = meanValueSum / endDetectionCorrelationFunctionLength;

        // recalculate correlation sum with next mean value
        areaCorrelationSum -= (nextMeanValue - meanValue) * (textEdge * amplitudeBeforeTextEdge +
            (endDetectionCorrelationFunctionLength - textEdge) * amplitudeAfterTextEdge);

        // write area correlation sum to reducedGrayAllocation
        rsSetElementAt_int(reducedGrayAllocation, areaCorrelationSum,
            i - endDetectionCorrelationFunctionLength + 1, 0);

        // calculate max correlation and max correlation index
        if (maxCorrelation < areaCorrelationSum) {
            maxCorrelation = areaCorrelationSum;
            maxCorrelationStartX = i - endDetectionCorrelationFunctionLength + 1;
        };

        // set mean value for the next loop
        meanValue = nextMeanValue;
    };

    // search for up to 10 highest local maximums in calculated end correlation
    int previousCorrelation;
    int currentCorrelation;
    int nextCorrelation;
    int previousSlope;
    int nextSlope;
    int raisingSlopeCount;
    int fallingSlopeCount;
    int CORRELATION_LOCAL_MAX_ARRAY_SIZE = 10;
    int correlationLocalMaxArray[CORRELATION_LOCAL_MAX_ARRAY_SIZE][2];
    int correlationLocalMaxArrayLastIndex = 0;

    for (
        int i = xTopStart + localMaxCalculationPadding;
        i < xTopEnd - endDetectionCorrelationFunctionLength - localMaxCalculationPadding;
        ++i
    ) {
        previousCorrelation = rsGetElementAt_int(reducedGrayAllocation, i - 1, 0);
        currentCorrelation = rsGetElementAt_int(reducedGrayAllocation, i, 0);
        nextCorrelation = rsGetElementAt_int(reducedGrayAllocation, i + 1, 0);
        previousSlope = currentCorrelation - previousCorrelation;
        nextSlope = nextCorrelation - currentCorrelation;

        if (previousSlope > 0 && nextSlope < 0) { // if found local max
            raisingSlopeCount = 1;
            fallingSlopeCount = 1;
            for (int j = i - 1; j > i - vicinityHalfWidth; --j) {
                int slope = rsGetElementAt_int(reducedGrayAllocation, j, 0) -
                    rsGetElementAt_int(reducedGrayAllocation, j - 1, 0);
                if (slope >= 0) {
                    ++raisingSlopeCount;
                } else {
                    break;
                };
            };
            for (int j = i + 1; j < i + vicinityHalfWidth; ++j) {
                int slope = rsGetElementAt_int(reducedGrayAllocation, j + 1, 0) -
                    rsGetElementAt_int(reducedGrayAllocation, j, 0);
                if (slope <= 0) {
                    ++fallingSlopeCount;
                } else {
                    break;
                };
            };

            // write the local max to correlationLocalMaxArray in descending order if it meets conditions
            if (raisingSlopeCount >= vicinityHalfWidth && fallingSlopeCount >= vicinityHalfWidth) {
                if (correlationLocalMaxArrayLastIndex == 0) {
                    correlationLocalMaxArray[0][0] = currentCorrelation;
                    correlationLocalMaxArray[0][1] = i; // position in columnSumAllocation
                    ++correlationLocalMaxArrayLastIndex;
                } else {
                    if (correlationLocalMaxArrayLastIndex < CORRELATION_LOCAL_MAX_ARRAY_SIZE) {
                        int j = correlationLocalMaxArrayLastIndex;
                        while (j > 0) {
                            if (correlationLocalMaxArray[j - 1][0] < currentCorrelation) {
                                correlationLocalMaxArray[j][0] = correlationLocalMaxArray[j - 1][0];
                                correlationLocalMaxArray[j][1] = correlationLocalMaxArray[j - 1][1];
                            } else {
                                break;
                            };
                            --j;
                        };
                        correlationLocalMaxArray[j][0] = currentCorrelation;
                        correlationLocalMaxArray[j][1] = i; // position in columnSumAllocation

                        ++correlationLocalMaxArrayLastIndex;
                    } else {  // correlationLocalMaxArrayLastIndex = CORRELATION_LOCAL_MAX_ARRAY_SIZE
                        if (correlationLocalMaxArray[CORRELATION_LOCAL_MAX_ARRAY_SIZE - 1][0] < currentCorrelation) {
                            int j = CORRELATION_LOCAL_MAX_ARRAY_SIZE - 1;
                            while (j > 0) {
                                if (correlationLocalMaxArray[j - 1][0] < currentCorrelation) {
                                    correlationLocalMaxArray[j][0] = correlationLocalMaxArray[j - 1][0];
                                    correlationLocalMaxArray[j][1] = correlationLocalMaxArray[j - 1][1];
                                } else {
                                    break;
                                };
                                --j;
                            };
                            correlationLocalMaxArray[j][0] = currentCorrelation;
                            correlationLocalMaxArray[j][1] = i; // position in columnSumAllocation
                        };
                    };
                };
            };

            // if local max was detected, then next local max search should be started on minimal
            // distance from the detected local max (raising slope can only begin after the end of
            // the falling slope, and after that there is a vicinityHalfWidth of minimal raising
            // slope duration)
            i += fallingSlopeCount + vicinityHalfWidth;
        };
    };

    // initialize default detected end position
    int stripEndPosition = xTopStart + localMaxCalculationPadding + textEdge;

    // if there are detected local maximums
    if (correlationLocalMaxArrayLastIndex > 0) {
        // override default detected end position with position of largest local maximum
        stripEndPosition = correlationLocalMaxArray[0][1] + textEdge;

        // iterate over local maximums to check if they meet conditions of text edge
        for (int j = 0; j < correlationLocalMaxArrayLastIndex; ++j) {
            // position in reducedColumnSumAllocation for column sum corresponding to the
            // calculated text edge position of the local maximum j in correlationLocalMaxArray
            int currentLocalMaxTextEdgePosition = correlationLocalMaxArray[j][1] + textEdge;
            int meanSum = 0;
            for (int i = currentLocalMaxTextEdgePosition - localMaxAreaMeanHalfWidth;
                i < currentLocalMaxTextEdgePosition + localMaxAreaMeanHalfWidth;
                ++i
            ) {
                meanSum += rsGetElementAt_int(reducedColumnSumAllocation, i, 0);
            };

            // initialize checkedSumPosition
            int checkedSumPosition = correlationLocalMaxArray[j][1] + directionSign * localMaxCalculationPadding;

            // determine if there is a local max, that is closer to the currentLocalMaxTextEdgePosition
            // than the initialized checkedSumPosition
            for (int i = 0; i < correlationLocalMaxArrayLastIndex; ++i) {
                if (i != j &&
                    ((correlationLocalMaxArray[j][1] - correlationLocalMaxArray[i][1]) *
                    (correlationLocalMaxArray[i][1] - checkedSumPosition) > 0)
                ) {
                    checkedSumPosition = correlationLocalMaxArray[i][1];
                };
            };
            checkedSumPosition += textEdge; // offset checkedSumPosition to the text edge

            int checkedSum = 0;
            for (int i = checkedSumPosition - ((1 + directionSign) / 2) * localMaxAreaMeanHalfWidth;
                i < checkedSumPosition + ((1 - directionSign) / 2) * localMaxAreaMeanHalfWidth;
                ++i
            ) {
                checkedSum += rsGetElementAt_int(reducedColumnSumAllocation, i, 0);
            };

            // override detected end position with position of a local maximum, that meets conditions
            if (meanSum < 2 * checkedSum) {
                stripEndPosition = correlationLocalMaxArray[j][1] + textEdge;
                break;
            };
        };
    };

    // write point of detected strip end to detectedLineEndsX and detectedLineEndsY in nonreduced format
    detectedLineEndsX[y] = stripEndPosition * meanPullKernelWidth;
    detectedLineEndsY[y] = (detectedLineY + ((stripEndPosition - detectedLineX) *
        detectedLineAngleY) / detectedLineAngleX) * meanPullKernelWidth;

    // write detected end x coordinate to fbAllocation in nonreduced format
    rsSetElementAt_int(fbAllocation, detectedLineEndsX[y], y, 0);
}

void RS_KERNEL endMarkerDrawKernel(int in, uint32_t x, uint32_t y) {
    uchar4 out;
    int directionSign = rsGetElementAt_int(endDetectionCorrelationFunctionAllocation,
        endDetectionDirectionSignColumn, y);

    // calculate marker origin
    int xMarkerOrigin = detectedLineEndsX[y] + yuvToGrayKernelShiftX - yuvToOutKernelShiftX;
    int yMarkerOrigin = detectedLineEndsY[y] + yuvToGrayKernelShiftY - yuvToOutKernelShiftY;

    // set marker color
    out[0] = clamp(colorOfMarkerIncludedInLine[0], 0, 255);
    out[1] = clamp(colorOfMarkerIncludedInLine[1], 0, 255);
    out[2] = clamp(colorOfMarkerIncludedInLine[2], 0, 255);
    out[3] = clamp(colorOfMarkerIncludedInLine[3], 0, 255);

    // draw horizontal top dash
    drawRectangleWithAllocationBordesCheck_uchar4(
        &outAllocation,
        out,
        xMarkerOrigin - endMarkerDashThickness,
        xMarkerOrigin + endMarkerDashThickness,
        yMarkerOrigin - endMarkerDashThickness,
        yMarkerOrigin
    );
    // draw vertical dash
    int yMarkerBottomDashStart = yMarkerOrigin + textStripBottomEdge - textStripTopEdge;
    drawRectangleWithAllocationBordesCheck_uchar4(
        &outAllocation,
        out,
        xMarkerOrigin - ((1 + directionSign) / 2) * endMarkerDashThickness,
        xMarkerOrigin + ((1 - directionSign) / 2) * endMarkerDashThickness,
        yMarkerOrigin,
        yMarkerBottomDashStart
    );
    // draw horizontal bottom dash
    drawRectangleWithAllocationBordesCheck_uchar4(
        &outAllocation,
        out,
        xMarkerOrigin - endMarkerDashThickness,
        xMarkerOrigin + endMarkerDashThickness,
        yMarkerBottomDashStart,
        yMarkerBottomDashStart + endMarkerDashThickness
    );
}


static void drawRectangleWithAllocationBordesCheck_uchar4(
    rs_allocation *allocation,
    uchar4 value,
    int xStart,
    int xEnd,
    int yStart,
    int yEnd
) {
    for (int xCoordinate = xStart; xCoordinate < xEnd; ++xCoordinate) {
        for (int yCoordinate = yStart; yCoordinate < yEnd; ++yCoordinate) {
            setElementWithAllocationBordesCheck_uchar4(allocation, value, xCoordinate, yCoordinate);
        };
    };
};

static void setElementWithAllocationBordesCheck_uchar4(
    rs_allocation *allocation,
    uchar4 value,
    int xCoordinate,
    int yCoordinate
) {
    // check if pixel coordinates are within allocation borders
    if (
        (xCoordinate >= 0) &&
        (xCoordinate < rsAllocationGetDimX(*allocation)) &&
        (yCoordinate >= 0) &&
        (yCoordinate < rsAllocationGetDimY(*allocation))
    ) {
        rsSetElementAt_uchar4(*allocation, value, xCoordinate, yCoordinate);
    };
};

calculateStripCorrelation(
    rs_allocation *inputAllocation,
    rs_allocation *correlationCoreAllocation,
    int xCoordinate,
    int yStart,
    int yLength,
    int stripHalfWidth,
    float maxAmplitudeToMeanRatio
) {
    int sum = 0;
    int mean;
    int meanAmplitude;
    int numberOfSumElements = ((2 * stripHalfWidth + 1) * yLength);

    // calculate mean value on input local deviation strip
    int jStart = xCoordinate - stripHalfWidth;
    int jStop = xCoordinate + stripHalfWidth;
    for (int i = 0; i < yLength; ++i) {
        for (int j = jStart; j <= jStop; ++j) {
            sum += rsGetElementAt_int(*inputAllocation, j, yStart + i);
        };
    };
    mean = sum / numberOfSumElements;

    // calculate mean amplitude of input local deviation strip with subtracted mean value
    sum = 0;
    for (int i = 0; i < yLength; ++i) {
        for (int j = jStart; j <= jStop; ++j) {
            sum += abs(rsGetElementAt_int(*inputAllocation, j, yStart + i) - mean);
        };
    };
    meanAmplitude = sum / numberOfSumElements;

    // calculate correlation (sum all limited pixels multiplied by corresponding correlation function pixels)
    sum = 0;
    float amplitudeLimit = maxAmplitudeToMeanRatio * ((float) meanAmplitude);
    for (int i = 0; i < yLength; ++i) {
        for (int j = jStart; j <= jStop; ++j) {
            int inputValue = rsGetElementAt_int(*inputAllocation, j, yStart + i) - mean;
            // limit abs() of input value by mean amplitude multiplied by ratio of max amplitude of
            // correlation function to mean value of correlation function
            if (((float) abs(inputValue)) > amplitudeLimit) {
                inputValue = clamp(inputValue, (int) (-amplitudeLimit), (int) amplitudeLimit);
            };
            // add correlation of this limited pixel to sum
            sum += inputValue * rsGetElementAt_int(*correlationCoreAllocation, 0, i);
        };
    };

    // return correlation
    return sum / numberOfSumElements;
};

calculateLocalDeviation(
    rs_allocation *allocation,
    uint32_t xCoordinate,
    uint32_t yCoordinate,
    int kernelHalfWidth,
    int kernelWidth,
    int numberOfSumElements,
    int mean
) {
    int sum = 0;
    for (
        int i = xCoordinate - kernelHalfWidth;
        i <= xCoordinate + kernelHalfWidth;
        ++i
    ) {
        for (
            int j = yCoordinate - kernelHalfWidth;
            j <= yCoordinate + kernelHalfWidth;
            ++j
        ) {
            sum += pown(rsGetElementAt_int(*allocation, i, j) - mean, 2);
        }
    }
    return (int) sqrt((float) (sum / numberOfSumElements));
};

