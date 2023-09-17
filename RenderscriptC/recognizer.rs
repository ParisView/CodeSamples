

#pragma version(1)

#pragma rs java_package_name(com.code.sample.readmrz)


// allocations
rs_allocation fbAllocation;
rs_allocation detectedCharactersAllocation;
rs_allocation recognizerTextStripAllocation;
rs_allocation inputSizesAllocation;
rs_allocation pasteShiftsAllocation;
rs_allocation recognizerNormalizedInputAllocation;

const int RECOGNIZER_STAGE1_NUMBER_OF_FEATURES = 16;
const int RECOGNIZER_L1_NUMBER_OF_FEATURES = RECOGNIZER_STAGE1_NUMBER_OF_FEATURES;
rs_allocation recognizerL1OutAllocations[RECOGNIZER_L1_NUMBER_OF_FEATURES];
rs_allocation recognizerL1WeightsAllocations[RECOGNIZER_L1_NUMBER_OF_FEATURES];

rs_allocation recognizerL1BiasAllocation;
rs_allocation recognizerL1BatchNormAllocation;

const int RECOGNIZER_L2_NUMBER_OF_FEATURES = RECOGNIZER_STAGE1_NUMBER_OF_FEATURES;
rs_allocation recognizerL2OutAllocations[RECOGNIZER_L2_NUMBER_OF_FEATURES];
rs_allocation recognizerL2WeightsAllocations[RECOGNIZER_L1_NUMBER_OF_FEATURES * RECOGNIZER_L2_NUMBER_OF_FEATURES];

rs_allocation recognizerL2BiasAllocation;
rs_allocation recognizerL2BatchNormAllocation;

const int RECOGNIZER_L3_NUMBER_OF_FEATURES = RECOGNIZER_STAGE1_NUMBER_OF_FEATURES;
rs_allocation recognizerL3OutAllocations[RECOGNIZER_L3_NUMBER_OF_FEATURES];
rs_allocation recognizerL3WeightsAllocations[RECOGNIZER_L2_NUMBER_OF_FEATURES * RECOGNIZER_L3_NUMBER_OF_FEATURES];

rs_allocation recognizerL3BiasAllocation;
rs_allocation recognizerL3BatchNormAllocation;

const int RECOGNIZER_L4_NUMBER_OF_FEATURES = 32;
rs_allocation recognizerL4OutAllocations[RECOGNIZER_L4_NUMBER_OF_FEATURES];
rs_allocation recognizerL4WeightsAllocations[RECOGNIZER_L3_NUMBER_OF_FEATURES * RECOGNIZER_L4_NUMBER_OF_FEATURES];

rs_allocation recognizerL4BiasAllocation;
rs_allocation recognizerL4BatchNormAllocation;

const int RECOGNIZER_L5_NUMBER_OF_FEATURES = RECOGNIZER_L4_NUMBER_OF_FEATURES;
rs_allocation recognizerL5MaxPoolAllocations[RECOGNIZER_L5_NUMBER_OF_FEATURES];

const int RECOGNIZER_L6_NUMBER_OF_FEATURES = 32;
rs_allocation recognizerL6OutAllocations[RECOGNIZER_L6_NUMBER_OF_FEATURES];
rs_allocation recognizerL6WeightsAllocations[RECOGNIZER_L5_NUMBER_OF_FEATURES * RECOGNIZER_L6_NUMBER_OF_FEATURES];

rs_allocation recognizerL6BiasAllocation;
rs_allocation recognizerL6BatchNormAllocation;

const int RECOGNIZER_L7_NUMBER_OF_FEATURES = 64;
rs_allocation recognizerL7OutAllocations[RECOGNIZER_L7_NUMBER_OF_FEATURES];
rs_allocation recognizerL7WeightsAllocations[RECOGNIZER_L6_NUMBER_OF_FEATURES * RECOGNIZER_L7_NUMBER_OF_FEATURES];

rs_allocation recognizerL7BiasAllocation;
rs_allocation recognizerL7BatchNormAllocation;

const int RECOGNIZER_L8_NUMBER_OF_FEATURES = RECOGNIZER_L7_NUMBER_OF_FEATURES;
rs_allocation recognizerL8MaxPoolAllocations[RECOGNIZER_L8_NUMBER_OF_FEATURES];

const int RECOGNIZER_L9_NUMBER_OF_FEATURES = 128;
rs_allocation recognizerL9OutAllocation;
rs_allocation recognizerL9WeightsAllocations[RECOGNIZER_L8_NUMBER_OF_FEATURES * RECOGNIZER_L9_NUMBER_OF_FEATURES];

rs_allocation recognizerL9BiasAllocation;
rs_allocation recognizerL9BatchNormAllocation;
rs_allocation recognizerL10OutAllocation;
rs_allocation recognizerL10WeightsAllocation;
rs_allocation recognizerL10BiasAllocation;
rs_allocation recognizerL11OutAllocation;
rs_allocation recognizerL11WeightsAllocation;
rs_allocation recognizerL11BiasAllocation;

// feedback input parameters
int fbAllocationCharacterClassColumn;

// inputSizesCalculationKernel input parameters
int characterStartRow;
int characterEndRow;
int characterLowestPointRow;
int characterHighestPointRow;
int pasteShiftXStartRow;
int pasteShiftXEndRow;
int pasteShiftYStartRow;
int pasteShiftYEndRow;
int inputSizesCalculationThresholdWH1x1;
int inputSizesCalculationStepWH;

// recognizerInputNormalizationKernel input parameters
int characterIndex;
float recognizerCharacterImageMean;
float recognizerCharacterImageSTDReciprocal;

// recognizerL1ConvolutionKernel input parameters
int recognizerConvolutionKernelWidth;
int recognizerConvolutionOutputShift;
int recognizerBatchNormEffectiveWeightRow;
int recognizerBatchNormEffectiveBiasRow;

// recognizerL9ConvolutionKernel input parameters
int recognizerL9MaxPoolWidth;
int recognizerL9MaxPoolHeight;

// recognizerL10LinearKernel input parameters
int recognizerL10NumberOfInputs;

// recognizerL11LinearKernel input parameters
int recognizerL11NumberOfInputs;

// recognizerPredictionKernel input parameters
int recognizerPredictionNumberOfInputs;


void RS_KERNEL inputSizesCalculationKernel(int in, uint32_t x, uint32_t y) {
    // calculate width and height of character cutout
    int cutoutWidth = rsGetElementAt_int(detectedCharactersAllocation, x, characterEndRow) -
        rsGetElementAt_int(detectedCharactersAllocation, x, characterStartRow);
    int cutoutHeight = rsGetElementAt_int(detectedCharactersAllocation, x, characterHighestPointRow) -
        rsGetElementAt_int(detectedCharactersAllocation, x, characterLowestPointRow);

    // calculate width and height of allocation, where character cutout will be pasted
    int calculatedWidth;
    int calculatedHeight;
    if (cutoutWidth <= inputSizesCalculationThresholdWH1x1) {
        calculatedWidth = inputSizesCalculationThresholdWH1x1;
    } else {
        calculatedWidth = cutoutWidth;
        int reminder = (cutoutWidth - inputSizesCalculationThresholdWH1x1) % inputSizesCalculationStepWH;
        if (reminder > 0) {
            calculatedWidth += inputSizesCalculationStepWH - reminder;
        };
    };
    if (cutoutHeight <= inputSizesCalculationThresholdWH1x1) {
        calculatedHeight = inputSizesCalculationThresholdWH1x1;
    } else {
        calculatedHeight = cutoutHeight;
        int reminder = (cutoutHeight - inputSizesCalculationThresholdWH1x1) % inputSizesCalculationStepWH;
        if (reminder > 0) {
            calculatedHeight += inputSizesCalculationStepWH - reminder;
        };
    };

    // write calculated width and height to inputSizesAllocation
    rsSetElementAt_int(inputSizesAllocation, calculatedWidth, x, 0);
    rsSetElementAt_int(inputSizesAllocation, calculatedHeight, x, 1);

    // calculate shifts of start and end along x and y, that will be used when pasting character cutout
    int pasteShiftXStart = (calculatedWidth - cutoutWidth) / 2;
    int pasteShiftXEnd = pasteShiftXStart + cutoutWidth;
    int pasteShiftYStart = (calculatedHeight - cutoutHeight) / 2;
    int pasteShiftYEnd = pasteShiftYStart + cutoutHeight;

    // write calculated shifts to pasteShiftsAllocation
    rsSetElementAt_int(pasteShiftsAllocation, pasteShiftXStart, x, pasteShiftXStartRow);
    rsSetElementAt_int(pasteShiftsAllocation, pasteShiftXEnd, x, pasteShiftXEndRow);
    rsSetElementAt_int(pasteShiftsAllocation, pasteShiftYStart, x, pasteShiftYStartRow);
    rsSetElementAt_int(pasteShiftsAllocation, pasteShiftYEnd, x, pasteShiftYEndRow);
}

void RS_KERNEL recognizerInputNormalizationKernel(float in, uint32_t x, uint32_t y) {
    float pixelValue;

    // load character cutout borders and paste shifts
    int cutoutShiftX = rsGetElementAt_int(detectedCharactersAllocation, characterIndex, characterStartRow);
    int cutoutShiftY = rsGetElementAt_int(detectedCharactersAllocation, characterIndex, characterLowestPointRow);
    int pasteShiftXStart = rsGetElementAt_int(pasteShiftsAllocation, characterIndex, pasteShiftXStartRow);
    int pasteShiftXEnd = rsGetElementAt_int(pasteShiftsAllocation, characterIndex, pasteShiftXEndRow);
    int pasteShiftYStart = rsGetElementAt_int(pasteShiftsAllocation, characterIndex, pasteShiftYStartRow);
    int pasteShiftYEnd = rsGetElementAt_int(pasteShiftsAllocation, characterIndex, pasteShiftYEndRow);

    // image in recognizerNormalizedInputAllocation is padded with 0 outside character cutout
    if (x < pasteShiftXStart || y < pasteShiftYStart) {
        pixelValue = 0.0;
    } else {
        if (x >= pasteShiftXEnd || y >= pasteShiftYEnd) {
            pixelValue = 0.0;
        } else {
            pixelValue = rsGetElementAt_float(recognizerTextStripAllocation,
                x - pasteShiftXStart + cutoutShiftX, y - pasteShiftYStart + cutoutShiftY);
        };
    };

    // normalize and write pixel value to recognizerNormalizedInputAllocation
    rsSetElementAt_float(recognizerNormalizedInputAllocation, (pixelValue - recognizerCharacterImageMean
        ) * recognizerCharacterImageSTDReciprocal, x, y);
}

void RS_KERNEL recognizerStage1PadWithZeroKernel(float in, uint32_t x, uint32_t y) {
    // for each output feature map write 0 to pixel with coordinates x and y
    for (int n = 0; n < RECOGNIZER_STAGE1_NUMBER_OF_FEATURES; ++n) {
        rsSetElementAt_float(recognizerL1OutAllocations[n], 0.0, x, y);
        rsSetElementAt_float(recognizerL2OutAllocations[n], 0.0, x, y);
        rsSetElementAt_float(recognizerL3OutAllocations[n], 0.0, x, y);
    };
}

void RS_KERNEL recognizerL1ConvolutionKernel(float in, uint32_t x, uint32_t y) {
    float correlation;

    // for each output feature map calculate convolution for pixel with coordinates x and y
    for (int n = 0; n < RECOGNIZER_L1_NUMBER_OF_FEATURES; ++n) {
        // calculate correlation of input and convolution kernel
        correlation = 0.0;
        for (int i = 0; i < recognizerConvolutionKernelWidth; ++i) {
            for (int j = 0; j < recognizerConvolutionKernelWidth; ++j) {
                correlation += rsGetElementAt_float(recognizerNormalizedInputAllocation, x + j, y + i) *
                    rsGetElementAt_float(recognizerL1WeightsAllocations[n], j, i);
            };
        };
        // add bias to correlation
        correlation += rsGetElementAt_float(recognizerL1BiasAllocation, n, 0);
        // multiply correlation by batch norm effective weight
        correlation *= rsGetElementAt_float(
            recognizerL1BatchNormAllocation, n, recognizerBatchNormEffectiveWeightRow);
        // add batch norm effective bias to correlation
        correlation += rsGetElementAt_float(
            recognizerL1BatchNormAllocation, n, recognizerBatchNormEffectiveBiasRow);
        // apply ReLU function to correlation
        if (correlation < 0.0) correlation = 0.0;
        // write correlation to recognizerL1OutAllocations shifted due to zero padding on next layer
        rsSetElementAt_float(recognizerL1OutAllocations[n], correlation,
            x + recognizerConvolutionOutputShift, y + recognizerConvolutionOutputShift);
    };
}

void RS_KERNEL recognizerL2ConvolutionKernel(float in, uint32_t x, uint32_t y) {
    float correlation;

    // for each output feature map calculate sum of convolutions for pixel with coordinates x and y
    for (int n = 0; n < RECOGNIZER_L2_NUMBER_OF_FEATURES; ++n) {
        correlation = 0;
        // for each input channel
        for (int k = 0; k < RECOGNIZER_L1_NUMBER_OF_FEATURES; ++k) {
            int weightsAllocationIndex = n * RECOGNIZER_L1_NUMBER_OF_FEATURES + k;
            // calculate correlation of input and convolution kernel
            for (int i = 0; i < recognizerConvolutionKernelWidth; ++i) {
                for (int j = 0; j < recognizerConvolutionKernelWidth; ++j) {
                    correlation += rsGetElementAt_float(recognizerL1OutAllocations[k], x + j, y + i) *
                        rsGetElementAt_float(recognizerL2WeightsAllocations[weightsAllocationIndex], j, i);
                };
            };
        };
        // add bias to correlation
        correlation += rsGetElementAt_float(recognizerL2BiasAllocation, n, 0);
        // multiply correlation by batch norm effective weight
        correlation *= rsGetElementAt_float(
            recognizerL2BatchNormAllocation, n, recognizerBatchNormEffectiveWeightRow);
        // add batch norm effective bias to correlation
        correlation += rsGetElementAt_float(
            recognizerL2BatchNormAllocation, n, recognizerBatchNormEffectiveBiasRow);
        // apply ReLU function to correlation
        if (correlation < 0) correlation = 0;
        // write correlation to recognizerL2OutAllocations shifted due to zero padding on next layer
        rsSetElementAt_float(recognizerL2OutAllocations[n], correlation,
            x + recognizerConvolutionOutputShift, y + recognizerConvolutionOutputShift);
    };
}

void RS_KERNEL recognizerL3ConvolutionKernel(float in, uint32_t x, uint32_t y) {
    float correlation;

    // for each output feature map calculate sum of convolutions for pixel with coordinates x and y
    for (int n = 0; n < RECOGNIZER_L3_NUMBER_OF_FEATURES; ++n) {
        correlation = 0;
        // for each input channel
        for (int k = 0; k < RECOGNIZER_L2_NUMBER_OF_FEATURES; ++k) {
            int weightsAllocationIndex = n * RECOGNIZER_L2_NUMBER_OF_FEATURES + k;
            // calculate correlation of input and convolution kernel
            for (int i = 0; i < recognizerConvolutionKernelWidth; ++i) {
                for (int j = 0; j < recognizerConvolutionKernelWidth; ++j) {
                    correlation += rsGetElementAt_float(recognizerL2OutAllocations[k], x + j, y + i) *
                        rsGetElementAt_float(recognizerL3WeightsAllocations[weightsAllocationIndex], j, i);
                };
            };
        };
        // add bias to correlation
        correlation += rsGetElementAt_float(recognizerL3BiasAllocation, n, 0);
        // multiply correlation by batch norm effective weight
        correlation *= rsGetElementAt_float(
            recognizerL3BatchNormAllocation, n, recognizerBatchNormEffectiveWeightRow);
        // add batch norm effective bias to correlation
        correlation += rsGetElementAt_float(
            recognizerL3BatchNormAllocation, n, recognizerBatchNormEffectiveBiasRow);
        // apply ReLU function to correlation
        if (correlation < 0) correlation = 0;
        // write correlation to recognizerL3OutAllocations shifted due to zero padding on next layer
        rsSetElementAt_float(recognizerL3OutAllocations[n], correlation,
            x + recognizerConvolutionOutputShift, y + recognizerConvolutionOutputShift);
    };
}

void RS_KERNEL recognizerL4ConvolutionKernel(float in, uint32_t x, uint32_t y) {
    float correlation;

    // for each output feature map calculate sum of convolutions for pixel with coordinates x and y
    for (int n = 0; n < RECOGNIZER_L4_NUMBER_OF_FEATURES; ++n) {
        correlation = 0;
        // for each input channel
        for (int k = 0; k < RECOGNIZER_L3_NUMBER_OF_FEATURES; ++k) {
            int weightsAllocationIndex = n * RECOGNIZER_L3_NUMBER_OF_FEATURES + k;
            // calculate correlation of input and convolution kernel
            for (int i = 0; i < recognizerConvolutionKernelWidth; ++i) {
                for (int j = 0; j < recognizerConvolutionKernelWidth; ++j) {
                    correlation += rsGetElementAt_float(recognizerL3OutAllocations[k], x + j, y + i) *
                        rsGetElementAt_float(recognizerL4WeightsAllocations[weightsAllocationIndex], j, i);
                };
            };
        };

        // bias and batch norm from this layer are carried to the next layer calculation for optimization

        // write correlation to recognizerL4OutAllocations
        rsSetElementAt_float(recognizerL4OutAllocations[n], correlation, x, y);
    };
}

void RS_KERNEL recognizerL5MaxPoolKernel(float in, uint32_t x, uint32_t y) {
    // for each output feature map calculate max value of 2 by 2 pixel area in recognizerL4OutAllocations
    int inputX = 2 * x;
    int inputY = 2 * y;
    for (int n = 0; n < RECOGNIZER_L5_NUMBER_OF_FEATURES; ++n) {
        float maxValue = fmax(
            rsGetElementAt_float(recognizerL4OutAllocations[n], inputX, inputY),
            rsGetElementAt_float(recognizerL4OutAllocations[n], inputX + 1, inputY)
        );
        maxValue = fmax(maxValue, rsGetElementAt_float(recognizerL4OutAllocations[n], inputX, inputY + 1));
        maxValue = fmax(maxValue, rsGetElementAt_float(recognizerL4OutAllocations[n], inputX + 1, inputY + 1));

        // add bias (from previous layer for optimization) to maxValue
        maxValue += rsGetElementAt_float(recognizerL4BiasAllocation, n, 0);
        // multiply maxValue by batch norm effective weight (from previous layer for optimization)
        maxValue *= rsGetElementAt_float(
            recognizerL4BatchNormAllocation, n, recognizerBatchNormEffectiveWeightRow);
        // add batch norm effective bias to maxValue (from previous layer for optimization)
        maxValue += rsGetElementAt_float(
            recognizerL4BatchNormAllocation, n, recognizerBatchNormEffectiveBiasRow);
        // apply ReLU function to maxValue (from previous layer for optimization)
        if (maxValue < 0) maxValue = 0;

        // write calculated max value to recognizerL5MaxPoolAllocations
        rsSetElementAt_float(recognizerL5MaxPoolAllocations[n], maxValue, x, y);
    };
}

void RS_KERNEL recognizerL6PadWithZeroKernel(float in, uint32_t x, uint32_t y) {
    // for each output feature map write 0 to pixel with coordinates x and y
    for (int n = 0; n < RECOGNIZER_L6_NUMBER_OF_FEATURES; ++n) {
        rsSetElementAt_float(recognizerL6OutAllocations[n], 0.0, x, y);
    };
}

void RS_KERNEL recognizerL6ConvolutionKernel(float in, uint32_t x, uint32_t y) {
    float correlation;

    // for each output feature map calculate sum of convolutions for pixel with coordinates x and y
    for (int n = 0; n < RECOGNIZER_L6_NUMBER_OF_FEATURES; ++n) {
        correlation = 0;
        // for each input channel
        for (int k = 0; k < RECOGNIZER_L5_NUMBER_OF_FEATURES; ++k) {
            int weightsAllocationIndex = n * RECOGNIZER_L5_NUMBER_OF_FEATURES + k;
            // calculate correlation of input and convolution kernel
            for (int i = 0; i < recognizerConvolutionKernelWidth; ++i) {
                for (int j = 0; j < recognizerConvolutionKernelWidth; ++j) {
                    correlation += rsGetElementAt_float(recognizerL5MaxPoolAllocations[k], x + j, y + i) *
                        rsGetElementAt_float(recognizerL6WeightsAllocations[weightsAllocationIndex], j, i);
                };
            };
        };
        // add bias to correlation
        correlation += rsGetElementAt_float(recognizerL6BiasAllocation, n, 0);
        // multiply correlation by batch norm effective weight
        correlation *= rsGetElementAt_float(
            recognizerL6BatchNormAllocation, n, recognizerBatchNormEffectiveWeightRow);
        // add batch norm effective bias to correlation
        correlation += rsGetElementAt_float(
            recognizerL6BatchNormAllocation, n, recognizerBatchNormEffectiveBiasRow);
        // apply ReLU function to correlation
        if (correlation < 0) correlation = 0;
        // write correlation to recognizerL6OutAllocations shifted due to zero padding on next layer
        rsSetElementAt_float(recognizerL6OutAllocations[n], correlation,
            x + recognizerConvolutionOutputShift, y + recognizerConvolutionOutputShift);
    };
}

void RS_KERNEL recognizerL7ConvolutionKernel(float in, uint32_t x, uint32_t y) {
    float correlation;

    // for each output feature map calculate sum of convolutions for pixel with coordinates x and y
    for (int n = 0; n < RECOGNIZER_L7_NUMBER_OF_FEATURES; ++n) {
        correlation = 0;
        // for each input channel
        for (int k = 0; k < RECOGNIZER_L6_NUMBER_OF_FEATURES; ++k) {
            int weightsAllocationIndex = n * RECOGNIZER_L6_NUMBER_OF_FEATURES + k;
            // calculate correlation of input and convolution kernel
            for (int i = 0; i < recognizerConvolutionKernelWidth; ++i) {
                for (int j = 0; j < recognizerConvolutionKernelWidth; ++j) {
                    correlation += rsGetElementAt_float(recognizerL6OutAllocations[k], x + j, y + i) *
                        rsGetElementAt_float(recognizerL7WeightsAllocations[weightsAllocationIndex], j, i);
                };
            };
        };

        // bias and batch norm from this layer are carried to the next layer calculation for optimization

        // write correlation to recognizerL7OutAllocations
        rsSetElementAt_float(recognizerL7OutAllocations[n], correlation, x, y);
    };
}

void RS_KERNEL recognizerL8MaxPoolKernel(float in, uint32_t x, uint32_t y) {
    // for each output feature map calculate max value of 2 by 2 pixel area in recognizerL7OutAllocations
    int inputX = 2 * x;
    int inputY = 2 * y;
    for (int n = 0; n < RECOGNIZER_L8_NUMBER_OF_FEATURES; ++n) {
        float maxValue = fmax(
            rsGetElementAt_float(recognizerL7OutAllocations[n], inputX, inputY),
            rsGetElementAt_float(recognizerL7OutAllocations[n], inputX + 1, inputY)
        );
        maxValue = fmax(maxValue, rsGetElementAt_float(recognizerL7OutAllocations[n], inputX, inputY + 1));
        maxValue = fmax(maxValue, rsGetElementAt_float(recognizerL7OutAllocations[n], inputX + 1, inputY + 1));

        // add bias (from previous layer for optimization) to maxValue
        maxValue += rsGetElementAt_float(recognizerL7BiasAllocation, n, 0);
        // multiply maxValue by batch norm effective weight (from previous layer for optimization)
        maxValue *= rsGetElementAt_float(
            recognizerL7BatchNormAllocation, n, recognizerBatchNormEffectiveWeightRow);
        // add batch norm effective bias to maxValue (from previous layer for optimization)
        maxValue += rsGetElementAt_float(
            recognizerL7BatchNormAllocation, n, recognizerBatchNormEffectiveBiasRow);
        // apply ReLU function to maxValue (from previous layer for optimization)
        if (maxValue < 0) maxValue = 0;

        // write calculated max value to recognizerL8MaxPoolAllocations
        rsSetElementAt_float(recognizerL8MaxPoolAllocations[n], maxValue, x, y);
    };
}

void RS_KERNEL recognizerL9ConvolutionKernel(float in, uint32_t x, uint32_t y) {
    float correlation;
    float maxValue;

    // calculate sum of convolutions for pixel with coordinates x and y and calculate their max value
    for (int maxPoolX = 0; maxPoolX < recognizerL9MaxPoolWidth; ++maxPoolX) {
        for (int maxPoolY = 0; maxPoolY < recognizerL9MaxPoolHeight; ++maxPoolY) {
            correlation = 0;
            // for each input channel
            for (int k = 0; k < RECOGNIZER_L8_NUMBER_OF_FEATURES; ++k) {
                int weightsAllocationIndex = x * RECOGNIZER_L8_NUMBER_OF_FEATURES + k;
                // calculate correlation of input and convolution kernel
                for (int i = 0; i < recognizerConvolutionKernelWidth; ++i) {
                    for (int j = 0; j < recognizerConvolutionKernelWidth; ++j) {
                        correlation += rsGetElementAt_float(recognizerL8MaxPoolAllocations[k],
                            maxPoolX + j, maxPoolY + i) *
                            rsGetElementAt_float(recognizerL9WeightsAllocations[weightsAllocationIndex], j, i);
                    };
                };
            };
            // calculate max value
            if (maxPoolX == 0 && maxPoolY == 0) {
                maxValue = correlation;
            } else {
                if (maxValue < correlation) {maxValue = correlation;};
            };
        };
    };
    // add bias to maxValue
    maxValue += rsGetElementAt_float(recognizerL9BiasAllocation, x, 0);
    // multiply maxValue by batch norm effective weight
    maxValue *= rsGetElementAt_float(
        recognizerL9BatchNormAllocation, x, recognizerBatchNormEffectiveWeightRow);
    // add batch norm effective bias to maxValue
    maxValue += rsGetElementAt_float(
        recognizerL9BatchNormAllocation, x, recognizerBatchNormEffectiveBiasRow);
    // apply ReLU function to maxValue
    if (maxValue < 0) maxValue = 0;
    // write maxValue to recognizerL9OutAllocations
    rsSetElementAt_float(recognizerL9OutAllocation, maxValue, x, y);
}

void RS_KERNEL recognizerL10LinearKernel(float in, uint32_t x, uint32_t y) {
    float out = 0;
    // calculate dot product of recognizerL9OutAllocation with row x of recognizerL10WeightsAllocation
    for (int i = 0; i < recognizerL10NumberOfInputs; ++i) {
        out += rsGetElementAt_float(recognizerL9OutAllocation, i, y) *
            rsGetElementAt_float(recognizerL10WeightsAllocation, i, x);
    };
    // add bias to out
    out += rsGetElementAt_float(recognizerL10BiasAllocation, x, 0);
    // apply ReLU function to out
    if (out < 0) out = 0;
    // write out to recognizerL10OutAllocation
    rsSetElementAt_float(recognizerL10OutAllocation, out, x, y);
}

void RS_KERNEL recognizerL11LinearKernel(float in, uint32_t x, uint32_t y) {
    float out = 0;
    // calculate dot product of recognizerL10OutAllocation with row x of recognizerL11WeightsAllocation
    for (int i = 0; i < recognizerL11NumberOfInputs; ++i) {
        out += rsGetElementAt_float(recognizerL10OutAllocation, i, y) *
            rsGetElementAt_float(recognizerL11WeightsAllocation, i, x);
    };
    // add bias to out
    out += rsGetElementAt_float(recognizerL11BiasAllocation, x, 0);
    // write out to recognizerL11OutAllocation
    rsSetElementAt_float(recognizerL11OutAllocation, out, x, y);
}

void RS_KERNEL recognizerPredictionKernel(float in, uint32_t x, uint32_t y) {
    float maxValue = rsGetElementAt_float(recognizerL11OutAllocation, 0, 0);
    int maxValueIndex = 0;
    for (int i = 1; i < recognizerPredictionNumberOfInputs; ++i) {
        float nextValue = rsGetElementAt_float(recognizerL11OutAllocation, i, 0);
        if (maxValue < nextValue) {
            maxValue = nextValue;
            maxValueIndex = i;
        };
    };
    // write max value index as character class to fbAllocation
    rsSetElementAt_int(fbAllocation, maxValueIndex, fbAllocationCharacterClassColumn, 0);
}




