

#pragma version(1)

#pragma rs java_package_name(com.code.sample.readmrz)


// allocations
rs_allocation segmenterNormalizedInputAllocation;

const int SEGMENTER_L1_NUMBER_OF_FEATURES = 8;
rs_allocation segmenterL1OutAllocations[SEGMENTER_L1_NUMBER_OF_FEATURES];
rs_allocation segmenterL1WeightsAllocations[SEGMENTER_L1_NUMBER_OF_FEATURES];

rs_allocation segmenterL1BiasAllocation;
rs_allocation segmenterL1BatchNormAllocation;

const int SEGMENTER_L2_NUMBER_OF_FEATURES = 8;
rs_allocation segmenterL2OutAllocations[SEGMENTER_L2_NUMBER_OF_FEATURES];
rs_allocation segmenterL2WeightsAllocations[SEGMENTER_L1_NUMBER_OF_FEATURES * SEGMENTER_L2_NUMBER_OF_FEATURES];

rs_allocation segmenterL2BiasAllocation;
rs_allocation segmenterL2BatchNormAllocation;
rs_allocation segmenterL3MaxPoolAllocation;
rs_allocation segmenterL3BatchNormAllocation;
rs_allocation segmenterL4OutAllocation;
rs_allocation segmenterL4WeightsAllocation;
rs_allocation segmenterL4BiasAllocation;
rs_allocation segmenterL4BatchNormAllocation;
rs_allocation segmenterL5OutAllocation;
rs_allocation segmenterL5WeightsAllocation;
rs_allocation segmenterL5BiasAllocation;
rs_allocation segmenterPredictionAllocation;

// segmenterInputNormalizationKernel input parameters
float segmenterTextStripImageMean;
float segmenterTextStripImageSTDReciprocal;

// segmenterL1ConvolutionKernel input parameters
int segmenterL1ConvolutionKernelWidth;
int segmenterBatchNormEffectiveWeightRow;
int segmenterBatchNormEffectiveBiasRow;

// segmenterL2ConvolutionKernel input parameters
int segmenterL2ConvolutionKernelWidth;


void RS_KERNEL segmenterInputNormalizationKernel(float in, uint32_t x, uint32_t y) {
    rsSetElementAt_float(segmenterNormalizedInputAllocation, (in - segmenterTextStripImageMean
        ) * segmenterTextStripImageSTDReciprocal, x, y);
}

void RS_KERNEL segmenterL1ConvolutionKernel(float in, uint32_t x, uint32_t y) {
    float correlation;

    // for each output feature map calculate convolution for pixel with coordinates x and y
    for (int n = 0; n < SEGMENTER_L1_NUMBER_OF_FEATURES; ++n) {
        // calculate correlation of input and convolution kernel
        correlation = 0;
        for (int i = 0; i < segmenterL1ConvolutionKernelWidth; ++i) {
            for (int j = 0; j < segmenterL1ConvolutionKernelWidth; ++j) {
                correlation += rsGetElementAt_float(segmenterNormalizedInputAllocation, x + j, y + i) *
                    rsGetElementAt_float(segmenterL1WeightsAllocations[n], j, i);
            };
        };
        // add bias to correlation
        correlation += rsGetElementAt_float(segmenterL1BiasAllocation, n, 0);
        // multiply correlation by batch norm effective weight
        correlation *= rsGetElementAt_float(
            segmenterL1BatchNormAllocation, n, segmenterBatchNormEffectiveWeightRow);
        // add batch norm effective bias to correlation
        correlation += rsGetElementAt_float(
            segmenterL1BatchNormAllocation, n, segmenterBatchNormEffectiveBiasRow);
        // apply ReLU function to correlation
        if (correlation < 0) correlation = 0;
        // write correlation to segmenterL1OutAllocations
        rsSetElementAt_float(segmenterL1OutAllocations[n], correlation, x, y);
    };
}

void RS_KERNEL segmenterL2ConvolutionKernel(float in, uint32_t x, uint32_t y) {
    float correlation;

    // for each output feature map calculate sum of convolutions for pixel with coordinates x and y
    for (int n = 0; n < SEGMENTER_L2_NUMBER_OF_FEATURES; ++n) {
        correlation = 0;
        // for each input channel
        for (int k = 0; k < SEGMENTER_L1_NUMBER_OF_FEATURES; ++k) {
            int weightsAllocationIndex = n * SEGMENTER_L1_NUMBER_OF_FEATURES + k;

            // calculate correlation of input and convolution kernel
            for (int i = 0; i < segmenterL2ConvolutionKernelWidth; ++i) {
                for (int j = 0; j < segmenterL2ConvolutionKernelWidth; ++j) {
                    correlation += rsGetElementAt_float(segmenterL1OutAllocations[k], x + j, y + i) *
                        rsGetElementAt_float(segmenterL2WeightsAllocations[weightsAllocationIndex], j, i);
                };
            };
        };
        // add bias to correlation
        correlation += rsGetElementAt_float(segmenterL2BiasAllocation, n, 0);
        // multiply correlation by batch norm effective weight
        correlation *= rsGetElementAt_float(
            segmenterL2BatchNormAllocation, n, segmenterBatchNormEffectiveWeightRow);
        // add batch norm effective bias to correlation
        correlation += rsGetElementAt_float(
            segmenterL2BatchNormAllocation, n, segmenterBatchNormEffectiveBiasRow);
        // apply ReLU function to correlation
        if (correlation < 0) correlation = 0;
        // write correlation to segmenterL2OutAllocations
        rsSetElementAt_float(segmenterL2OutAllocations[n], correlation, x, y);
    };
};

void RS_KERNEL segmenterL3MaxPool1DKernel(float in, uint32_t x, uint32_t y) {
    // calculate max value of x column in allocation segmenterL2OutAllocations[y]
    float maxValue = fmax(
        rsGetElementAt_float(segmenterL2OutAllocations[y], x, 0),
        rsGetElementAt_float(segmenterL2OutAllocations[y], x, 1)
    );
    int l2OutHeight = rsAllocationGetDimY(segmenterL2OutAllocations[y]);
    for(int i = 2; i < l2OutHeight; ++i) {
        maxValue = fmax(maxValue, rsGetElementAt_float(segmenterL2OutAllocations[y], x, i));
    };
    // multiply max value by batch norm effective weight
    maxValue *= rsGetElementAt_float(
        segmenterL3BatchNormAllocation, y, segmenterBatchNormEffectiveWeightRow);
    // add batch norm effective bias to max value
    maxValue += rsGetElementAt_float(
        segmenterL3BatchNormAllocation, y, segmenterBatchNormEffectiveBiasRow);
    // write max value to segmenterL3MaxPoolAllocation
    rsSetElementAt_float(segmenterL3MaxPoolAllocation, maxValue, x, y);
};

void RS_KERNEL segmenterL4LinearKernel(float in, uint32_t x, uint32_t y) {
    float out = 0;

    // calculate dot product of column x of segmenterL3MaxPoolAllocation with row y of
    // segmenterL4WeightsAllocation
    int l3MaxPoolHeight = rsAllocationGetDimY(segmenterL3MaxPoolAllocation);
    for (int i = 0; i < l3MaxPoolHeight; ++i) {
        out += rsGetElementAt_float(segmenterL3MaxPoolAllocation, x, i) *
            rsGetElementAt_float(segmenterL4WeightsAllocation, i, y);
    };
    // add bias to out
    out += rsGetElementAt_float(segmenterL4BiasAllocation, y, 0);
    // multiply out by batch norm effective weight
    out *= rsGetElementAt_float(
        segmenterL4BatchNormAllocation, y, segmenterBatchNormEffectiveWeightRow);
    // add batch norm effective bias to out
    out += rsGetElementAt_float(
        segmenterL4BatchNormAllocation, y, segmenterBatchNormEffectiveBiasRow);
    // apply ReLU function to out
    if (out < 0) out = 0;

    // write out to segmenterL4OutAllocation
    rsSetElementAt_float(segmenterL4OutAllocation, out, x, y);
}

void RS_KERNEL segmenterL5LinearKernel(float in, uint32_t x, uint32_t y) {
    float out = 0;

    // calculate dot product of column x of segmenterL4OutAllocation with row y of
    // segmenterL5WeightsAllocation
    int l4OutHeight = rsAllocationGetDimY(segmenterL4OutAllocation);
    for (int i = 0; i < l4OutHeight; ++i) {
        out += rsGetElementAt_float(segmenterL4OutAllocation, x, i) *
            rsGetElementAt_float(segmenterL5WeightsAllocation, i, y);
    };
    // add bias to out
    out += rsGetElementAt_float(segmenterL5BiasAllocation, y, 0);

    // write out to segmenterL5OutAllocation
    rsSetElementAt_float(segmenterL5OutAllocation, out, x, y);
}

void RS_KERNEL segmenterPredictionKernel(int in, uint32_t x, uint32_t y) {
    // calculate index of max value in column x of segmenterL5OutAllocation
    int maxValueIndex = 0;
    float maxValue = rsGetElementAt_float(segmenterL5OutAllocation, x, 0);
    int l5OutHeight = rsAllocationGetDimY(segmenterL5OutAllocation);
    for (int i = 1; i < l5OutHeight; ++i) {
        float nextValue = rsGetElementAt_float(segmenterL5OutAllocation, x, i);
        if (maxValue < nextValue) {
            maxValue = nextValue;
            maxValueIndex = i;
        };
    };

    // write index of max value to segmenterPredictionAllocation
    rsSetElementAt_int(segmenterPredictionAllocation, maxValueIndex, x, y);
}



