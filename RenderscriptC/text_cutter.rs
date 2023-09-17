

#pragma version(1)

#pragma rs java_package_name(com.code.sample.readmrz)


// allocations
rs_allocation grayAllocation;
rs_allocation localMeanAllocation;
rs_allocation textStrip1CutoutAllocation;
rs_allocation textStrip2CutoutAllocation;

// textCutoutKernel input parameters
int detectedLineAngleX = 100; // line tg = detectedLineAngleY / detectedLineAngleX
int detectedLineAngleY = 0;
int textCutoutStartShiftX;
int textCutoutStartShiftY;
int textStripWidth;
int2 detectedLineEndsX;
int2 detectedLineEndsY;


void RS_KERNEL textCutoutKernel(float in, uint32_t x, uint32_t y) {
    // calculate coordinates of the top pixel of the cutout at this particular x
    int xSigned = x;
    int xTopCoordinate = detectedLineEndsX[0] - textCutoutStartShiftX + xSigned;
    int yTopCoordinate = detectedLineEndsY[0] - textCutoutStartShiftY + (
        xSigned * detectedLineAngleY) / detectedLineAngleX;

    // for each pixel in the x-th column of the cutout get the pixel, subtract local mean and
    // cast to float to be used as recognition model input
    int strip1CutoutStop = textStripWidth;
    int strip2CutoutStop = strip1CutoutStop + textStripWidth;
    // cut out strip 1
    for (int i = 0; i < strip1CutoutStop; ++i) {
        // calculate coordinates for each pixel in the column
        int yCoordinate = yTopCoordinate + i;
        int xCoordinate = xTopCoordinate - (i * detectedLineAngleY) / detectedLineAngleX;

        // calculate output value from input pixel
        int pixel = rsGetElementAt_int(grayAllocation, xCoordinate, yCoordinate);
        pixel -= rsGetElementAt_int(localMeanAllocation, xCoordinate, yCoordinate);
        float out = ((float) pixel) / 127.0;

        // write output value to textStrip1CutoutAllocation
        rsSetElementAt_float(textStrip1CutoutAllocation, out, x, i);
    };
    // cut out strip 2
    for (int i = strip1CutoutStop; i < strip2CutoutStop; ++i) {
        // calculate coordinates for each pixel in the column
        int yCoordinate = yTopCoordinate + i;
        int xCoordinate = xTopCoordinate - (i * detectedLineAngleY) / detectedLineAngleX;

        // calculate output value from input pixel
        int pixel = rsGetElementAt_int(grayAllocation, xCoordinate, yCoordinate);
        pixel -= rsGetElementAt_int(localMeanAllocation, xCoordinate, yCoordinate);
        float out = ((float) pixel) / 127.0;

        // write output value to textStrip2CutoutAllocation
        rsSetElementAt_float(textStrip2CutoutAllocation, out, x, i - strip1CutoutStop);
    };
}

