

#pragma version(1)

#pragma rs java_package_name(com.code.sample.readmrz)


// allocations
rs_allocation yuvInputAllocation;
rs_allocation outAllocation;
rs_allocation grayAllocation;
rs_allocation columnSumAllocation;
rs_allocation localMeanAllocation;

// yuvToOutKernel input parameters
int yuvToOutKernelShiftX;
int yuvToOutKernelShiftY;

// drawColorToOutKernel input parameters
int singleColorRectanglePixelColor[4];

// drawDashedHorizontalRectangleToOutKernel and
// drawDashedVerticalRectangleToOutKernel input parameters
int dashedRectanglePixelColor1[4];
int dashedRectanglePixelColor2[4];
int imaginaryFirstLeftDashStart;
int imaginaryFirstTopDashStart;
int dashLength = 1;
int doubleDashLength = 2;

// yuvToGrayKernel input parameters
int yuvToGrayKernelShiftX;
int yuvToGrayKernelShiftY;

// localMeanColumnSumKernel and localMeanRowSumKernel input parameters
int localMeanKernelHalfWidth;
int localMeanKernelWidth;
int numberOfSummedElements;
int computationFieldWidth;
int computationFieldHeight;


void RS_KERNEL yuvToOutKernel(uchar4 in, uint32_t x, uint32_t y) {
    uchar3 yuvPixel;
    uint32_t yuv_x = yuvToOutKernelShiftX + x;
    uint32_t yuv_y = yuvToOutKernelShiftY + y;
    yuvPixel.r = rsGetElementAtYuv_uchar_Y(yuvInputAllocation, yuv_x, yuv_y);
    yuvPixel.g = rsGetElementAtYuv_uchar_U(yuvInputAllocation, yuv_x, yuv_y);
    yuvPixel.b = rsGetElementAtYuv_uchar_V(yuvInputAllocation, yuv_x, yuv_y);

    int4 rgb;
    rgb.r = yuvPixel.r +
        ((yuvPixel.b * 1436) >> 10) - 179;
    rgb.g = yuvPixel.r -
        ((yuvPixel.g * 46549) >> 17) -
        ((yuvPixel.b * 93604) >> 17) + 135;
    rgb.b = yuvPixel.r +
        ((yuvPixel.g * 1814) >> 10) - 227;
    rgb.a = 255;
    uchar4 out = convert_uchar4(clamp(rgb, 0, 255));

    rsSetElementAt_uchar4(outAllocation, out, x, y);
}

void RS_KERNEL drawColorToOutKernel(uchar4 in, uint32_t x, uint32_t y) {
    uchar4 out;
    out[0] = clamp(singleColorRectanglePixelColor[0], 0, 255);
    out[1] = clamp(singleColorRectanglePixelColor[1], 0, 255);
    out[2] = clamp(singleColorRectanglePixelColor[2], 0, 255);
    out[3] = clamp(singleColorRectanglePixelColor[3], 0, 255);
    rsSetElementAt_uchar4(outAllocation, out, x, y);
}

void RS_KERNEL drawDashedHorizontalRectangleToOutKernel(uchar4 in, uint32_t x, uint32_t y) {
    uchar4 out;
    int coordinateInsideDoubleDashGroup = (x - imaginaryFirstLeftDashStart) % doubleDashLength;
    if (coordinateInsideDoubleDashGroup < dashLength) {
        out[0] = clamp(dashedRectanglePixelColor1[0], 0, 255);
        out[1] = clamp(dashedRectanglePixelColor1[1], 0, 255);
        out[2] = clamp(dashedRectanglePixelColor1[2], 0, 255);
        out[3] = clamp(dashedRectanglePixelColor1[3], 0, 255);
    } else {
        out[0] = clamp(dashedRectanglePixelColor2[0], 0, 255);
        out[1] = clamp(dashedRectanglePixelColor2[1], 0, 255);
        out[2] = clamp(dashedRectanglePixelColor2[2], 0, 255);
        out[3] = clamp(dashedRectanglePixelColor2[3], 0, 255);
    };
    rsSetElementAt_uchar4(outAllocation, out, x, y);
}

void RS_KERNEL drawDashedVerticalRectangleToOutKernel(uchar4 in, uint32_t x, uint32_t y) {
    uchar4 out;
    int coordinateInsideDoubleDashGroup = (y - imaginaryFirstTopDashStart) % doubleDashLength;
    if (coordinateInsideDoubleDashGroup < dashLength) {
        out[0] = clamp(dashedRectanglePixelColor1[0], 0, 255);
        out[1] = clamp(dashedRectanglePixelColor1[1], 0, 255);
        out[2] = clamp(dashedRectanglePixelColor1[2], 0, 255);
        out[3] = clamp(dashedRectanglePixelColor1[3], 0, 255);
    } else {
        out[0] = clamp(dashedRectanglePixelColor2[0], 0, 255);
        out[1] = clamp(dashedRectanglePixelColor2[1], 0, 255);
        out[2] = clamp(dashedRectanglePixelColor2[2], 0, 255);
        out[3] = clamp(dashedRectanglePixelColor2[3], 0, 255);
    };
    rsSetElementAt_uchar4(outAllocation, out, x, y);
}

void RS_KERNEL yuvToGrayKernel(int in, uint32_t x, uint32_t y) {
    uchar3 yuvPixel;
    uint32_t yuv_x = yuvToGrayKernelShiftX + x;
    uint32_t yuv_y = yuvToGrayKernelShiftY + y;
    yuvPixel.r = rsGetElementAtYuv_uchar_Y(yuvInputAllocation, yuv_x, yuv_y);
    yuvPixel.g = rsGetElementAtYuv_uchar_U(yuvInputAllocation, yuv_x, yuv_y);
    yuvPixel.b = rsGetElementAtYuv_uchar_V(yuvInputAllocation, yuv_x, yuv_y);

    int out;
    out = yuvPixel.r + ((yuvPixel.b * 1436) >> 10) - 306;
    out += yuvPixel.r - ((yuvPixel.g * 46549) >> 17) - ((yuvPixel.b * 93604) >> 17) + 8;
    out += yuvPixel.r + ((yuvPixel.g * 1814) >> 10) - 354;
    out /= 3;
    rsSetElementAt_int(grayAllocation, out, x, y);
}

void RS_KERNEL localMeanColumnSumKernel(int in, uint32_t x, uint32_t y) {
    int columnSum = 0;
    int heightIndex;

    // calculate and write first column sum element on grayAllocation
    for (heightIndex = 0; heightIndex < localMeanKernelWidth; ++heightIndex) {
        columnSum += rsGetElementAt_int(grayAllocation, x, heightIndex);
    };
    rsSetElementAt_int(columnSumAllocation, columnSum, x, localMeanKernelHalfWidth);

    // calculate and write the rest of the elements of the column
    for (heightIndex = localMeanKernelWidth; heightIndex < computationFieldHeight; ++heightIndex) {
        columnSum -= rsGetElementAt_int(grayAllocation, x, heightIndex - localMeanKernelWidth);
        columnSum += rsGetElementAt_int(grayAllocation, x, heightIndex);
        rsSetElementAt_int(columnSumAllocation, columnSum, x,
            heightIndex - localMeanKernelHalfWidth);
    }
}

void RS_KERNEL localMeanRowSumKernel(int in, uint32_t x, uint32_t y) {
    int rowSum = 0;
    int widthIndex;
    int localMeanValue;

    // calculate first row sum element on columnSumAllocation
    for (widthIndex = 0; widthIndex < localMeanKernelWidth; ++widthIndex) {
        rowSum += rsGetElementAt_int(columnSumAllocation, widthIndex, y);
    };
    // calculate the first element of the row of local mean
    localMeanValue = rowSum / numberOfSummedElements;

    // write first element of the row to localMeanAllocation
    rsSetElementAt_int(localMeanAllocation, localMeanValue, localMeanKernelHalfWidth, y);

    // calculate the rest of row sum elements on columnSumAllocation
    for (widthIndex = localMeanKernelWidth; widthIndex < computationFieldWidth; ++widthIndex) {
        rowSum += rsGetElementAt_int(columnSumAllocation, widthIndex, y);
        rowSum -= rsGetElementAt_int(columnSumAllocation,
            widthIndex - localMeanKernelWidth, y);

        // calculate local mean
        localMeanValue = rowSum / numberOfSummedElements;

        // write local mean to localMeanAllocation
        rsSetElementAt_int(localMeanAllocation, localMeanValue,
            widthIndex - localMeanKernelHalfWidth, y);
    };
}

