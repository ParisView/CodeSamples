package com.code.sample.readmrz

import android.os.SystemClock
import android.renderscript.Allocation
import android.renderscript.Script
import android.util.Size
import android.view.Surface
import android.view.TextureView

// dimensions
private const val LOCAL_MEAN_KERNEL_WIDTH = 2f // mm, about letter size


class BasicParameters(private val processingScript: ScriptC_processing) {
    private lateinit var yuvAllocation: Allocation
    private lateinit var grayAllocation: Allocation
    private lateinit var localMeanAllocation: Allocation
    private lateinit var columnSumAllocation: Allocation
    private lateinit var outAllocation: Allocation
    private lateinit var outputTextureViewSurface: Surface
    private val localMeanColumnSumKernelLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val localMeanRowSumKernelLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val rectangle = RectangleParameters()

    fun variablesSetUp(
        inputAllocationSize: Size,
        inputListener: Allocation.OnBufferAvailableListener,
        computationFieldSize: Size,
        outAllocationSize: Size,
        outputTextureView: TextureView,
        mmToPixelFactor: Float,
        createYuvAllocation: (Size) -> Allocation,
        createIntAllocation: (Size) -> Allocation,
        createOutAllocation: (Size) -> Allocation,
        successfulEventColor: IntArray,
        dashedRectangleColor1: IntArray,
        dashedRectangleColor2: IntArray
    ) {
        // creating basic allocations
        yuvAllocation = createYuvAllocation(inputAllocationSize)
        yuvAllocation.setOnBufferAvailableListener(inputListener)
        grayAllocation = createIntAllocation(computationFieldSize)
        localMeanAllocation = createIntAllocation(computationFieldSize)
        columnSumAllocation = createIntAllocation(computationFieldSize)
        outAllocation = createOutAllocation(outAllocationSize)
        outputTextureViewSurface = Surface(outputTextureView.surfaceTexture)
        outAllocation.surface = outputTextureViewSurface

        // setting input YUV allocation
        processingScript._yuvInputAllocation = yuvAllocation

        // setting grayscale allocation and its coordinates shift within input allocation borders
        processingScript._grayAllocation = grayAllocation
        val yuvToGrayKernelShift = calculateInToOutKernelShift(inputAllocationSize, computationFieldSize)
        processingScript._yuvToGrayKernelShiftX = yuvToGrayKernelShift.width
        processingScript._yuvToGrayKernelShiftY = yuvToGrayKernelShift.height

        // setting output RGBA allocation and its coordinates shift within input allocation borders
        processingScript._outAllocation = outAllocation
        val yuvToOutKernelShift = calculateInToOutKernelShift(inputAllocationSize, outAllocationSize)
        processingScript._yuvToOutKernelShiftX = yuvToOutKernelShift.width
        processingScript._yuvToOutKernelShiftY = yuvToOutKernelShift.height

        // setting local mean calculation parameters
        processingScript._columnSumAllocation = columnSumAllocation
        processingScript._computationFieldWidth = computationFieldSize.width
        processingScript._computationFieldHeight = computationFieldSize.height
        processingScript._localMeanAllocation = localMeanAllocation
        val localMeanKernelHalfWidthInPixels = ((LOCAL_MEAN_KERNEL_WIDTH / 2.0) * mmToPixelFactor).toInt()
        val kernelWidthInPixels = localMeanKernelHalfWidthInPixels * 2 + 1
        processingScript._localMeanKernelHalfWidth = localMeanKernelHalfWidthInPixels
        processingScript._localMeanKernelWidth = kernelWidthInPixels
        processingScript._numberOfSummedElements = kernelWidthInPixels * kernelWidthInPixels
        localMeanColumnSumKernelLaunchOptions.setY(0, 1)
        localMeanRowSumKernelLaunchOptions.setX(0, 1)
        localMeanRowSumKernelLaunchOptions.setY(localMeanKernelHalfWidthInPixels,
            computationFieldSize.height - localMeanKernelHalfWidthInPixels)

        // setting rectangle colors
        processingScript._singleColorRectanglePixelColor = successfulEventColor
        processingScript._dashedRectanglePixelColor1 = dashedRectangleColor1
        processingScript._dashedRectanglePixelColor2 = dashedRectangleColor2

        // calculating rectangle parameters
        rectangle.calculateParameters(outAllocationSize, mmToPixelFactor)

        // setting drawDashedHorizontalRectangleToOutKernel and
        // drawDashedVerticalRectangleToOutKernel input parameters
        processingScript._dashLength = rectangle.dashLength
        processingScript._doubleDashLength = rectangle.doubleDashLength
    }

    fun variablesClose() {
        yuvAllocation.setOnBufferAvailableListener(null)
        yuvAllocation.destroy()
        grayAllocation.destroy()
        localMeanAllocation.destroy()
        columnSumAllocation.destroy()
        outAllocation.destroy()

        outputTextureViewSurface.release()
    }

    fun ioReceive() {
        yuvAllocation.ioReceive()
    }

    fun performProcessing() {
        processingScript.forEach_yuvToOutKernel(outAllocation)
        processingScript.forEach_yuvToGrayKernel(grayAllocation)
        processingScript.forEach_localMeanColumnSumKernel(
            columnSumAllocation, localMeanColumnSumKernelLaunchOptions)
        processingScript.forEach_localMeanRowSumKernel(
            columnSumAllocation, localMeanRowSumKernelLaunchOptions)
    }

    fun ioSend() {
        outAllocation.ioSend()
    }

    fun getInputSurface(): Surface {
        return yuvAllocation.surface
    }

    fun showDashedRectangle() {
        val millisInPeriod = (SystemClock.uptimeMillis() % rectangle.dashShiftPeriod).toInt()
        var currentDashStartInsideDoubleDashGroup =
            rectangle.doubleDashLength * millisInPeriod / rectangle.dashShiftPeriod

        processingScript._imaginaryFirstLeftDashStart =
            rectangle.rectangleDashStartX + currentDashStartInsideDoubleDashGroup
        processingScript.forEach_drawDashedHorizontalRectangleToOutKernel(
            outAllocation, rectangle.drawTopLaunchOptions)

        currentDashStartInsideDoubleDashGroup += rectangle.horizontalLineConstShift
        processingScript._imaginaryFirstTopDashStart =
            rectangle.rectangleDashStartY + currentDashStartInsideDoubleDashGroup
        processingScript.forEach_drawDashedVerticalRectangleToOutKernel(
            outAllocation, rectangle.drawRightLaunchOptions)

        currentDashStartInsideDoubleDashGroup += rectangle.verticalLineConstShift
        currentDashStartInsideDoubleDashGroup += rectangle.horizontalLineConstShift
        processingScript._imaginaryFirstLeftDashStart =
            rectangle.rectangleDashStartX - currentDashStartInsideDoubleDashGroup
        processingScript.forEach_drawDashedHorizontalRectangleToOutKernel(
            outAllocation, rectangle.drawBottomLaunchOptions)

        currentDashStartInsideDoubleDashGroup += rectangle.verticalLineConstShift
        processingScript._imaginaryFirstTopDashStart =
            rectangle.rectangleDashStartY - currentDashStartInsideDoubleDashGroup
        processingScript.forEach_drawDashedVerticalRectangleToOutKernel(
            outAllocation, rectangle.drawLeftLaunchOptions)
    }

    fun showSingleColorRectangle() {
        processingScript.forEach_drawColorToOutKernel(outAllocation, rectangle.drawTopLaunchOptions)
        processingScript.forEach_drawColorToOutKernel(outAllocation, rectangle.drawBottomLaunchOptions)
        processingScript.forEach_drawColorToOutKernel(outAllocation, rectangle.drawLeftLaunchOptions)
        processingScript.forEach_drawColorToOutKernel(outAllocation, rectangle.drawRightLaunchOptions)
    }

    private fun calculateInToOutKernelShift(inAllocationSize: Size, outAllocationSize: Size): Size {
        return Size(
            (inAllocationSize.width - outAllocationSize.width) / 2,
            (inAllocationSize.height - outAllocationSize.height) / 2
        )
    }
}


