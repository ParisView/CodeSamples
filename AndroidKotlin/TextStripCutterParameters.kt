package com.code.sample.readmrz

import android.renderscript.Allocation
import android.renderscript.Script
import android.util.Size

private const val TEXT_CUTOUT_KERNEL_PADDING_X_MM = 1.5f // mm
private const val TEXT_CUTOUT_KERNEL_PADDING_Y_MM = 0.8f // mm 0.8mm - for 36 pixel width on 800x600 input


class TextStripCutterParameters(private val processingScript: ScriptC_processing) {
    private var textStripWidth: Int = 30
    private var xPaddingInPixels: Int = 5

    lateinit var textStrip1CutoutAllocation: Allocation
    lateinit var textStrip2CutoutAllocation: Allocation
    private val textCutoutKernelLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()

    fun generalVariablesSetUp(cutoutWidth: Int, mmToPixelFactor: Float): Int {
        xPaddingInPixels = (TEXT_CUTOUT_KERNEL_PADDING_X_MM * mmToPixelFactor).toInt()
        var yPaddingInPixels = (TEXT_CUTOUT_KERNEL_PADDING_Y_MM * mmToPixelFactor).toInt()

        // calculating text strip width taking padding Y into account
        textStripWidth = cutoutWidth / 2 + yPaddingInPixels
        // tuning text strip width to fit recognition model requirements
        val yPaddingDelta = checkConvolution3LModelRequirements(textStripWidth)
        yPaddingInPixels += yPaddingDelta
        textStripWidth += yPaddingDelta

        // setting launch options
        textCutoutKernelLaunchOptions.setY(0, 1)

        // setting parameters to the script
        processingScript._textCutoutStartShiftX = xPaddingInPixels
        processingScript._textCutoutStartShiftY = yPaddingInPixels
        processingScript._textStripWidth = textStripWidth

        return textStripWidth
    }

    fun currentFrameVariablesSetUp(
        cutoutLength: Int,
        createFloatAllocation: (Size) -> Allocation
    ): Int {
        // recalculating detected text strip length taking padding X into account
        var textStripLength = cutoutLength + 2 * xPaddingInPixels
        // tuning text strip length to fit recognition model requirements
        val xPaddingDelta = checkConvolution3LModelRequirements(textStripLength)
        textStripLength += xPaddingDelta

        // creating allocations for the text strip 1 and text strip 2 cutouts
        textStrip1CutoutAllocation = createFloatAllocation(Size(textStripLength, textStripWidth))
        textStrip2CutoutAllocation = createFloatAllocation(Size(textStripLength, textStripWidth))

        // setting parameters to the script
        processingScript._textStrip1CutoutAllocation = textStrip1CutoutAllocation
        processingScript._textStrip2CutoutAllocation = textStrip2CutoutAllocation

        return textStripLength
    }

    fun cutoutTextStrips() {
        processingScript.forEach_textCutoutKernel(
            textStrip1CutoutAllocation,
            textCutoutKernelLaunchOptions
        )
    }

    fun currentFrameVariablesClose() {
        textStrip1CutoutAllocation.destroy()
        textStrip2CutoutAllocation.destroy()
    }

    private fun checkConvolution3LModelRequirements(dimensionValue: Int): Int{
        // dimension value must stay an Integer after all Convolution and Pull layers of the model,
        // and the model must have a regular structure on all layers in order to have the same
        // computations during segmentation at the ends and in the middle of the text strip
        return (4 - dimensionValue % 4) % 4
    }
}

