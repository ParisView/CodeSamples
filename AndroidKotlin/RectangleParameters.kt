package com.code.sample.readmrz

import android.renderscript.Script
import android.util.Size

private const val TWO_STRINGS_TEXT_AREA_WIDTH = 118f// mm
private const val TWO_STRINGS_TEXT_AREA_HEIGHT = 15f // mm
private const val RECTANGLE_LINE_WIDTH = 0.5f // mm
private const val RECTANGLE_DASH_LENGTH = 2.0f // mm
private const val RECTANGLE_DASH_VELOCITY = 4.0f // mm/sec


class RectangleParameters {
    private val secondToMillisecondRatio = 1000.0
    val drawTopLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    val drawBottomLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    val drawLeftLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    val drawRightLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    var dashLength = 1
    var doubleDashLength = 2
    var dashShiftPeriod = 2
    var rectangleDashStartX = 0
    var rectangleDashStartY = 0
    var horizontalLineConstShift = 0
    var verticalLineConstShift = 0

    fun calculateParameters(size: Size, mmToPixelFactor: Float) {
        val textAreaWidth = TWO_STRINGS_TEXT_AREA_WIDTH // mm
        val textAreaHeight = TWO_STRINGS_TEXT_AREA_HEIGHT // mm
        val rectangleLineWidth = RECTANGLE_LINE_WIDTH // mm
        val dashLengthMM = RECTANGLE_DASH_LENGTH // mm
        var rectangleWidth = (textAreaWidth * mmToPixelFactor).toInt()
        var rectangleHeight = (textAreaHeight * mmToPixelFactor).toInt()
        val rectangleLineWidthInPixels = (rectangleLineWidth * mmToPixelFactor).toInt()
        dashLength = (dashLengthMM * mmToPixelFactor).toInt()
        doubleDashLength = 2 * dashLength
        // dash shift period in milliseconds
        dashShiftPeriod = (secondToMillisecondRatio * 2.0 * dashLengthMM / RECTANGLE_DASH_VELOCITY).toInt()

        // calculating rectangle outer border coordinates
        var rectangleBorderOuterLeft = (size.width - rectangleWidth - 2 * rectangleLineWidthInPixels) / 2
        if (rectangleBorderOuterLeft < 0) rectangleBorderOuterLeft = 0
        var rectangleBorderOuterRight = size.width - rectangleBorderOuterLeft
        var rectangleBorderOuterTop = (size.height - rectangleHeight - 2 * rectangleLineWidthInPixels) / 2
        if (rectangleBorderOuterTop < 0) rectangleBorderOuterTop = 0
        val rectangleBorderOuterBottom = size.height - rectangleBorderOuterTop

        // tuning rectangle coordinates to fit integer number of dash and space groups
        rectangleHeight = rectangleBorderOuterBottom - rectangleBorderOuterTop
        rectangleWidth = rectangleBorderOuterRight - rectangleBorderOuterLeft
        val nOfDashes = (rectangleHeight + rectangleWidth) / dashLength
        rectangleWidth = nOfDashes * dashLength - rectangleHeight
        val rectangleBorderLeftShift = (rectangleBorderOuterRight - rectangleBorderOuterLeft -
                rectangleWidth) / 2
        val rectangleBorderRightShift = (rectangleBorderOuterRight - rectangleBorderOuterLeft -
                rectangleWidth) - rectangleBorderLeftShift
        rectangleBorderOuterLeft += rectangleBorderLeftShift
        rectangleBorderOuterRight -= rectangleBorderRightShift

        // calculating rectangle inner border coordinates
        val rectangleBorderInnerLeft = rectangleBorderOuterLeft + rectangleLineWidthInPixels
        val rectangleBorderInnerRight = rectangleBorderOuterRight - rectangleLineWidthInPixels
        val rectangleBorderInnerTop = rectangleBorderOuterTop + rectangleLineWidthInPixels
        val rectangleBorderInnerBottom = rectangleBorderOuterBottom - rectangleLineWidthInPixels

        // calculating dash shift parameters
        rectangleDashStartX = rectangleBorderOuterLeft - 2 * doubleDashLength
        rectangleDashStartY = rectangleBorderInnerTop - 2 * doubleDashLength
        horizontalLineConstShift = (rectangleBorderOuterRight - rectangleBorderOuterLeft) % doubleDashLength
        verticalLineConstShift = (rectangleBorderInnerBottom - rectangleBorderInnerTop) % doubleDashLength

        // setting drawing kernel launch options
        drawTopLaunchOptions.setY(rectangleBorderOuterTop, rectangleBorderInnerTop)
        drawTopLaunchOptions.setX(rectangleBorderOuterLeft, rectangleBorderOuterRight)
        drawBottomLaunchOptions.setY(rectangleBorderInnerBottom, rectangleBorderOuterBottom)
        drawBottomLaunchOptions.setX(rectangleBorderOuterLeft, rectangleBorderOuterRight)
        drawLeftLaunchOptions.setY(rectangleBorderInnerTop, rectangleBorderInnerBottom)
        drawLeftLaunchOptions.setX(rectangleBorderOuterLeft, rectangleBorderInnerLeft)
        drawRightLaunchOptions.setY(rectangleBorderInnerTop, rectangleBorderInnerBottom)
        drawRightLaunchOptions.setX(rectangleBorderInnerRight, rectangleBorderOuterRight)
    }
}


