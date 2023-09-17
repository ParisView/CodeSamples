package com.code.sample.readmrz

import android.renderscript.Allocation
import android.renderscript.Script
import android.util.Log
import android.util.Size
import kotlin.math.abs
import kotlin.math.roundToInt

// reduced kernel parameters
private const val MEAN_PULL_KERNEL_WIDTH_MM = 2f // mm, about letter size
private const val REDUCED_KERNEL_HALF_WIDTH_IN_PIXELS = 1 // in pixels
private const val REDUCED_KERNEL_WIDTH_IN_PIXELS = 3 // in pixels
private const val LOCAL_DEVIATION_LIMIT = 42 // third part of half of pixel value range (255/2=127, 127/3=42)

// text search correlation function parameters
private const val TEXT_SEARCH_CORRELATION_FUNCTION_LENGTH_MM = 20.5f // mm
private const val TEXT_HEIGHT = 4.5f // mm
private const val DISTANCE_FROM_START_TO_TEXT = 5f // mm
private const val SPACE_BETWEEN_TEXT_LINES = 1.5f // mm
private const val CORRELATION_FUNCTION_AMPLITUDE = 3027 // in range 0 to 127 to be in -1.0 to 1.0 in GL
private const val TEXT_SEARCH_CORRELATION_STRIP_HALF_WIDTH_MM = 3f // mm
private const val TEXT_SEARCH_CORRELATION_NUMBER_OF_RUNS = 9 // number of passes along width of reduced allocation

// line detection parameters
private const val LINE_DETECTION_ALLOCATION_TEXT_TOP_EDGE_Y_ROW_NUMBER = 0 // first row of the allocation
private const val TEXT_LINE_MAX_ANGLE_COTANGENT = 10 // cotangent = delta width / delta height
private const val TEXT_LINE_MAX_DELTA_ANGLE_COTANGENT = 50 // cotangent = delta width / delta height
private const val MARKER_INCLUDE_IN_LINE_FLAG = 27
private const val MIN_POINTS_FOR_DETECTED_LINE = 4 // must be less than actual number of points by 1
private const val LINE_MARKER_HALF_WIDTH_MM = 1.5f // mm
private const val LINE_MARKER_HEIGHT_MM = 2f // mm
private const val LINE_MARKER_DASH_THICKNESS_MM = 1f // mm

// end detection correlation function parameters
private const val END_DETECTION_CORRELATION_FUNCTION_LENGTH_MM = 20f // mm
// distance from outer edge of end detection correlation function to text edge in mm
private const val END_DETECTION_CORRELATION_FUNCTION_TEXT_EDGE_MM = 1f
// end detection correlation function allocation rows
private const val NUMBER_OF_ROWS = 2 // row 0 is for text strip start parameters
                                     // row 1 is for text strip end parameters
// end detection correlation function allocation column indices
private const val NUMBER_OF_COLUMNS = 6 // text edge, start, end, amplitude before text edge,
                                        // amplitude after text edge, direction sign
private const val TEXT_EDGE_COLUMN_INDEX = 0
private const val CORRELATION_START_COLUMN_INDEX = 1
private const val CORRELATION_END_COLUMN_INDEX = 2
private const val AMPLITUDE_BEFORE_TEXT_EDGE_COLUMN_INDEX = 3
private const val AMPLITUDE_AFTER_TEXT_EDGE_COLUMN_INDEX = 4
private const val DIRECTION_SIGN_COLUMN_INDEX = 5
private const val END_DETECTION_LOCAL_MAX_VICINITY_HALF_WIDTH = 0.5f // mm
private const val END_DETECTION_LOCAL_MAX_AREA_MEAN_HALF_WIDTH = 1f // mm
private const val END_MARKER_DASH_THICKNESS_MM = LINE_MARKER_DASH_THICKNESS_MM


class TextStripFinderParameters(private val processingScript: ScriptC_processing) {
    private lateinit var reducedGrayAllocation: Allocation
    private lateinit var reducedColumnSumAllocation: Allocation
    private lateinit var reducedLocalMeanAllocation: Allocation
    private lateinit var reducedLocalDeviationAllocation: Allocation
    private lateinit var textSearchCorrelationFunctionAllocation: Allocation
    private lateinit var lineDetectionAllocation: Allocation
    private lateinit var endDetectionCorrelationFunctionAllocation: Allocation
    private val reducedLocalMeanColumnSumKernelLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val reducedLocalMeanRowSumKernelLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val reducedLocalDeviationKernelLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val textSearchCorrelationKernelLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val lineDetectionKernelLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val lineMarkerDrawKernelLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val endDetectionKernelLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val endMarkerDrawKernelLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()

    fun variablesSetUp(
        inputAllocationSize: Size,
        mmToPixelFactor: Float,
        createIntAllocation: (Size) -> Allocation,
        successfulEventColor: IntArray,
        neutralColor: IntArray
    ): Int {
        // calculating mean pull kernel width and reduced allocation size
        val meanPullKernelWidth = (MEAN_PULL_KERNEL_WIDTH_MM * mmToPixelFactor /
                REDUCED_KERNEL_WIDTH_IN_PIXELS).roundToInt()
        val reducedAllocationSize = calculateReducedAllocationSize(
                inputAllocationSize, meanPullKernelWidth)

        // setting processing field reduction parameters to the script
        processingScript._reducedAllocationWidth = reducedAllocationSize.width
        processingScript._reducedAllocationHeight = reducedAllocationSize.height
        processingScript._meanPullKernelWidth = meanPullKernelWidth
        processingScript._numberOfMeanPullSumElements = meanPullKernelWidth * meanPullKernelWidth
        processingScript._reducedKernelWidth = REDUCED_KERNEL_WIDTH_IN_PIXELS
        processingScript._reducedKernelHalfWidth = REDUCED_KERNEL_HALF_WIDTH_IN_PIXELS
        processingScript._numberOfReducedKernelSumElements =
            REDUCED_KERNEL_WIDTH_IN_PIXELS * REDUCED_KERNEL_WIDTH_IN_PIXELS
        // setting local deviation parameters to the script
        processingScript._localDeviationLimit = LOCAL_DEVIATION_LIMIT

        // creating allocations
        reducedGrayAllocation = createIntAllocation(reducedAllocationSize)
        reducedColumnSumAllocation = createIntAllocation(reducedAllocationSize)
        reducedLocalMeanAllocation = createIntAllocation(reducedAllocationSize)
        reducedLocalDeviationAllocation = createIntAllocation(reducedAllocationSize)

        // setting kernel launch options
        reducedLocalMeanColumnSumKernelLaunchOptions.setY(0, 1)
        reducedLocalMeanRowSumKernelLaunchOptions.setX(0, 1)
        reducedLocalMeanRowSumKernelLaunchOptions.setY(REDUCED_KERNEL_HALF_WIDTH_IN_PIXELS,
                reducedAllocationSize.height - REDUCED_KERNEL_HALF_WIDTH_IN_PIXELS)
        reducedLocalDeviationKernelLaunchOptions.setX(REDUCED_KERNEL_HALF_WIDTH_IN_PIXELS,
                reducedAllocationSize.width - REDUCED_KERNEL_HALF_WIDTH_IN_PIXELS)
        reducedLocalDeviationKernelLaunchOptions.setY(REDUCED_KERNEL_HALF_WIDTH_IN_PIXELS,
                reducedAllocationSize.height - REDUCED_KERNEL_HALF_WIDTH_IN_PIXELS)
        textSearchCorrelationKernelLaunchOptions.setY(0, 1)
        lineDetectionKernelLaunchOptions.setY(0, 1)
        lineDetectionKernelLaunchOptions.setX(0, 1)
        lineMarkerDrawKernelLaunchOptions.setY(0, 1)
        endDetectionKernelLaunchOptions.setX(0, 1)
        endMarkerDrawKernelLaunchOptions.setX(0, 1)

        // creating text search correlation function
        val textSearchCorrelationFunctionLength = (
                TEXT_SEARCH_CORRELATION_FUNCTION_LENGTH_MM * mmToPixelFactor / meanPullKernelWidth
                ).roundToInt()
        val correlationFunctionTextEdges = calculateCorrelationFunctionTextEdges(
                mmToPixelFactor, meanPullKernelWidth)
        val (allocation, correlationAmplitudePeakToMeanRatio) = createTextSearchCorrelationFunctionAllocation(
                textSearchCorrelationFunctionLength, correlationFunctionTextEdges, createIntAllocation)
        textSearchCorrelationFunctionAllocation = allocation
        // setting text search correlation function parameters to the script
        processingScript._textSearchCorrelationKernelLength = textSearchCorrelationFunctionLength
        processingScript._correlationAmplitudePeakToMeanRatio = correlationAmplitudePeakToMeanRatio
        processingScript._correlationFunctionTextTopEdge = correlationFunctionTextEdges[0]
        val textStripTopEdge = correlationFunctionTextEdges[0] * meanPullKernelWidth
        val textStripBottomEdge = correlationFunctionTextEdges[3] * meanPullKernelWidth
        processingScript._textStripTopEdge = textStripTopEdge
        processingScript._textStripBottomEdge = textStripBottomEdge
        val textTopToBottomWidth = textStripBottomEdge - textStripTopEdge
        processingScript._textSearchCorrelationKernelStripHalfWidth =
                calculateTextSearchCorrelationStripHalfWidth(mmToPixelFactor, meanPullKernelWidth)
        processingScript._textSearchCorrelationKernelXStep =
                reducedAllocationSize.width / (TEXT_SEARCH_CORRELATION_NUMBER_OF_RUNS + 1)

        // creating line detection allocation
        lineDetectionAllocation = createIntAllocation(
                Size(TEXT_SEARCH_CORRELATION_NUMBER_OF_RUNS, TEXT_SEARCH_CORRELATION_NUMBER_OF_RUNS + 2))
        // setting line detection parameters to the script
        processingScript._lineDetectionAllocationTextTopEdgeYRowNumber =
            LINE_DETECTION_ALLOCATION_TEXT_TOP_EDGE_Y_ROW_NUMBER
        processingScript._textSearchCorrelationKernelNumberOfRuns = TEXT_SEARCH_CORRELATION_NUMBER_OF_RUNS
        processingScript._calculatedMeanAngleXBase = calculateMeanAngleXBase()
        processingScript._lineDetectionKernelMaxAngleCtg = TEXT_LINE_MAX_ANGLE_COTANGENT
        processingScript._lineDetectionKernelMaxDeltaAngleCtg = TEXT_LINE_MAX_DELTA_ANGLE_COTANGENT
        processingScript._lineDetectionAllocationIncludeFlagRowNumber = TEXT_SEARCH_CORRELATION_NUMBER_OF_RUNS
        processingScript._lineDetectionAllocationTempBufferRowNumber = TEXT_SEARCH_CORRELATION_NUMBER_OF_RUNS + 1
        processingScript._minPointsForDetectedLine = MIN_POINTS_FOR_DETECTED_LINE
        // setting line detection markers parameters to the script
        processingScript._markerIncludedInLineFlag = MARKER_INCLUDE_IN_LINE_FLAG
        processingScript._colorOfMarkerIncludedInLine = successfulEventColor
        processingScript._colorOfMarkerNotIncludedInLine = neutralColor
        val lineMarkerHalfWidth = (LINE_MARKER_HALF_WIDTH_MM * mmToPixelFactor).roundToInt()
        processingScript._lineMarkerHalfWidth = lineMarkerHalfWidth
        val lineMarkerDashThickness =
                calculateDashThicknessInPixels(LINE_MARKER_DASH_THICKNESS_MM, mmToPixelFactor)
        processingScript._lineMarkerDashThickness = lineMarkerDashThickness
        processingScript._lineMarkerHeight = (LINE_MARKER_HEIGHT_MM * mmToPixelFactor).roundToInt()
        processingScript._lineMarkerVerticalDashXShift = lineMarkerHalfWidth - lineMarkerDashThickness / 2

        // creating end detection correlation function
        val endDetectionCorrelationFunctionLength = (
                END_DETECTION_CORRELATION_FUNCTION_LENGTH_MM * mmToPixelFactor / meanPullKernelWidth
                ).roundToInt()
        endDetectionCorrelationFunctionAllocation = createEndDetectionCorrelationFunctionAllocation(
            endDetectionCorrelationFunctionLength,
            TEXT_EDGE_COLUMN_INDEX,
            CORRELATION_START_COLUMN_INDEX,
            CORRELATION_END_COLUMN_INDEX,
            AMPLITUDE_BEFORE_TEXT_EDGE_COLUMN_INDEX,
            AMPLITUDE_AFTER_TEXT_EDGE_COLUMN_INDEX,
            DIRECTION_SIGN_COLUMN_INDEX,
            mmToPixelFactor,
            meanPullKernelWidth,
            reducedAllocationSize,
            createIntAllocation
        )
        // setting end detection parameters to the script
        processingScript._endDetectionCorrelationFunctionLength = endDetectionCorrelationFunctionLength
        processingScript._endDetectionTextEdgeColumn = TEXT_EDGE_COLUMN_INDEX
        processingScript._endDetectionCorrelationStartColumn = CORRELATION_START_COLUMN_INDEX
        processingScript._endDetectionCorrelationEndColumn = CORRELATION_END_COLUMN_INDEX
        processingScript._endDetectionAmplitudeBeforeTextEdgeColumn =
            AMPLITUDE_BEFORE_TEXT_EDGE_COLUMN_INDEX
        processingScript._endDetectionAmplitudeAfterTextEdgeColumn =
            AMPLITUDE_AFTER_TEXT_EDGE_COLUMN_INDEX
        processingScript._endDetectionDirectionSignColumn = DIRECTION_SIGN_COLUMN_INDEX
        processingScript._endDetectionCorrelationStripWidth =
                correlationFunctionTextEdges[3] - correlationFunctionTextEdges[0]
        processingScript._vicinityHalfWidth =
                calculateEndDetectionVicinityHalfWidth(mmToPixelFactor, meanPullKernelWidth)
        val endDetectionLocalMaxAreaMeanHalfWidth =
                calculateEndDetectionLocalMaxAreaMeanHalfWidth(mmToPixelFactor, meanPullKernelWidth)
        processingScript._localMaxAreaMeanHalfWidth = endDetectionLocalMaxAreaMeanHalfWidth
        processingScript._localMaxCalculationPadding = 3 * endDetectionLocalMaxAreaMeanHalfWidth
        processingScript._endMarkerDashThickness =
                calculateDashThicknessInPixels(END_MARKER_DASH_THICKNESS_MM, mmToPixelFactor)

        // setting allocations to the script
        processingScript._reducedGrayAllocation = reducedGrayAllocation
        processingScript._reducedColumnSumAllocation = reducedColumnSumAllocation
        processingScript._reducedLocalMeanAllocation = reducedLocalMeanAllocation
        processingScript._reducedLocalDeviationAllocation = reducedLocalDeviationAllocation
        processingScript._textSearchCorrelationFunctionAllocation =
            textSearchCorrelationFunctionAllocation
        processingScript._lineDetectionAllocation = lineDetectionAllocation
        processingScript._endDetectionCorrelationFunctionAllocation =
            endDetectionCorrelationFunctionAllocation

        return textTopToBottomWidth
    }

    fun variablesClose() {
        reducedGrayAllocation.destroy()
        reducedColumnSumAllocation.destroy()
        reducedLocalMeanAllocation.destroy()
        reducedLocalDeviationAllocation.destroy()
        textSearchCorrelationFunctionAllocation.destroy()
        lineDetectionAllocation.destroy()
        endDetectionCorrelationFunctionAllocation.destroy()
    }

    fun detectLine() {
        processingScript.forEach_reductionKernel(reducedGrayAllocation)
        processingScript.forEach_reducedLocalMeanColumnSumKernel(
            reducedColumnSumAllocation,
            reducedLocalMeanColumnSumKernelLaunchOptions
        )
        processingScript.forEach_reducedLocalMeanRowSumKernel(
            reducedColumnSumAllocation,
            reducedLocalMeanRowSumKernelLaunchOptions
        )
        processingScript.forEach_reducedLocalDeviationKernel(
            reducedLocalMeanAllocation,
            reducedLocalDeviationKernelLaunchOptions
        )
        processingScript.forEach_textSearchCorrelationKernel(
            lineDetectionAllocation,
            textSearchCorrelationKernelLaunchOptions
        )
        processingScript.forEach_lineDetectionKernel(
            lineDetectionAllocation,
            lineDetectionKernelLaunchOptions
        )
        processingScript.forEach_lineMarkerDrawKernel(
            lineDetectionAllocation,
            lineMarkerDrawKernelLaunchOptions
        )
    }

    fun detectLineEnds() {
        processingScript.forEach_endDetectionKernel(
            endDetectionCorrelationFunctionAllocation,
            endDetectionKernelLaunchOptions
        )
        processingScript.forEach_endMarkerDrawKernel(
            endDetectionCorrelationFunctionAllocation,
            endMarkerDrawKernelLaunchOptions
        )
    }

    private fun createEndDetectionCorrelationFunctionAllocation(
            correlationFunctionLength: Int,
            textEdgeColumn: Int,
            correlationStartColumn: Int,
            correlationEndColumn: Int,
            amplitudeBeforeTextEdgeColumn: Int,
            amplitudeAfterTextEdgeColumn: Int,
            directionSignColumn: Int,
            mmToPixelFactor: Float,
            meanPullKernelWidth: Int,
            computationFieldSize: Size,
            createIntAllocation: (Size) -> Allocation
    ): Allocation {
        val numberOfRows = NUMBER_OF_ROWS
        val numberOfColumns = NUMBER_OF_COLUMNS
        val allocation = createIntAllocation(Size(numberOfColumns, numberOfRows))
        val array = IntArray(numberOfColumns * numberOfRows)

        // setting parameters for the text strip start in allocation row 0
        array[textEdgeColumn] = (END_DETECTION_CORRELATION_FUNCTION_TEXT_EDGE_MM * mmToPixelFactor
                ).roundToInt() / meanPullKernelWidth
        array[correlationStartColumn] = 0
        array[correlationEndColumn] = computationFieldSize.width / 3
        // Unit amplitude must be guaranteed to be > 0 after modulo division, therefor the "+ 1" at the end.
        // The actual value of the unit amplitude does not matter, and is set approximately.
        val unitAmplitude = CORRELATION_FUNCTION_AMPLITUDE / correlationFunctionLength + 1
        val amplitudeBeforeTextEdge = - unitAmplitude * (correlationFunctionLength - array[textEdgeColumn])
        val amplitudeAfterTextEdge = unitAmplitude * array[textEdgeColumn]
        array[amplitudeBeforeTextEdgeColumn] = amplitudeBeforeTextEdge
        array[amplitudeAfterTextEdgeColumn] = amplitudeAfterTextEdge
        array[directionSignColumn] = 1

        // setting parameters for the text strip end in allocation row 1
        array[numberOfColumns + textEdgeColumn] = correlationFunctionLength - array[textEdgeColumn]
        array[numberOfColumns + correlationStartColumn] = computationFieldSize.width * 2 / 3
        array[numberOfColumns + correlationEndColumn] = computationFieldSize.width
        array[numberOfColumns + amplitudeBeforeTextEdgeColumn] = amplitudeAfterTextEdge
        array[numberOfColumns + amplitudeAfterTextEdgeColumn] = amplitudeBeforeTextEdge
        array[numberOfColumns + directionSignColumn] = -1

        // copy data from array to allocation
        allocation.copyFrom(array)

        return allocation
    }

    private fun calculateEndDetectionVicinityHalfWidth(
            mmToPixelFactor: Float, meanPullKernelWidth: Int
    ): Int {
        var value = (
                END_DETECTION_LOCAL_MAX_VICINITY_HALF_WIDTH * mmToPixelFactor / meanPullKernelWidth
                ).roundToInt()
        if (value < 1) { value = 1 }
        return value
    }

    private fun calculateEndDetectionLocalMaxAreaMeanHalfWidth(
            mmToPixelFactor: Float, meanPullKernelWidth: Int
    ): Int {
        var value = (
                END_DETECTION_LOCAL_MAX_AREA_MEAN_HALF_WIDTH * mmToPixelFactor / meanPullKernelWidth
                ).roundToInt()
        if (value < 1) { value = 1 }
        return value
    }

    private fun calculateDashThicknessInPixels(dashThicknessMM: Float, mmToPixelFactor: Float): Int {
        // dash thickness must be odd to ensure marker symmetry when operating with half widths
        return ((dashThicknessMM * mmToPixelFactor).roundToInt() / 2) * 2 + 1
    }

    private fun calculateTextSearchCorrelationStripHalfWidth(
            mmToPixelFactor: Float, meanPullKernelWidth: Int
    ): Int {
        return (TEXT_SEARCH_CORRELATION_STRIP_HALF_WIDTH_MM * mmToPixelFactor).roundToInt() / meanPullKernelWidth
    }

    private fun calculateMeanAngleXBase(): Int {
        // calculating calculatedMeanAngleXBase for lineDetectionKernel
        var calculatedMeanAngleXBase = TEXT_SEARCH_CORRELATION_NUMBER_OF_RUNS - 1
        for (i in 2 until TEXT_SEARCH_CORRELATION_NUMBER_OF_RUNS - 1) {
            calculatedMeanAngleXBase *= i
        }
        return calculatedMeanAngleXBase
    }

    private fun calculateCorrelationFunctionTextEdges(
            mmToPixelFactor: Float, meanPullKernelWidth: Int
    ): IntArray {
        val textHeight = TEXT_HEIGHT
        val distanceFromStartToText = DISTANCE_FROM_START_TO_TEXT
        val spaceBetweenTextLines = SPACE_BETWEEN_TEXT_LINES
        val textEdges = IntArray(4)

        textEdges[0] = (
                distanceFromStartToText * mmToPixelFactor / meanPullKernelWidth).roundToInt()
        textEdges[1] = ((distanceFromStartToText + textHeight
                ) * mmToPixelFactor / meanPullKernelWidth).roundToInt()
        textEdges[2] = ((distanceFromStartToText + textHeight +
                spaceBetweenTextLines) * mmToPixelFactor / meanPullKernelWidth).roundToInt()
        textEdges[3] = ((distanceFromStartToText + textHeight + spaceBetweenTextLines +
                textHeight) * mmToPixelFactor / meanPullKernelWidth).roundToInt()
        return textEdges
    }

    private fun createTextSearchCorrelationFunctionAllocation(
            correlationFunctionLength: Int, textEdges: IntArray, createIntAllocation: (Size) -> Allocation
    ): Pair<Allocation, Float> {
        val correlationFunctionAmplitude = CORRELATION_FUNCTION_AMPLITUDE
        val array = IntArray(correlationFunctionLength)

        for (i in 0 until textEdges[0]) {
            array[i] = 0
        }
        for (i in textEdges[0] until textEdges[1]) {
            array[i] = correlationFunctionAmplitude
        }
        for (i in textEdges[1] until textEdges[2]) {
            array[i] = 0
        }
        for (i in textEdges[2] until textEdges[3]) {
            array[i] = correlationFunctionAmplitude
        }
        for (i in textEdges[3] until correlationFunctionLength) {
            array[i] = 0
        }
        var sum = 0
        for (i in 0 until correlationFunctionLength) {
            sum += array[i]
        }
        sum /= correlationFunctionLength
        var maxAmplitude = 0
        for (i in 0 until correlationFunctionLength) {
            array[i] -= sum
            if (maxAmplitude < abs(array[i])) {maxAmplitude = abs(array[i])
            }
        }
        val correlationAmplitudePeakToMeanRatio = maxAmplitude.toFloat() / sum.toFloat()
        val allocation = createIntAllocation(Size(1, correlationFunctionLength))
        allocation.copyFrom(array)

        return Pair(allocation, correlationAmplitudePeakToMeanRatio)
    }

    private fun calculateReducedAllocationSize(size: Size, meanPullKernelWidth: Int): Size {
        return Size(size.width / meanPullKernelWidth,
            size.height / meanPullKernelWidth)
    }
}


