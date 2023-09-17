package com.code.sample.readmrz

import android.content.Context
import android.graphics.ImageFormat
import android.os.SystemClock
import android.renderscript.*
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import kotlin.math.abs
import kotlin.math.min

// colors
private val COLOR_WHITE = intArrayOf(255, 255, 255, 255)
private val COLOR_BLACK = intArrayOf(0, 0, 0, 255)
private val COLOR_GREEN_A400 = intArrayOf(0, 230, 118, 255) // #00E676
private val NEUTRAL_COLOR = COLOR_WHITE
private val SUCCESSFUL_EVENT_COLOR = COLOR_GREEN_A400
private val DASHED_RECTANGLE_COLOR_1 = COLOR_WHITE
private val DASHED_RECTANGLE_COLOR_2 = COLOR_BLACK

// dimensions
private val OPTIMAL_COMPUTATION_FIELD_SIZE: Size = Size(800, 300)
private const val STANDARD_DOCUMENT_PAGE_WIDTH = 125f // mm
private const val STANDARD_DOCUMENT_PAGE_HEIGHT = 88f // mm
private const val AREA_EXPANSION_FACTOR = 1.05f // expanding area to see space around document page


class ProcessingModel(private val currentContext: Context) {
    // camera output size selection parameters
    val BEST_INPUT_SIZE: Size = Size(800, 600)
    val RATIO_INFLUENCE_ON_INPUT_SIZE: Float = 1.5f //larger -> more influence
    val LARGER_SQUARE_FACTOR: Float = 4f //larger -> larger square has more advantage

    private val rs: RenderScript = RenderScript.create(currentContext)
    private val processingScript: ScriptC_processing = ScriptC_processing(rs)
    private val basicProcessor = BasicParameters(processingScript)
    private val textStripFinder = TextStripFinderParameters(processingScript)
    private val fbBuffer = FeedbackBuffer(processingScript)
    private val textCutter = TextStripCutterParameters(processingScript)
    private val segmenter = SegmenterModelParameters(processingScript, currentContext)
    private val charactersCutter = CharactersCutterParameters(processingScript)
    private val recognizer = RecognizerModelParameters(processingScript, currentContext)
    private val mrzVerifier = MRZVerifier()


    fun variablesSetUp(
            inputAllocationSize: Size,
            inputListener: Allocation.OnBufferAvailableListener,
            outputTextureView: TextureView
    ): Surface {
        // calculating dimension parameters
        val computationFieldSize = calculateComputationFieldSize(inputAllocationSize)
        val mmToPixelFactor = calculateMMToPixelFactor(computationFieldSize)
        val outAllocationSize = calculateOutAllocationSize(computationFieldSize.width,
            inputAllocationSize.height, outputTextureView.width, outputTextureView.height)

        // setting basic processor parameters
        basicProcessor.variablesSetUp(
            inputAllocationSize,
            inputListener,
            computationFieldSize,
            outAllocationSize,
            outputTextureView,
            mmToPixelFactor,
            ::createYuvAllocation,
            ::createOneLayerIntAllocation,
            ::createOutAllocation,
            SUCCESSFUL_EVENT_COLOR,
            DASHED_RECTANGLE_COLOR_1,
            DASHED_RECTANGLE_COLOR_2
        )

        // setting text strip position finder parameters
        val textTopToBottomWidth = textStripFinder.variablesSetUp(computationFieldSize,
            mmToPixelFactor, ::createOneLayerIntAllocation, SUCCESSFUL_EVENT_COLOR, NEUTRAL_COLOR)

        // setting feedback buffer parameters
        fbBuffer.variablesSetUp(::createOneLayerIntAllocation)

        // setting text cutter parameters
        val textStripWidth = textCutter.generalVariablesSetUp(textTopToBottomWidth, mmToPixelFactor)

        // setting text strip segmenter parameters
        segmenter.generalVariablesSetUp(textStripWidth, ::createOneLayerFloatAllocation)

        // setting characters cutter parameters
        val numberOfColumnsInDetectedCharactersAllocation =
            charactersCutter.generalVariablesSetUp(::createOneLayerIntAllocation)

        // setting character recognizer parameters
        recognizer.generalVariablesSetUp(numberOfColumnsInDetectedCharactersAllocation,
                ::createOneLayerFloatAllocation, ::createOneLayerIntAllocation)

        // return an input surface for camera output
        return basicProcessor.getInputSurface()
    }

    fun variablesClose() {
        basicProcessor.variablesClose()
        textStripFinder.variablesClose()
        fbBuffer.variablesClose()
        segmenter.generalVariablesClose()
        charactersCutter.generalVariablesClose()
        recognizer.generalVariablesClose()
    }

    fun processInputImage(): MRZDetectionResult {
        var correctMRZDetected = false
        var recognizedText: Array<Array<String>> = arrayOf(arrayOf(""))

        basicProcessor.ioReceive()
        basicProcessor.performProcessing()
        textStripFinder.detectLine()

        // if text line was detected then perform further processing
        fbBuffer.copyAllocationToArray()
        if (fbBuffer.feedbackArray[fbBuffer.lineDetectionFlagColumn] == fbBuffer.lineDetectedFlag) {
            textStripFinder.detectLineEnds()
            fbBuffer.copyAllocationToArray()

            // cutout text strips
            val textStripLength = textCutter.currentFrameVariablesSetUp(
                abs(fbBuffer.feedbackArray[fbBuffer.textStripEndXColumn] -
                        fbBuffer.feedbackArray[fbBuffer.textStripStartXColumn]),
                ::createOneLayerFloatAllocation
            )
            textCutter.cutoutTextStrips()

            // set up segmenter variables
            segmenter.currentFrameVariablesSetUp(
                    textStripLength,
                    ::createOneLayerFloatAllocation,
                    ::createOneLayerIntAllocation
            )
            // segment text strip 1
            segmenter.segmentTextStrip(textCutter.textStrip1CutoutAllocation)

            // switch charactersCutter to strip 1 and detect character groups
            charactersCutter.switchToStrip1()
            charactersCutter.detectCharacterGroups()

            // if number of detected characters is correct then perform further processing
            fbBuffer.copyAllocationToArray()
            val nDetectedCharacters = fbBuffer.feedbackArray[fbBuffer.nDetectedCharactersColumn]
            if (nDetectedCharacters in mrzVerifier.correctStringLength) {
                // set height detector launch options
                charactersCutter.currentFrameVariablesSetUp(nDetectedCharacters)
                // detect characters heights for strip 1 based on segmenterNormalizedInputAllocation
                // that contains strip 1 normalized cutout
                charactersCutter.detectCharactersHeights()

                // segment text strip 2
                segmenter.segmentTextStrip(textCutter.textStrip2CutoutAllocation)

                // switch charactersCutter to strip 2 and detect character groups
                charactersCutter.switchToStrip2()
                charactersCutter.detectCharacterGroups()

                // if numbers of detected characters in strip 1 and strip 2 are equal
                // then perform further processing
                fbBuffer.copyAllocationToArray()
                if (nDetectedCharacters == fbBuffer.feedbackArray[fbBuffer.nDetectedCharactersColumn]) {
                    recognizedText = Array(2) {(Array(nDetectedCharacters) {""})}
                    mrzVerifier.detectedStringLength = nDetectedCharacters

                    basicProcessor.showSingleColorRectangle()
                    basicProcessor.ioSend()

                    // detect characters heights for strip 2 based on segmenterNormalizedInputAllocation
                    // that contains strip 2 normalized cutout
                    charactersCutter.detectCharactersHeights()

                    // switch recognizer input to strip 2 and calculate input allocation sizes
                    // for each character in the strip
                    recognizer.setTextStripAllocationAndCalculateInputSizes(
                            textCutter.textStrip2CutoutAllocation,
                            charactersCutter.detectedCharactersProcessingLaunchOptions
                    )

                    // recognize characters in strip 2
                    for (characterIndex in 0 until nDetectedCharacters) {
                        recognizer.currentCharacterVariablesSetUp(
                            characterIndex, ::createOneLayerFloatAllocation)
                        recognizer.recognizeCharacter()

                        fbBuffer.copyAllocationToArray()
                        val characterClass = fbBuffer.feedbackArray[fbBuffer.characterClassColumn]
                        recognizedText[1][characterIndex] = recognizer.characterClasses[characterClass]

                        // rs.finish() ensures that all data is processed before destroying allocations
                        rs.finish()
                        recognizer.currentCharacterVariablesClose()
                    }

                    // check detected string 2 that it is a correct MRZ string, and perform recognition
                    // of string 1 only if string 2 is correct
                    val passedString2PreliminaryCheck =
                            mrzVerifier.mrzString2PreliminaryCheck(recognizedText)
                    if (passedString2PreliminaryCheck) {
                        // switch charactersCutter back to strip 1
                        charactersCutter.switchToStrip1()

                        // switch recognizer input to strip 1 and calculate input allocation sizes
                        // for each character in the strip
                        recognizer.setTextStripAllocationAndCalculateInputSizes(
                                textCutter.textStrip1CutoutAllocation,
                                charactersCutter.detectedCharactersProcessingLaunchOptions
                        )

                        // recognize characters in strip 1
                        for (characterIndex in 0 until nDetectedCharacters) {
                            recognizer.currentCharacterVariablesSetUp(
                                    characterIndex, ::createOneLayerFloatAllocation)
                            recognizer.recognizeCharacter()

                            fbBuffer.copyAllocationToArray()
                            val characterClass = fbBuffer.feedbackArray[fbBuffer.characterClassColumn]
                            recognizedText[0][characterIndex] = recognizer.characterClasses[characterClass]

                            // rs.finish() ensures that all data is processed before destroying allocations
                            rs.finish()
                            recognizer.currentCharacterVariablesClose()
                        }

                        // check detected string 1 that it is a correct MRZ string, and
                        // if it is correct, then set correctMRZDetected flag
                        correctMRZDetected = mrzVerifier.mrzCompositeCheck(recognizedText)
                    }
                } else { // wrong number of detected characters in string 1
                    basicProcessor.showDashedRectangle()
                    basicProcessor.ioSend()
                }
            } else { // wrong number of detected characters in string 2
                basicProcessor.showDashedRectangle()
                basicProcessor.ioSend()
            }

            // rs.finish() ensures that all data is processed before destroying allocations
            rs.finish()
            textCutter.currentFrameVariablesClose()
            segmenter.currentFrameVariablesClose()

        } else { // line is not detected
            basicProcessor.showDashedRectangle()
            basicProcessor.ioSend()
        }

        return MRZDetectionResult(correctMRZDetected, recognizedText)
    }

    private fun createYuvAllocation(size: Size): Allocation {
        val yuvTypeBuilder = Type.Builder(rs, Element.YUV(rs))
        yuvTypeBuilder.setX(size.width)
        yuvTypeBuilder.setY(size.height)
        yuvTypeBuilder.setYuvFormat(ImageFormat.YUV_420_888)
        return Allocation.createTyped(rs, yuvTypeBuilder.create(),
                Allocation.USAGE_IO_INPUT or Allocation.USAGE_SCRIPT)
    }

    private fun createOneLayerIntAllocation(size: Size): Allocation {
        val oneLayerIntTypeBuilder = Type.Builder(rs, Element.I32(rs))
        oneLayerIntTypeBuilder.setX(size.width)
        oneLayerIntTypeBuilder.setY(size.height)
        return Allocation.createTyped(rs, oneLayerIntTypeBuilder.create(),
                Allocation.USAGE_SCRIPT)
    }

    private fun createOneLayerFloatAllocation(size: Size): Allocation {
        val oneLayerFloatTypeBuilder = Type.Builder(rs, Element.F32(rs))
        oneLayerFloatTypeBuilder.setX(size.width)
        oneLayerFloatTypeBuilder.setY(size.height)
        return Allocation.createTyped(rs, oneLayerFloatTypeBuilder.create(),
                Allocation.USAGE_SCRIPT)
    }

    private fun calculateOutAllocationSize(
            inputWidth: Int, inputHeight: Int, textureWidth: Int, textureHeight: Int
    ): Size {
        return if (inputHeight < inputWidth * textureHeight / textureWidth) {
            Size(inputHeight * textureWidth / textureHeight, inputHeight)
        } else {
            Size(inputWidth, inputWidth * textureHeight / textureWidth)
        }
    }

    private fun createOutAllocation(size: Size): Allocation  {
        val rgbTypeBuilder = Type.Builder(rs, Element.RGBA_8888(rs))
        rgbTypeBuilder.setX(size.width)
        rgbTypeBuilder.setY(size.height)
        return Allocation.createTyped(rs, rgbTypeBuilder.create(),
                Allocation.USAGE_IO_OUTPUT or Allocation.USAGE_SCRIPT)
    }

    private fun calculateComputationFieldSize(
            allocationSize: Size): Size {
        return Size(min(allocationSize.width, OPTIMAL_COMPUTATION_FIELD_SIZE.width),
                min(allocationSize.height, OPTIMAL_COMPUTATION_FIELD_SIZE.height))
    }

    private fun calculateMMToPixelFactor(allocationSize: Size): Float {
        val pageWidth = STANDARD_DOCUMENT_PAGE_WIDTH // mm
        val areaExpansion = AREA_EXPANSION_FACTOR // expanding area to see space around document page

        // allocation width must completely cover width of the document page and expansion area around it
        return allocationSize.width.toFloat() / (pageWidth * areaExpansion)
    }
}



