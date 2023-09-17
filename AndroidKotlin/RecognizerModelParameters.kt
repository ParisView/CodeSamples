package com.code.sample.readmrz

import android.content.Context
import android.renderscript.Allocation
import android.renderscript.Script
import android.util.Size
import java.io.BufferedReader


// size calculation kernel parameters
private const val INPUT_SIZES_ALLOCATION_NUMBER_OF_ROWS = 2 // row 0 for width (x), row 1 for height (y)
private const val PASTE_SHIFTS_ALLOCATION_NUMBER_OF_ROWS = 4
private const val PASTE_SHIFT_X_START_ROW = 0
private const val PASTE_SHIFT_X_END_ROW = 1
private const val PASTE_SHIFT_Y_START_ROW = 2
private const val PASTE_SHIFT_Y_END_ROW = 3

// recognition model parameters
private const val CONVOLUTION_KERNEL_HALF_WIDTH = 1
private const val CONVOLUTION_KERNEL_WIDTH = 2 * CONVOLUTION_KERNEL_HALF_WIDTH + 1
private const val CONVOLUTION_KERNEL_ALLOCATION_SHRINK = 2 * CONVOLUTION_KERNEL_HALF_WIDTH
private const val CONVOLUTION_PADDING_WIDTH = CONVOLUTION_KERNEL_HALF_WIDTH
private const val CONVOLUTION_OUTPUT_SHIFT = CONVOLUTION_KERNEL_HALF_WIDTH
private const val BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS = 2
private const val BATCH_NORM_EFFECTIVE_WEIGHT_ROW_INDEX = 0
private const val BATCH_NORM_EFFECTIVE_BIAS_ROW_INDEX = 1
private const val L1_NUMBER_OF_INPUTS = 1
private const val STAGE_1_NUMBER_OF_OUTPUT_FEATURES = 16
private const val L1_NUMBER_OF_OUTPUT_FEATURES = STAGE_1_NUMBER_OF_OUTPUT_FEATURES
private const val L2_NUMBER_OF_OUTPUT_FEATURES = STAGE_1_NUMBER_OF_OUTPUT_FEATURES
private const val L3_NUMBER_OF_OUTPUT_FEATURES = STAGE_1_NUMBER_OF_OUTPUT_FEATURES
private const val L4_NUMBER_OF_OUTPUT_FEATURES = 32
private const val L5_NUMBER_OF_OUTPUT_FEATURES = L4_NUMBER_OF_OUTPUT_FEATURES
private const val L6_NUMBER_OF_OUTPUT_FEATURES = 32
private const val L7_NUMBER_OF_OUTPUT_FEATURES = 64
private const val L8_NUMBER_OF_OUTPUT_FEATURES = L7_NUMBER_OF_OUTPUT_FEATURES
private const val L9_NUMBER_OF_OUTPUT_FEATURES = 128
private const val L10_NUMBER_OF_OUTPUT_FEATURES = 128
private const val L11_NUMBER_OF_OUTPUT_FEATURES = 37

// min width and height of input allocation (for calculations producing 1x1 sized result)
private const val W_H_THRESHOLD_1X1 = 18 // 18x18 pixels for 1x1 result of nn calculations
// width and height increment step (each step increments 1 to the size of the result)
private const val W_H_STEP = 4

// image mean and reciprocal to standard deviation calculated on whole recognition model dataset
private const val CHARACTER_IMAGE_MEAN = -0.02598962001502514f
private const val CHARACTER_IMAGE_STD_RECIPROCAL = 2.9873864364394844f


class RecognizerModelParameters(
        private val processingScript: ScriptC_processing,
        private val currentContext: Context
) {
    private lateinit var inputSizesAllocation: Allocation
    private lateinit var pasteShiftsAllocation: Allocation

    private lateinit var normalizedInputAllocation: Allocation
    private lateinit var l1ConvolutionOutAllocationsArray: Array<Allocation>
    private lateinit var l1Convolution3x3WeightsAllocationsArray: Array<Allocation>
    private lateinit var l1Convolution3x3BiasAllocation: Allocation
    private lateinit var l1BatchNormAllocation: Allocation
    private lateinit var l2ConvolutionOutAllocationsArray: Array<Allocation>
    private lateinit var l2Convolution3x3WeightsAllocationsArray: Array<Allocation>
    private lateinit var l2Convolution3x3BiasAllocation: Allocation
    private lateinit var l2BatchNormAllocation: Allocation
    private lateinit var l3ConvolutionOutAllocationsArray: Array<Allocation>
    private lateinit var l3Convolution3x3WeightsAllocationsArray: Array<Allocation>
    private lateinit var l3Convolution3x3BiasAllocation: Allocation
    private lateinit var l3BatchNormAllocation: Allocation
    private lateinit var l4ConvolutionOutAllocationsArray: Array<Allocation>
    private lateinit var l4Convolution3x3WeightsAllocationsArray: Array<Allocation>
    private lateinit var l4Convolution3x3BiasAllocation: Allocation
    private lateinit var l4BatchNormAllocation: Allocation
    private lateinit var l5MaxPoolAllocationsArray: Array<Allocation>
    private lateinit var l6ConvolutionOutAllocationsArray: Array<Allocation>
    private lateinit var l6Convolution3x3WeightsAllocationsArray: Array<Allocation>
    private lateinit var l6Convolution3x3BiasAllocation: Allocation
    private lateinit var l6BatchNormAllocation: Allocation
    private lateinit var l7ConvolutionOutAllocationsArray: Array<Allocation>
    private lateinit var l7Convolution3x3WeightsAllocationsArray: Array<Allocation>
    private lateinit var l7Convolution3x3BiasAllocation: Allocation
    private lateinit var l7BatchNormAllocation: Allocation
    private lateinit var l8MaxPoolAllocationsArray: Array<Allocation>
    private lateinit var l9ConvolutionOutAllocation: Allocation
    private lateinit var l9Convolution3x3WeightsAllocationsArray: Array<Allocation>
    private lateinit var l9Convolution3x3BiasAllocation: Allocation
    private lateinit var l9BatchNormAllocation: Allocation
    private lateinit var l10LinearOutAllocation: Allocation
    private lateinit var l10LinearWeightsAllocation: Allocation
    private lateinit var l10LinearBiasAllocation: Allocation
    private lateinit var l11LinearOutAllocation: Allocation
    private lateinit var l11LinearWeightsAllocation: Allocation
    private lateinit var l11LinearBiasAllocation: Allocation

    private val padWithZeroStartLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val padWithZeroLowestLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val padWithZeroStage1EndLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val padWithZeroStage1HighestLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val recognizerStage1ConvolutionLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val padWithZeroL6EndLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val padWithZeroL6HighestLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val recognizerL6ConvolutionLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    private val predictionKernelLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()

    private lateinit var inputSizesArray: IntArray
    private var sizesAllocationNumberOfColumns = 1
    private val pasteShiftXStartRow = PASTE_SHIFT_X_START_ROW
    private val pasteShiftXEndRow = PASTE_SHIFT_X_END_ROW
    private val pasteShiftYStartRow = PASTE_SHIFT_Y_START_ROW
    private val pasteShiftYEndRow = PASTE_SHIFT_Y_END_ROW
    private val thresholdWH1x1 = W_H_THRESHOLD_1X1
    private val stepWH = W_H_STEP
    private val characterImageMean: Float = CHARACTER_IMAGE_MEAN
    private val characterImageSTDReciprocal: Float = CHARACTER_IMAGE_STD_RECIPROCAL
    private val convolutionKernelWidth: Int = CONVOLUTION_KERNEL_WIDTH
    private val convolutionPaddingWidth = CONVOLUTION_PADDING_WIDTH
    private val convolutionOutputShift = CONVOLUTION_OUTPUT_SHIFT
    private val batchNormEffectiveWeightRow: Int = BATCH_NORM_EFFECTIVE_WEIGHT_ROW_INDEX
    private val batchNormEffectiveBiasRow: Int = BATCH_NORM_EFFECTIVE_BIAS_ROW_INDEX
    private var l9MaxPoolWidth: Int = 0
    private var l9MaxPoolHeight: Int = 0
    private var l10NumberOfInputs: Int = 0
    private var l11NumberOfInputs: Int = 0
    private var predictionNumberOfInputs: Int = 0

    fun generalVariablesSetUp(
            columnsInDetectedCharacters: Int,
            createFloatAllocation: (Size) -> Allocation,
            createIntAllocation: (Size) -> Allocation
    ) {
        // creating allocations, that will contain sizes of input allocations and shifts of
        // cutout areas for each character
        sizesAllocationNumberOfColumns = columnsInDetectedCharacters
        inputSizesAllocation = createIntAllocation(Size(
                columnsInDetectedCharacters, INPUT_SIZES_ALLOCATION_NUMBER_OF_ROWS))
        pasteShiftsAllocation = createIntAllocation(Size(
                columnsInDetectedCharacters, PASTE_SHIFTS_ALLOCATION_NUMBER_OF_ROWS))
        inputSizesArray = IntArray(columnsInDetectedCharacters * INPUT_SIZES_ALLOCATION_NUMBER_OF_ROWS)

        // setting launch options
        padWithZeroStartLaunchOptions.setX(0, convolutionPaddingWidth)
        padWithZeroLowestLaunchOptions.setY(0, convolutionPaddingWidth)
        predictionKernelLaunchOptions.setX(0, 1)
        predictionKernelLaunchOptions.setY(0, 1)

        // setting model layers sizes
        val batchNormAllocationNumberOfRows = BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS
        val l1NumberOfInputs = L1_NUMBER_OF_INPUTS
        val l1NumberOfFeatures = L1_NUMBER_OF_OUTPUT_FEATURES
        val l2NumberOfFeatures = L2_NUMBER_OF_OUTPUT_FEATURES
        val l3NumberOfFeatures = L3_NUMBER_OF_OUTPUT_FEATURES
        val l4NumberOfFeatures = L4_NUMBER_OF_OUTPUT_FEATURES
        val l5NumberOfFeatures = L5_NUMBER_OF_OUTPUT_FEATURES
        val l6NumberOfFeatures = L6_NUMBER_OF_OUTPUT_FEATURES
        val l7NumberOfFeatures = L7_NUMBER_OF_OUTPUT_FEATURES
        val l8NumberOfFeatures = L8_NUMBER_OF_OUTPUT_FEATURES
        val l9NumberOfFeatures = L9_NUMBER_OF_OUTPUT_FEATURES
        l10NumberOfInputs = l9NumberOfFeatures
        val l10NumberOfFeatures = L10_NUMBER_OF_OUTPUT_FEATURES
        l11NumberOfInputs = l10NumberOfFeatures
        val l11NumberOfFeatures = L11_NUMBER_OF_OUTPUT_FEATURES
        predictionNumberOfInputs = l11NumberOfFeatures

        // creating raw model parameters reader
        val modelParametersRaw = currentContext.resources.openRawResource(R.raw.recognizer_parameters)
        val modelParametersReader = modelParametersRaw.bufferedReader()

        // creating array of layer 1 output allocations filling it with temporary allocations, because
        // their size is different for every character, and then destroying temporary allocations
        l1ConvolutionOutAllocationsArray = createOutputAllocationsArray(
                l1NumberOfFeatures, createFloatAllocation)
        // creating array of layer 1 weights allocations and filling them with model parameters
        l1Convolution3x3WeightsAllocationsArray = createAndFillConvolutionWeightsAllocationsArray(
                l1NumberOfInputs,
                l1NumberOfFeatures,
                convolutionKernelWidth,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 1 bias allocation and filling it with model parameters
        l1Convolution3x3BiasAllocation = createAndFillBiasAllocation(
                l1NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 1 batch norm allocation and filling it with model parameters
        l1BatchNormAllocation = createAndFillBatchNormAllocation(
                l1NumberOfFeatures,
                batchNormAllocationNumberOfRows,
                modelParametersReader,
                createFloatAllocation
        )
        // creating array of layer 2 output allocations filling it with temporary allocations, because
        // their size is different for every character, and then destroying temporary allocations
        l2ConvolutionOutAllocationsArray = createOutputAllocationsArray(
                l2NumberOfFeatures, createFloatAllocation)
        // creating array of layer 2 weights allocations and filling them with model parameters
        l2Convolution3x3WeightsAllocationsArray = createAndFillConvolutionWeightsAllocationsArray(
                l1NumberOfFeatures,
                l2NumberOfFeatures,
                convolutionKernelWidth,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 2 bias allocation and filling it with model parameters
        l2Convolution3x3BiasAllocation = createAndFillBiasAllocation(
                l2NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 2 batch norm allocation and filling it with model parameters
        l2BatchNormAllocation = createAndFillBatchNormAllocation(
                l2NumberOfFeatures,
                batchNormAllocationNumberOfRows,
                modelParametersReader,
                createFloatAllocation
        )
        // creating array of layer 3 output allocations filling it with temporary allocations, because
        // their size is different for every character, and then destroying temporary allocations
        l3ConvolutionOutAllocationsArray = createOutputAllocationsArray(
                l3NumberOfFeatures, createFloatAllocation)
        // creating array of layer 3 weights allocations and filling them with model parameters
        l3Convolution3x3WeightsAllocationsArray = createAndFillConvolutionWeightsAllocationsArray(
                l2NumberOfFeatures,
                l3NumberOfFeatures,
                convolutionKernelWidth,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 3 bias allocation and filling it with model parameters
        l3Convolution3x3BiasAllocation = createAndFillBiasAllocation(
                l3NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 3 batch norm allocation and filling it with model parameters
        l3BatchNormAllocation = createAndFillBatchNormAllocation(
                l3NumberOfFeatures,
                batchNormAllocationNumberOfRows,
                modelParametersReader,
                createFloatAllocation
        )
        // creating array of layer 4 output allocations filling it with temporary allocations, because
        // their size is different for every character, and then destroying temporary allocations
        l4ConvolutionOutAllocationsArray = createOutputAllocationsArray(
                l4NumberOfFeatures, createFloatAllocation)
        // creating array of layer 4 weights allocations and filling them with model parameters
        l4Convolution3x3WeightsAllocationsArray = createAndFillConvolutionWeightsAllocationsArray(
                l3NumberOfFeatures,
                l4NumberOfFeatures,
                convolutionKernelWidth,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 4 bias allocation and filling it with model parameters
        l4Convolution3x3BiasAllocation = createAndFillBiasAllocation(
                l4NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 4 batch norm allocation and filling it with model parameters
        l4BatchNormAllocation = createAndFillBatchNormAllocation(
                l4NumberOfFeatures,
                batchNormAllocationNumberOfRows,
                modelParametersReader,
                createFloatAllocation
        )
        // creating array of layer 5 output allocations filling it with temporary allocations, because
        // their size is different for every character, and then destroying temporary allocations
        l5MaxPoolAllocationsArray = createOutputAllocationsArray(
                l5NumberOfFeatures, createFloatAllocation)
        // creating array of layer 6 output allocations filling it with temporary allocations, because
        // their size is different for every character, and then destroying temporary allocations
        l6ConvolutionOutAllocationsArray = createOutputAllocationsArray(
                l6NumberOfFeatures, createFloatAllocation)
        // creating array of layer 6 weights allocations and filling them with model parameters
        l6Convolution3x3WeightsAllocationsArray = createAndFillConvolutionWeightsAllocationsArray(
                l5NumberOfFeatures,
                l6NumberOfFeatures,
                convolutionKernelWidth,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 6 bias allocation and filling it with model parameters
        l6Convolution3x3BiasAllocation = createAndFillBiasAllocation(
                l6NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 6 batch norm allocation and filling it with model parameters
        l6BatchNormAllocation = createAndFillBatchNormAllocation(
                l6NumberOfFeatures,
                batchNormAllocationNumberOfRows,
                modelParametersReader,
                createFloatAllocation
        )
        // creating array of layer 7 output allocations filling it with temporary allocations, because
        // their size is different for every character, and then destroying temporary allocations
        l7ConvolutionOutAllocationsArray = createOutputAllocationsArray(
                l7NumberOfFeatures, createFloatAllocation)
        // creating array of layer 7 weights allocations and filling them with model parameters
        l7Convolution3x3WeightsAllocationsArray = createAndFillConvolutionWeightsAllocationsArray(
                l6NumberOfFeatures,
                l7NumberOfFeatures,
                convolutionKernelWidth,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 7 bias allocation and filling it with model parameters
        l7Convolution3x3BiasAllocation = createAndFillBiasAllocation(
                l7NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 7 batch norm allocation and filling it with model parameters
        l7BatchNormAllocation = createAndFillBatchNormAllocation(
                l7NumberOfFeatures,
                batchNormAllocationNumberOfRows,
                modelParametersReader,
                createFloatAllocation
        )
        // creating array of layer 8 output allocations filling it with temporary allocations, because
        // their size is different for every character, and then destroying temporary allocations
        l8MaxPoolAllocationsArray = createOutputAllocationsArray(
                l8NumberOfFeatures, createFloatAllocation)
        // creating layer 9 output allocation
        l9ConvolutionOutAllocation = createFloatAllocation(Size(l9NumberOfFeatures, 1))
        // creating array of layer 9 weights allocations and filling them with model parameters
        l9Convolution3x3WeightsAllocationsArray = createAndFillConvolutionWeightsAllocationsArray(
                l8NumberOfFeatures,
                l9NumberOfFeatures,
                convolutionKernelWidth,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 9 bias allocation and filling it with model parameters
        l9Convolution3x3BiasAllocation = createAndFillBiasAllocation(
                l9NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 9 batch norm allocation and filling it with model parameters
        l9BatchNormAllocation = createAndFillBatchNormAllocation(
                l9NumberOfFeatures,
                batchNormAllocationNumberOfRows,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 10 output allocation
        l10LinearOutAllocation = createFloatAllocation(Size(l10NumberOfFeatures, 1))
        // creating layer 10 weights allocation and filling it with model parameters
        l10LinearWeightsAllocation = createAndFillLinearWeightsAllocation(
                l10NumberOfInputs,
                l10NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 10 bias allocation and filling it with model parameters
        l10LinearBiasAllocation = createAndFillBiasAllocation(
                l10NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 11 output allocation
        l11LinearOutAllocation = createFloatAllocation(Size(l11NumberOfFeatures, 1))
        // creating layer 11 weights allocation and filling it with model parameters
        l11LinearWeightsAllocation = createAndFillLinearWeightsAllocation(
                l11NumberOfInputs,
                l11NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )
        // creating layer 11 bias allocation and filling it with model parameters
        l11LinearBiasAllocation = createAndFillBiasAllocation(
                l11NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )

        // closing raw model parameters reader
        modelParametersRaw.close()

        // setting character recognizer parameters to the script
        processingScript._inputSizesAllocation = inputSizesAllocation
        processingScript._pasteShiftsAllocation = pasteShiftsAllocation
        processingScript._pasteShiftXStartRow = pasteShiftXStartRow
        processingScript._pasteShiftXEndRow = pasteShiftXEndRow
        processingScript._pasteShiftYStartRow = pasteShiftYStartRow
        processingScript._pasteShiftYEndRow = pasteShiftYEndRow
        processingScript._inputSizesCalculationThresholdWH1x1 = thresholdWH1x1
        processingScript._inputSizesCalculationStepWH = stepWH
        processingScript._recognizerCharacterImageMean = characterImageMean
        processingScript._recognizerCharacterImageSTDReciprocal = characterImageSTDReciprocal
        processingScript._recognizerConvolutionKernelWidth = convolutionKernelWidth
        processingScript._recognizerConvolutionOutputShift = convolutionOutputShift
        processingScript._recognizerBatchNormEffectiveWeightRow = batchNormEffectiveWeightRow
        processingScript._recognizerBatchNormEffectiveBiasRow = batchNormEffectiveBiasRow
        processingScript._recognizerL1WeightsAllocations = l1Convolution3x3WeightsAllocationsArray
        processingScript._recognizerL1BiasAllocation = l1Convolution3x3BiasAllocation
        processingScript._recognizerL1BatchNormAllocation = l1BatchNormAllocation
        processingScript._recognizerL2WeightsAllocations = l2Convolution3x3WeightsAllocationsArray
        processingScript._recognizerL2BiasAllocation = l2Convolution3x3BiasAllocation
        processingScript._recognizerL2BatchNormAllocation = l2BatchNormAllocation
        processingScript._recognizerL3WeightsAllocations = l3Convolution3x3WeightsAllocationsArray
        processingScript._recognizerL3BiasAllocation = l3Convolution3x3BiasAllocation
        processingScript._recognizerL3BatchNormAllocation = l3BatchNormAllocation
        processingScript._recognizerL4WeightsAllocations = l4Convolution3x3WeightsAllocationsArray
        processingScript._recognizerL4BiasAllocation = l4Convolution3x3BiasAllocation
        processingScript._recognizerL4BatchNormAllocation = l4BatchNormAllocation
        processingScript._recognizerL6WeightsAllocations = l6Convolution3x3WeightsAllocationsArray
        processingScript._recognizerL6BiasAllocation = l6Convolution3x3BiasAllocation
        processingScript._recognizerL6BatchNormAllocation = l6BatchNormAllocation
        processingScript._recognizerL7WeightsAllocations = l7Convolution3x3WeightsAllocationsArray
        processingScript._recognizerL7BiasAllocation = l7Convolution3x3BiasAllocation
        processingScript._recognizerL7BatchNormAllocation = l7BatchNormAllocation
        processingScript._recognizerL9OutAllocation = l9ConvolutionOutAllocation
        processingScript._recognizerL9WeightsAllocations = l9Convolution3x3WeightsAllocationsArray
        processingScript._recognizerL9BiasAllocation = l9Convolution3x3BiasAllocation
        processingScript._recognizerL9BatchNormAllocation = l9BatchNormAllocation
        processingScript._recognizerL10OutAllocation = l10LinearOutAllocation
        processingScript._recognizerL10WeightsAllocation = l10LinearWeightsAllocation
        processingScript._recognizerL10BiasAllocation = l10LinearBiasAllocation
        processingScript._recognizerL10NumberOfInputs = l10NumberOfInputs
        processingScript._recognizerL11OutAllocation = l11LinearOutAllocation
        processingScript._recognizerL11WeightsAllocation = l11LinearWeightsAllocation
        processingScript._recognizerL11BiasAllocation = l11LinearBiasAllocation
        processingScript._recognizerL11NumberOfInputs = l11NumberOfInputs
        processingScript._recognizerPredictionNumberOfInputs = predictionNumberOfInputs
    }

    fun currentCharacterVariablesSetUp(
        characterIndex: Int,
        createFloatAllocation: (Size) -> Allocation,
    ) {
        val convolutionOutputShrink: Int = CONVOLUTION_KERNEL_ALLOCATION_SHRINK
        val stage1AllocationSize = Size(inputSizesArray[characterIndex],
                inputSizesArray[characterIndex + sizesAllocationNumberOfColumns])

        // creating l1, l2 and l3 output allocations for recognition model
        normalizedInputAllocation = createFloatAllocation(stage1AllocationSize)
        for (i in l1ConvolutionOutAllocationsArray.indices) {
            l1ConvolutionOutAllocationsArray[i] = createFloatAllocation(stage1AllocationSize)
        }
        for (i in l2ConvolutionOutAllocationsArray.indices) {
            l2ConvolutionOutAllocationsArray[i] = createFloatAllocation(stage1AllocationSize)
        }
        for (i in l3ConvolutionOutAllocationsArray.indices) {
            l3ConvolutionOutAllocationsArray[i] = createFloatAllocation(stage1AllocationSize)
        }

        // setting stage 1 (l1, l2, l3) padding launch options at the end and at the highest allocation point
        padWithZeroStage1EndLaunchOptions.setX(
                stage1AllocationSize.width - convolutionPaddingWidth, stage1AllocationSize.width)
        padWithZeroStage1HighestLaunchOptions.setY(
                stage1AllocationSize.height - convolutionPaddingWidth, stage1AllocationSize.height)

        // shrinking l4 allocation size
        val l4AllocationSize = Size(stage1AllocationSize.width - convolutionOutputShrink,
                stage1AllocationSize.height - convolutionOutputShrink)

        // setting stage 1 (l1, l2, l3) convolution launch options
        recognizerStage1ConvolutionLaunchOptions.setX(0, l4AllocationSize.width)
        recognizerStage1ConvolutionLaunchOptions.setY(0, l4AllocationSize.height)

        // creating l4 output allocations for recognition model
        for (i in l4ConvolutionOutAllocationsArray.indices) {
            l4ConvolutionOutAllocationsArray[i] = createFloatAllocation(l4AllocationSize)
        }

        // calculating l5 allocation size
        val l5AllocationSize = Size(l4AllocationSize.width / 2,l4AllocationSize.height / 2)

        // creating l5 max pool allocations for recognition model
        for (i in l5MaxPoolAllocationsArray.indices) {
            l5MaxPoolAllocationsArray[i] = createFloatAllocation(l5AllocationSize)
        }

        // setting l6 allocation size
        val l6AllocationSize = l5AllocationSize

        // creating l6 output allocations for recognition model
        for (i in l6ConvolutionOutAllocationsArray.indices) {
            l6ConvolutionOutAllocationsArray[i] = createFloatAllocation(l6AllocationSize)
        }

        // setting l6 padding launch options at the end and at the highest allocation point
        padWithZeroL6EndLaunchOptions.setX(
                l6AllocationSize.width - convolutionPaddingWidth, l6AllocationSize.width)
        padWithZeroL6HighestLaunchOptions.setY(
                l6AllocationSize.height - convolutionPaddingWidth, l6AllocationSize.height)

        // shrinking l7 allocation size
        val l7AllocationSize = Size(l6AllocationSize.width - convolutionOutputShrink,
                l6AllocationSize.height - convolutionOutputShrink)

        // setting l6 convolution launch options
        recognizerL6ConvolutionLaunchOptions.setX(0, l7AllocationSize.width)
        recognizerL6ConvolutionLaunchOptions.setY(0, l7AllocationSize.height)

        // creating l7 output allocations for recognition model
        for (i in l7ConvolutionOutAllocationsArray.indices) {
            l7ConvolutionOutAllocationsArray[i] = createFloatAllocation(l7AllocationSize)
        }

        // calculating l8 allocation size
        val l8AllocationSize = Size(l7AllocationSize.width / 2,l7AllocationSize.height / 2)

        // creating l8 max pool allocations for recognition model
        for (i in l8MaxPoolAllocationsArray.indices) {
            l8MaxPoolAllocationsArray[i] = createFloatAllocation(l8AllocationSize)
        }

        l9MaxPoolWidth = l8AllocationSize.width - convolutionOutputShrink
        l9MaxPoolHeight = l8AllocationSize.height - convolutionOutputShrink

        // setting character recognizer parameters to the script
        processingScript._characterIndex = characterIndex
        processingScript._recognizerNormalizedInputAllocation = normalizedInputAllocation
        processingScript._recognizerL1OutAllocations = l1ConvolutionOutAllocationsArray
        processingScript._recognizerL2OutAllocations = l2ConvolutionOutAllocationsArray
        processingScript._recognizerL3OutAllocations = l3ConvolutionOutAllocationsArray
        processingScript._recognizerL4OutAllocations = l4ConvolutionOutAllocationsArray
        processingScript._recognizerL5MaxPoolAllocations = l5MaxPoolAllocationsArray
        processingScript._recognizerL6OutAllocations = l6ConvolutionOutAllocationsArray
        processingScript._recognizerL7OutAllocations = l7ConvolutionOutAllocationsArray
        processingScript._recognizerL8MaxPoolAllocations = l8MaxPoolAllocationsArray
        processingScript._recognizerL9MaxPoolWidth = l9MaxPoolWidth
        processingScript._recognizerL9MaxPoolHeight = l9MaxPoolHeight
    }

    fun currentCharacterVariablesClose() {
        normalizedInputAllocation.destroy()
        for (allocation in l1ConvolutionOutAllocationsArray) {
            allocation.destroy()
        }
        for (allocation in l2ConvolutionOutAllocationsArray) {
            allocation.destroy()
        }
        for (allocation in l3ConvolutionOutAllocationsArray) {
            allocation.destroy()
        }
        for (allocation in l4ConvolutionOutAllocationsArray) {
            allocation.destroy()
        }
        for (allocation in l5MaxPoolAllocationsArray) {
            allocation.destroy()
        }
        for (allocation in l6ConvolutionOutAllocationsArray) {
            allocation.destroy()
        }
        for (allocation in l7ConvolutionOutAllocationsArray) {
            allocation.destroy()
        }
        for (allocation in l8MaxPoolAllocationsArray) {
            allocation.destroy()
        }
    }

    fun generalVariablesClose() {
        inputSizesAllocation.destroy()
        pasteShiftsAllocation.destroy()
        for (allocation in l1Convolution3x3WeightsAllocationsArray) {
            allocation.destroy()
        }
        l1Convolution3x3BiasAllocation.destroy()
        l1BatchNormAllocation.destroy()
        for (allocation in l2Convolution3x3WeightsAllocationsArray) {
            allocation.destroy()
        }
        l2Convolution3x3BiasAllocation.destroy()
        l2BatchNormAllocation.destroy()
        for (allocation in l3Convolution3x3WeightsAllocationsArray) {
            allocation.destroy()
        }
        l3Convolution3x3BiasAllocation.destroy()
        l3BatchNormAllocation.destroy()
        for (allocation in l4Convolution3x3WeightsAllocationsArray) {
            allocation.destroy()
        }
        l4Convolution3x3BiasAllocation.destroy()
        l4BatchNormAllocation.destroy()
        for (allocation in l6Convolution3x3WeightsAllocationsArray) {
            allocation.destroy()
        }
        l6Convolution3x3BiasAllocation.destroy()
        l6BatchNormAllocation.destroy()
        for (allocation in l7Convolution3x3WeightsAllocationsArray) {
            allocation.destroy()
        }
        l7Convolution3x3BiasAllocation.destroy()
        l7BatchNormAllocation.destroy()
        l9ConvolutionOutAllocation.destroy()
        for (allocation in l9Convolution3x3WeightsAllocationsArray) {
            allocation.destroy()
        }
        l9Convolution3x3BiasAllocation.destroy()
        l9BatchNormAllocation.destroy()
        l10LinearOutAllocation.destroy()
        l10LinearWeightsAllocation.destroy()
        l10LinearBiasAllocation.destroy()
        l11LinearOutAllocation.destroy()
        l11LinearWeightsAllocation.destroy()
        l11LinearBiasAllocation.destroy()
    }

    fun recognizeCharacter() {
        processingScript.forEach_recognizerInputNormalizationKernel(
                normalizedInputAllocation
        )
        processingScript.forEach_recognizerStage1PadWithZeroKernel(
                l1ConvolutionOutAllocationsArray[0],
                padWithZeroStartLaunchOptions
        )
        processingScript.forEach_recognizerStage1PadWithZeroKernel(
                l1ConvolutionOutAllocationsArray[0],
                padWithZeroLowestLaunchOptions
        )
        processingScript.forEach_recognizerStage1PadWithZeroKernel(
                l1ConvolutionOutAllocationsArray[0],
                padWithZeroStage1EndLaunchOptions
        )
        processingScript.forEach_recognizerStage1PadWithZeroKernel(
                l1ConvolutionOutAllocationsArray[0],
                padWithZeroStage1HighestLaunchOptions
        )
        processingScript.forEach_recognizerL1ConvolutionKernel(
                l1ConvolutionOutAllocationsArray[0],
                recognizerStage1ConvolutionLaunchOptions
        )
        processingScript.forEach_recognizerL2ConvolutionKernel(
                l2ConvolutionOutAllocationsArray[0],
                recognizerStage1ConvolutionLaunchOptions
        )
        processingScript.forEach_recognizerL3ConvolutionKernel(
                l3ConvolutionOutAllocationsArray[0],
                recognizerStage1ConvolutionLaunchOptions
        )
        processingScript.forEach_recognizerL4ConvolutionKernel(
                l4ConvolutionOutAllocationsArray[0]
        )
        processingScript.forEach_recognizerL5MaxPoolKernel(
                l5MaxPoolAllocationsArray[0]
        )
        processingScript.forEach_recognizerL6PadWithZeroKernel(
                l6ConvolutionOutAllocationsArray[0],
                padWithZeroStartLaunchOptions
        )
        processingScript.forEach_recognizerL6PadWithZeroKernel(
                l6ConvolutionOutAllocationsArray[0],
                padWithZeroLowestLaunchOptions
        )
        processingScript.forEach_recognizerL6PadWithZeroKernel(
                l6ConvolutionOutAllocationsArray[0],
                padWithZeroL6EndLaunchOptions
        )
        processingScript.forEach_recognizerL6PadWithZeroKernel(
                l6ConvolutionOutAllocationsArray[0],
                padWithZeroL6HighestLaunchOptions
        )
        processingScript.forEach_recognizerL6ConvolutionKernel(
                l6ConvolutionOutAllocationsArray[0],
                recognizerL6ConvolutionLaunchOptions
        )
        processingScript.forEach_recognizerL7ConvolutionKernel(
                l7ConvolutionOutAllocationsArray[0]
        )
        processingScript.forEach_recognizerL8MaxPoolKernel(
                l8MaxPoolAllocationsArray[0]
        )
        processingScript.forEach_recognizerL9ConvolutionKernel(
                l9ConvolutionOutAllocation
        )
        processingScript.forEach_recognizerL10LinearKernel(
                l10LinearOutAllocation
        )
        processingScript.forEach_recognizerL11LinearKernel(
                l11LinearOutAllocation
        )
        processingScript.forEach_recognizerPredictionKernel(
                l11LinearOutAllocation,
                predictionKernelLaunchOptions
        )
    }

    fun setTextStripAllocationAndCalculateInputSizes(
            textStripAllocation: Allocation,
            launchOptions: Script.LaunchOptions
    ) {
        // switch recognizerTextStripAllocation to textStripAllocation
        processingScript._recognizerTextStripAllocation = textStripAllocation

        // calculate input allocations sizes
        processingScript.forEach_inputSizesCalculationKernel(
                inputSizesAllocation,
                launchOptions
        )
        inputSizesAllocation.copyTo(inputSizesArray)
    }

    private fun createOutputAllocationsArray(
            numberOfAllocations: Int,
            createFloatAllocation: (Size) -> Allocation
    ): Array<Allocation> {
        // creating array of output allocations filling it with temporary allocations,
        // because their size is different for every character
        val outputAllocationsArray = Array(numberOfAllocations) {
            createFloatAllocation(Size(1, 1))
        }

        // destroying temporary allocations
        for (allocation in outputAllocationsArray) {
            allocation.destroy()
        }

        return outputAllocationsArray
    }

    private fun createAndFillConvolutionWeightsAllocationsArray(
            numberOfInputs: Int,
            numberOfOutputs: Int,
            kernelWidth: Int,
            parametersReader: BufferedReader,
            createFloatAllocation: (Size) -> Allocation
    ): Array<Allocation> {
        // creating array of convolution weights allocations
        val numberOfWeightsAllocations = numberOfInputs * numberOfOutputs
        val convolutionWeightsAllocationsArray = Array(numberOfWeightsAllocations) {
            createFloatAllocation(Size(kernelWidth, kernelWidth))
        }

        // filling convolution weights allocations with parameters from parameters reader
        val numberOfKernelElements = kernelWidth * kernelWidth
        for (i in convolutionWeightsAllocationsArray.indices) {
            val weightsArray = FloatArray(numberOfKernelElements)
            for (j in weightsArray.indices) {
                weightsArray[j] = parametersReader.readLine().toFloat()
            }
            convolutionWeightsAllocationsArray[i].copyFrom(weightsArray)
        }

        return convolutionWeightsAllocationsArray
    }

    private fun createAndFillBiasAllocation(
            numberOfColumns: Int,
            parametersReader: BufferedReader,
            createFloatAllocation: (Size) -> Allocation
    ): Allocation {
        // creating bias allocation
        val numberOfRows = 1
        val biasAllocation = createFloatAllocation(Size(numberOfColumns, numberOfRows))

        // filling bias allocation with parameters from parameters reader
        val biasArray = FloatArray(numberOfColumns)
        for (i in biasArray.indices) { biasArray[i] = parametersReader.readLine().toFloat() }
        biasAllocation.copyFrom(biasArray)

        return biasAllocation
    }

    private fun createAndFillBatchNormAllocation(
            numberOfColumns: Int,
            numberOfRows: Int,
            parametersReader: BufferedReader,
            createFloatAllocation: (Size) -> Allocation
    ): Allocation {
        // creating batch norm allocation
        val batchNormAllocation = createFloatAllocation(Size(numberOfColumns, numberOfRows))

        // filling batch norm allocation with parameters from parameters reader
        val batchNormArray = FloatArray(numberOfColumns * numberOfRows)
        for (i in 0 until numberOfColumns) {
            batchNormArray[i + batchNormEffectiveWeightRow * numberOfColumns] =
                    parametersReader.readLine().toFloat()
            batchNormArray[i + batchNormEffectiveBiasRow * numberOfColumns] =
                    parametersReader.readLine().toFloat()
        }
        batchNormAllocation.copyFrom(batchNormArray)

        return batchNormAllocation
    }

    private fun createAndFillLinearWeightsAllocation(
            numberOfInputs: Int,
            numberOfOutputs: Int,
            parametersReader: BufferedReader,
            createFloatAllocation: (Size) -> Allocation
    ): Allocation {
        // creating linear weights allocation
        val linearWeightsAllocation = createFloatAllocation(Size(numberOfInputs, numberOfOutputs))

        // filling linear weights allocation with parameters from parameters reader
        val weightsArray = FloatArray(numberOfInputs * numberOfOutputs)
        for (i in weightsArray.indices) { weightsArray[i] = parametersReader.readLine().toFloat() }
        linearWeightsAllocation.copyFrom(weightsArray)

        return linearWeightsAllocation
    }

    val characterClasses = arrayOf<String>(
            "A", // 0
            "B", // 1
            "C", // 2
            "D", // 3
            "E", // 4
            "F", // 5
            "G", // 6
            "H", // 7
            "I", // 8
            "J", // 9
            "K", // 10
            "L", // 11
            "M", // 12
            "N", // 13
            "O", // 14
            "P", // 15
            "Q", // 16
            "R", // 17
            "S", // 18
            "T", // 19
            "U", // 20
            "V", // 21
            "W", // 22
            "X", // 23
            "Y", // 24
            "Z", // 25

            "0", // 26
            "1", // 27
            "2", // 28
            "3", // 29
            "4", // 30
            "5", // 31
            "6", // 32
            "7", // 33
            "8", // 34
            "9", // 35

            "<", // 36
    )
}


