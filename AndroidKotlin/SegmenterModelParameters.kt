package com.code.sample.readmrz

import android.content.Context
import android.renderscript.Allocation
import android.renderscript.Script
import android.util.Size
import java.io.BufferedReader

// segmentation model parameters
private const val L1_NUMBER_OF_INPUTS = 1
private const val L1_NUMBER_OF_OUTPUT_FEATURES = 8
private const val L1_CONVOLUTION_KERNEL_WIDTH = 3
private const val L1_CONVOLUTION_KERNEL_ALLOCATION_SHRINK = L1_CONVOLUTION_KERNEL_WIDTH - 1
private const val BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS = 2
private const val BATCH_NORM_EFFECTIVE_WEIGHT_ROW_INDEX = 0
private const val BATCH_NORM_EFFECTIVE_BIAS_ROW_INDEX = 1
private const val L1_BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS = BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS
private const val L2_NUMBER_OF_OUTPUT_FEATURES = 8
private const val L2_CONVOLUTION_KERNEL_WIDTH = 3
private const val L2_CONVOLUTION_KERNEL_ALLOCATION_SHRINK = L2_CONVOLUTION_KERNEL_WIDTH - 1
private const val L2_BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS = BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS
private const val L3_NUMBER_OF_OUTPUT_FEATURES = L2_NUMBER_OF_OUTPUT_FEATURES
private const val L3_BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS = BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS
private const val L4_NUMBER_OF_OUTPUT_FEATURES = 8
private const val L4_BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS = BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS
private const val L5_NUMBER_OF_OUTPUT_FEATURES = 4
private const val PREDICTION_ALLOCATION_NUMBER_OF_ROWS = 1

// image mean and reciprocal to standard deviation calculated on whole segmentation model dataset
private const val TEXT_STRIP_IMAGE_MEAN = 0.0031279860995709896f
private const val TEXT_STRIP_IMAGE_STD_RECIPROCAL = 3.5935380458831787f


class SegmenterModelParameters(
    private val processingScript: ScriptC_processing,
    private val currentContext: Context
) {
    private lateinit var normalizedInputAllocation: Allocation
    private lateinit var l1ConvolutionOutAllocationsArray: Array<Allocation>
    private lateinit var l1Convolution3x3WeightsAllocationsArray: Array<Allocation>
    private lateinit var l1Convolution3x3BiasAllocation: Allocation
    private lateinit var l1BatchNormAllocation: Allocation
    private lateinit var l2ConvolutionOutAllocationsArray: Array<Allocation>
    private lateinit var l2Convolution3x3WeightsAllocationsArray: Array<Allocation>
    private lateinit var l2Convolution3x3BiasAllocation: Allocation
    private lateinit var l2BatchNormAllocation: Allocation
    private lateinit var l3MaxPoolAllocation: Allocation
    private lateinit var l3BatchNormAllocation: Allocation
    private lateinit var l4LinearOutAllocation: Allocation
    private lateinit var l4LinearWeightsAllocation: Allocation
    private lateinit var l4LinearBiasAllocation: Allocation
    private lateinit var l4BatchNormAllocation: Allocation
    private lateinit var l5LinearOutAllocation: Allocation
    private lateinit var l5LinearWeightsAllocation: Allocation
    private lateinit var l5LinearBiasAllocation: Allocation
    private lateinit var predictionAllocation: Allocation

    private var textStripWidth: Int = 5
    private val l1KernelWidth: Int = L1_CONVOLUTION_KERNEL_WIDTH
    private val l2KernelWidth: Int = L2_CONVOLUTION_KERNEL_WIDTH
    private val textStripImageMean: Float = TEXT_STRIP_IMAGE_MEAN
    private val textStripImageSTDReciprocal: Float = TEXT_STRIP_IMAGE_STD_RECIPROCAL
    private val batchNormEffectiveWeightRow: Int = BATCH_NORM_EFFECTIVE_WEIGHT_ROW_INDEX
    private val batchNormEffectiveBiasRow: Int = BATCH_NORM_EFFECTIVE_BIAS_ROW_INDEX
    private val l3NumberOfFeatures = L3_NUMBER_OF_OUTPUT_FEATURES
    private val l4NumberOfFeatures = L4_NUMBER_OF_OUTPUT_FEATURES
    private val l5NumberOfFeatures = L5_NUMBER_OF_OUTPUT_FEATURES
    private val predictionToCutoutShift = (L1_CONVOLUTION_KERNEL_ALLOCATION_SHRINK +
            L2_CONVOLUTION_KERNEL_ALLOCATION_SHRINK) / 2

    fun generalVariablesSetUp(
            textStripWidthInput: Int,
            createFloatAllocation: (Size) -> Allocation
    ) {
        textStripWidth = textStripWidthInput

        // setting model layers sizes
        val l1NumberOfInputs = L1_NUMBER_OF_INPUTS
        val l1NumberOfFeatures = L1_NUMBER_OF_OUTPUT_FEATURES
        val l1BatchNormAllocationNumberOfRows = L1_BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS
        val l2NumberOfFeatures = L2_NUMBER_OF_OUTPUT_FEATURES
        val l2BatchNormAllocationNumberOfRows = L2_BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS
        val l3BatchNormAllocationNumberOfRows = L3_BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS
        val l4BatchNormAllocationNumberOfRows = L4_BATCH_NORM_ALLOCATION_NUMBER_OF_ROWS

        // creating raw model parameters reader
        val modelParametersRaw = currentContext.resources.openRawResource(R.raw.segmenter_parameters)
        val modelParametersReader = modelParametersRaw.bufferedReader()

        // creating array of layer 1 output allocations filling it with temporary allocations,
        // because their size is changed every frame, and then destroying temporary allocations
        l1ConvolutionOutAllocationsArray = createOutputAllocationsArray(
                l1NumberOfFeatures, createFloatAllocation)

        // creating array of layer 1 weights allocations and filling them with model parameters
        l1Convolution3x3WeightsAllocationsArray = createAndFillConvolutionWeightsAllocationsArray(
                l1NumberOfInputs,
                l1NumberOfFeatures,
                l1KernelWidth,
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
                l1BatchNormAllocationNumberOfRows,
                modelParametersReader,
                createFloatAllocation
        )

        // creating array of layer 2 output allocations filling it with temporary allocations,
        // because their size is changed every frame, and then destroying temporary allocations
        l2ConvolutionOutAllocationsArray = createOutputAllocationsArray(
                l2NumberOfFeatures, createFloatAllocation)

        // creating array of layer 2 weights allocations and filling them with model parameters
        l2Convolution3x3WeightsAllocationsArray = createAndFillConvolutionWeightsAllocationsArray(
                l1NumberOfFeatures,
                l2NumberOfFeatures,
                l2KernelWidth,
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
                l2BatchNormAllocationNumberOfRows,
                modelParametersReader,
                createFloatAllocation
        )

        // creating layer 3 batch norm allocation and filling it with model parameters
        l3BatchNormAllocation = createAndFillBatchNormAllocation(
                l3NumberOfFeatures,
                l3BatchNormAllocationNumberOfRows,
                modelParametersReader,
                createFloatAllocation
        )

        // creating layer 4 linear weights allocation and filling it with model parameters
        l4LinearWeightsAllocation = createAndFillLinearWeightsAllocation(
                l3NumberOfFeatures,
                l4NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )

        // creating layer 4 bias allocation and filling it with model parameters
        l4LinearBiasAllocation = createAndFillBiasAllocation(
                l4NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )

        // creating layer 4 batch norm allocation and filling it with model parameters
        l4BatchNormAllocation = createAndFillBatchNormAllocation(
                l4NumberOfFeatures,
                l4BatchNormAllocationNumberOfRows,
                modelParametersReader,
                createFloatAllocation
        )

        // creating layer 5 linear weights allocation and filling it with model parameters
        l5LinearWeightsAllocation = createAndFillLinearWeightsAllocation(
                l4NumberOfFeatures,
                l5NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )

        // creating layer 5 bias allocation and filling it with model parameters
        l5LinearBiasAllocation = createAndFillBiasAllocation(
                l5NumberOfFeatures,
                modelParametersReader,
                createFloatAllocation
        )

        // closing raw model parameters reader
        modelParametersRaw.close()

        // setting text strip segmenter parameters
        processingScript._segmenterL1WeightsAllocations = l1Convolution3x3WeightsAllocationsArray
        processingScript._segmenterL1BiasAllocation = l1Convolution3x3BiasAllocation
        processingScript._segmenterL1BatchNormAllocation = l1BatchNormAllocation
        processingScript._segmenterL1ConvolutionKernelWidth = l1KernelWidth
        processingScript._segmenterTextStripImageMean = textStripImageMean
        processingScript._segmenterTextStripImageSTDReciprocal = textStripImageSTDReciprocal
        processingScript._segmenterBatchNormEffectiveWeightRow = batchNormEffectiveWeightRow
        processingScript._segmenterBatchNormEffectiveBiasRow = batchNormEffectiveBiasRow
        processingScript._segmenterL2WeightsAllocations = l2Convolution3x3WeightsAllocationsArray
        processingScript._segmenterL2BiasAllocation = l2Convolution3x3BiasAllocation
        processingScript._segmenterL2BatchNormAllocation = l2BatchNormAllocation
        processingScript._segmenterL2ConvolutionKernelWidth = l2KernelWidth
        processingScript._segmenterL3BatchNormAllocation = l3BatchNormAllocation
        processingScript._segmenterL4WeightsAllocation = l4LinearWeightsAllocation
        processingScript._segmenterL4BiasAllocation = l4LinearBiasAllocation
        processingScript._segmenterL4BatchNormAllocation = l4BatchNormAllocation
        processingScript._segmenterL5WeightsAllocation = l5LinearWeightsAllocation
        processingScript._segmenterL5BiasAllocation = l5LinearBiasAllocation
        processingScript._segmenterPredictionToCutoutShift = predictionToCutoutShift
    }

    fun currentFrameVariablesSetUp(
            textStripLength: Int,
            createFloatAllocation: (Size) -> Allocation,
            createIntAllocation: (Size) -> Allocation
    ) {
        val l1OutAllocationShrink: Int = L1_CONVOLUTION_KERNEL_ALLOCATION_SHRINK
        val l2OutAllocationShrink: Int = L2_CONVOLUTION_KERNEL_ALLOCATION_SHRINK
        var allocationLength = textStripLength
        var allocationHeight = textStripWidth
        val numberOfRowsInPrediction = PREDICTION_ALLOCATION_NUMBER_OF_ROWS

        // creating allocations
        normalizedInputAllocation = createFloatAllocation(Size(allocationLength, allocationHeight))
        allocationLength -= l1OutAllocationShrink
        allocationHeight -= l1OutAllocationShrink
        for (i in l1ConvolutionOutAllocationsArray.indices) {
            l1ConvolutionOutAllocationsArray[i] = createFloatAllocation(Size(
                    allocationLength,
                    allocationHeight
            ))
        }
        allocationLength -= l2OutAllocationShrink
        allocationHeight -= l2OutAllocationShrink
        for (i in l2ConvolutionOutAllocationsArray.indices) {
            l2ConvolutionOutAllocationsArray[i] = createFloatAllocation(Size(
                    allocationLength,
                    allocationHeight
            ))
        }
        l3MaxPoolAllocation = createFloatAllocation(Size(allocationLength, l3NumberOfFeatures))
        l4LinearOutAllocation = createFloatAllocation(Size(allocationLength, l4NumberOfFeatures))
        l5LinearOutAllocation = createFloatAllocation(Size(allocationLength, l5NumberOfFeatures))
        predictionAllocation = createIntAllocation(Size(allocationLength, numberOfRowsInPrediction))

        // setting segmenter parameters to the script
        processingScript._segmenterNormalizedInputAllocation = normalizedInputAllocation
        processingScript._segmenterL1OutAllocations = l1ConvolutionOutAllocationsArray
        processingScript._segmenterL2OutAllocations = l2ConvolutionOutAllocationsArray
        processingScript._segmenterL3MaxPoolAllocation = l3MaxPoolAllocation
        processingScript._segmenterL4OutAllocation = l4LinearOutAllocation
        processingScript._segmenterL5OutAllocation = l5LinearOutAllocation
        processingScript._segmenterPredictionAllocation = predictionAllocation
    }

    fun currentFrameVariablesClose() {
        normalizedInputAllocation.destroy()
        for (allocation in l1ConvolutionOutAllocationsArray) {
            allocation.destroy()
        }
        for (allocation in l2ConvolutionOutAllocationsArray) {
            allocation.destroy()
        }
        l3MaxPoolAllocation.destroy()
        l4LinearOutAllocation.destroy()
        l5LinearOutAllocation.destroy()
        predictionAllocation.destroy()
    }

    fun generalVariablesClose() {
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
        l3BatchNormAllocation.destroy()
        l4LinearWeightsAllocation.destroy()
        l4LinearBiasAllocation.destroy()
        l4BatchNormAllocation.destroy()
        l5LinearWeightsAllocation.destroy()
        l5LinearBiasAllocation.destroy()
    }

    fun segmentTextStrip(sourceAllocation: Allocation) {
        processingScript.forEach_segmenterInputNormalizationKernel(
            sourceAllocation,
        )
        processingScript.forEach_segmenterL1ConvolutionKernel(
            l1ConvolutionOutAllocationsArray[0]
        )
        processingScript.forEach_segmenterL2ConvolutionKernel(
            l2ConvolutionOutAllocationsArray[0]
        )
        processingScript.forEach_segmenterL3MaxPool1DKernel(
            l3MaxPoolAllocation
        )
        processingScript.forEach_segmenterL4LinearKernel(
            l4LinearOutAllocation
        )
        processingScript.forEach_segmenterL5LinearKernel(
            l5LinearOutAllocation
        )
        processingScript.forEach_segmenterPredictionKernel(
            predictionAllocation
        )
    }

    private fun createOutputAllocationsArray(
            numberOfAllocations: Int,
            createFloatAllocation: (Size) -> Allocation
    ): Array<Allocation> {
        // creating array of output allocations filling it with temporary allocations,
        // because their size is changed every frame
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
}

