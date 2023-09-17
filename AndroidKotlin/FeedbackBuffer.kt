package com.code.sample.readmrz

import android.renderscript.Allocation
import android.util.Size

// only one row in feedback allocation
// columns of feedback allocation are rewritten starting from column 0 in each kernel, that uses feedback
private const val NUMBER_OF_ROWS = 1
private const val NUMBER_OF_COLUMNS = 2 // it is the max number of used feedback parameters in one single kernel

// line detection kernel feedback columns
private const val LINE_DETECTION_FLAG_COLUMN_INDEX = 0
private const val LINE_DETECTED_FLAG = 37

// end detection kernel feedback columns
// in end detection kernel: row indices of end detection allocation correspond to column
// indices of feedback allocation
private const val TEXT_STRIP_START_X_COLUMN_INDEX = 0
private const val TEXT_STRIP_END_X_COLUMN_INDEX = 1

// character groups detection kernel feedback columns
private const val N_DETECTED_CHARACTERS_COLUMN_INDEX = 0

// recognizer prediction kernel feedback columns
private const val CHARACTER_CLASS_COLUMN_INDEX = 0


class FeedbackBuffer(private val processingScript: ScriptC_processing) {
    private lateinit var feedbackAllocation: Allocation
    val feedbackArray = IntArray(NUMBER_OF_COLUMNS * NUMBER_OF_ROWS)
    val lineDetectionFlagColumn: Int = LINE_DETECTION_FLAG_COLUMN_INDEX
    val lineDetectedFlag: Int = LINE_DETECTED_FLAG
    val textStripStartXColumn: Int = TEXT_STRIP_START_X_COLUMN_INDEX
    val textStripEndXColumn: Int = TEXT_STRIP_END_X_COLUMN_INDEX
    val nDetectedCharactersColumn: Int = N_DETECTED_CHARACTERS_COLUMN_INDEX
    val characterClassColumn: Int = CHARACTER_CLASS_COLUMN_INDEX

    fun variablesSetUp(
            createIntAllocation: (Size) -> Allocation
    ) {
        // creating feedback allocation
        feedbackAllocation = createIntAllocation(Size(NUMBER_OF_COLUMNS, NUMBER_OF_ROWS))

        // setting parameters in renderscript
        processingScript._fbAllocation = feedbackAllocation
        processingScript._fbAllocationLineDetectionFlagColumn = lineDetectionFlagColumn
        processingScript._fbAllocationLineDetectedFlag = lineDetectedFlag
        processingScript._fbAllocationNDetectedCharactersColumn = nDetectedCharactersColumn
        processingScript._fbAllocationCharacterClassColumn = characterClassColumn
    }

    fun variablesClose() {
        feedbackAllocation.destroy()
    }

    fun copyAllocationToArray() {
        feedbackAllocation.copyTo(feedbackArray)
    }
}

