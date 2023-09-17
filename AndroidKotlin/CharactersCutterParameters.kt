package com.code.sample.readmrz

import android.renderscript.Allocation
import android.renderscript.Script
import android.util.Size

private const val MAX_DETECTED_CHARACTERS_IN_TEXT_STRIP = 50

private const val DETECTED_CHARACTERS_ALLOCATION_NUMBER_OF_COLUMNS = MAX_DETECTED_CHARACTERS_IN_TEXT_STRIP
private const val DETECTED_CHARACTERS_ALLOCATION_NUMBER_OF_ROWS = 5
private const val CHARACTER_PRESENT_FLAG_ROW = 0
private const val CHARACTER_START_ROW = 1
private const val CHARACTER_END_ROW = 2
private const val CHARACTER_LOWEST_POINT_ROW = 3
private const val CHARACTER_HIGHEST_POINT_ROW = 4

private const val CHARACTER_PRESENT_FLAG = 37

private const val PREDICTION_SPACE = 0
private const val PREDICTION_CHARACTER_START = 1
private const val PREDICTION_CHARACTER_MIDDLE = 2
private const val PREDICTION_CHARACTER_END = 3
private const val END_OF_PREDICTION = -7

private const val MAX_ALLOWED_ERRORS_IN_GROUP = 2
private const val MAX_NEXT_PREDICTIONS_TO_CHECK = MAX_ALLOWED_ERRORS_IN_GROUP + 1
private const val MAX_END_SEARCH_ROUNDS = 3
private const val MIN_GROUP_WIDTH = 5

private const val CHARACTER_CUTOUT_PADDING = 2


class CharactersCutterParameters(private val processingScript: ScriptC_processing) {
    private lateinit var strip1DetectedCharactersAllocation: Allocation
    private lateinit var strip2DetectedCharactersAllocation: Allocation
    private lateinit var currentAllocation: Allocation
    private val groupDetectorLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()
    val detectedCharactersProcessingLaunchOptions: Script.LaunchOptions = Script.LaunchOptions()

    fun generalVariablesSetUp(
            createIntAllocation: (Size) -> Allocation
    ): Int {
        // creating detected characters allocation for strip 1 and strip 2
        strip1DetectedCharactersAllocation = createIntAllocation(Size(
            DETECTED_CHARACTERS_ALLOCATION_NUMBER_OF_COLUMNS,
            DETECTED_CHARACTERS_ALLOCATION_NUMBER_OF_ROWS
        ))
        strip2DetectedCharactersAllocation = createIntAllocation(Size(
            DETECTED_CHARACTERS_ALLOCATION_NUMBER_OF_COLUMNS,
            DETECTED_CHARACTERS_ALLOCATION_NUMBER_OF_ROWS
        ))

        // setting kernel launch options
        groupDetectorLaunchOptions.setY(0, 1)
        groupDetectorLaunchOptions.setX(0, 1)
        detectedCharactersProcessingLaunchOptions.setY(0, 1)

        // setting characters cutter parameters
        processingScript._characterPresentFlagRow = CHARACTER_PRESENT_FLAG_ROW
        processingScript._characterStartRow = CHARACTER_START_ROW
        processingScript._characterEndRow = CHARACTER_END_ROW
        processingScript._characterLowestPointRow = CHARACTER_LOWEST_POINT_ROW
        processingScript._characterHighestPointRow = CHARACTER_HIGHEST_POINT_ROW
        processingScript._characterPresentFlag = CHARACTER_PRESENT_FLAG
        processingScript._predictionCharacterStart = PREDICTION_CHARACTER_START
        processingScript._predictionCharacterMiddle = PREDICTION_CHARACTER_MIDDLE
        processingScript._predictionCharacterEnd = PREDICTION_CHARACTER_END
        processingScript._predictionSpace = PREDICTION_SPACE
        processingScript._endOfPrediction = END_OF_PREDICTION
        processingScript._maxAllowedErrorsInGroup = MAX_ALLOWED_ERRORS_IN_GROUP
        processingScript._maxNextPredictionsToCheck = MAX_NEXT_PREDICTIONS_TO_CHECK
        processingScript._maxEndSearchRounds = MAX_END_SEARCH_ROUNDS
        processingScript._minGroupWidth = MIN_GROUP_WIDTH
        processingScript._characterCutoutPadding = CHARACTER_CUTOUT_PADDING

        return DETECTED_CHARACTERS_ALLOCATION_NUMBER_OF_COLUMNS
    }

    fun currentFrameVariablesSetUp(nDetectedCharacters: Int) {
        detectedCharactersProcessingLaunchOptions.setX(0, nDetectedCharacters)
    }

    fun generalVariablesClose() {
        strip1DetectedCharactersAllocation.destroy()
        strip2DetectedCharactersAllocation.destroy()
    }

    fun switchToStrip1() {
        // switch currentAllocation to strip 1
        currentAllocation = strip1DetectedCharactersAllocation
        // set detectedCharactersAllocation to currentAllocation
        processingScript._detectedCharactersAllocation = currentAllocation
    }

    fun switchToStrip2() {
        // switch currentAllocation to strip 2
        currentAllocation = strip2DetectedCharactersAllocation
        // set detectedCharactersAllocation to currentAllocation
        processingScript._detectedCharactersAllocation = currentAllocation
    }

    fun detectCharacterGroups() {
        // detect character groups
        processingScript.forEach_characterGroupDetectionKernel(
            currentAllocation,
            groupDetectorLaunchOptions
        )
    }

    fun detectCharactersHeights() {
        processingScript.forEach_characterHeightDetectionKernel(
            currentAllocation,
            detectedCharactersProcessingLaunchOptions
        )
    }
}


