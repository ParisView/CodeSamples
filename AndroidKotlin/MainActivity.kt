package com.code.sample.readmrz

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.camera2.*
import android.hardware.camera2.CaptureRequest.CONTROL_AF_MODE
import android.hardware.camera2.CaptureResult.CONTROL_AF_MODE
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.renderscript.*
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import kotlin.math.abs
import kotlin.math.sqrt

const val TAG: String = "RMRZapp"
private const val REQUEST_CAMERA_PERMISSION_CODE: Int = 52
private const val RETURN_RESULT_INPUT_INTENT_ACTION =
        "com.readmrz.conditionalIntents.uniqueActions.READ_MRZ"
private const val RETURN_RESULT_OUTPUT_INTENT_ACTION =
        "com.readmrz.conditionalIntents.uniqueActions.RETURN_MRZ_LINES"
private const val RETURN_RESULT_EXTRA_DATA_NAME = "MRZ lines"


class MainActivity : AppCompatActivity() {
    private val mainCoroutineScope = MainScope()
    private var currentCameraSessionID = 0L
    private var startCameraJob: Job? = null
    private var cameraPermissionIsGranted = false
    private var awaitingCameraPermission = false
    private var openedCameraDevice: CameraDevice? = null
    private var cameraCaptureSession: CameraCaptureSession? = null
    private var previewRequest: CaptureRequest.Builder? = null
    private lateinit var previewTextureView: TextureView
    private lateinit var cameraOutputSurface: Surface
    private lateinit var startPauseImageButton: ImageButton
    private lateinit var infoButton: Button
    private lateinit var returnResultButton: Button
    private var recognitionIsOn = true
    private lateinit var thisActivity: MainActivity
    private val charactersTextViewArrayRows = 2
    private val charactersTextViewArrayColumns = 44
    private lateinit var charactersTextViewArray: Array<Array<TextView>>
    private lateinit var processingModel: ProcessingModel
    private val returnResultNumberOfStrings = 2
    private var returnResultStringArray = Array(returnResultNumberOfStrings) { "" }
    private var returnResultStatus = RESULT_CANCELED

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        thisActivity = this
        processingModel = ProcessingModel(thisActivity)
        previewTextureView = findViewById(R.id.preview_texture_view)
        startPauseImageButton = findViewById(R.id.button_start)
        startPauseImageButton.setOnClickListener { startPauseImageButtonOnClick() }
        startPauseImageButton.isClickable = false
        infoButton = findViewById(R.id.button_info)
        infoButton.setOnClickListener { infoButtonOnClick() }
        infoButton.isClickable = false

        charactersTextViewArray = Array(charactersTextViewArrayRows) {
            Array(charactersTextViewArrayColumns) { TextView(thisActivity) } }

        for (i in 0 until charactersTextViewArrayRows) {
            for (j in 0 until charactersTextViewArrayColumns) {
                val textViewName = "recognized_string_r${i}_c${j}_text_view"
                val textViewID = resources.getIdentifier(textViewName, "id", packageName)
                charactersTextViewArray[i][j] = findViewById(textViewID)
            }
        }

        if (thisActivity.intent.action == RETURN_RESULT_INPUT_INTENT_ACTION) {
            returnResultButton = findViewById(R.id.button_return_result)
            returnResultButton.visibility = View.VISIBLE
            returnResultButton.setOnClickListener { returnResultButtonOnClick() }
            returnResultButton.isClickable = true
        }

        if (applicationContext.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) {
            if (
                    ContextCompat.checkSelfPermission(
                            baseContext,
                            android.Manifest.permission.CAMERA
                    ) == PackageManager.PERMISSION_GRANTED
            ) {
                cameraPermissionIsGranted = true
            } else {
                awaitingCameraPermission = true
                ActivityCompat.requestPermissions(this,
                        arrayOf(android.Manifest.permission.CAMERA),
                        REQUEST_CAMERA_PERMISSION_CODE)
            }
        }
    }

    override fun onRequestPermissionsResult(
            requestCode: Int,
            permissions: Array<out String>,
            grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            REQUEST_CAMERA_PERMISSION_CODE -> {
                if (grantResults.isNotEmpty() &&
                        (grantResults[0] == PackageManager.PERMISSION_GRANTED)
                ) {
                    cameraPermissionIsGranted = true
                }
                awaitingCameraPermission = false
            }
        }
    }

    override fun onResume() {
        super.onResume()
        currentCameraSessionID = SystemClock.uptimeMillis()
        startCameraJob = mainCoroutineScope.launch {
            while (awaitingCameraPermission) { delay(200) }
            if (cameraPermissionIsGranted) {
                if (recognitionIsOn) {
                    resetTextInRecognizedCharactersTextViews()
                }
                while (!previewTextureView.isAvailable) { delay(200) }
                startCamera()
                if (recognitionIsOn) {
                    startPauseImageButton.setImageResource(R.drawable.pause_rectangles_icon)
                } else {
                    startPauseImageButton.setImageResource(R.drawable.play_triangle_icon)
                }
                startPauseImageButton.isClickable = true
            }
            infoButton.isClickable = true
        }
    }

    override fun onPause() {
        currentCameraSessionID = 0L
        infoButton.isClickable = false
        startPauseImageButton.isClickable = false
        startCameraJob?.cancel()
        stopCamera()
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        mainCoroutineScope.cancel()
    }

    private fun startPauseImageButtonOnClick() {
        if (recognitionIsOn) {
            recognitionIsOn = false
            startPauseImageButton.setImageResource(R.drawable.play_triangle_icon)
        } else {
            recognitionIsOn = true
            resetTextInRecognizedCharactersTextViews()
            returnResultStatus = RESULT_CANCELED
            startPauseImageButton.setImageResource(R.drawable.pause_rectangles_icon)

            previewRequest?.build()?.let { request -> cameraCaptureSession?.capture(
                    request, null, null)}
        }
    }

    private fun infoButtonOnClick() {
        val startInfoActivityIntent = Intent(thisActivity, InfoActivity::class.java)
        startActivity(startInfoActivityIntent)
    }

    private fun returnResultButtonOnClick() {
        if (returnResultStatus == RESULT_OK) {
            Intent(RETURN_RESULT_OUTPUT_INTENT_ACTION).also {
                it.putExtra(RETURN_RESULT_EXTRA_DATA_NAME, returnResultStringArray)
                setResult(returnResultStatus, it)
            }
        } else {
            setResult(returnResultStatus)
        }
        finish()
    }

    private fun startCamera() {
        val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            val cameraIDList = cameraManager.cameraIdList
            var cameraFacingBackID: String? = null
            var cameraFacingFrontID: String? = null
            for (cameraID in cameraIDList) {
                val characteristics = cameraManager.getCameraCharacteristics(cameraID)
                when (characteristics.get(CameraCharacteristics.LENS_FACING)) {
                    CameraCharacteristics.LENS_FACING_BACK -> {
                        cameraFacingBackID = cameraID
                        break
                    }
                    CameraCharacteristics.LENS_FACING_FRONT -> {
                        if (cameraFacingFrontID == null) {
                            cameraFacingFrontID = cameraID
                        }
                    }
                }
            }
            val selectedCameraID = cameraFacingBackID ?: cameraFacingFrontID
            if (selectedCameraID != null) {
                try {
                    cameraStateCallback.sessionID = currentCameraSessionID
                    cameraManager.openCamera(selectedCameraID, cameraStateCallback, null)
                } catch (e: SecurityException) {
                    Log.e(TAG, "could not open camera", e)
                }
            }
        } catch (e: CameraAccessException) {
            Log.e(TAG, "could not open camera", e)
        } catch (e: IllegalArgumentException) {
            Log.e(TAG, "could not open camera", e)
        }
    }

    private val cameraStateCallback = object: CameraDevice.StateCallback() {
        var sessionID = 0L

        override fun onOpened(camera: CameraDevice) {
            if (sessionID == currentCameraSessionID && sessionID != 0L) {
                sessionID = 0L
                openedCameraDevice = camera
                val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
                val characteristics = cameraManager.getCameraCharacteristics(camera.id)
                val bestSize = processingModel.BEST_INPUT_SIZE
                val ratioInfluence = processingModel.RATIO_INFLUENCE_ON_INPUT_SIZE
                val largerSquareFactor = processingModel.LARGER_SQUARE_FACTOR
                val inputAllocationSize = selectSuitableCameraOutputSize(
                        characteristics, bestSize, ratioInfluence, largerSquareFactor)
                cameraOutputSurface = processingModel.variablesSetUp(
                        inputAllocationSize,
                        onYuvInputBufferAvailableCallback,
                        previewTextureView
                )

                @Suppress("DEPRECATION")
                camera.createCaptureSession(
                        listOf(cameraOutputSurface),
                        object: CameraCaptureSession.StateCallback() {
                            override fun onConfigured(session: CameraCaptureSession) {
                                try {
                                    previewRequest = camera.createCaptureRequest(
                                            CameraDevice.TEMPLATE_PREVIEW
                                    ).also {
                                        it.addTarget(cameraOutputSurface)
                                        it.set(CaptureRequest.CONTROL_AF_MODE,
                                                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                                        if (recognitionIsOn) {
                                            session.capture(it.build(), null, null)
                                        }
                                    }
                                    cameraCaptureSession = session
                                } catch (e: IllegalStateException) {
                                    Log.e(TAG, "could not create capture request", e)
                                }
                            }

                            override fun onConfigureFailed(session: CameraCaptureSession) {
                                Log.d(TAG, "camera session configure failed")
                            }
                        },
                        null
                )
            }
        }

        override fun onDisconnected(camera: CameraDevice) {
            camera.close()
        }

        override fun onError(camera: CameraDevice, error: Int) {
            Log.d(TAG, "camera ${camera.id} error, code: $error")
            onDisconnected(camera)
        }
    }

    private fun stopCamera() {
        // automatically close cameraCapture Request and Session with cameraDevice closing
        previewRequest = null
        cameraCaptureSession = null
        openedCameraDevice?.let {
            it.close()
            processingModel.variablesClose()
        }
        openedCameraDevice = null
    }

    private val onYuvInputBufferAvailableCallback = Allocation.OnBufferAvailableListener {
        val recognitionResult = processingModel.processInputImage()

        if (recognitionResult.detectedCorrectMRZ) {
            mainCoroutineScope.launch(Dispatchers.Main) {
                setTextInRecognizedCharactersTextViews(recognitionResult.textArray)
                setTextInReturnResultStringArray(recognitionResult.textArray)
                returnResultStatus = RESULT_OK
                recognitionIsOn = false
                startPauseImageButton.setImageResource(R.drawable.play_triangle_icon)
            }
        } else {
            if (recognitionIsOn) {
                previewRequest?.build()?.let { request -> cameraCaptureSession?.capture(
                        request, null, null)}
            }
        }
    }

    private fun selectSuitableCameraOutputSize(
            characteristics: CameraCharacteristics, bestSize: Size,
            ratioInfluence: Float, largerSquareFactor: Float
    ): Size {
        val outputSizesArray = characteristics.get(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
        )?.getOutputSizes(Allocation::class.java) ?: arrayOf()
        var outputSize: Size
        var outputSizeSuitabilityMetric: Float
        var nextSizeSuitabilityMetric: Float

        if (outputSizesArray.isNotEmpty()) {
            outputSize = outputSizesArray[0]
            outputSizeSuitabilityMetric = computeSuitabilityMetric(
                    outputSize, bestSize, ratioInfluence, largerSquareFactor)

            for (nextSize in outputSizesArray) {
                nextSizeSuitabilityMetric = computeSuitabilityMetric(
                        nextSize, bestSize, ratioInfluence, largerSquareFactor)
                if (nextSizeSuitabilityMetric > outputSizeSuitabilityMetric) {
                    outputSize = nextSize
                    outputSizeSuitabilityMetric = nextSizeSuitabilityMetric
                }
            }
        } else {
            return bestSize
        }
        return outputSize
    }

    private fun computeSuitabilityMetric(size: Size, bestSize: Size, ratioInfluence: Float,
                                         largerSquareFactor: Float): Float {
        val largerOrSmallerSquareFactor = if (sqrtFromHW(size) >= sqrtFromHW(bestSize)) {
            largerSquareFactor } else { 1f }

        val squareFactor = 1f / (1f + abs(
                (sqrtFromHW(size) - sqrtFromHW(bestSize)) /
                        (largerOrSmallerSquareFactor * sqrtFromHW(bestSize))
        ))

        val ratioFactor = 1f / (1f + ratioInfluence * abs(
                (sizeLongToShortRatio(size) - sizeLongToShortRatio(bestSize)) /
                        sizeLongToShortRatio(bestSize)
        ))

        return (squareFactor * ratioFactor) / (
                0.0001f + squareFactor + ratioFactor)
    }

    private fun sqrtFromHW(size: Size): Float {
        return sqrt(size.height.toFloat() * size.width.toFloat())
    }

    private fun sizeLongToShortRatio(size: Size): Float {
        return if (size.width > size.height) {
            size.width.toFloat() / size.height.toFloat()
        } else {
            size.height.toFloat() / size.width.toFloat()
        }
    }

    private fun setTextInRecognizedCharactersTextViews(textArray: Array<Array<String>>) {
        for (i in 0 until charactersTextViewArrayRows) {
            for (j in 0 until charactersTextViewArrayColumns) {
                if (i < textArray.size && j < textArray[0].size) {
                    charactersTextViewArray[i][j].text = textArray[i][j]
                } else {
                    charactersTextViewArray[i][j].text = ""
                }
            }
        }
    }

    private fun resetTextInRecognizedCharactersTextViews() {
        for (i in 0 until charactersTextViewArrayRows) {
            for (j in 0 until charactersTextViewArrayColumns) {
                charactersTextViewArray[i][j].text = ""
            }
        }
    }

    private fun setTextInReturnResultStringArray(textArray: Array<Array<String>>) {
        for (i in 0 until returnResultNumberOfStrings) {
            var textString = ""
            if (i < textArray.size) {
                for (item in textArray[i]) {
                    textString += item
                }
            }
            returnResultStringArray[i] = textString
        }
    }
}

