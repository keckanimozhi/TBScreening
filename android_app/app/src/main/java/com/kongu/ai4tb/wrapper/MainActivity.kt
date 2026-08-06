package com.kongu.ai4tb.wrapper

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

/**
 * Native on-device screening screen (Task 18): symptom checklist +
 * optional X-ray photo pick + optional 1s cough recording, all scored
 * fully offline via ONNX Runtime (X-ray/cough CNNs) and the ported
 * logistic regression (symptom/fusion) -- no network calls, no server.
 *
 * Mirrors the Streamlit app's (app/app.py) UX and risk thresholds
 * (Low<0.3, Medium<0.7, High>=0.7) for consistency, but does not attempt
 * to reproduce its Grad-CAM visualization or the GIS dashboard --
 * ONNX Runtime Mobile is inference-only (no backprop), so Grad-CAM was
 * out of scope for this first on-device pass; see docs/PROJECT_CHECKLIST.md.
 * The full dashboard is still reachable via "Open web dashboard" below,
 * which launches WebViewActivity (Task 14's original deliverable).
 */
class MainActivity : AppCompatActivity() {

    // Training-set mean p_xray/p_cough (from data/processed/fusion_train.csv),
    // used to impute a missing modality -- matches src/fusion/predict.py's
    // behavior exactly, just hardcoded since there's no training data on-device.
    private val meanPXray = 0.42373492416044845
    private val meanPCough = 0.5595705102481272

    private lateinit var xrayClassifier: XrayClassifier
    private lateinit var coughClassifier: CoughClassifier
    private lateinit var melBasis: Array<FloatArray>
    private val executor = Executors.newSingleThreadExecutor()

    private var xrayBitmap: Bitmap? = null
    private var coughLogMel: FloatArray? = null

    private lateinit var symptomChecks: List<CheckBox>
    private lateinit var xrayPreview: ImageView
    private lateinit var coughStatus: TextView
    private lateinit var resultText: TextView

    private val pickXrayLauncher = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        if (uri != null) {
            contentResolver.openInputStream(uri)?.use { stream ->
                xrayBitmap = BitmapFactory.decodeStream(stream)
                xrayPreview.setImageBitmap(xrayBitmap)
                xrayPreview.visibility = ImageView.VISIBLE
            }
        }
    }

    private val requestAudioPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) recordCough() else Toast.makeText(this, "Microphone permission needed to record cough", Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        xrayClassifier = XrayClassifier(this)
        coughClassifier = CoughClassifier(this)
        melBasis = MelSpectrogram.loadMelBasis(this)

        xrayPreview = findViewById(R.id.xrayPreview)
        coughStatus = findViewById(R.id.coughStatus)
        resultText = findViewById(R.id.resultText)

        val symptomContainer = findViewById<LinearLayout>(R.id.symptomContainer)
        symptomChecks = ModelWeights.symptomLabels.map { label ->
            CheckBox(this).apply {
                text = label.replaceFirstChar { it.uppercase() }
            }
        }
        symptomChecks.forEach { symptomContainer.addView(it) }

        findViewById<Button>(R.id.pickXrayButton).setOnClickListener {
            pickXrayLauncher.launch(
                androidx.activity.result.PickVisualMediaRequest(
                    ActivityResultContracts.PickVisualMedia.ImageOnly
                )
            )
        }

        findViewById<Button>(R.id.recordCoughButton).setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED
            ) {
                recordCough()
            } else {
                requestAudioPermission.launch(Manifest.permission.RECORD_AUDIO)
            }
        }

        findViewById<Button>(R.id.runScreeningButton).setOnClickListener { runScreening() }

        findViewById<Button>(R.id.openWebButton).setOnClickListener {
            startActivity(Intent(this, WebViewActivity::class.java))
        }
    }

    private fun recordCough() {
        coughStatus.text = "Recording..."
        executor.execute {
            val audio = CoughRecorder.recordOneSecond()
            val logMel = MelSpectrogram.computeLogMel(audio, melBasis)
            coughLogMel = logMel
            runOnUiThread { coughStatus.text = "Recorded (1.0s captured)" }
        }
    }

    private fun runScreening() {
        val symptomValues = DoubleArray(symptomChecks.size) { i -> if (symptomChecks[i].isChecked) 1.0 else 0.0 }
        val pSymptom = ModelWeights.predictSymptomProbability(symptomValues)

        val bitmap = xrayBitmap
        val logMel = coughLogMel

        executor.execute {
            val pXray = bitmap?.let { xrayClassifier.predict(it) }
            val pCough = logMel?.let { coughClassifier.predict(it) }

            val fused = ModelWeights.predictFusedProbability(
                pXray ?: meanPXray,
                pCough ?: meanPCough,
                pSymptom
            )

            val band = when {
                fused >= 0.7 -> "High"
                fused >= 0.3 -> "Medium"
                else -> "Low"
            }
            val recommendation = when (band) {
                "High" -> "Refer for confirmatory testing (GeneXpert/TrueNat/sputum microscopy)."
                "Medium" -> "Clinical review recommended; consider confirmatory testing."
                else -> "Low presumptive risk from available inputs; routine follow-up."
            }

            val sb = StringBuilder()
            sb.appendLine("X-ray: ${pXray?.let { "%.0f%%".format(it * 100) } ?: "not provided (used average)"}")
            sb.appendLine("Cough: ${pCough?.let { "%.0f%%".format(it * 100) } ?: "not provided (used average)"}")
            sb.appendLine("Symptoms: %.0f%%".format(pSymptom * 100))
            sb.appendLine()
            sb.appendLine("Fused risk: %.0f%% (${band})".format(fused * 100))
            sb.appendLine()
            sb.appendLine("Recommendation: $recommendation")

            runOnUiThread { resultText.text = sb.toString() }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        xrayClassifier.close()
        coughClassifier.close()
        executor.shutdown()
    }
}
