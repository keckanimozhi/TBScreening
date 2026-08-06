package com.kongu.ai4tb.wrapper

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.exp

private fun sigmoid(x: Float): Double = 1.0 / (1.0 + exp(-x.toDouble()))

/** ImageNet normalization, matching src/xray/dataset.py IMAGENET_MEAN/STD. */
private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
private const val IMAGE_SIZE = 224

/**
 * Runs the on-device X-ray ONNX model (exported by
 * src/export/export_xray.py, verified to match the PyTorch model to
 * ~1e-4 logit difference on the test set).
 *
 * Note on image preprocessing: `Bitmap.createScaledBitmap`'s resize
 * filtering is not guaranteed to numerically match PIL/torchvision's
 * resize used during training -- there was no way to verify bit-exact
 * parity here without an Android device to run this class on. The
 * ImageNet normalization step below IS an exact match to
 * src/xray/dataset.py. If on-device X-ray predictions look off compared
 * to the same image run through src/xray/evaluate.py, this resize step
 * is the first place to check.
 */
class XrayClassifier(context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val bytes = context.assets.open("xray.onnx").readBytes()
        session = env.createSession(bytes)
    }

    fun predict(bitmap: Bitmap): Double {
        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val floatBuffer = FloatBuffer.allocate(1 * 3 * IMAGE_SIZE * IMAGE_SIZE)

        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        // NCHW layout: channel-major, matching the ONNX model's expected input order.
        for (c in 0 until 3) {
            for (y in 0 until IMAGE_SIZE) {
                for (x in 0 until IMAGE_SIZE) {
                    val pixel = pixels[y * IMAGE_SIZE + x]
                    val channelValue = when (c) {
                        0 -> (pixel shr 16) and 0xFF // R
                        1 -> (pixel shr 8) and 0xFF  // G
                        else -> pixel and 0xFF        // B
                    }
                    val normalized = (channelValue / 255.0f - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
                    floatBuffer.put(normalized)
                }
            }
        }
        floatBuffer.rewind()

        val inputTensor = OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
        inputTensor.use {
            val inputName = session.inputNames.iterator().next()
            session.run(mapOf(inputName to it)).use { results ->
                val logit = (results[0].value as Array<FloatArray>)[0][0]
                return sigmoid(logit)
            }
        }
    }

    fun close() {
        session.close()
    }
}

/**
 * Runs the on-device cough ONNX model over a log-mel spectrogram
 * produced by MelSpectrogram.computeLogMel.
 */
class CoughClassifier(context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val bytes = context.assets.open("cough.onnx").readBytes()
        session = env.createSession(bytes)
    }

    fun predict(logMel: FloatArray): Double {
        val buffer = FloatBuffer.wrap(logMel)
        val inputTensor = OnnxTensor.createTensor(
            env, buffer,
            longArrayOf(1, 1, MelSpectrogram.N_MELS.toLong(), MelSpectrogram.N_FRAMES.toLong())
        )
        inputTensor.use {
            val inputName = session.inputNames.iterator().next()
            session.run(mapOf(inputName to it)).use { results ->
                val logit = (results[0].value as Array<FloatArray>)[0][0]
                return sigmoid(logit)
            }
        }
    }

    fun close() {
        session.close()
    }
}
