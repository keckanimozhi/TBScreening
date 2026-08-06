package com.kongu.ai4tb.wrapper

import android.content.Context
import java.io.DataInputStream
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.PI

/**
 * On-device log-mel spectrogram extraction, matching src/cough/features.py
 * (librosa.feature.melspectrogram + librosa.power_to_db + standardization)
 * closely enough to feed the same trained cough CNN.
 *
 * This is a line-for-line port of the algorithm verified against librosa
 * in src/export/verify_mel_algorithm.py (max abs difference ~1e-6 on the
 * standardized output over 10 real cough clips) -- NOT independently
 * re-derived here. There is no Android device/emulator in the build
 * environment to test this Kotlin file directly, so correctness rests on
 * that Python-side verification plus a careful, unmodified port. If cough
 * predictions on-device look wrong, re-check this file against
 * verify_mel_algorithm.py's `manual_log_mel` step by step before assuming
 * the trained model itself is at fault.
 *
 * Key parameters (must match src/cough/features.py exactly):
 *   sampleRate = 44100, nFft = 1024, hopLength = 256, nMels = 64,
 *   duration = 1.0s (44100 samples) -> 173 frames.
 * Padding: librosa >=0.10 defaults to pad_mode="constant" (zero-padding),
 * NOT "reflect" -- this was the one bug found during Python-side
 * verification (librosa's own docs/older tutorials often assume reflect).
 */
object MelSpectrogram {

    const val SAMPLE_RATE = 44100
    const val N_FFT = 1024
    const val HOP_LENGTH = 256
    const val N_MELS = 64
    const val N_FREQ_BINS = N_FFT / 2 + 1 // 513
    const val N_FRAMES = 173

    private var melBasis: Array<FloatArray>? = null

    /** Loads the precomputed librosa mel filterbank (64x513 float32),
     * bundled as an asset rather than re-derived on-device -- see the
     * module doc comment in verify_mel_algorithm.py for why. */
    fun loadMelBasis(context: Context): Array<FloatArray> {
        melBasis?.let { return it }

        val input = DataInputStream(context.assets.open("mel_filterbank_64x513.bin"))
        val basis = Array(N_MELS) { FloatArray(N_FREQ_BINS) }
        val buffer = ByteArray(4)
        for (m in 0 until N_MELS) {
            for (f in 0 until N_FREQ_BINS) {
                input.readFully(buffer)
                // little-endian float32, matching numpy's tofile() on this platform
                val bits = (buffer[0].toInt() and 0xFF) or
                    ((buffer[1].toInt() and 0xFF) shl 8) or
                    ((buffer[2].toInt() and 0xFF) shl 16) or
                    ((buffer[3].toInt() and 0xFF) shl 24)
                basis[m][f] = Float.fromBits(bits)
            }
        }
        input.close()
        melBasis = basis
        return basis
    }

    private fun periodicHann(n: Int): DoubleArray =
        DoubleArray(n) { i -> 0.5 - 0.5 * cos(2.0 * PI * i / n) }

    /** In-place iterative radix-2 Cooley-Tukey FFT. `real`/`imag` length
     * must be a power of two (N_FFT = 1024 here). */
    private fun fft(real: DoubleArray, imag: DoubleArray) {
        val n = real.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val tr = real[i]; real[i] = real[j]; real[j] = tr
                val ti = imag[i]; imag[i] = imag[j]; imag[j] = ti
            }
        }

        var length = 2
        while (length <= n) {
            val half = length / 2
            val angleStep = -2.0 * PI / length
            var start = 0
            while (start < n) {
                for (k in 0 until half) {
                    val angle = angleStep * k
                    val wr = cos(angle)
                    val wi = sin(angle)
                    val evenIdx = start + k
                    val oddIdx = start + k + half
                    val er = real[evenIdx]; val ei = imag[evenIdx]
                    val tr = real[oddIdx] * wr - imag[oddIdx] * wi
                    val ti = real[oddIdx] * wi + imag[oddIdx] * wr
                    real[evenIdx] = er + tr
                    imag[evenIdx] = ei + ti
                    real[oddIdx] = er - tr
                    imag[oddIdx] = ei - ti
                }
                start += length
            }
            length *= 2
        }
    }

    /**
     * audio: exactly SAMPLE_RATE (44100) samples, normalized to [-1, 1]
     * (as librosa.load returns -- if you have 16-bit PCM shorts, divide
     * by 32768.0 first).
     *
     * Returns a flattened (N_MELS * N_FRAMES) FloatArray in row-major
     * [mel][frame] order, ready to reshape into the (1, 1, 64, 173) ONNX
     * input tensor.
     */
    fun computeLogMel(audio: DoubleArray, melBasis: Array<FloatArray>): FloatArray {
        require(audio.size == SAMPLE_RATE) { "expected exactly $SAMPLE_RATE samples, got ${audio.size}" }

        val pad = N_FFT / 2
        val padded = DoubleArray(audio.size + 2 * pad)
        System.arraycopy(audio, 0, padded, pad, audio.size)
        // zero-padding on both ends (librosa's default pad_mode="constant")

        val window = periodicHann(N_FFT)
        val melSpec = Array(N_MELS) { DoubleArray(N_FRAMES) }

        val frameReal = DoubleArray(N_FFT)
        val frameImag = DoubleArray(N_FFT)

        for (t in 0 until N_FRAMES) {
            val start = t * HOP_LENGTH
            for (i in 0 until N_FFT) {
                frameReal[i] = padded[start + i] * window[i]
                frameImag[i] = 0.0
            }
            fft(frameReal, frameImag)

            val power = DoubleArray(N_FREQ_BINS)
            for (f in 0 until N_FREQ_BINS) {
                val re = frameReal[f]
                val im = frameImag[f]
                power[f] = re * re + im * im
            }

            for (m in 0 until N_MELS) {
                var acc = 0.0
                val row = melBasis[m]
                for (f in 0 until N_FREQ_BINS) {
                    acc += row[f] * power[f]
                }
                melSpec[m][t] = acc
            }
        }

        val amin = 1e-10
        val topDb = 80.0

        var maxMel = 0.0
        for (m in 0 until N_MELS) for (t in 0 until N_FRAMES) maxMel = max(maxMel, melSpec[m][t])
        val refDb = 10.0 * log10(max(amin, maxMel))

        var maxLogSpec = Double.NEGATIVE_INFINITY
        val logSpec = Array(N_MELS) { DoubleArray(N_FRAMES) }
        for (m in 0 until N_MELS) {
            for (t in 0 until N_FRAMES) {
                val db = 10.0 * log10(max(amin, melSpec[m][t])) - refDb
                logSpec[m][t] = db
                maxLogSpec = max(maxLogSpec, db)
            }
        }

        val floor = maxLogSpec - topDb
        var sum = 0.0
        var count = 0
        for (m in 0 until N_MELS) {
            for (t in 0 until N_FRAMES) {
                val clamped = max(logSpec[m][t], floor)
                logSpec[m][t] = clamped
                sum += clamped
                count++
            }
        }
        val mean = sum / count
        var sqSum = 0.0
        for (m in 0 until N_MELS) for (t in 0 until N_FRAMES) {
            val d = logSpec[m][t] - mean
            sqSum += d * d
        }
        val std = kotlin.math.sqrt(sqSum / count)

        val out = FloatArray(N_MELS * N_FRAMES)
        for (m in 0 until N_MELS) {
            for (t in 0 until N_FRAMES) {
                out[m * N_FRAMES + t] = ((logSpec[m][t] - mean) / (std + 1e-8)).toFloat()
            }
        }
        return out
    }
}
