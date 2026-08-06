package com.kongu.ai4tb.wrapper

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder

/**
 * Records exactly 1.0s of mono 44.1kHz 16-bit PCM audio (matching
 * src/cough/features.py's SAMPLE_RATE/DURATION_SEC) and returns it
 * normalized to [-1, 1] doubles, ready for MelSpectrogram.computeLogMel.
 *
 * Blocking/synchronous -- call from a background thread, not the UI
 * thread (recording 1s of audio will otherwise freeze the UI for that
 * second, on top of risking an ANR).
 */
object CoughRecorder {

    private const val SAMPLE_RATE = MelSpectrogram.SAMPLE_RATE

    @SuppressLint("MissingPermission") // caller must check RECORD_AUDIO permission first
    fun recordOneSecond(): DoubleArray {
        val minBufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        val bufferSize = maxOf(minBufferSize, SAMPLE_RATE * 2) // at least 1s worth of 16-bit samples

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        val shortBuffer = ShortArray(SAMPLE_RATE)
        try {
            recorder.startRecording()
            var offset = 0
            while (offset < shortBuffer.size) {
                val read = recorder.read(shortBuffer, offset, shortBuffer.size - offset)
                if (read <= 0) break
                offset += read
            }
        } finally {
            recorder.stop()
            recorder.release()
        }

        return DoubleArray(shortBuffer.size) { i -> shortBuffer[i] / 32768.0 }
    }
}
