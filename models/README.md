# Models

Training checkpoints (`*.pth`) are written here but not versioned. The exported ONNX
models below are versioned and ready to ship in a mobile app.

| File                      | Parameters | Size   | Validation accuracy | Extra recordings |
|---------------------------|------------|--------|---------------------|------------------|
| `snore_mobile_tiny.onnx`  | 9 889      | 161 KB | 0.98                | 10 / 11          |
| `snore_mobile_small.onnx` | 35 137     | 258 KB | 0.975               | 8 / 11           |

"Extra recordings" are the 11 files of `data/inference/` (6 snoring, 5 other sounds),
which are not part of the training data. `snore_mobile_tiny.onnx` is the recommended
one. Both sets are small, so treat these numbers as a first indication only.

## Interface

Both models have the same interface and embed the spectrogram computation, so the app
only has to pass raw audio.

- **Input** `waveform`: `float32`, shape `[batch, 16000]` - one second of mono audio at
  16 kHz, with samples in `[-1, 1]` (16-bit PCM divided by 32768).
- **Output** `probability`: `float32`, shape `[batch, 1]` - probability that the window
  contains snoring. Use 0.5 as a starting threshold.

Each window is normalised on its own, so the result does not depend on the microphone
gain, and a silent window gives a probability close to 0. The models use ONNX opset 18
(ONNX Runtime 1.14 or later).

## Using a model on Android

Add the `com.microsoft.onnxruntime:onnxruntime-android` dependency, copy the `.onnx`
file to `app/src/main/assets/`, then:

```kotlin
val env = OrtEnvironment.getEnvironment()
val session = env.createSession(context.assets.open("snore_mobile_tiny.onnx").readBytes())

/** [pcm] holds one second of 16 kHz mono PCM-16 audio (16000 samples). */
fun snoreProbability(pcm: ShortArray): Float {
    val input = FloatBuffer.allocate(16000)
    for (i in 0 until 16000) input.put(pcm[i] / 32768f)
    input.rewind()
    OnnxTensor.createTensor(env, input, longArrayOf(1, 16000)).use { tensor ->
        session.run(mapOf("waveform" to tensor)).use { result ->
            @Suppress("UNCHECKED_CAST")
            return (result[0].value as Array<FloatArray>)[0][0]
        }
    }
}
```

Create the session once and reuse it. For a recording longer than one second, run the
model on overlapping one-second windows (for example every 0.5 s) and average.

## Rebuilding the models

```bash
python -m src.data.preprocess
python -m src.train --model mobile_tiny
python -m src.export --model mobile_tiny
```

The export step compares the ONNX model with the PyTorch one on the validation set and
fails if they disagree.
