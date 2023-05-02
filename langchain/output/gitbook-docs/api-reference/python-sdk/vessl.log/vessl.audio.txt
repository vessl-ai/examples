# vessl.Audio

Use the `vessl.Audio` class to log audio data. This takes the audio data and saves it as a local WAV file in the `vessl-media/audio` directory with randomly generated names.



| Parameter      | Description                                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------------------ |
| `data_or_path` | <p>Supported types<br> - <code>numpy.ndarray</code> : the audio data</p><p> - <code>str</code>: the audio path</p> |
| `sample_rate`  | The sample rate of the audio file. Required if the `numpy.ndarray` of audio data is provided as `data_or_path`     |
| `caption`      | Label of the given audio                                                                                           |

### `numpy.ndarray`

```python
import vessl
import soundfile as sf

audio_path = "sample.wav"
data, sample_rate = sf.read(audio_path)


# Sample rate is required if numpy.ndarray is provided 
vessl.log(
  payload={
    "test-audio": [
      vessl.Audio(data, sample_rate=sample_rate, caption="audio with data example")
    ]
  }
)
```

### `str`

```python
import vessl 

vessl.log(
  payload={
    "test-audio": [
      vessl.Audio(audio_path, caption="audio with path example")
    ]
  }
)
```
