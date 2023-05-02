# vessl.log

Use `vessl.log` in a training or testing loop to log a dictionary of metrics. Provide the step parameter for the loop unit – like the epoch value – and any metrics you want to log as a dictionary in the `row` parameter.&#x20;

You can also log images or audio types of objects. Provide a list of `vessl.Image` objects or `vessl.Audio` with data and captions as the `payload` parameter with any dictionary key. Note that only the first key will be logged.

| Parameter | Description                                                                       |
| --------- | --------------------------------------------------------------------------------- |
| `step`    | Unit size of the loop                                                             |
| `payload` | Dictionary of metrics or a list of `vessl.Image` objects or `vessl.Audio` objects |

### Logging metrics

```python
# Logging loss values for each epoch in PyTorch

import vessl

for epoch in range(epochs):
    ...
    vessl.log(step=epoch, payload={'loss': loss.item})
```

### Logging image objects

```python
# Logging images in PyTorch

import vessl

def test(model, test_loader, ...):
    ...
    test_images = []
    with torch.no_grad():
        for data, target in test_loader:
            ...
            output = model(data)
            ...
            test_images.append(
                vessl.Image(
                    data[0], 
                    caption=f'Pred: {output[0].item()} Truth: {target[0]}'
                )
            )
    ...
    vessl.log(payload={"test-images": test_images})
```

### Logging audio objects

```python
# Logging audio

import vessl
import soundfile as sf

audio_path = "sample.wav"
data, sample_rate = sf.read(audio_path)

# Log audio with data
vessl.log(
  payload={
    "test-audio": [
      vessl.Audio(data, sample_rate=sample_rate, caption="audio with data example")
    ]
  }
)
```
