# ExperimentCallback

`ExperimentCallback` extends Keras' callback class. Add `ExperimentCallback` as a callback parameter in the `fit` function to automatically track Keras metrics at the end of each epoch. You can also log image objects using `ExperimentCallback`.

| Parameter         | Description                                                                                                                        |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `data_type`       | Use `image` to log image objects                                                                                                   |
| `validation_data` | Tuple of `(validation_data, validation_labels)`                                                                                    |
| `labels`          | <p>List of labels to get the caption from the inferred logits.</p><p>The argmax value will be used if labels are not provided.</p> |
| `num_images`      | Number of images to log in the validation data                                                                                     |

### Logging metrics

```python
# Logging loss and accuracy for each epoch in Keras
from vessl.integration.keras import ExperimentCallback

...
model.fit(..., callbacks=[ExperimentCallback()])
...
```

### Logging image objects

```python
# Logging images along with the loss and accuracy for each epoch in Keras
from vessl.keras import ExperimentCallback

...
model.fit(
    ...,
    callbacks=[ExperimentCallback(
        data_type='image',
        validation_data=(x_val, y_val),
        num_images=5,
    )]
)
...
```
