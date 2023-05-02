# TensorBoard

Using VESSL's Python SDK, you can view and interact with metrics and media logged to TensorBoard directly on VESSL. We currently support scalars, images, and audio.&#x20;

{% hint style="info" %}
VESSL supports TensorBoard with TensorFlow, PyTorch (TensorBoard > 1.14), and TensorBoardX.
{% endhint %}

You can integrate TensorBoard by simply adding `vessl.init(tensorboard=True)`to your code.&#x20;

Note that this should be called **before creating the file writer**. This is because VESSL auto-detects the TensorBoard `logdir` upon writer creation but cannot do so if the writer has already been created. &#x20;

### TensorFlow

```python
import tensorflow as tf
import vessl

vessl.init(tensorboard=True) # Must be called before tf.summary.create_file_writer

w = tf.summary.create_file_writer("./logdir")
...
```

### PyTorch

```python
from torch.utils.tensorboard import SummaryWriter
import vessl

vessl.init(tensorboard=True) # Must be called before SummaryWriter

writer = SummaryWriter("newdir")
...
```

### TensorBoardX

```python
from tensorboardX import SummaryWriter
import vessl

vessl.init(tensorboard=True) # Must be called before SummaryWriter

writer = SummaryWriter("newdir")
...
```
