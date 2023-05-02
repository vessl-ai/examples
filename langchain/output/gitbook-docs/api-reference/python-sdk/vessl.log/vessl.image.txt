# vessl.Image

Use the `vessl.Image` class to log image data. This takes the image data and saves it as a local PNG file in the `vessl-media/image` directory with randomly generated names.

| Parameter | Description                                                                                                                                                                                                                                      |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data`    | <p>Supported types<br> - <code>PIL Image</code>: the <code>Image</code> module of Pillow</p><p> - <code>torch.Tensor</code>: a PyTorch tensor </p><p> - <code>numpy.ndarray</code>: a NumPy array </p><p> - <code>str</code>: the image path</p> |
| `caption` | Label of the given image                                                                                                                                                                                                                         |

### `PIL Image`

```python
import vessl
from PIL import Image

my_PIL_image = Image.open('my-image.png')
vessl.Image(
    data=my_PIL_image,
    caption='my-caption',
)
```

### `torch.Tensor`

```python
import vessl
import torch

vessl.Image()
test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=10, shuffle=True)
for data, target in test_loader:
    vessl.Image(
        data=data[0], 
        caption=f'Target:{target[0]}',
    )
```

### `numpy.ndarray`

```python
import vessl
import numpy as np

my_np_image = np.array([[0,1,1,0],[1,0,0,1],[0,1,1,0]]) 
vessl.Image(
    data= my_np_image,
    caption='my-caption',
)
```

### `str`

```python
import vessl

my_image_path = 'my-image.png'
vessl.Image(
    data=my_image_path,
    caption='my-caption',
)

```
