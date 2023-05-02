# vessl.progress

Use `vessl.progress` to track the progress of your experiment. VESSL provides an estimate of a remaining training time by calculating the average elapsed time of previous epochs or batch sizes. You can view this information by hovering over the status of a running experiment. This can be used in both VESSL's managed server or in a local environment.&#x20;

| Parameter | Description                                        |
| --------- | -------------------------------------------------- |
| `value`   | Amount of progress (decimal value between 0 and 1) |

### Examples

```python
import vessl

for epoch in range(epochs):
    ...
    
    # Update experiment progress every epoch
    vessl.progress((epoch+1) / epochs)
```

```python
def train(model, device, train_loader, optimizer, epoch, start_epoch):
    model.train()
    loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        ...

        # Update experiment progress every batch
        vessl.progress(
            ((epoch+1)*batch_size + batch_idx) / (batch_size * epochs))
        )
```
