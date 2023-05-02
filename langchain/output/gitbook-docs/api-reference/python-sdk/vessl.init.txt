# vessl.init

Use `vessl.init` to initialize a [local experiment](../../user-guide/experiment/local-experiments.md). This will create a new experiment which you can view on VESSL's web interface.

You can also continue with previously initialized experiments using the `experiment_name_or_number` parameter. Note that this experiment must not be in a VESSL-managed experiment and must be in the running state.

| Parameter                   | Description                              |
| --------------------------- | ---------------------------------------- |
| `experiment_name_or_number` | Experiment to reinitialize               |
| `message`                   | Experiment message (for new experiments) |
| `hp`                        | Experiment hyperparameters (for record)  |

{% hint style="info" %}
`vessl.init` will have no effect in a VESSL-managed experiment.
{% endhint %}

### Example

```python
import vessl

if __name__ == '__main__':
    vessl.init()
    ... # Rest of your code
```

