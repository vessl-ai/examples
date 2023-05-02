# vessl.upload

Use `vessl.upload` to upload output files. In a VESSL-managed experiment, output files are uploaded by default. However, in a [local experiment](../../user-guide/experiment/local-experiments.md), you need to call this method explicitly. Call this method from within your experiment.

| Parameter | Description    |
| --------- | -------------- |
| `path`    | Path to upload |

### Example

```python
import vessl

if __name__ == '__main__':
    vessl.init()
    ...
    vessl.upload("./output/")  # Uploads the folder to experiment output volume
```

