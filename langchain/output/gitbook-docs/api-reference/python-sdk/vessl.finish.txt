# vessl.finish

Use `vessl.finish` to manually stop your [local experiment](../../user-guide/experiment/local-experiments.md).

{% hint style="info" %}
`vessl.finish` will have no effect in a VESSL-managed experiment.
{% endhint %}

### Example

```python
import vessl

if __name__ == '__main__':
    vessl.init()
    ...
    vessl.finish()
    ...  # Rest of your code
```

