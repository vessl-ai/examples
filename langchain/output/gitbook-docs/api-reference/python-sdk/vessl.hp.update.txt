# vessl.hp.update

To record hyperparameters in VESSL **experiments**, set `vessl.hp` and update with `vessl.hp.update` as follows.

#### Option 1: record hyperparameters with Python dictionary

```python
import vessl

d = {"lr": 0.1, "optimizer": "sgd"}
vessl.hp.update(d)
```

#### Option 2: record hyperparameters with Python argparse module

```python
import argparse
import vessl

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_layers', type=int, default=3)
args = parser.parse_args(args=[])

vessl.hp.update(args)
```

#### Option 3: record hyperparameters with [vessl.init()](vessl.init.md)

{% hint style="info" %}
vessl.init will have no effect in a VESSL-managed experiment&#x20;
{% endhint %}

You can pass hyperparameters as a parameter of init.

```python
import vessl

d = {"lr": 0.1, "optimizer": "sgd"}
vessl.init(hp=d)
```

Or, you can call `vessl.init()` first, set `vessl.hp`, and call `vessl.hp.update()` without any parameters.

```python
import vessl

vessl.init()

vessl.hp.lr = 0.1
vessl.hp.optimizer = "sgd"  # vessl.hp = {'lr': '0.1', 'optimizer': 'sgd'}

vessl.hp.update()
```
