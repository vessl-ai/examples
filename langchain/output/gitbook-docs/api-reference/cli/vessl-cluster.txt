# vessl cluster

### Overview

Run `vessl cluster --help` to view the list of commands, or `vessl cluster [COMMAND] --help` to view individual command instructions.

{% hint style="info" %}
Certain commands only apply to custom kernel clusters.
{% endhint %}

### Delete a custom kernel cluster

```
vessl cluster delete NAME
```

| Argument | Description                |
| -------- | -------------------------- |
| `NAME`   | Custom kernel cluster name |

### List all kernel clusters

```
vessl cluster list
```

### List nodes in a kernel cluster

```
vessl cluster list-nodes NAME
```

| Argument | Description                |
| -------- | -------------------------- |
| `NAME`   | Custom kernel cluster name |

### View information on a kernel cluster

```
vessl cluster read NAME
```

| Argument | Description         |
| -------- | ------------------- |
| `NAME`   | Kernel cluster name |

### Rename a custom kernel cluster

```
vessl cluster rename NAME NEW_NAME
```

| Argument   | Description                |
| ---------- | -------------------------- |
| `NAME`     | Custom kernel cluster name |
| `NEW_NAME` | New kernel cluster name    |
