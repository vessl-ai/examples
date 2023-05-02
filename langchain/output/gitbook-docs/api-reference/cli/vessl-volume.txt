# vessl volume

### Overview

Run `vessl volume --help` to view the list of commands, or `vessl volume [COMMAND] --help` to view individual command instructions.

{% hint style="info" %}
In most cases, you will not interact with volume files directly.&#x20;
{% endhint %}

## Commands

### Copy a volume file&#x20;

You can copy a volume file either from local to remote, remote to local, or remote to remote.&#x20;

```
vessl volume copy [OPTIONS]
```

| Option              | Description                                                                                  |
| ------------------- | -------------------------------------------------------------------------------------------- |
| `--source-id`       | Source volume file id. If not specified, source is assumed to be local.                      |
| `--source-path`     | If `--source-id` is empty, local source path. Otherwise, remote source path.                 |
| `--dest-id`         | Destination volume file id. If not specified, destination is assumed to be local.            |
| `--dest-path`       | If `--dest-id` is empty, local destination path. Otherwise, remote destination path.         |
| `-r`, `--recursive` | Required both `--source-id` and `--dest-id` are specified, and `--source-id` is a directory. |

### Delete the volume file

```
vessl volume delete [OPTIONS] ID
```

| Argument | Description    |
| -------- | -------------- |
| `ID`     | Volume file ID |

| Option              | Description                            |
| ------------------- | -------------------------------------- |
| `-r`, `--recursive` | Required if volume file is a directory |

### List volume files

```
vessl volume list [OPTIONS] ID
```

| Argument | Description    |
| -------- | -------------- |
| `ID`     | Volume file ID |

| Option              | Description                                |
| ------------------- | ------------------------------------------ |
| `-p`, `--path`      | Path within volume file (defaults to root) |
| `-r`, `--recursive` | Required if path is a directory            |
