# vessl dataset

### Overview

Run `vessl dataset --help` to view the list of commands, or `vessl dataset [COMMAND] --help` to view individual command instructions.

### Copy files within a dataset

{% hint style="warning" %}
This is not supported for externally sourced datasets.
{% endhint %}

```
vessl dataset copy [OPTIONS] NAME SOURCE DEST
```

| Argument | Description                    |
| -------- | ------------------------------ |
| `NAME`   | Dataset name                   |
| `SOURCE` | Source path within the dataset |
| `DEST`   | Destination file path within   |

| Option              | Description                            |
| ------------------- | -------------------------------------- |
| `-r`, `--recursive` | Required if source file is a directory |

### Create a dataset

```
vessl dataset create [OPTIONS] NAME
```

| Argument | Description  |
| -------- | ------------ |
| `NAME`   | Dataset name |

| Option                  | Description                               |
| ----------------------- | ----------------------------------------- |
| `-m`, `--description`   | Dataset description                       |
| `-e`, `--external-path` | AWS S3 or Google Cloud Storage bucket URL |
| `--aws-role-arn`        | AWS Role ARN to access S3                 |
| `--enable-versioning`   | Enable versioning                         |
| `--version-path`        | Versioning bucket path                    |

### Delete a file within a dataset

{% hint style="warning" %}
This is not supported for externally sourced datasets.
{% endhint %}

```
vessl dataset delete-file [OPTIONS] NAME PATH
```

| Argument | Description  |
| -------- | ------------ |
| `NAME`   | Dataset name |
| `PATH`   | File path    |

| Option              | Description                     |
| ------------------- | ------------------------------- |
| `-r`, `--recursive` | Required if file is a directory |

### Download dataset files

```
vessl dataset download [OPTIONS] NAME SOURCE DEST
```

| Argument | Description                    |
| -------- | ------------------------------ |
| `NAME`   | Dataset name                   |
| `SOURCE` | Source path within the dataset |
| `DEST`   | Local destination path         |

### List all datasets

```
vessl dataset list
```

### List dataset files

```
vessl dataset list-files [OPTIONS] NAME
```

| Argument | Description  |
| -------- | ------------ |
| `NAME`   | Dataset name |

| Option              | Description                               |
| ------------------- | ----------------------------------------- |
| `-p`, `--path`      | Directory path to list (defaults to root) |
| `-r`, `--recursive` | List files recursively                    |

### View information on the dataset

```
vessl dataset read [OPTIONS] NAME
```

| Argument | Description  |
| -------- | ------------ |
| `NAME`   | Dataset name |

### Upload files to a dataset

```
vessl dataset upload NAME SOURCE DEST
```

| Argument | Description                         |
| -------- | ----------------------------------- |
| `NAME`   | Dataset name                        |
| `SOURCE` | Local source path                   |
| `DEST`   | Destination path within the dataset |
