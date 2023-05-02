# vessl model

### Overview

Run `vessl model-repository --help` to view the list of commands related to model repositories, or `vessl model-repository [COMMAND] --help` to view individual command instructions.

Run `vessl model --help` to view the list of commands related to models, or `vessl model [COMMAND] --help` to view individual command instructions.

### Create a model repository

```
vessl model-repository [OPTIONS] NAME
```

| Argument | Description           |
| -------- | --------------------- |
| `NAME`   | Model repository name |

| Option                | Description                  |
| --------------------- | ---------------------------- |
| `-m`, `--description` | Model repository description |

### List all model repositories

```
vessl model-repository list
```

### View information on the model repository

```
vessl model-repository read NAME
```

| Argument | Description           |
| -------- | --------------------- |
| `NAME`   | Model repository name |

### Create a model

```
vessl model create [OPTIONS] REPOSITORY_NAME
```

| Argument          | Description           |
| ----------------- | --------------------- |
| `REPOSITORY_NAME` | Model repository name |

| Option            | Description                             |
| ----------------- | --------------------------------------- |
| `--model-name`    | Model name                              |
| `--source`        | Model source (experiment or local)      |
| `--experiment-id` | Experiment id to create a model         |
| `--paths`         | Paths to create model. Default: `["/"]` |

### Delete a file within a model

```
vessl model delete-file [OPTIONS] REPOSITORY_NAME MODEL_NUMBER PATH
```

| Argument          | Description           |
| ----------------- | --------------------- |
| `REPOSITORY_NAME` | Model repository name |
| `MODEL_NUMBER`    | Model number          |
| `PATH`            | File path             |

| Option            | Description                     |
| ----------------- | ------------------------------- |
| `-r, --recursive` | Required if file is a directory |

### Download the model

```
vessl model download REPOSITORY_NAME MODEL_NUMBER SOURCE DEST
```

| Argument          | Description                  |
| ----------------- | ---------------------------- |
| `REPOSITORY_NAME` | Model repository name        |
| `MODEL_NUMBER`    | Model number                 |
| `SOURCE`          | Source path within the model |
| `DEST`            | Local destination path       |

### List all models <a href="#usd-savvihub-image-list" id="usd-savvihub-image-list"></a>

```
vessl model list
```

### List model files

```
vessl model list-files [OPTIONS] REPOSITORY_NAME MODEL_NUMBER
```

| Argument          | Description           |
| ----------------- | --------------------- |
| `REPOSITORY_NAME` | Model repository name |
| `MODEL_NUMBER`    | Model number          |

| Option            | Description                               |
| ----------------- | ----------------------------------------- |
| `-p, --path`      | Directory path to list (defaults to root) |
| `-r, --recursive` | List files recursively                    |

### View information on the model

```
 vessl model read REPOSITORY_NAME MODEL_NUMBER
```

| Argument          | Description           |
| ----------------- | --------------------- |
| `REPOSITORY_NAME` | Model repository name |
| `MODEL_NUMBER`    | Model number          |

### Upload files to a model

```
vessl model upload REPOSITORY_NAME MODEL_NUMBER SOURCE DEST
```

| Argument          | Description                        |
| ----------------- | ---------------------------------- |
| `REPOSITORY_NAME` | Model repository name              |
| `MODEL_NUMBER`    | Model number                       |
| `SOURCE`          | Local source path                  |
| `DEST`            | Destinataion path within the model |
