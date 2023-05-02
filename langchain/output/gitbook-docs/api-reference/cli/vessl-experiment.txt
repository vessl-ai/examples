# vessl experiment

### Overview

Run `vessl experiment --help` to view the list of commands, `vessl experiment [COMMAND] -help` to view individual command instructions.

### Create an experiment

```
vessl experiment create [OPTIONS]
```

| Option                                | Description                                                                                                                                                |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-c`, `--cluster`                     | Cluster name (must be specified before other options)                                                                                                      |
| `-x`, `--command`                     | Start command to execute in experiment container                                                                                                           |
| `-r`, `--resource`                    | Resource type to run an experiment (for managed cluster only)                                                                                              |
| `--processor-type`                    | `CPU` or `GPU` (for custom cluster only)                                                                                                                   |
| `--cpu-limit`                         | Number of vCPUs (for custom cluster only)                                                                                                                  |
| `--memory-limit`                      | Memory limit in GiB (for custom cluster only)                                                                                                              |
| `--gpu-type`                          | <p>GPU type (for custom cluster only)</p><p>ex. <code>Tesla-K80</code></p>                                                                                 |
| `--gpu-limit`                         | Number of GPU cores (for custom cluster only)                                                                                                              |
| `--upload-local-file` _(multiple)_    | <p>Upload local file. Format: [local_path] or [local_path]:[remote_path].</p><p>ex. <code>--upload-local-file my-project:/home/vessl/my-project</code></p> |
| `--upload-local-git-diff`             | Upload local git commit hash and diff (only works in project repositories)                                                                                 |
| `-i`, `--image-url`                   | <p>Kernel docker image URL</p><p>ex. <code>vessl/kernels:py36.full-cpu</code></p>                                                                          |
| `-m`, `--message`                     | Message                                                                                                                                                    |
| `--termination-protection`            | Enable termination protection                                                                                                                              |
| `-h`, `--hyperparameter` (_multiple_) | <p>Hyperparameters in the form of <code>[key]=[value]</code></p><p>ex.  <code>-h lr=0.01 -h epochs=100</code></p>                                          |
| `--dataset` (_multiple_)              | <p>Dataset mounts in the form of <code>[mount_path] [dataset_name]</code></p><p>ex. <code>--dataset /input mnist</code></p>                                |
| `--root-volume-size`                  | Root volume size (defaults to `20Gi`)                                                                                                                      |
| `--working-dir`                       | Working directory path (defaults to `/home/vessl/`)                                                                                                        |
| `--output-dir`                        | Output directory path (defaults to `/output`                                                                                                               |
| `--local-project`                     | Local project file URL                                                                                                                                     |
| `--worker-count`                      | Number of workers (for distributed experiment only)                                                                                                        |
| `--framework-type`                    | Specify `pytorch` or `tensorflow`(for distributed experiment only)                                                                                         |

### Download experiment output files

Each user can define experiment output files. You can save validation results, trained checkpoints, best performing models and other artifacts.

```
vessl experiment download-output [OPTIONS] NAME
```

| Argument | Description     |
| -------- | --------------- |
| `NAME`   | Experiment name |

| Option            | Description                                     |
| ----------------- | ----------------------------------------------- |
| `-p`, `--path`    | Local download path (defaults to`./output`)     |
| `--worker-number` | Worker number (for distributed experiment only) |

### List all experiments

```
vessl experiment list
```

### List experiment output files

Each user can define experiment output files. You can save validation results, trained checkpoints, best models, and other artifacts.

```
vessl experiment list-output [OPTIONS] NAME
```

| Argument | Description     |
| -------- | --------------- |
| `NAME`   | Experiment name |

| Option              | Description                                     |
| ------------------- | ----------------------------------------------- |
| `-r`, `--recursive` | List files recursively                          |
| `--worker-number`   | Worker number (for distributed experiment only) |



### View logs of the experiment container

```
vessl experiment logs [OPTIONS] NAME
```

| Argument | Description     |
| -------- | --------------- |
| `NAME`   | Experiment name |

| Option            | Description                                               |
| ----------------- | --------------------------------------------------------- |
| `--tail`          | Number of lines to display from the end (defaults to 200) |
| `--worker-number` | Worker number (for distributed experiment only)           |



### View information on the experiment

```
vessl experiment read NAME
```

| Argument | Description     |
| -------- | --------------- |
| `NAME`   | Experiment name |

### Terminate an experiment

```
vessl experiment terminate NAME
```

| Argument | Description     |
| -------- | --------------- |
| `NAME`   | Experiment name |
