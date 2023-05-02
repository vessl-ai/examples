# vessl sweep

### Overview

Run `vessl sweep --help` to view the list of commands, or `vessl sweep [COMMAND] --help` to view individual command instructions.

### Create a sweep

```
vessl sweep create [OPTIONS]
```

| Option                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-T`, `--objective-type`              | `minimize` or `maximize`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `-G`, `--objective-goal`              | <p>Objective goal<br>ex. 0.99</p>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `-M`, `--objective-metric`            | <p>Objective metric<br>ex. <code>val_accuracy</code></p>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `--num-experiments`                   | Maximum number of experiments                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `--num-parallel`                      | Number of experiments to be run in parallel                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `--num-failed`                        | Maximum number of experiments to allow to fail                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `-a`, `--algorithm`                   | `grid`, `random`, or `bayesian`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `-p`, `--parameter` _(multiple)_      | <p>Search space parameters in the form of <code>[name] [type] [range_type] [values...]</code>. <br><code>[type]</code> must be one of <code>categorical</code>, <code>int</code>, or <code>double</code>. <br><code>[range_type]</code> must be either <code>space</code> or  <code>list</code>.  If <code>space</code>, <code>[values...]</code> is a 3-tuple of <code>[min] [max] [step]</code>. If <code>list</code> , <code>[values...]</code> is a list of values to search.</p><p>ex. <code>-p epochs int space 100 1000 50</code> </p> |
| `-c`, `--cluster`                     | Cluster name (must be specified before other options)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `-x`, `--command`                     | Start command to execute in experiment container                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `-r`, `--resource`                    | Resource type to run an experiment (for managed cluster only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `--processor`                         | `CPU` or `GPU` (for custom cluster only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `--cpu-limit`                         | Number of vCPUs (for custom cluster only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `--memory-limit`                      | Memory limit in GiB (for custom cluster only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `--gpu-type`                          | <p>GPU type (for custom cluster only)</p><p>ex. <code>Tesla-K80</code></p>                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `--gpu-limit`                         | Number of GPU cores (for custom cluster only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `-i`, `--image-url`                   | <p>Kernel docker image URL</p><p>ex. <code>vessl/kernels:py36.full-cpu</code></p>                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `--early-stopping-name`               | Early stopping algorithm name                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `--early-stopping-settings`           | <p>Early stopping algorithm settings in the format of <code>[key] [value]</code><br>ex. <code>--early-stopping-settings start_step 4</code></p>                                                                                                                                                                                                                                                                                                                                                                                               |
| `--message`                           | Message                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `-h`, `--hyperparameter` (_multiple_) | <p>Hyperparameters in the form of <code>[key]=[value]</code></p><p>ex.  <code>-h lr=0.01 -h epochs=100</code></p>                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `--dataset` (_multiple_)              | <p>Dataset mounts in the form of <code>[mount_path] [dataset_name]</code></p><p>ex. <code>--dataset /input mnist</code></p>                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `--root-volume-size`                  | Root volume size (defaults to `20Gi`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `--working-dir`                       | Working directory path (defaults to `/work/[project_name]`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `--output-dir`                        | Output directory path (defaults to `/output`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `--local-project`                     | Local project file URL                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

### List all sweeps

```
vessl sweep list
```

### View logs of the sweep container

```
vessl sweep logs [OPTIONS] NAME
```

| Argument | Description |
| -------- | ----------- |
| `NAME`   | Sweep name  |

| Option   | Description                                               |
| -------- | --------------------------------------------------------- |
| `--tail` | Number of lines to display from the end (defaults to 200) |

### View information on the sweep

```
vessl sweep read NAME
```

| Argument | Description |
| -------- | ----------- |
| `NAME`   | sweep name  |

### Terminate the sweep

```
vessl sweep terminate NAME
```

| Argument | Description |
| -------- | ----------- |
| `NAME`   | sweep name  |

### Find the best sweep experiment&#x20;

```
vessl sweep best-experiment
```

| Argument | Description |
| -------- | ----------- |
| `NAME`   | sweep name  |
