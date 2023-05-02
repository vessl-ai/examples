# vessl workspace

### Overview

Run `vessl workspace --help` to view the list of commands, or `vessl workspace [COMMAND] --help` to view individual command instructions.

### Create a workspace

```bash
vessl workspace create [OPTIONS] NAME
```

| Argument | Description    |
| -------- | -------------- |
| `NAME`   | workspace name |

| Option                             | Description                                                                                                                                                |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-c`, `--cluster`                  | Cluster name (must be specified before other options)                                                                                                      |
| `--node`                           | Cluster nodes. Defaults to all nodes in cluster.                                                                                                           |
| `-r`, `--resource`                 | Resource type to run an experiment (for managed cluster only)                                                                                              |
| `--processor-type`                 | `CPU` or `GPU` (for custom cluster only)                                                                                                                   |
| `--cpu-limit`                      | Number of vCPUs (for custom cluster only)                                                                                                                  |
| `--memory-limit`                   | Memory limit in GiB (for custom cluster only)                                                                                                              |
| `--gpu-type`                       | <p>GPU type (for custom cluster only)</p><p>ex. <code>Tesla-K80</code></p>                                                                                 |
| `--gpu-limit`                      | Number of GPU cores (for custom cluster only)                                                                                                              |
| `-i`, `--image-url`                | <p>Kernel docker image URL</p><p>ex. <code>vessl/kernels:py36.full-cpu</code></p>                                                                          |
| `--max-hours`                      | Maximum number of hours to run workspace. Defaults to 24.                                                                                                  |
| `--dataset` (_multiple_)           | <p>Dataset mounts in the form of <code>[mount_path]:[dataset_name]</code></p><p>ex. <code>--dataset /input:mnist</code></p>                                |
| `--upload-local-file` _(multiple)_ | <p>Upload local file. Format: [local_path] or [local_path]:[remote_path].</p><p>ex. <code>--upload-local-file my-project:/home/vessl/my-project</code></p> |
| `--root-volume-size`               | Root volume size (defaults to `100Gi`)                                                                                                                     |
| `-p`, `--port` _(multiple)_        | Format: \[expose\_type] \[port] \[name], ex. `-p 'tcp 22 ssh'`. Jupyter and SSH ports exist by default.                                                    |
| `--init-script`                    | Custom init script                                                                                                                                         |

###

### Connect to a running workspace

```bash
vessl workspace ssh [OPTIONS]
```

| Option     | Description          |
| ---------- | -------------------- |
| --key-path | SSH private key path |

```bash
$ vessl workspace ssh
The authenticity of host '[tcp.apne2-prod1-cluster.vessl.com]:31123 ([52.78.240.117]:31123)' can't be established.
ECDSA key fingerprint is SHA256:ugLx91zLE9ELAqT19uNjQ.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added '[tcp.apne2.vessl.com]:31123,[52.78.240.117]:31123' (ECDSA) to the list of known hosts.
Linux workspace-x1hczjvygiql-0 4.14.225-169.362.amzn2.x86_64 #1 SMP Mon Mar 22 20:14:50 UTC 2021 x86_64

The programs included with the Debian GNU/Linux system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Debian GNU/Linux comes with ABSOLUTELY NO WARRANTY, to the extent
permitted by applicable law.
john@workspace-x1hczjvygiql-0:~$ 
```



### Connect to workspaces via VSCode Remote-SSH <a href="#savvihub-workspace-vscode" id="savvihub-workspace-vscode"></a>

```bash
vessl workspace vscode [OPTIONS]
```

| Option     | Description          |
| ---------- | -------------------- |
| --key-path | SSH private key path |

```bash
$ vessl workspace vscode
Updated '/Users/johndoe/.ssh/config'.
```

### Backup the home directory of the workspace <a href="#savvihub-workspace-backup" id="savvihub-workspace-backup"></a>

Create a zip file at `/tmp/workspace-backup.zip` and uploads the backup to VESSL server.

{% hint style="info" %}
You should run this command inside a running workspace.
{% endhint %}

```bash
vessl workspace backup
```

```bash
$ vessl workspace backup
Successfully uploaded 1 out of 1 file(s).
```

### Restore workspace home directory from a backup.  <a href="#savvihub-workspace-restore" id="savvihub-workspace-restore"></a>

Download the zip file to `/tmp/workspace-backup.zip` and extract to `/home/vessl/`.

{% hint style="info" %}
You should run this command inside a running workspace.
{% endhint %}

```bash
vessl workspace restore
```

```bash
$ vessl workspace restore
[?] Select workspace: rash-uncle (backup created 13 minutes ago)
 > rash-uncle (backup created 13 minutes ago)
   hazel-saver (backup created 2 days ago)

Successfully downloaded 1 out of 1 file(s).
```

### List all workspaces <a href="#usd-savvihub-image-list" id="usd-savvihub-image-list"></a>

```
vessl workspace list
```

### View information on the workspace

```
vessl workspace read ID
```

| Argument | Description  |
| -------- | ------------ |
| `ID`     | Workspace ID |

### View logs of the workspace container

```
vessl workspace logs ID
```

| Argument | Description  |
| -------- | ------------ |
| `ID`     | Workspace ID |

| Option   | Description                                               |
| -------- | --------------------------------------------------------- |
| `--tail` | Number of lines to display from the end (defaults to 200) |

### Start a workspace container

```
vessl workspace start ID
```

| Argument | Description  |
| -------- | ------------ |
| `ID`     | Workspace ID |

### Stop a workspace container

```
vessl workspace stop ID
```

| Argument | Description  |
| -------- | ------------ |
| `ID`     | Workspace ID |

### Terminate a workspace container

```
vessl workspace terminate ID
```

| Argument | Description  |
| -------- | ------------ |
| `ID`     | Workspace ID |
