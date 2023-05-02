# Volume Mount

Users can mount datasets, projects, and files to either an experiment or a service.&#x20;

### 1. Volume mount to a VESSL cluster

When you create experiments or services on a VESSL cluster, you can find the volume mount tree in the form of a file directory starting from the root (`/`). You can mount different objects to specified repository paths as following:

| Repository path | Mount objects                            | Note                              |
| --------------- | ---------------------------------------- | --------------------------------- |
| /input          | Dataset, Project, a trained model, files | A default mount path for datasets |
| /output         | Artifacts                                | A fixed immutable empty directory |
| /work           | Project                                  | A working directory               |

![](<../../.gitbook/assets/image (237).png>)

### 2. Volume mount to a custom cluster

You can start experiments or services on a custom cluster by clicking **ADD NFS** and add datasets and files by the clicking corresponding buttons. The NFS mount options are as follows:

| Option      | Note                                           | Examples         |
| ----------- | ---------------------------------------------- | ---------------- |
| Server URL  | NFS server endpoint                            | 10.10.10.10      |
| Server Path | NFS directory absolute path to mount           | /volume1/sharing |
| Mount Path  | Mount destination path to experiment container | /input           |

![](<../../.gitbook/assets/image (223).png>)

{% hint style="warning" %}
You can only mount NFS volume to **custom clusters.**
{% endhint %}



