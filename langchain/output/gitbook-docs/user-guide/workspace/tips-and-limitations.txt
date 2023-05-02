# Tips & Limitations

### pip

If you install a python package with pip but you cannot find it, then add the following path as [the official jupyterlab document](https://jupyterlab.readthedocs.io/en/stable/getting\_started/installation.html#pip) stated.

```
export PATH="/home/name/.local/bin:$PATH"
```



### conda

If you want to use conda in your notebook, run the following code as the first executable cell.

```
!pip install -q git+https://github.com/vessl-ai/condacolab.git
import condacolab
condacolab.install()
```

VESSL provides a python package to install conda based on condacolab. For more information, see the [github repository](https://github.com/vessl-ai/condavessl).



### Disk persistency and node affinity problem

In VESSL,  `/home/vessl` **is the only persistent directory**. Other directories reset eveytime you restart a workspace. If you need libraries or packages that should be installed outside  `/home/vessl`, fill the `init` script with the install commands. You can also build your own docker image on top of [VESSL managed docker images](tips-and-limitations.md#undefined).&#x20;

![](<../../.gitbook/assets/image (114).png>)

VESSL provides disk persistency in two ways:

* For cloud providers (managed cluster and custom cluster provided by AWS or GCP), VESSL uses storage provisioners like `aws-efs-csi-driver`. It automatically attach EFS volumes to container when started, and stored persistently until the workspace is terminated.
* For on-premise cluster, VESSL use storage provisioners like `local-path-provisioner`. It stores data on the host machine assigned to when the workspace is created. (So it fixes to one machine due to storage persistency).&#x20;
  * VESSL does online backup/restore to resolve this issue. VESSL automatically backup and upload all contents in `/home/vessl` when the workspace is stopped. All contents will be downloaded and restored when the workspace is resumed.
  * If `/home/vessl/` is larger than 15GB, VESSL does not online backup/restore, so it fixes to one machine.
  * (For enterprise plan) Organization admin can specify the detail rules of online backup, such as the backup location.

#### Backup & Restore manually

You can manually backup & restore `/home/vessl/` with CLI. This feature is useful in the following situations:

* Move the workspace to another cluster
* Clone the workspace

You can proceed the following order:

* Run [`vessl workspace backup`](../../api-reference/cli/vessl-workspace.md#savvihub-workspace-backup) from the source workspace
* Run [`vessl workspace restore`](../../api-reference/cli/vessl-workspace.md#savvihub-workspace-restore) from the destination workspace
  * `/home/vessl/` folder should be empty in the destination workspace

If `/home/vessl/` is larger than 15GB, VESSL CLI does not support backup/restore.



### Docker

VESSL workspaces are docker containers running on Kubernetes. Docker daemon inside a docker container is not supported unless specifically privileged. VESSL does not support privileged containers for security reasons.



### File Owners (EFS, managed cluster)

On managed clusters, Amazon EFS is mounted to `/home/vessl` and owners and groups of all files inside this directory are managed by [Amazon EFS](https://docs.aws.amazon.com/efs/latest/ug/accessing-fs-nfs-permissions.html). Amazon EFS uses numberic identifies to represent file ownership and check permissions when a user attempts to access a file system object. This can cause a problem in some cases. For example, vim creates `.viminfo` file with permission `600`. However, because the owner mapped to a random numeric value – not `vessl`, vim cannot write to `.viminfo`. In this case, run `chmod o+w .viminfo`.&#x20;



###

