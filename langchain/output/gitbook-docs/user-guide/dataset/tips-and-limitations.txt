# Tips & Limitations

### CIFS mount

VESSL providers FlexVolume storage type to support CIFS mount.

1. Install [CIFS FlexVolume plugin](https://github.com/fstab/cifs).&#x20;
2. Create `secret.yml` for CIFS mount.
3. Fill the options in the create dataset dialog.

![](<../../.gitbook/assets/image (190).png>)



### Use other mount options not supported by VESSL

By using HostPath mount, you can work around to use other mount options which are not supported by VESSL.

1. Mount the storage on all host machines, in the same path. (e.g. `/mnt/s3fs-mnist-data`)
2. Mount dataset with the HostPath option.
