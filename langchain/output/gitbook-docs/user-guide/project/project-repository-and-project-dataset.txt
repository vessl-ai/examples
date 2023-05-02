# Project Repository & Project Dataset

Project provides a way to connect code repositories and datasets. VESSL provides the following with project repositories & project datasets&#x20;

* Download codes & datasets when a experiment / a sweep is created
* Track versions and file diffs between experiments

### Add Project Repository

Project Repository can be configured in creating project and project settings.&#x20;

![](<../../.gitbook/assets/image (117).png>)

Add project repository and select github repositories in the dialog. If you have not integrated github with VESSL, you should integrate github first in the organization settings (link will be given in the add dialog.)

![](<../../.gitbook/assets/image (217).png>)

### Add Project Dataset

Project dataset can be configured in the same way as the project repository. Unlike project repository, project dataset is allowed to specify the mount path in the experiment/sweep.

![](<../../.gitbook/assets/image (176).png>)

Once after you connected repositories and datasets, they are mounted by default when creating an experiment / a sweep.

![](<../../.gitbook/assets/image (156).png>)

#### Connect cluster-scoped local dataset ([docs](../dataset/adding-new-datasets.md))

You can connect the cluster-scoped dataset to project dataset. If you use different cluster from the cluster specified in the dataset during creating an experiment, an error may occur. To resolve it, you need to choose the cluster specified in the dataset or continue without the certain dataset.

![](<../../.gitbook/assets/image (99).png>)
