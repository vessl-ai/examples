# Managing Datasets

Under Datasets, you can view the file tree of your dataset. Here, you can also upload file, create folder.

![](<../../.gitbook/assets/image (98).png>)

### Dataset Versioning (Enterprise only)

**Dataset Version** is a specific snapshot of a dataset captured at a particular point in time. To enable this feature, you have to check `Enable Versioning` when creating dataset.

Dataset Version can be created by yourself on the `VERSIONS` tab, or be automatically created when an experiment is created to provider reproducibility of the experiment. You can also choose the specific dataset version to use when creating an experiment.&#x20;

{% hint style="danger" %}
This feature is currently unavailable for AWS S3 or GCS dataset source types.
{% endhint %}



#### How it works

If you enable versioning when creating a dataset, all dataset files are incrementally saved as blob. Each Version stores the mapping from blob to actual file path.

Each blob is stored incrementally by hashsum - the dataset size does not increase even if dataset version is created frequently, unless the files are changed.

