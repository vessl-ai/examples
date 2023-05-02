# Adding New Datasets

When you click the **NEW DATASET** under **DATASET** page, you will be asked to add new dataset either from local or external data source. You have three data provider options: Vessl, Amazon Simple Storage Service, and Google Cloud Storage.

![](<../../.gitbook/assets/image (95).png>)

![](<../../.gitbook/assets/image (109).png>)

{% tabs %}
{% tab title="VESSL" %}
When you select a VESSL dataset, you can upload data from the local disk. To create a VESSL dataset:

1. Enter **Dataset Name**
2. Click **Upload Files**
3. Click **Submit**
{% endtab %}

{% tab title="AWS S3" %}
You can retrieve dataset from S3 by selecting Amazon Simple Storage Service. To create a dataset from S3

1. Enter **Dataset Name**
2. Enter **ARN**
3. Enter **Bucket Path**
4. Click **Create**
{% endtab %}

{% tab title="GCS" %}
You also have an option to retrieve dataset from Google Cloud Storage. To create a dataset from GCS:

1. Enter **Dataset Name**
2. Enter **Bucket Path**
3. Click **Create** button
{% endtab %}

{% tab title="Local (cluster-scoped)" %}
If the dataset exists inside the cluster (NAS, host machine, etc.) and you want to mount it only inside the cluster, you can select Local Storage option. In this case, VESSL only stores the location of the dataset, and mounts the path when an experiment is created.



VESSL supports 3 types of local mounts

* NFS
* HostPath
* FlexVolume (e.g. [CIFS mount](tips-and-limitations.md#cifs-mount))

![](<../../.gitbook/assets/image (107).png>)



Since **VESSL does not have access to the local dataset**, you cannot browse local dataset files on VESSL.
{% endtab %}
{% endtabs %}



{% hint style="info" %}
A detailed integration guide is provided on each **Create Dataset** dialog.
{% endhint %}
