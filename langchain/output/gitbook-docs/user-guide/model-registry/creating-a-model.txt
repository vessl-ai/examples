# Creating a Model

You can add a model to a model registry by selecting a previous experiment or by uploading local checkpoint files. To create a model, create a model repository first under **MODELS**. We recommend following our naming conventions to improve project maintainability.

![](<../../.gitbook/assets/image (159).png>)

### Creating a model from experiment

{% hint style="info" %}
Only`completed`experiments can be sourced to create models.
{% endhint %}

There are two entry points to create a model in the repository.

#### 1. Create a model from the model repository page

You can create models on the model repository page. Click the `New Model` button, set the model description and tag, find the experiment you want, and choose the desired directory you want to put in the model.&#x20;

![](<../../.gitbook/assets/image (142).png>)

#### 2. Create a model from the experiment detail page

If you find an experiment that you want to create a model from its output files, you can create one by clicking the `Create Model` button under the `Actions` button on the experiment detail page. Select the model repository and click`SELECT` on the dialog.&#x20;

![](<../../.gitbook/assets/image (194).png>)

Then, set the model description and tag, and choose the desired directory among the output files of the experiment on the model create page. You can include or exclude specific directories in the output files checkbox section.

![](<../../.gitbook/assets/image (192).png>)

### Creating a model from local files

Uploading the checkpoint files on your local machine to VESSL is another way to utilize the model registry feature. If you select the `model from local` type when selecting **Source** in `Models>Model Repository>New Model`, you can create a model by uploading a local file.

![](<../../.gitbook/assets/image (183).png>)
