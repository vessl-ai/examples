# Creating an Experiment

To create an experiment, first specify a few options such as cluster, resource, image, and start command. Here is an explanation of the config options.

![](../../.gitbook/assets/Experiments.001.jpeg)

### Cluster & Resource (Required) <a href="#runtime" id="runtime"></a>

You can run your experiment on either VESSL's managed cluster or your custom cluster. Start by selecting a cluster.&#x20;

{% tabs %}
{% tab title="VESSL's Managed Cluster" %}
Once you selected VESSL's managed cluster, you can view a list of available resources under the dropdown menu.&#x20;

![](<../../.gitbook/assets/image (218).png>)

You also have an option to use spot instances.

{% content-ref url="../commons/running-spot-instances.md" %}
[running-spot-instances.md](../commons/running-spot-instances.md)
{% endcontent-ref %}

Check out the full list of resource types and corresponding prices:

{% content-ref url="../organization/organization-settings/billing-information.md" %}
[billing-information.md](../organization/organization-settings/billing-information.md)
{% endcontent-ref %}
{% endtab %}

{% tab title="Custom Cluster" %}
Your custom cluster can be either on-premise or on-cloud. For on-premise clusters, you can specify the processor type and resource requirements. The experiment will be assigned automatically to an available node based on the input resource requirements.&#x20;

![](<../../.gitbook/assets/image (116).png>)
{% endtab %}
{% endtabs %}

### Distribution Mode (Optional) <a href="#image" id="image"></a>

You have an option to use multi-node distributed training. The default option is single-node training.&#x20;

![](<../../.gitbook/assets/image (209).png>)

### Image (Required) <a href="#image" id="image"></a>

Select the Docker image that the experiment container will use. You can either use a managed image provided by VESSL or your own custom image.&#x20;

{% tabs %}
{% tab title="Managed Image" %}
Managed images are pre-pulled images provided by VESSL. You can find the available image tags at VESSL's [Amazon ECR Public Gallery](https://gallery.ecr.aws/vessl/kernels)_._&#x20;

![](<../../.gitbook/assets/image (119).png>)
{% endtab %}

{% tab title="Custom Image" %}
You can pull your own custom images from either [Docker Hub](https://hub.docker.com) or [Amazon ECR](https://aws.amazon.com/ecr/).&#x20;

#### Public Images

To pull images from the public Docker registry, simply pass the image URL. The example below demonstrates pulling the official TensorFlow development GPU image from Docker Hub.&#x20;

![](<../../.gitbook/assets/image (208).png>)

#### Private Images

To pull images from the private Docker registry, you should first integrate your credentials in organization settings.

{% content-ref url="../organization/organization-settings/add-integrations.md" %}
[add-integrations.md](../organization/organization-settings/add-integrations.md)
{% endcontent-ref %}

Then, check the private image checkbox, fill in the image URL, and select the credential.

![](<../../.gitbook/assets/image (161).png>)
{% endtab %}
{% endtabs %}

### Start Command (Required) <a href="#start-command" id="start-command"></a>

Specify the start command in the experiment container. Write a running script with command-line arguments just as you are using a terminal. You can put multiple commands by using the `&&` command or a new line separation.&#x20;

![](<../../.gitbook/assets/image (127).png>)

### Volume (Optional)

You can mount the project, dataset, and files to the experiment container.

![](<../../.gitbook/assets/image (221).png>)

Learn more about volume mount on the following page:

{% content-ref url="../commons/volume-mount.md" %}
[volume-mount.md](../commons/volume-mount.md)
{% endcontent-ref %}

### Hyperparameters

You can set hyperparameters as key-value pairs. The given hyperparameters are automatically added to the container as environment variables with the given key and value. A typical experiment will include hyperparameters like `learning_rate` and `optimizer`.&#x20;

![](<../../.gitbook/assets/image (174).png>)

You can also use them at runtime by appending them to the start command as follows.

```bash
python main.py  \
  --learning-rate $learning_rate  \
  --optimizer $optimizer
```

### Termination Protection&#x20;

Checking the termination protection option puts experiments in idle once it completes running, so you to access the container of a finished experiment.&#x20;

![](<../../.gitbook/assets/image (215).png>)
