# Creating a Workspace

## How to create a workspace

To **create** a **workspace**, you need to select few options including workspace name, resource, image and few advanced settings.

![](<../../.gitbook/assets/image (162).png>)

{% hint style="info" %}
Put the asterisk mark(\*) on the required option
{% endhint %}

### Workspace Name\*

Once you start to create a new workspace, the default workspace name will be randomly generated. Specify a good name to remember.

### Cluster\*

You can choose a cluster managed by VESSL or a custom cluster registered by yourself. (See [configure organization cluster](../organization/organization-settings/configure-clusters.md).) The managed cluster always on the cloud vendor server, whereas the custom cluster could be either on the cloud server or on the on-premise server.

### Resource\*

Choose the type of resource that the container will use. Select the resource among the dropdown option or specify the requirements manually.

### Image\*

You can choose the Docker image that the workspace container will use. There are two types of images: the **Managed Image** and the **Custom Image**. Select the Docker image type that you want to run on the workspace container.

{% tabs %}
{% tab title="Managed Image" %}
For the **Managed Image**, you can simply select such an option, then the image managed by VESSL will be used in default. You can run Jupyter services on the managed image.



![](<../../.gitbook/assets/image (229).png>)
{% endtab %}

{% tab title="Custom Image" %}
You can use any docker images from Docker Hub or [_AWS ECR_](https://aws.amazon.com/ecr/). To run a workspace with custom images, your custom images have to satisfy below requirements.

* Jupyterlab
  * VESSL runs Jupyterlab and expose port `8888`. Jupyter should be pre-installed in the container image.
* sshd
  * VESSL runs sshd and expose port `22` as NodePort. sshd package should be pre-installed in the container image.
* PVC mountable at `/home/vessl`
  * VESSL mounts a PVC at `/home/vessl` to keep state across Pod restarts.

For more information about building custom images, you can refer to [this guide](building-custom-images.md).

#### Public image

To pull images from the public Docker registry, you can simply pass the image URL as the below example.

![](<../../.gitbook/assets/스크린샷 2022-01-18 오후 9.53.50.png>)

#### Private image

To pull images from the private Docker registry or the private AWS ECR, you should [_integrate your credentials in organization settings first_](broken-reference). Then check the private image checkbox and select the credentials you have just integrated. Below is an example of a private image from the AWS ECR.

![](<../../.gitbook/assets/스크린샷 2022-01-18 오후 9.53.03.png>)
{% endtab %}
{% endtabs %}

{% hint style="warning" %}
Debian based images are compatible.
{% endhint %}

### Advanced Setting

#### Max Runtime

Specify the max runtime (default: 24 hours) for this workspace. After max runtime, workspace will be automatically stopped.

(For Enterprise Plan) Organization admin can limit the max runtime that users can input.

![](<../../.gitbook/assets/image (168).png>)

#### Disk

You can specify the **disk** **size** (default: 100GB) to use in your container. This will be the request storage size of your PVC. Disk size cannot be changed once the workspace is created.

{% hint style="warning" %}
Disk size can be ignored in a custom cluster due to limitation of kubernetes. ([official docs](https://kubernetes.io/docs/concepts/storage/volumes/#resources))
{% endhint %}

#### Port

You can customize **port** settings. By default, 8888 (jupyter) and 22 (ssh) are exposed.

#### Init script

**Init script** is a shell script that runs every time the workspace starts. Because `/home/vessl` is the only persistent directory, packages you installed outside the home directory may reset on stop & start. In this case, you can fill **init script** with install commands such as `apt-get update && apt-get install ripgrep -y`.

#### Volume (custom cluster only)

You can attach your NFS/Host machine volume.

![](<../../.gitbook/assets/image (178).png>)

