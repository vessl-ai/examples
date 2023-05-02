---
description: Run your first experiment on VESSL
---

# Quickstart

{% embed url="https://www.youtube.com/watch?v=9ak-qtpXOBw" %}

### 1. Sign up and create a Project

To run your first experiment on VESSL AI, first [sign up](https://vessl.ai) for a free account and add an organization. **Organization** is a shared working environment where you can find team assets such as datasets, models, and experiments.&#x20;

![](<../.gitbook/assets/Docs\_Quickstart.001 (7).png>)

While we are on the web console, let’s also add a project called "mnist". As you will see later, **Project** serves as a central repository equipped with a dashboard and visualizations for all of your experiments.

![](<../.gitbook/assets/Docs\_Quickstart.002 (3).png>)

### 2. Install VESSL AI Client

VESSL AI comes with a powerful **CLI** and Python SDK useful for managing ML assets and workflow. Install VESSL AI Client on your local machine using pip install.&#x20;

```
pip install vessl
```

In this guide, we will be using an [example](https://github.com/vessl-ai/examples) from our GitHub. Git clone the repository.

```
git clone https://github.com/vessl-ai/examples
cd examples
```

Let's access the project we created using your first VESSL AI CLI command. The command will guide you to a page that grants CLI access and [configure](https://docs.vessl.ai/api-reference/cli/getting-started) your default organization and project.&#x20;

```
vessl configure \
  --organization quickstart \
  --project mnist
```

![](<../.gitbook/assets/Docs\_Quickstart.001 (1).jpeg>)

### 3. Run an Experiment

Now that we have specified the project and obtained CLI access, you will run your first [experiment](quickstart.md#1.-sign-up-and-create-a-project) on VESSL AI. On a local machine, this is as simple as running a python script.&#x20;

```bash
# Install requirements and run VESSL experiments from local machine
pip install -r mnist/keras/requirements.txt && python mnist/keras/main.py --output-path=output --checkpoint-path=output/checkpoint --save-model --save-image
```

![](<../.gitbook/assets/Docs\_Quickstart.001 (2).png>)

You can also run experiments using VESSL AI's managed clusters by using the [`vessl run`](../api-reference/cli/vessl-run.md) command. The command will upload your current directory and run command on the cluster  asynchronously. You can use [`vessl experiment create`](../api-reference/cli/vessl-experiment.md#create-an-experiment) command instead of `vessl run` to specify detailed options (e.g. volume mounts) in one line.

{% tabs %}
{% tab title="run" %}
```bash
vessl run "pip install -r mnist/keras/requirements.txt && python mnist/keras/main.py --save-model --save-image"
```
{% endtab %}

{% tab title="experiment create (inquiry)" %}
```bash
$ vessl experiment create --upload-local-file .:/home/vessl/local

[?] Cluster: aws-apne2-prod1
 > aws-apne2-prod1

[?] Resource: v1.cpu-0.mem-1
 > v1.cpu-0.mem-1
   v1.cpu-2.mem-6
   v1.cpu-2.mem-6.spot
   v1.cpu-4.mem-13
   v1.cpu-4.mem-13.spot

[?] Image URL: public.ecr.aws/vessl/kernels:py36.full-cpu
 > public.ecr.aws/vessl/kernels:py36.full-cpu
   public.ecr.aws/vessl/kernels:py37.full-cpu
   public.ecr.aws/vessl/kernels:py36.full-cpu.jupyter
   public.ecr.aws/vessl/kernels:py37.full-cpu.jupyter
   tensorflow/tensorflow:1.14.0-py3
   tensorflow/tensorflow:1.15.5-py3
   tensorflow/tensorflow:2.0.4-py3
   tensorflow/tensorflow:2.2.1-py3
   
[?] Command: cd local && pip install -r mnist/keras/requirements.txt && python mnist/keras/main.py --save-model --save-image
```

{% hint style="info" %}
You should specify `--upload-local-file` option to upload your current directory. If you want to link a github repo instead of upload files from local, see this.
{% endhint %}
{% endtab %}

{% tab title="experiment create (one line)" %}
```bash
vessl experiment create \
  --cluster aws-apne2-prod1 \
  --resource v1.cpu-0.mem-1 \
  --image-url public.ecr.aws/vessl/kernels:py36.full-cpu \
  --upload-local-file .:/home/vessl/local
  --command 'cd local && pip install -r mnist/keras/requirements.txt && python mnist/keras/main.py --save-model --save-image'
```

{% hint style="info" %}
You should specify `--upload-local-file` option to upload your current directory. If you want to link a github repo instead of upload files from local, see this.
{% endhint %}
{% endtab %}
{% endtabs %}

Once the command completes, you will be given a link to **Experiments**. The experiment page stores logs, visualizations, and files specific to the experiment.&#x20;

![](../.gitbook/assets/Docs\_Quickstart.002.jpeg)

This metrics and images of the experiment was made possible by calling the [`init()`](https://docs.vessl.ai/api-reference/python-sdk/vessl.init) and [`log()`](quickstart.md#1.-sign-up-and-create-a-project) function from our **Python SDK**, which you can use in your code by simply importing the library as shown in the example [code](https://github.com/savvihub/examples/blob/vssl-2332/vessl-intro.ipynb).&#x20;

{% code title="" %}
```python
import vessl

# Initialize new experiment via VESSL SDK 
vessl.init(organization="quickstart", project="mnist")
```
{% endcode %}

```python
# Train function and log metrics to VESSL

def train(model, device, train_loader, optimizer, epoch, start_epoch):
    model.train()
    loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        ...

    # Logging loss metrics to VESSL
    vessl.log(
        step=epoch + start_epoch + 1,
        payload={'loss': loss.item()}
    )
```

### 4. Track and visualize experiments

When you click the project name on the navigation bar, you will be guided back to the project page. Under each tab, you can explore VESSL's main features:

* **Experiments** – unified dashboard for tracking experiments
* **Tracking** – visualization of model performance and system metrics
* **Sweeps** – scalable hyperparameter optimization
* **Models** – a repository for versioned models

![](../.gitbook/assets/Docs\_Quickstart.005.png)

![](<../.gitbook/assets/Docs\_Quickstart.006 (1).png>)

### 5. Develop state-of-the-art models on VESSL AI

Let's try building a model with the resources and datasets of your choice. Under **Datasets**, you can mount and manage datasets from local or cloud storage.&#x20;

![](../.gitbook/assets/Docs\_Quickstart.007.png)

Let's move over to **Workspaces** where you can configure a custom environment for Jupyter Notebooks with SSH. You can use either VESSL AI's managed cluster with spot instance support or your own custom clusters.

![](../.gitbook/assets/Docs\_Quickstart.008.png)

Launch a **Juypter Notebook**. Here, you will find an example Notebook which introduces how you can integrate local experiments with VESSL AI to empower your research workflow.

![](../.gitbook/assets/Docs\_Quickstart.009.png)

### Next Step

{% embed url="https://www.youtube.com/watch?v=En4y7kVHkGw" %}

Now that you are familiar with the overall workflow of VESSL AI, explore additional features available on our platform and start building!

* Use our [**Sweep**](https://docs.vessl.ai/user-guide/sweep) to automate model tuning.&#x20;
* Use [**distributed experiment**](quickstart.md#1.-sign-up-and-create-a-project) to take full advantage of your GPUs.&#x20;
* Explore [**organization settings**](quickstart.md#1.-sign-up-and-create-a-project) to set up and manage on-cloud or on-premise clusters.&#x20;
