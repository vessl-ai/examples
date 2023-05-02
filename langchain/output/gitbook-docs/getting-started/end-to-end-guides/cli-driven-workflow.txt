# CLI-driven Workflow

In this document, we will cover the common tasks involved in building a machine learning model and guide how you can mostly our [CLI commands](../../api-reference/what-is-the-vessl-cli-sdk.md) to accomplish these tasks:

* Build a baseline machine learning model with [Experiments](../../user-guide/experiment/).
* Optimize hyperparameters using [Sweep](../../user-guide/sweep/).
* Update and store models on [Model registry](../../user-guide/model-registry/).&#x20;

Here, we will use the MNIST database to create an image classification model. All our CLI commands are one-liners but you can also select from command option prompts.

### Requirements

To follow this guide, you should first have the following setup.&#x20;

* [Organization](../../user-guide/organization/) — a dedicated organization for you or your team
* [Project](../../user-guide/project/) — a space for your machine learning model and mounted datasets
* [VESSL Client](../../api-reference/what-is-the-vessl-cli-sdk.md) — Python SDK and CLI to manage ML workflows and resources on VESSL

If you have not created an Organization or a Project, first follow the instructions on the [end-to-end guides](https://docs.vessl.ai/getting-started/end-to-end-guides).&#x20;

### 1. Experiment — Build a baseline model

#### 1-1. Configure your default organization and project

Let's start by configuring the client with the default organization and project we have created earlier. This is done by executing [`vessl configure`](../../api-reference/cli/getting-started.md#setup-a-vessl-environment).&#x20;

{% tabs %}
{% tab title="one-liner" %}
```bash
vessl configure  \
    --organization "YOUR_ORGANIZATION_NAME"  \
    --project "YOUR_PROJECT_NAME"
```
{% endtab %}

{% tab title="prompter" %}
```bash
vessl configure

Please grant CLI access from the URL below.
https://vessl.ai/cli/grant-access?token=o0dyb21eu8fw

Waiting...

[?] Default organization: YOUR_ORGANIZATION_NAME
 > YOUR_ORGANIZATION_NAME

[?] Default project: YOUR_PROJECT_NAME
 > YOUR_PROJECT_NAME
```
{% endtab %}
{% endtabs %}

![](<../../.gitbook/assets/carbon (16).png>)

You also can check and re-configure your default organization and project by specifying options after `vessl configure`.&#x20;

```bash
vessl configure list # check your current default organization and project
vessl configure --renew-token # renew access token and default account
vessl configure organization # change your default organization
vessl configure project # change your default project
```

#### 1-2. Create and mount a dataset

To create a [dataset](../../user-guide/dataset/) on VESSL, run [`vessl dataset create`](../../api-reference/cli/vessl-dataset.md#create-a-dataset). Let's create a dataset from the public AWS S3 dataset we have prepared: `s3://savvihub-public-apne2/mnist`. You can check that your dataset was created successfully by clicking the output link.&#x20;

{% tabs %}
{% tab title="one-liner" %}
```bash
vessl dataset create "vessl-mnist" \
  --is-public --external-path "s3://savvihub-public-apne2/mnist"
```
{% endtab %}

{% tab title="prompter" %}
```bash
vessl dataset create --is-public -e "s3://savvihub-public-apne2/mnist"

Organization: YOUR_ORGANIZATION_NAME
Dataset name: YOUR_DATASET_NAME
```


{% endtab %}
{% endtabs %}

![](<../../.gitbook/assets/carbon (2).png>)

#### 1-3. Create a machine learning experiment&#x20;

To create an experiment, use [`vessl experiment create`](../../api-reference/cli/vessl-experiment.md#create-an-experiment). Let's run an experiment using VESSL's managed clusters. First, specify the cluster and resources options. Then, specify the image URL — in this case, we are pulling a Docker image from [VESSL's Amazon ECR Public Gallery](https://gallery.ecr.aws/vessl/kernels). Finally, we will specify the start command which will be executed in the experiment container — here we will use the MNIST Keras example from our [GitHub repository](https://github.com/vessl-ai/examples/tree/main/mnist).&#x20;

{% tabs %}
{% tab title="one-liner" %}
```bash
vessl experiment create \
  --cluster "aws-apne2-prod1" \
  --resource "v1.cpu-4.mem-13" \
  --image-url "public.ecr.aws/vessl/kernels:py36.full-cpu" \
  --dataset "/input:vessl-mnist"\
  --command "git clone https://github.com/vessl-ai/examples.git && pip install -r examples/mnist/keras/requirements.txt && python examples/mnist/keras/main.py --save-model --save-image"
```
{% endtab %}

{% tab title="prompter" %}
```
vessl experiment create --dataset /input:YOUR_DATASET_NAME

Organization: YOUR_ORGANIZATION_NAME
Project: YOUR_PROJECT_NAME
[?] Cluster: aws-apne2-prod1
 > aws-apne2-prod1

[?] Resource: v1.cpu-4.mem-13
   v1.cpu-0.mem-1
   v1.cpu-2.mem-6
   v1.cpu-2.mem-6.spot
 > v1.cpu-4.mem-13
   v1.cpu-4.mem-13.spot
   v1.t4-1.mem-13
   v1.t4-1.mem-13.spot
   v1.t4-1.mem-54
   v1.t4-1.mem-54.spot
   v1.t4-4.mem-163
   v1.t4-4.mem-163.spot
   v1.k80-1.mem-52
   v1.k80-1.mem-52.spot

[?] Image URL: public.ecr.aws/vessl/kernels:py36.full-cpu
 > public.ecr.aws/vessl/kernels:py36.full-cpu
   public.ecr.aws/vessl/kernels:py37.full-cpu
   public.ecr.aws/vessl/kernels:py36.full-cpu.jupyter
   public.ecr.aws/vessl/kernels:py37.full-cpu.jupyter
   tensorflow/tensorflow:1.14.0-py3
   tensorflow/tensorflow:1.15.5-py3
   tensorflow/tensorflow:2.0.4-py3
   tensorflow/tensorflow:2.2.1-py3
   tensorflow/tensorflow:2.3.2
   tensorflow/tensorflow:2.4.1
   tensorflow/tensorflow:2.3.0

Start command: git clone https://github.com/vessl-ai/examples.git && pip install -r examples/mnist/keras/requirements.txt && python examples/mnist/keras/main.py --save-model --save-image
```
{% endtab %}
{% endtabs %}

![](<../../.gitbook/assets/carbon (3) (1).png>)

You can also [integrate a GitHub repository](https://docs.vessl.ai/user-guide/organization/organization-settings/add-integrations#github) with your project so you don't have to `git clone` every time you create an experiment. For more information about those features, please refer to our doc's [project repository & project dataset](../../user-guide/project/project-repository-and-project-dataset.md) page.

#### 1-4. View experiment results

The experiment may take a few minutes to complete. You can get the details of the experiment, including its status, using [`vessl experiment read`](../../api-reference/cli/vessl-experiment.md#view-information-on-the-experiment) or by clicking the output link.&#x20;

{% tabs %}
{% tab title="one-liner" %}
```bash
vessl experiment read "YOUR_EXPERIMENT_NAME"
```
{% endtab %}

{% tab title="prompter" %}
```
vessl experiment read 

Organization: YOUR_ORGANIZATION_NAME
Project: YOUR_PROJECT_NAME
[?] Experiment: 1-quasar-bat #1
 > 1-quasar-bat #1
```
{% endtab %}
{% endtabs %}

![](<../../.gitbook/assets/carbon (4).png>)

#### 1-5. Create a model

In VESSL, you can create a [model from a completed experiment](../../user-guide/model-registry/creating-a-model.md#creating-a-model-from-experiment). First, create a model repository by `vessl model-repository create` by specifying the repository name.&#x20;

{% tabs %}
{% tab title="one-liner" %}
```bash
vessl model-repository create "tutorial-mnist"
```
{% endtab %}

{% tab title="prompter" %}
```
vessl model-repository create

Organization: YOUR_ORGANIZATION_NAME
Model repository name: YOUR_MODEL_REPOSITORY_NAME
```
{% endtab %}
{% endtabs %}

![](<../../.gitbook/assets/carbon (5) (1).png>)

Then, let's get a list of experiments in the project and their ID.&#x20;

```bash
vessl experiment list
```

![](<../../.gitbook/assets/carbon (6).png>)

Finally, run [`vessl model create`](../../api-reference/cli/vessl-model.md#create-a-model) with options including the destination repository and experiment ID. Make sure that the option value for `--experiment-id` is an integer, not a string.

{% tabs %}
{% tab title="one-liner" %}
```bash
vessl model create "tutorial-mnist" \
  --model-name "v0.0.1" \
  --source "experiment" \
  --experiment-id YOUR_EXPERIMENT_ID
```
{% endtab %}

{% tab title="prompter" %}
```bash
vessl model create --model-name "v0.0.1"

Organization: YOUR_ORGANIZATION_NAME
Project: YOUR_PROJECT_NAME
[?] Model repository: YOUR_MODEL_REPOSITORY_NAME
 > YOUR_MODEL_REPOSITORY_NAME

[?] Source: From an experiment
 > From an experiment
   From local files

[?] Experiment: 1-quasar-bat #1
 > 1-quasar-bat #1

[?] Paths (Press -> to select and <- to unselect):
   X my_model 0 B
   X my_model/keras_metadata.pb 7.7 KB
   X my_model/saved_model.pb 88.8 KB
   X my_model/variables 0 B
   X my_model/variables/variables.data-00000-of-00001 1.2 MB
 > X my_model/variables/variables.index 1.4 KB
```
{% endtab %}
{% endtabs %}

![](<../../.gitbook/assets/carbon (15).png>)

You can see that the model has been created successfully by specifying the repository name and selecting the model number.&#x20;

```
vessl model read "tutorial-mnist"
```

![](<../../.gitbook/assets/carbon (7).png>)

You can get a list of model repositories and models you have created inside the project by using the following commands.

```
vessl model-repository list # get a list of model repositories
vessl model list # get a list of models 
```

### 2. Sweep — Optimize hyperparameters

So far, we ran a single machine learning [experiment](../../user-guide/experiment/) and saved it as a [model](../../user-guide/model-registry/). Here, we will use a [sweep](../../user-guide/sweep/) to find the optimal hyperparameter value. First copy and paste the following command and while the sweep is running we will explain each options.&#x20;

{% tabs %}
{% tab title="one-liner" %}
```bash
vessl sweep create \
  --objective-type "maximize" \
  --objective-goal "0.99" \
  --objective-metric "val_accuracy" \
  --num-experiments 4 --num-parallel 2 --num-failed 2 \
  --parameter "optimizer categorical list adam sgd adadelta" \
  --parameter "batch_size int space 64 256 8" \
  --algorithm random \
  --cluster "aws-apne2-prod1" \
  --resource "v1.cpu-4.mem-13" \
  --image-url "public.ecr.aws/vessl/kernels:py36.full-cpu" \
  --dataset "/input:vessl-mnist" \
  --command "git clone https://github.com/vessl-ai/examples.git && pip install -r examples/mnist/keras/requirements.txt && python examples/mnist/keras/main.py --save-model --save-image"
```
{% endtab %}

{% tab title="prompter" %}
```
vessl sweep create --dataset "/input:YOUR_DATASET_NAME"

Organization: YOUR_ORGANIZATION_NAME
Project: YOUR_PROJECT_NAME
[?] Objective type: maximize
 > maximize
   minimize

Objective metric: val_accuracy
Objective goal: 0.99
Maximum number of experiments: 4
Number of experiments to be run in parallel: 2
Maximum number of experiments to allow to fail: 2
[?] Sweep algorithm: random
   grid
 > random
   bayesian

Parameter #1 name: optimizer
[?] Parameter #1 type: categorical
 > categorical
   int
   double

[?] Parameter #1 range type: list
   space
 > list

Parameter #1 values (space separated): adam sgd adadelta
Add another parameter (y/n): y
Parameter #2 name: batch_size
[?] Parameter #2 type: int
   categorical
 > int
   double

[?] Parameter #2 range type: space
 > space
   list

Parameter #2 values ([min] [max] [step]): 64 256 8
Add another parameter (y/n): n
[?] Cluster: aws-apne2-prod1
 > aws-apne2-prod1

[?] Resource: v1.cpu-4.mem-13
   v1.cpu-0.mem-1
   v1.cpu-2.mem-6
   v1.cpu-2.mem-6.spot
 > v1.cpu-4.mem-13
   v1.cpu-4.mem-13.spot
   v1.t4-1.mem-13
   v1.t4-1.mem-13.spot
   v1.t4-1.mem-54
   v1.t4-1.mem-54.spot
   v1.t4-4.mem-163
   v1.t4-4.mem-163.spot
   v1.k80-1.mem-52
   v1.k80-1.mem-52.spot

[?] Image URL: public.ecr.aws/vessl/kernels:py36.full-cpu
 > public.ecr.aws/vessl/kernels:py36.full-cpu
   public.ecr.aws/vessl/kernels:py37.full-cpu
   public.ecr.aws/vessl/kernels:py36.full-cpu.jupyter
   public.ecr.aws/vessl/kernels:py37.full-cpu.jupyter
   tensorflow/tensorflow:1.14.0-py3
   tensorflow/tensorflow:1.15.5-py3
   tensorflow/tensorflow:2.0.4-py3
   tensorflow/tensorflow:2.2.1-py3
   tensorflow/tensorflow:2.3.2
   tensorflow/tensorflow:2.4.1
   tensorflow/tensorflow:2.3.0

Start command: git clone https://github.com/vessl-ai/examples.git && pip install -r examples/mnist/keras/requirements.txt && python examples/mnist/keras/main.py --save-model --save-image
```
{% endtab %}
{% endtabs %}

The first part of the command defines the key objective and number of experiments.&#x20;

* `--objective-type` — target object (either to minimize or maximize the metric)
* `--objective-goal` — target metric name as defined and logged using `vessl.log()`
* `--objective-metric` — target metric value
* `--num-experiments` — total number of experiments
* `--num-parallel` — the number of experiments to run in parallel
* `--num-failed` — the number of failed experiments before the sweep terminates

Next, we specified the details of the parameters and which algorithm to use. In this example, the `optimizer` is a **** `categorical` type and the option values are listed as an array. The `batch_size` is an int value and the search `space` is set using max, min, and step.&#x20;

The command is then followed by cluster, resource, image, dataset, and command options, similar to the `vessl experiment create` explained above.&#x20;

You may find it easier to run `vessl sweep create` and specify the options through command prompts. For more information on sweep, refer to our [sweep documentation.](../../user-guide/sweep/creating-a-sweep.md)

![](<../../.gitbook/assets/carbon (9).png>)

### 3. Model Registry: Update and store the best model

Now that we ran multiple experiments using a [sweep](../../user-guide/sweep/), let's find the optimal experiment. [`vessl sweep best-experiment`](../../api-reference/cli/vessl-sweep.md#find-the-best-sweep-experiment) returns the experiment information with the best specified metric value. In this example, the command will return the details of an experiment with the maximum `val_accuracy`.&#x20;

{% tabs %}
{% tab title="one-liner" %}
```bash
vessl sweep best-experiment "grove-scowl"
```
{% endtab %}

{% tab title="prompter" %}
```
vessl sweep best-experiment

Organization: YOUR_ORGANIZATION_NAME
Project: YOUR_PROJECT_NAME
[?] Sweep: YOUR_SWEEP_NAME
 > YOUR_SWEEP_NAME
```
{% endtab %}
{% endtabs %}

![](<../../.gitbook/assets/carbon (12).png>)

Let's create a `v0.0.2` model with [`vessl model create`](../../api-reference/cli/vessl-model.md#create-a-model) from the output of the best sweep experiment. You can get the experiment ID using the [`vessl experiment read`](../../api-reference/cli/vessl-experiment.md#view-information-on-the-experiment) command.

{% tabs %}
{% tab title="one-liner" %}
```bash
vessl model create "tutorial-mnist" \
  --model-name "v0.0.2" \
  --source "experiment" \
  --experiment-id 8589948240
```
{% endtab %}

{% tab title="prompter" %}
```bash
vessl model create --model-name "v0.0.2"

Organization: YOUR_ORGANIZATION_NAME
Project: YOUR_PROJECT_NAME
[?] Model repository: YOUR_MODEL_REPOSITORY_NAME
 > YOUR_MODEL_REPOSITORY_NAME

[?] Source: From an experiment
 > From an experiment
   From local files

[?] Experiment: 1-quasar-bat #1
 > 1-quasar-bat #1

[?] Paths (Press -> to select and <- to unselect):
   X my_model 0 B
   X my_model/keras_metadata.pb 7.7 KB
   X my_model/saved_model.pb 88.8 KB
   X my_model/variables 0 B
   X my_model/variables/variables.data-00000-of-00001 1.2 MB
 > X my_model/variables/variables.index 1.4 KB
```
{% endtab %}
{% endtabs %}

Finally, you can view the performance of your model by using `vessl model read` and specifying the model repository followed by the model number.&#x20;

{% tabs %}
{% tab title="one-liner" %}
```bash
vessl model read "tutorial-mnist" "2"
```
{% endtab %}

{% tab title="prompter" %}
```
vessl model read

Organization: YOUR_ORGANIZATION_NAME
[?] Model repository: YOUR_MODEL_REPOSITORY_NAME
 > YOUR_MODEL_REPOSITORY_NAME

[?] Model: 1
 > 1
   2
```
{% endtab %}
{% endtabs %}

![](<../../.gitbook/assets/carbon (13).png>)

We covered the overall workflow of VESSL solely using the client CLI. We can also repeat the same process using the client SDK or through Web UI. Try this guide with your own code and dataset.&#x20;
