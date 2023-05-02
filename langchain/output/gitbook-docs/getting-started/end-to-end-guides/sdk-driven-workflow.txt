# SDK-driven Workflow

Building a machine learning model requires multiple processes. Those steps are slightly different depending on which task you want to accomplish. However, in this document, we will abstract them into the following three steps.

* [Experiment](../../user-guide/experiment/): Build a baseline machine learning model
* [Sweep](../../user-guide/sweep/): Optimize hyperparameters
* [Model registry](../../user-guide/model-registry/): Update the best model

The tutorial below is the process of creating a model for image classification with the MNIST dataset using the [VESSL Client SDK](../../api-reference/what-is-the-vessl-cli-sdk.md).

### Requirements

{% hint style="info" %}
If you haven't created an **organization** or a **project** on VESSL, please go [end-to-end programmatic workflows](./) and follow the instructions.
{% endhint %}

* [Organization](../../user-guide/organization/)
* [Project](../../user-guide/project/)
* [VESSL Client](../../api-reference/what-is-the-vessl-cli-sdk.md) (>= 0.1.34)

{% hint style="warning" %}
Don't forget to replace all **uppercase letters** below with your object names.
{% endhint %}

### Experiment: Build a baseline model

#### 1. Configure VESSL Client with the organization name and the project name

The first thing you need to do is to execute [`vessl.configure()`](broken-reference) which will configure the client **** with the default **organization** and **project** we have created earlier. When a configuration is completed, there is no need to provide project or organization options to run commands.

```python
import vessl

organization_name = "YOUR_ORGANIZATION_NAME"
project_name = "YOUR_PROJECT_NAME"
vessl.configure(
    organization_name=organization_name, 
    project_name=project_name
)
```

{% hint style="info" %}
You could change a default organization and/or project  by calling [`vessl.configure()`](broken-reference) anytime.
{% endhint %}

#### 2. Create a dataset

To create a [dataset](../../user-guide/dataset/) on VESSL, run [`vessl.create_dataset()`](broken-reference) . Let's create a VESSL dataset from the public S3 dataset. Here is the public bucket URL that VESSL provides.

* Public MNIST dataset path: `s3://savvihub-public-apne2/mnist`

```python
dataset = vessl.create_dataset(
  dataset_name="YOUR_DATASET_NAME", 
  is_public=True,
  external_path="s3://savvihub-public-apne2/mnist"
)
```

#### 3. Create a machine learning experiment&#x20;

You can use [`vessl.create_experiment()`](broken-reference) to create a machine learning [experiment](../../user-guide/experiment/). Let's run an experiment by selecting one of the [clusters](../../user-guide/organization/organization-settings/configure-clusters.md) connected to the organization and deciding how many [resources](../../user-guide/organization/organization-settings/billing-information.md) to use. For this experiment, we will use the `aws-apne2-prod1` cluster managed by VESSL, and `v1.cpu-4.mem-13` that allocates 4 cores of CPU and 13GB of memory. Next, specify the [image URL](https://gallery.ecr.aws/vessl/kernels) and [start command](https://github.com/vessl-ai/examples/tree/main/mnist) to be used in the experiment container, and mount the dataset you made earlier. For more information on dataset mounting, please visit the [volume mount](../../user-guide/commons/volume-mount.md) page.

* GitHub repository of VESSL examples: [vessl-ai/examples](https://github.com/vessl-ai/examples)

```python
github_repo = "https://github.com/vessl-ai/examples.git"
experiment = vessl.create_experiment(
    cluster_name="aws-apne2-prod1",
    kernel_resource_spec_name="v1.cpu-4.mem-13",
    kernel_image_url="public.ecr.aws/vessl/kernels:py36.full-cpu",
    dataset_mounts=[f"{/input/:{dataset.name}}"]
    start_command=f"git clone {github_repo} && pip install -r examples/mnist/keras/requirements.txt && python examples/mnist/keras/main.py --save-model --save-image",
)
```

Noted that you can mount a GitHub **repository** to [project](../../user-guide/project/) so that you don't have to add a `git clone` command in every experiment. Likewise, **datasets** can be registered in the project in advance, and the dataset will be automatically mounted without having to give the `dataset_mounts` option when creating an experiment. For more information about those features, please refer to the [project repository & project dataset](../../user-guide/project/project-repository-and-project-dataset.md) page.

#### 4. View experiment results

For the image classification task, the experiment may take some minutes to finish. After confirming that the experiment you made has been completed on VESSL Web Console, the result can be viewed using [`vessl.read_experiment()`](broken-reference).

```python
experiment = vessl.read_experiment(
    experiment_name_or_number=experiment.name
)
```

The metrics summary is in the form of a Python dictionary, and you can check the latest metric values `metrics_summary.latest` as follows.

```python
experiment.metrics_summary.latest["accuracy"].value
```

#### 5. Create a model

In VESSL, you can create a [model from the outputs of the experiment](../../user-guide/model-registry/creating-a-model.md#creating-a-model-from-experiment). Before creating a model, you need to create a model repository. To create one run `vessl.create_model_repository()` with the repository name.

```bash
model_repository = vessl.create_model_repository(
    name="YOUR_MODEL_REPOSITORY_NAME",
)
```

Then, run `vessl.create_model()`with few options including the model repository name that you have just made.&#x20;

```python
model = vessl.create_model(
  repository_name=model_repository.name, 
  experiment_id=experiment.id,
  model_name="v0.0.1",
)
```

### Sweep: Optimize hyperparameters

So far, we have run one machine learning [experiment](../../user-guide/experiment/) and made the results a [model](../../user-guide/model-registry/). Next, we will use the [sweep](../../user-guide/sweep/) to find the optimal hyperparameter value.

First, configure `sweep_objective` with the target **metric name** and target **value**. Note that the metric must be a value logged using [`vessl.log()`](../../api-reference/python-sdk/vessl.log/).

```python
sweep_objective = vessl.SweepObjective(
    type="maximize",        # maximize, minimize
    goal="0.99",            # a target value of metric
    metric="val_accuracy",  # the name of target metric 
)
```

Next, define the search space of `parameters`. As shown below, the `optimizer` is a **categorical** type and the option values are listed as an **array**. On the other hand, `batch_size` is an **int** value and the search space is set using **max**, **min**, and **step**. For more information on sweep see [creating a sweep](../../user-guide/sweep/creating-a-sweep.md).

```python
parameters = [
  vessl.SweepParameter(
    name="optimizer", 
    type="categorical",  # int, double, categorical
    range=vessl.SweepParameterRange(
      list=["adam", "sgd", "adadelta"]
    )
  ), 
  vessl.SweepParameter(
    name="batch_size",
    type="int",  # int, double, categorical
    range=vessl.SweepParameterRange(
      max="256",
      min="64",
      step="8",
    )
  )
]
```

Initiate hyperparameter searching using [`vessl.create_sweep()`](broken-reference). If you look at the options below, you can see that the cluster, resource spec, image, and start command are set the same as in the previous experiment.

```python
sweep = vessl.create_sweep(
    objective=sweep_objective,
    max_experiment_count=4,
    parallel_experiment_count=2,
    max_failed_experiment_count=2,
    algorithm="random",  # grid, random, bayesian 
    parameters=parameters,
    dataset_mounts = [f"/input/:{dataset.name}"],
    cluster_name=experiment.kernel_cluster.name,                     # same as the experiment  
    kernel_resource_spec_name=experiment.kernel_resource_spec.name,  # same as the experiment
    kernel_image_url=experiment.kernel_image.image_url,              # same as the experiment
    start_command=experiment.start_command,                          # same as the experiment
)
```

### Model Registry: Update the best model

Now that we have run several experiments using [sweep](../../user-guide/sweep/), let's find the **best** **experiment** using [`vessl.get_best_sweep_experiment()`](broken-reference) which returns the experiment information that recorded the **best** **value** among the metric values set in `sweep_objective`. In this example, since we have set to find the **maximum** value, the experiment with the **highest** **validation** **accuracy** will return.

```python
best_experiment = vessl.get_best_sweep_experiment(sweep_name=sweep.name)
```

Let's create a `v0.0.2` model with [`vessl.create_model()`](broken-reference) from the output of `best_experiment`.

```python
best_experiment = vessl.read_experiment(experiment_name_or_number=experiment.name)
model = vessl.create_model(
  repository_name="YOUR_MODEL_REPOSITORY_NAME", 
  experiment_id=best_experiment.id,
  model_name="v0.0.2",
)
```

If you want to view the performance of your model, you can check the latest metric value using `vessl.read_model()`.

```python
vessl.read_model(
    repository_name="YOUR_MODEL_REPOSITORY_NAME",
    model_number="YOUR_MODEL_NUMBER",
)
```

Congratulation! We have looked at the overall workflow of using the **VESSL Client SDK**. The same process as this can be done with the [VESSL Client CLI](cli-driven-workflow.md) or through the [VESSL Web Console](../quickstart.md). Now, solve challenging machine learning tasks out there with your own code and dataset with VESSL.
