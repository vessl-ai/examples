# Creating a Sweep

To create a Sweep you need to specify a few options including objective, parameters, algorithm, and runtime.

### Objective

{% hint style="warning" %}
You should log the objective metric with Python SDK in your code.
{% endhint %}

Specify the objective metric that you want to optimize. You can set either to maximize or minimize your target value.&#x20;

{% tabs %}
{% tab title="Maximize val_accuracy" %}
![](<../../.gitbook/assets/image (86).png>)
{% endtab %}

{% tab title="Minimize val_loss" %}
![](<../../.gitbook/assets/image (167).png>)
{% endtab %}
{% endtabs %}

### Common Parameters

Specify the following parameters with a positive integer:

* **Max experiment count**: The maximum number of experiments to run. A sweep will keep spawning a new experiment until the total number of experiments reaches the count.&#x20;
* **Parallel experiment count**: The number of parallel experiments to run. Sweep runs experiments concurrently up to the number of parallel experiment counts.&#x20;
* **Max failed experiment count**: The number of allowed failed experiments. If the number of failed experiments exceeds this number, the Sweep will no longer spawn new experiments.&#x20;

{% hint style="info" %}
Noted that both **Parallel experiment count** and **Max failed experiment count** should be less than or equal to **Max experiment count**.
{% endhint %}

![](<../../.gitbook/assets/image (92).png>)

### Parameters

#### Algorithm name

* [**Grid search**: ](https://en.wikipedia.org/wiki/Hyperparameter\_optimization#Random\_search)A simple exhaustive searching by all combinations through a specified search space. All search space of parameters should be discrete and bounded. If each of the two parameters has three possible values, the total number of possible experiments is six.
* ****[**Random search**](https://en.wikipedia.org/wiki/Hyperparameter\_optimization#Random\_search):  Randomly selecting the parameter values existing in the search space, and spawn an experiment with those parameters. The search space could be discrete, continuous, or mixed.
* ****[**Bayesian optimization**](https://en.wikipedia.org/wiki/Hyperparameter\_optimization#Bayesian\_optimization): A global optimization method for noisy black-box functions. Bayesian optimization will select the next parameter on the probabilistic model of the function mapping from hyperparameter values to the objective.

#### Search space

{% hint style="warning" %}
Parameters are set as **hyperparameters** of the experiment. [See example code](https://github.com/savvihub/examples/blob/main/mnist/pytorch/main.py#L153-L156) on our GitHub repository.&#x20;
{% endhint %}

* **Name**: The name of the parameter that is applied to the experiment as an environment variable at runtime.
* **Type**: Choose between categorical, int, or double type of the parameter.
* **Range**: You can choose between search space and list options. For a categorical type, only a list option is available.&#x20;
* **Value**: The input form of the value is determined by the range type. For a search space, a continuous space is defined with min, max, and step, and for a list option, a search space is defined with discrete values.

![](<../../.gitbook/assets/image (101).png>)

### Early Stopping (Optional)

You can set early stopping to prevent overfitting on the training dataset. It supports the median algorithm which takes two input values, `min_experiment_required` and `start_step`. VESSL examines the metric value for each step after `start_step` and compares it to the median value of the completed experiment to decide whether to trigger early stopping.&#x20;

![](<../../.gitbook/assets/image (100).png>)

### Runtime

Configuring the runtime option is similar to creating an experiment:

{% content-ref url="../experiment/creating-an-experiment.md" %}
[creating-an-experiment.md](../experiment/creating-an-experiment.md)
{% endcontent-ref %}

You can retrieve the configuration of prior experiments by clicking **Configure from Prior Experiments**.

![](<../../.gitbook/assets/image (132).png>)
