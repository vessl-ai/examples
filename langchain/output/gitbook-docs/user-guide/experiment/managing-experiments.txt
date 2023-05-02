# Managing Experiments

Under the experiments page, you can view the details of each experiment such as experiment status and logs. Here, you can also terminate or reproduce experiments.&#x20;

![](<../../.gitbook/assets/image (203).png>)

### Experiment Status

| Type          | Description                                                                                                           |
| ------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Pending**   | An experiment is created with a pending status until the experiment node is ready. (VESSL-managed experiment only)    |
| **Running**   | The experiment is running.                                                                                            |
| **Completed** | The experiment has successfully finished (exited in 0).                                                               |
| **Idle**      | The experiment is completed but still approachable due to the termination protection. (VESSL-managed experiment only) |
| **Failed**    | The experiment has unsuccessfully finished.                                                                           |

{% hint style="info" %}
VESSL-managed experiments' status depends on its [Kubernetes pod lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/).
{% endhint %}

{% hint style="success" %}
To track the progress of your running experiment, use [`vessl.progress`](../../api-reference/python-sdk/vessl.progress.md). VESSL will calculate the remaining running time, which you can view by hovering over the status mark.
{% endhint %}

### Experiment Terminal

If you activate the **TERMINAL**, you can SSH access the experiment container through a web terminal. You can directly attach the SSH terminal to the experiment process or open a new experiment shell.&#x20;

#### Attaching to the experiment process

By attaching SSH directly to the experiment process, you can view the same logs displayed on the Web Console under the **LOGS** tab. You can take various commands such as interrupting the process.&#x20;

#### Creating a new shell

Opening a new SSH terminal allows you to navigate the experiment container to see where the datasets or projects are mounted.

### Reproducing Experiments

One of the great features of VESSL is that all the experiments can be reproduced. VESSL keeps track of all experiment configurations including the dataset snapshot and source code version. and allows you to reproduce any experiment with just a single click. You can reproduce experiments either on the Web Console or via VESSL CLI. &#x20;

![](<../../.gitbook/assets/image (153).png>)

### Terminating Experiments

You can stop running the experiment and delete the experiment pod.

### Unpushed Changes

A warning titled **UNPUSHED CHANGES** will appear in the experiment details if you run an experiment through CLI without pushing the local changes to GitHub. To solve this issue, download the `.patch` file containing `git diff` and apply it by running the following commands.&#x20;

```
# Change directory to your project
cd path/to/project

# Checkout your recent commit with SHA
git checkout YOUR_RECENT_COMMIT_SHA

# Apply .patch file to the commit
git apply your_git_diff.patch
```
