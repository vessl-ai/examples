# Project Overview

**Project Overview** provides a bird's-eye view of the progress of your machine learning projects. On the project overview dashboard, you can manage and track key information about your project:

* **Key Metrics**: Keep track of essential evaluation metrics of your experiments such as accuracy, loss, and MSE.
* **Sample Media**: Log images, audio, and other media from your experiment and explore your model's prediction results to compare your experiments visually.
* **Starred Experiments**: Star and keep track of meaningful experiments.&#x20;
* **Project Notes**: Make a note of important information about your project and share it with your teammates – similar to README.md of Git codebase.&#x20;

![](<../../.gitbook/assets/image (155).png>)

## Key Metrics

VESSL AI automatically marks metrics of best-performing experiments as key metrics. You can also manually bookmark key metrics and keep track of your model's meaningful evaluation metrics.&#x20;

To add or remove Key Metrics

1. Click the settings icon(![](<../../.gitbook/assets/image (122).png>)) on top of the Key Metrics card.

![](<../../.gitbook/assets/image (94).png>)

2\. Select **up to 4 metrics** and choose whether your goal is to minimize or maximize the target value.&#x20;

* If you select **Minimize**, an experiment with the smallest target value will be updated to the key metric charts.
* If you select **Maximize**, an experiment with the greatest target value will be updated to the key metric chart.&#x20;

&#x20;![](<../../.gitbook/assets/image (113).png>)

## Sample Media

You can log images or audio clips generated from your experiment to explore your model's prediction results and make visual (or auditory) comparisons.&#x20;

{% hint style="info" %}
For more information about logging media during your experiment, refer to [`vessl.log`](../../api-reference/python-sdk/vessl.log/) in our Python SDK.&#x20;
{% endhint %}

To see the media file, select an experiment and specify the media type using the dropdown menu on the upper right corner of Sample Media card.&#x20;

![](<../../.gitbook/assets/image (137).png>)

## Starred Experiment

You can mark important experiments as **Starred Experiments** to keep track of meaningful achievements in the project. Starred Experiments displayed with the tags and key metrics.

![](<../../.gitbook/assets/image (105).png>)

To star or unstar experiments

1. Go to the experiment tracking dashboard
2. Select experiments
3. Click 'Star' or 'Unstar' on the dropdown menu.&#x20;

![](<../../.gitbook/assets/image (239).png>)

You can also star or unstar experiments on the experiment summary page.&#x20;

&#x20;![](<../../.gitbook/assets/image (138).png>)



## Project Notes

**Project Notes** is a place for noting and sharing important information about the project together with your team. It works like README.md of your Git codebase.

![](<../../.gitbook/assets/image (187).png>)

To modify project note, click the settings icon(![](<../../.gitbook/assets/image (122).png>)) on top of Project Notes card. You will be given a markdown editor to update your notes.&#x20;

![](<../../.gitbook/assets/image (180).png>)
