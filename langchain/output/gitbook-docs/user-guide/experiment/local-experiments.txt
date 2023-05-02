# Local Experiments

On VESSL, you can also monitor experiments that you have run locally. This can easily be done by adding a few lines to your code.

Start monitoring your experiment using [`vessl.init`](../../api-reference/python-sdk/vessl.init.md) . This will launch a new experiment under your project. You can view the experiment output under [**LOGS**](experiment-results.md#logs), just like you would in a VESSL-managed experiment. Your local environment's system metrics are also monitored and can be viewed under [**PLOTS**](experiment-results.md#plots).

![System Metrics](<../../.gitbook/assets/Screen Shot 2021-11-04 at 3.23.32 PM.png>)



In a VESSL-managed experiment, files under the output volume are saved by default. In a local experiment, you can use [`vessl.upload`](../../api-reference/python-sdk/vessl.upload.md) to upload any output files. You can view these files under [**FILES**](experiment-results.md#files).

By default, VESSL will stop monitoring your local experiment when your program exits. If you wish to stop it manually, you can use [`vessl.finish`](../../api-reference/python-sdk/vessl.finish.md).
