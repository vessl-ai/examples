# vessl run

### Overview

`vessl run` is a simple command for running an experiment in the cluster. You can simple add `vessl run` in front of your command.

![](<../../.gitbook/assets/carbon (1) (1).png>)

This will inquiry information to run on the cluster and create an experiment with

* Upload current directory to `/home/vessl/local`
* Run command in `/home/vessl/local`

Once the command completes, you will be given a link to the experiment **** and logs from experiments.&#x20;

![](<../../.gitbook/assets/carbon (2) (1).png>)

At this stage, you can exit the command with Ctrl+C; This will not terminate the running experiment. To terminate the experiment, click the experiment link and select terminate on the page.



This command is equivalent to

```
vessl experiment create
  --command ${command}
  --upload-local-git-diff false
  --working-dir /home/vessl/local
  --upload-local-file .:/home/vessl/local
  
vessl experiment logs ${experiment-name}
```

&#x20;
