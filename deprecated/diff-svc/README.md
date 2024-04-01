# Sing songs with your voice! 
This repository explains how you can train your own diff-svc (singing voice conversion) model in end-to-end manner. Voice files are all you need! 

## Diff-SVC
Diff-SVC is a diffusion probabilistic model for singing voice conversion. The code is adopted from [here](https://github.com/prophesier/diff-svc) and manipulated.

Specifically, we simplify the settings and integrated VESSL codes in order to build the end-to-end pipeline. Through `vessl.log()` functions, you will be able to manage all your checkpoints and converted files via your vessl workspace.

<p align="center">
<img src="tmp/diff-svc.png"  width="400" height="300">
</p>
By training Diff-SVC model with a series of your acapella singings, the Diff-SVC model is able to convert any acapella song using your voice. You won't have to worry if you are not familiar with Pytorch. We will walk you through an end-to-end pipeline for training the Diff-SVC model. You will only need to prepare wav files that will be used for training (your voice), and the target song you want to sing.

## End-To-End pipeline
### Training

This project example experiment uses the `CSD-example` dataset we manipulated from open source [CSD dataset](https://zenodo.org/record/4785016). The dataset contains 40 songs for children sung by one Korean female singer. If you want to want to use your own voice for training, follow the below steps. 

1. Generate a new VESSL experiment. (Or, you can click the prebuilt experiment and reproduce it.)
2. Go to `Volume Mount` section. 
3. Press `Add File`
4. Upload your voices in .wav format. Set mount path to `/voice`.

### Start Command

```
cd examples/diff-svc && bash setup.sh && bash train.sh vessl-org project-name exp-name num-gpu max-epoch
```

`vessl-org` : Name of your vessl organization

`project-name` : Name of your vessl project

`exp-name` : Name of your experiment (name of the singer)

`num-gpu` : Number of the GPU. (Number of V100 machines)

`max-epoch` : Number of epochs you will train your model.

Caveat: All checkpoints will be uploaded after all the experiments have been closed. Please make sure you do not terminate the experiment while running!

### Inference pipeline

For inference, follow below steps to upload inference .wav files.

1. From your previous training, download the checkpoint you want to inference with.
2. Create a new experiment using the same setting, go to `Volume Mount` section. 
3. Press `Add File`
4. Upload all wav files that you want to translate. Mount path should be `/infer`.

### Start Command

```
cd examples/diff-svc && bash infer.sh vessl-org project-name exp-name
```

`vessl-org` : Name of your vessl organization

`project-name` : Name of your vessl project

`exp-name` : Name of your experiment (name of the singer)

After all your inference has been finished, you will able to see / download converted .wav files from the experiment page.
