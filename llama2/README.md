# Finetuning Llama2 and Serving with VESSL

## Overview
This guide provides steps for fine-tuning the Llama2 model and deploying it as a service using VESSL. Follow the prerequisites to set up your environment, then proceed with fine-tuning, registering the model, testing inference locally, and finally deploying the service.

## Prerequisite
Ensure you have VESSL installed and configured in your environment. Run the following commands in your terminal:

```bash
pip install vessl
vessl configure
```

### Finetuning Llama2
To fine-tune the Llama2 model, use the provided YAML configuration file with VESSL. Execute the command below:

```bash
vessl run create -f finetuning.yaml
```

### Register the fine-tuned model to VESSL with Servo
After the fine-tuning process is complete, you need to register the model in VESSL using the Servo library. Replace {MODEL_REPOSITORY_NAME} with the name of repository you want to register and {RUN_ID} with the ID obtained from your fine-tuning run. Use the following command to register the model:

```bash
vessl model register {MODEL_REPOSITORY_NAME} --run-id={RUN_ID}
```

### Test inference in local
To test the inference of your model locally, replace {MODEL_ID} with the ID of your registered model and run the following command:
```bash
vessl model serve {MODEL_ID} -i prompt="Plan a trip to explore Madagascar for three days."
```

### Deploy a service
Finally, to deploy your fine-tuned model as a service, replace {MODEL_ID} with the ID of your registered model. Use the serve.yaml configuration file with the command below:
`{MODEL_ID}`: Replace with id from registered model.
```bash
vessl serve create -f serve.yaml
```
