#!/bin/bash

echo "\$MODEL_NAME: $MODEL_NAME"

envsubst < axolotl-finetune.yaml
