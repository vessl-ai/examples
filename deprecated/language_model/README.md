# Language model
Run language model example on [VESSL](https://vessl.ai):
> Noted that you should add [hyperparameters](../README.md) as arguments to the start command

## PyTorch
### Dataset mount
  1. Create a new dataset with a public S3 bucket directory `s3://vessl-public-apne2/wikitext-2`.
  2. Mount the dataset to `/input` at the experiment create form.
### Start Command
  ```bash
  pip install -r examples/language_model/pytorch/requirements.txt && python examples/language_model/pytorch/main.py
  ```
### Hyperparameters
  ```bash
    model_type # RNN_TANH, RNN_RELU, LSTM, GRU, or Transformer [default: 'LSTM']
    emsize # size of word embeddings [default: 200]
    nhid # number of hidden units per layer [default: 200]
    nlayers # number of layers [default: 2]
    lr # initial learning rate [default: 20]
    clip # gradient clipping [default: 0.25]
    epochs # upper epoch limit [default: 40]
    batch_size # batch size [default: 20]
    dropout # dropout applied to layers (0 = no dropout) [default: 0.2]
    nhead # the number of heads in the transformer model [default: 2]
  ```