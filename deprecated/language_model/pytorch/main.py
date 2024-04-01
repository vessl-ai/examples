# coding: utf-8
import argparse
import math
import os
import time

import data
import model
import torch
import torch.nn as nn
import torch.onnx
import vessl


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def train(
    model_type,
    model,
    corpus,
    train_data,
    batch_size,
    bptt,
    clip,
    log_interval,
    dry_run,
    epoch,
):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.0
    train_loss = 0.0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if model_type != "Transformer":
        hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if model_type == "Transformer":
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()
        train_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time

            try:
                ppl = math.exp(cur_loss)
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                    "loss {:5.2f} | ppl {:8.2f}".format(
                        epoch,
                        batch,
                        len(train_data) // bptt,
                        lr,
                        elapsed * 1000 / log_interval,
                        cur_loss,
                        ppl,
                    )
                )
            except OverflowError:
                continue

            total_loss = 0
            start_time = time.time()
        if dry_run:
            break

    # Logging metrics to Vessl
    loss = train_loss / (len(train_data) // bptt)

    try:
        ppl = math.exp(loss)
        vessl.log(step=epoch, payload={"loss": loss, "ppl": ppl})
    except OverflowError:
        return


def evaluate(model_type, model, corpus, data_source, bptt):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    ntokens = len(corpus.dictionary)
    if model_type != "Transformer":
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if model_type == "Transformer":
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def save(model, path):
    if not os.path.exists(path):
        print(f" [*] Make directories : {path}")
        os.makedirs(path)
    artifact_path = os.path.join(path, "model.pt")
    with open(artifact_path, "wb") as f:
        torch.save(model, f)
    print(f" [*] Saved model in : {artifact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model"
    )
    parser.add_argument(
        "--input-path", type=str, default="/input", help="location of the data corpus"
    )
    parser.add_argument(
        "--output-path", type=str, default="/output", help="output files path"
    )
    parser.add_argument(
        "--tied", action="store_true", help="tie the word embedding and softmax weights"
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument("--bptt", type=int, default=35, help="sequence length")
    parser.add_argument(
        "--log-interval", type=int, default=200, metavar="N", help="report interval"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="verify the code and the model"
    )
    args = parser.parse_args()

    # hyperparameters
    model_type = str(
        os.environ.get("model_type", "LSTM")
    )  # RNN_TANH, RNN_RELU, LSTM, GRU, or Transformer
    emsize = int(os.environ.get("emsize", 200))  # size of word embeddings
    nhid = int(os.environ.get("nhid", 200))  # number of hidden units per layer
    nlayers = int(os.environ.get("nlayers", 2))  # number of layers
    lr = float(os.environ.get("lr", 20))  # initial learning rate
    clip = float(os.environ.get("clip", 0.25))  # gradient clipping
    epochs = int(os.environ.get("epochs", 40))  # upper epoch limit
    batch_size = int(os.environ.get("batch_size", 20))  # batch size
    dropout = float(
        os.environ.get("dropout", 0.2)
    )  # dropout applied to layers (0 = no dropout)
    nhead = int(
        os.environ.get("nhead", 2)
    )  # the number of heads in the transformer model

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda"
            )

    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Device: {device}")
    print(f"Device count: {torch.cuda.device_count()}")

    # Load data
    corpus = data.Corpus(args.input_path)
    eval_batch_size = 10
    train_data = batchify(corpus.train, batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # build model
    ntokens = len(corpus.dictionary)
    if model_type == "Transformer":
        model = model.TransformerModel(
            ntokens, emsize, nhead, nhid, nlayers, dropout
        ).to(device)
    else:
        model = model.RNNModel(
            model_type, ntokens, emsize, nhid, nlayers, dropout, args.tied
        ).to(device)

    print(f"model: {model}")

    criterion = nn.NLLLoss()

    # Training code
    best_val_loss = None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(
            model_type,
            model,
            corpus,
            train_data,
            batch_size,
            args.bptt,
            clip,
            args.log_interval,
            args.dry_run,
            epoch,
        )
        val_loss = evaluate(model_type, model, corpus, val_data, args.bptt)

        try:
            val_ppl = math.exp(val_loss)
            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f}".format(
                    epoch, (time.time() - epoch_start_time), val_loss, val_ppl
                )
            )
            print("-" * 89)

            # Logging metrics to Vessl
            vessl.log(step=epoch, payload={"val_loss": val_loss, "val_ppl": val_ppl})
        except OverflowError:
            continue

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            save(model, args.output_path)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

    # Load the best saved model.
    with open(os.path.join(args.output_path, "model.pt"), "rb") as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if model_type in ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]:
            model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(model_type, model, corpus, test_data, args.bptt)
    print("=" * 89)
    print(
        "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
            test_loss, math.exp(test_loss)
        )
    )
    print("=" * 89)
