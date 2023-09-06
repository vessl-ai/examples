import argparse
import os
import random

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import vessl

from model import Net, MyRunner

vessl.init()

def load_data(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    raw_data = pd.read_csv(data_path, dtype=np.float32)
    return raw_data


def preprocess(raw_data):
    label = raw_data.label.values
    data = raw_data.loc[:, raw_data.columns != "label"].values
    data = data / 255.0
    data = data.reshape(-1, 28, 28)
    return label, data


def get_dataloader(data, label, batch_size, shuffle):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(data).unsqueeze(1),
        torch.from_numpy(label).type(torch.LongTensor),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_checkpoint(checkpoint_file_path):
    print(f"=> Loading checkpoint '{checkpoint_file_path}' ...")
    if device == "cuda":
        checkpoint = torch.load(checkpoint_file_path)
    else:
        checkpoint = torch.load(
            checkpoint_file_path, map_location=lambda storage, loc: storage
        )
    model.load_state_dict(checkpoint.get("state_dict"))
    print(f"=> Loaded checkpoint (trained for {checkpoint.get('epoch')} epochs)")
    return checkpoint.get("epoch"), checkpoint.get("best_accuracy")


def save_checkpoint(state, is_best, filename):
    if is_best:
        print(f"=> Saving a new best to {filename}")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


def train(model, device, train_loader, optimizer, epoch, start_epoch):
    model.train()
    loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 128 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}".format(
                    epoch + 1,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    # Logging loss metrics to VESSL
    vessl.log(step=epoch + start_epoch + 1, payload={"loss": loss.item()})


def test(model, device, test_loader, save_image):
    model.eval()
    test_loss = 0
    correct = 0
    test_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            random_i = random.randint(0, len(data) - 1)
            test_images.append(
                vessl.Image(
                    data[random_i],
                    caption="Pred: {} Truth: {}".format(
                        pred[random_i].item(), target[random_i]
                    ),
                )
            )

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        "  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(test_loader.dataset), test_accuracy
        )
    )
    vessl.log(payload={"test_loss": test_loss, "test_accuracy": test_accuracy})

    if save_image:
        vessl.log({"Examples": test_images})

    return test_accuracy


def save(model, path):
    if not os.path.exists(path):
        print(f" [*] Make directories : {path}")
        os.makedirs(path)
    artifact_path = os.path.join(path, "model.pt")
    torch.save(model.state_dict(), artifact_path)
    print(f" [*] Saved model in : {artifact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--input-path", type=str, default="/input", help="input dataset path"
    )
    parser.add_argument(
        "--output-path", type=str, default="/output", help="output files path"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/output/checkpoint",
        help="checkpoint path",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For saving the current model",
    )
    parser.add_argument(
        "--save-image", action="store_true", default=False, help="For saving the images"
    )
    args = parser.parse_args()

    # hyperparameters
    epochs = int(os.environ.get("epochs", 5))
    batch_size = int(os.environ.get("batch_size", 128))
    optimizer = str(os.environ.get("optimizer", "adadelta"))
    learning_rate = float(os.environ.get("learning_rate", 0.1))

    # Validate device
    device_type = "cpu"
    device_count = 1
    if torch.backends.mps.is_available():
        device_type = "mps"
    if torch.cuda.is_available():
        device_type = "cuda"
        device_count = torch.cuda.device_count()
    device = torch.device(device_type)
    print(f"Device: {device}")
    print(f"Device count: {device_count}")
    model = Net().to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Load dataset
    use_mount_dataset = False
    if os.path.exists(os.path.join(args.input_path, "train.csv")) and os.path.exists(
        os.path.join(args.input_path, "test.csv")
    ):
        use_mount_dataset = True

    if use_mount_dataset:
        print("=> Mount dataset found!")
        train_df = load_data(args.input_path, "train.csv")
        test_df = load_data(args.input_path, "test.csv")
        y_train, x_train = preprocess(train_df)
        y_test, x_test = preprocess(test_df)

        # Prepare dataloader
        train_dataloader = get_dataloader(x_train, y_train, batch_size, True)
        test_dataloader = get_dataloader(x_test, y_test, batch_size, False)
    else:
        print("=> Mount dataset not found! Use torchvision dataset instead.")

        train_kwargs = {"batch_size": batch_size}
        test_kwargs = {"batch_size": batch_size}

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(
            args.input_path, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(args.input_path, train=False, transform=transform)

        # Prepare dataloader
        train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    # Load checkpoint if exists
    checkpoint_file_path = os.path.join(args.checkpoint_path, "checkpoints.pt")
    if os.path.exists(args.checkpoint_path) and os.path.isfile(checkpoint_file_path):
        start_epoch, best_accuracy = load_checkpoint(checkpoint_file_path)
    else:
        print("=> No checkpoint has been found! Train from scratch.")
        start_epoch, best_accuracy = 0, torch.FloatTensor([0])
        if not os.path.exists(args.checkpoint_path):
            print(f" [*] Make directories : {args.checkpoint_path}")
            os.makedirs(args.checkpoint_path)

    for epoch in range(epochs):
        train(model, device, train_dataloader, optimizer, epoch, start_epoch)
        test_accuracy = test(model, device, test_dataloader, args.save_image)

        # Save the best checkpoint
        test_accuracy = torch.FloatTensor([test_accuracy])
        is_best = bool(test_accuracy.numpy() > best_accuracy.numpy())
        best_accuracy = torch.FloatTensor(
            max(test_accuracy.numpy(), best_accuracy.numpy())
        )
        save_checkpoint(
            {
                "epoch": start_epoch + epoch + 1,
                "state_dict": model.state_dict(),
                "best_accuracy": best_accuracy,
            },
            is_best,
            checkpoint_file_path,
        )

        scheduler.step()

    if args.save_model:
        print(f"=> Saving the model to {args.output_path}")
        save(model, args.output_path)
