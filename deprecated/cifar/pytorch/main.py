import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import vessl
from torch import optim
from torch.utils.data import random_split


class Net(nn.Module):
    def __init__(self, l1, l2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data(input_path):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_data = torchvision.datasets.CIFAR10(
        root=input_path, train=True, download=True, transform=transform
    )

    test_data = torchvision.datasets.CIFAR10(
        root=input_path, train=False, download=True, transform=transform
    )

    return train_data, test_data


def get_dataloader(data, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


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
        print("=> Saving a new best")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


def train(model, device, train_dataloader, optimizer, epoch, start_epoch):
    model.train()
    loss = 0
    for batch_idx, (data, labels) in enumerate(train_dataloader):
        inputs, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 128 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    batch_idx * len(data),
                    len(train_dataloader.dataset),
                    100.0 * batch_idx / len(train_dataloader),
                    loss.item(),
                )
            )

    # Logging loss metrics to Vessl
    vessl.log(step=epoch + start_epoch + 1, payload={"loss": loss.item()})


def valid(model, device, val_dataloader, start_epoch):
    model.eval()
    val_loss = 0.0
    total = 0
    correct = 0
    for i, (data, labels) in enumerate(val_dataloader):
        with torch.no_grad():
            inputs, labels = data.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            val_loss += loss.cpu().numpy()

    val_loss /= len(val_dataloader.dataset)
    val_accuracy = 100.0 * correct / len(val_dataloader.dataset)

    print(
        "\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            val_loss, correct, len(val_dataloader.dataset), val_accuracy
        )
    )

    # Logging loss metrics to Vessl
    vessl.log(
        step=epoch + start_epoch + 1,
        payload={"val_loss": val_loss, "val_accuracy": val_accuracy},
    )

    return val_accuracy


def save(model, path):
    if not os.path.exists(path):
        print(f" [*] Make directories : {path}")
        os.makedirs(path)
    artifact_path = os.path.join(path, "model.pt")
    torch.save(model.state_dict(), artifact_path)
    print(f" [*] Saved model in : {artifact_path}")


def test_accuracy(model, device, test_data):
    test_dataloader = get_dataloader(test_data, 4, False, 2)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch cifar Example")
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
    l1 = int(os.environ.get("l1", 2))
    l2 = int(os.environ.get("l2", 2))
    lr = float(os.environ.get("lr", 0.01))
    epochs = int(os.environ.get("epochs", 10))
    batch_size = int(os.environ.get("batch_size", 128))

    # Load data from input
    train_data, test_data = load_data(args.input_path)

    # Split train and validation sets
    test_abs = int(len(train_data) * 0.8)
    train_subset, val_subset = random_split(
        train_data, [test_abs, len(train_data) - test_abs]
    )

    print(f"The number of train data: {len(train_subset)}")
    print(f"The number of val data: {len(val_subset)}")
    print(f"The number of test set: {len(test_data)}")

    # Prepare dataloader
    train_dataloader = get_dataloader(train_subset, batch_size, True, 8)
    val_dataloader = get_dataloader(val_subset, batch_size, True, 8)

    # Validate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Device count: {torch.cuda.device_count()}")

    model = Net(l1, l2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

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
        val_accuracy = valid(model, device, val_dataloader, start_epoch)

        # Save the best checkpoint
        val_accuracy = torch.FloatTensor([val_accuracy])
        is_best = bool(val_accuracy.numpy() > best_accuracy.numpy())
        best_accuracy = torch.FloatTensor(
            max(val_accuracy.numpy(), best_accuracy.numpy())
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

    if args.save_model:
        save(model, args.output_path)

    _, _ = load_checkpoint(checkpoint_file_path)
    test_acc = test_accuracy(model, device, test_data)
    print("Best test data accuracy: {}".format(test_acc))
