import argparse
import os
import savvihub

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.25)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return self.softmax(x)


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


def train(model, device, train_loader, optimizer, epoch):
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # Logging to Savvihub
    savvihub.log(
        step=epoch,
        row={'loss': loss.item()}
    )


def test(model, device, test_loader, save_image):
    model.eval()
    test_loss = 0
    correct = 0
    test_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_images.append(savvihub.Image(
                data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if save_image:
        savvihub.log({
            "Examples": test_images,
        })


def save(model, path):
    if not os.path.exists(path):
        print(f" [*] Make directories : {path}")
        os.makedirs(path)
    artifact_path = os.path.join(path, "model.pt")
    torch.save(model.state_dict(), artifact_path)
    print(f" [*] Saved model in : {artifact_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--input-path', type=str, default='/input', help='input dataset path')
    parser.add_argument('--output-path', type=str, default='/output', help='output files path')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For saving the current model')
    parser.add_argument('--save-image', action='store_true', default=False,
                        help='For saving the images')
    args = parser.parse_args()

    train_df = load_data(args.input_path, "train.csv")
    test_df = load_data(args.input_path, 'test.csv')

    train_label, train_data = preprocess(train_df)
    test_label, test_data = preprocess(test_df)

    print(f'The shape of train data: {train_data.shape}')
    print(f'The shape of test data: {test_data.shape}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    print(f'Device count: {torch.cuda.device_count()}')

    model = Net().to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_data = torch.from_numpy(train_data).unsqueeze(1)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor)
    test_data = torch.from_numpy(test_data).unsqueeze(1)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(0, args.epochs):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, args.save_image)
        scheduler.step()

    if args.save_model:
        save(model, args.output_path)
