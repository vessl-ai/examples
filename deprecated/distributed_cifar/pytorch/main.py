import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import vessl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def evaluate(model, device, test_dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def main():
    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="Local rank. Necessary for using the torch.distributed.launch utility.",
    )
    parser.add_argument(
        "--backend", type=str, help="Distributed backend (NCCL or gloo)", default="nccl"
    )
    parser.add_argument(
        "--num_epochs", type=int, help="Number of training epochs.", default=10
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size for one process.",
        default=128,
    )
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate.", default=0.1
    )
    parser.add_argument(
        "--accum_iter",
        type=int,
        help="Number of accumulate batches iteration",
        default=32,
    )
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=0)
    parser.add_argument(
        "--model_dir", type=str, help="Directory for saving models.", default="/output"
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        help="Model filename.",
        default="resnet_distributed.pth",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from saved checkpoint."
    )
    args = parser.parse_args()

    model_filepath = os.path.join(args.model_dir, args.model_filename)

    # Set random seeds to initialize the model
    set_random_seeds(random_seed=args.random_seed)

    # Initializes the distributed backend to synchronize nodes/GPUs
    torch.distributed.init_process_group(backend=args.backend)

    # Use resnet18 with DDP
    model = torchvision.models.resnet18(pretrained=False)
    device = torch.device("cuda:{}".format(args.local_rank))
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )

    # To resume, load model from "cuda:0"
    if args.resume:
        map_location = {"cuda:0": "cuda:{}".format(args.local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    # Prepare dataset and dataloader
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Data should be prefetched from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    train_set = torchvision.datasets.CIFAR10(
        root="/input", train=True, download=False, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="/input", train=False, download=False, transform=transform
    )

    # Using distributed sampler for training dataset
    train_sampler = DistributedSampler(dataset=train_set)
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=8,
    )

    # Skip sampler for test_dataset
    test_dataloader = DataLoader(
        dataset=test_set, batch_size=128, shuffle=False, num_workers=8
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        ddp_model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5
    )

    durations = []
    for epoch in range(args.num_epochs):
        print(f"Local Rank: {args.local_rank} Epoch: {epoch}, training started.")
        loss = 0.0
        start = time.time()

        ddp_model.train()

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            if (batch_idx + 1) % args.accum_iter == 0:
                # Sync model params in every accu_iter
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels) / args.accum_iter
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                # Do not sync model params
                with ddp_model.no_sync():
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = ddp_model(inputs)
                    loss = criterion(outputs, labels) / args.accum_iter
                    loss.backward()

        end = time.time()
        duration = end - start
        durations.append(duration)

        # Save and evaluate model only in local_rank 0
        if args.local_rank == 0:
            accuracy = evaluate(
                model=ddp_model, device=device, test_dataloader=test_dataloader
            )
            torch.save(ddp_model.state_dict(), model_filepath)
            print("-" * 75)
            print(
                f"Epoch: {epoch}, Accuracy: {accuracy}, Loss: {loss:.2f}, Elapsed: {duration:.2f}s"
            )
            print("-" * 75)

            # Logging to vessl
            vessl.log(
                step=epoch,
                payload={"accuracy": accuracy, "loss": loss, "elapsed": duration},
            )

    print(f"Total training time: {sum(durations):.2f}s")
    print(f"Average training time : {sum(durations) / len(durations):.2f}s")


if __name__ == "__main__":
    main()
