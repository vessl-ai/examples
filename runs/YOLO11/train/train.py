import argparse
import re

import vessl
from ultralytics import YOLO


def vessl_callback(trainer):
    metrics = trainer.metrics
    for key in list(metrics.keys()).copy():
        if key.endswith("(B)"):
            key_split = re.split("/|\(", key)
            new_key = f"{key_split[0]}/box_{key_split[1]}"
            metrics[new_key] = metrics.pop(key)
        elif key.endswith("(M)"):
            key_split = re.split("/|\(", key)
            new_key = f"{key_split[0]}/mask_{key_split[1]}"
            metrics[new_key] = metrics.pop(key)
    metrics["fitness"] = trainer.fitness
    vessl.log(
        payload=metrics,
        step=trainer.epoch,
    )


def main(args: argparse.Namespace):
    model = YOLO(args.model)
    model.add_callback("on_fit_epoch_end", vessl_callback)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        project=args.output_path,
        name=args.run_name,
        lr0=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="YOLO Trainer",
        description="Train YOLO models.",
    )
    parser.add_argument("--model", default="yolo11n.pt", type=str, help="YOLO model checkpoint path.")
    parser.add_argument("--data", default="coco.yaml", type=str, help="Dataset path to train the model with.")
    parser.add_argument("--epochs", default=100, type=int, help="Training epochs.")
    parser.add_argument("--lr", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--weight-decay", default=0.0005, type=float, help="Penalize large weights to prevent overfitting.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate for regularization in classification tasks.")
    parser.add_argument("--output-path", default="/output", type=str, help="Name of the directory where training outputs are saved.")
    parser.add_argument("--run-name", default="detect", help="Name of the training run. Used for creating a subdirectory within the output folder, where training logs and outputs are stored.")
    args = parser.parse_args()

    main(args)