import argparse

from ultralytics import YOLO


def main(args: argparse.Namespace):
    model = YOLO(args.model)
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