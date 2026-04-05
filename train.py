from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import save_artifacts, train_face_recognition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AT&T face recognition model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/att_faces"), help="Dataset folder")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory for saved artifacts")
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_size = (args.image_width, args.image_height)
    artifacts, metrics, history, _ = train_face_recognition(
        data_dir=args.data_dir,
        image_size=image_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_artifacts(artifacts, args.output_dir)

    summary = {
        **metrics,
        "loss_history": history.losses,
        "accuracy_history": history.accuracies,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
