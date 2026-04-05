from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .att_faces import download_att_faces, load_att_faces
from .backprop_model import SimpleBackpropClassifier, TrainingHistory


@dataclass
class FaceRecognitionArtifacts:
    model: SimpleBackpropClassifier
    label_encoder: LabelEncoder
    image_size: tuple[int, int]


def prepare_dataset(data_dir: Path, image_size: tuple[int, int] = (64, 64)) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    download_att_faces(data_dir)
    images, labels = load_att_faces(data_dir, image_size=image_size)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    X = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
    return X, y, label_encoder


def train_face_recognition(
    data_dir: Path,
    image_size: tuple[int, int] = (64, 64),
    hidden_size: int = 128,
    learning_rate: float = 0.01,
    epochs: int = 100,
    batch_size: int = 32,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[FaceRecognitionArtifacts, dict[str, float], TrainingHistory, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    X, y, label_encoder = prepare_dataset(data_dir, image_size=image_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = SimpleBackpropClassifier(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    )
    history = model.fit(X_train, y_train)

    metrics = {
        "train_accuracy": model.score(X_train, y_train),
        "test_accuracy": model.score(X_test, y_test),
        "train_size": float(X_train.shape[0]),
        "test_size": float(X_test.shape[0]),
    }

    artifacts = FaceRecognitionArtifacts(model=model, label_encoder=label_encoder, image_size=image_size)
    return artifacts, metrics, history, (X_train, X_test, y_train, y_test)


def save_artifacts(artifacts: FaceRecognitionArtifacts, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts.model.save(output_dir / "model.pkl")
    joblib.dump(artifacts.label_encoder, output_dir / "label_encoder.pkl")
    metadata = {"image_size": list(artifacts.image_size)}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_artifacts(output_dir: Path) -> FaceRecognitionArtifacts:
    output_dir = Path(output_dir)
    model = SimpleBackpropClassifier.load(output_dir / "model.pkl")
    label_encoder = joblib.load(output_dir / "label_encoder.pkl")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    image_size = tuple(metadata["image_size"])
    return FaceRecognitionArtifacts(model=model, label_encoder=label_encoder, image_size=(int(image_size[0]), int(image_size[1])))


def preprocess_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    if image.ndim == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image

    resized = cv2.resize(grayscale, image_size, interpolation=cv2.INTER_AREA)
    return resized.reshape(1, -1).astype(np.float32) / 255.0


def predict_bgr_image(image: np.ndarray, artifacts: FaceRecognitionArtifacts) -> tuple[str, float, np.ndarray]:
    features = preprocess_image(image, artifacts.image_size)
    probabilities = artifacts.model.predict_proba(features)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = artifacts.label_encoder.inverse_transform([predicted_index])[0]
    predicted_confidence = float(probabilities[predicted_index])
    return predicted_label, predicted_confidence, probabilities
