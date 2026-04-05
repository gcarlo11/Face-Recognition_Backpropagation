from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass
class TrainingHistory:
    losses: list[float]
    accuracies: list[float]


class SimpleBackpropClassifier:
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.w1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.w2: np.ndarray | None = None
        self.b2: np.ndarray | None = None
        self.n_classes_: int | None = None

    def _initialize_parameters(self, output_size: int) -> None:
        w1_scale = np.sqrt(2.0 / self.input_size)
        w2_scale = np.sqrt(2.0 / self.hidden_size)
        self.w1 = self.rng.normal(0.0, w1_scale, size=(self.input_size, self.hidden_size)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_size, dtype=np.float32)
        self.w2 = self.rng.normal(0.0, w2_scale, size=(self.hidden_size, output_size)).astype(np.float32)
        self.b2 = np.zeros(output_size, dtype=np.float32)
        self.n_classes_ = output_size

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0)

    @staticmethod
    def _relu_gradient(values: np.ndarray) -> np.ndarray:
        return (values > 0).astype(np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)

    @staticmethod
    def _one_hot(labels: np.ndarray, output_size: int) -> np.ndarray:
        one_hot = np.zeros((labels.shape[0], output_size), dtype=np.float32)
        one_hot[np.arange(labels.shape[0]), labels] = 1.0
        return one_hot

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingHistory:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        output_size = int(np.max(y)) + 1

        self._initialize_parameters(output_size)
        assert self.w1 is not None and self.b1 is not None and self.w2 is not None and self.b2 is not None

        y_one_hot = self._one_hot(y, output_size)
        history = TrainingHistory(losses=[], accuracies=[])
        sample_count = X.shape[0]
        eps = 1e-8

        for _ in range(self.epochs):
            permutation = self.rng.permutation(sample_count)
            X_shuffled = X[permutation]
            y_shuffled = y_one_hot[permutation]

            for start in range(0, sample_count, self.batch_size):
                stop = min(start + self.batch_size, sample_count)
                batch_X = X_shuffled[start:stop]
                batch_y = y_shuffled[start:stop]

                z1 = batch_X @ self.w1 + self.b1
                a1 = self._relu(z1)
                z2 = a1 @ self.w2 + self.b2
                probabilities = self._softmax(z2)

                batch_size = batch_X.shape[0]
                grad_z2 = (probabilities - batch_y) / batch_size
                grad_w2 = a1.T @ grad_z2
                grad_b2 = np.sum(grad_z2, axis=0)
                grad_a1 = grad_z2 @ self.w2.T
                grad_z1 = grad_a1 * self._relu_gradient(z1)
                grad_w1 = batch_X.T @ grad_z1
                grad_b1 = np.sum(grad_z1, axis=0)

                self.w2 -= self.learning_rate * grad_w2
                self.b2 -= self.learning_rate * grad_b2
                self.w1 -= self.learning_rate * grad_w1
                self.b1 -= self.learning_rate * grad_b1

            train_probabilities = self.predict_proba(X)
            loss = -np.mean(np.sum(y_one_hot * np.log(np.clip(train_probabilities, eps, 1.0)), axis=1))
            accuracy = float(np.mean(self.predict(X) == y))
            history.losses.append(float(loss))
            history.accuracies.append(accuracy)

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float32)
        z1 = X @ self.w1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.w2 + self.b2
        return self._softmax(z2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.int64)
        return float(np.mean(self.predict(X) == y))

    def save(self, path: Path) -> None:
        self._ensure_fitted()
        payload = {
            "config": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "random_state": self.random_state,
                "n_classes_": self.n_classes_,
            },
            "weights": {
                "w1": self.w1,
                "b1": self.b1,
                "w2": self.w2,
                "b2": self.b2,
            },
        }
        with Path(path).open("wb") as file_handle:
            pickle.dump(payload, file_handle)

    @classmethod
    def load(cls, path: Path) -> "SimpleBackpropClassifier":
        with Path(path).open("rb") as file_handle:
            payload = pickle.load(file_handle)

        config = payload["config"]
        model = cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            learning_rate=config["learning_rate"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            random_state=config["random_state"],
        )
        model.n_classes_ = config["n_classes_"]
        weights = payload["weights"]
        model.w1 = weights["w1"]
        model.b1 = weights["b1"]
        model.w2 = weights["w2"]
        model.b2 = weights["b2"]
        return model

    def _ensure_fitted(self) -> None:
        if any(parameter is None for parameter in (self.w1, self.b1, self.w2, self.b2)):
            raise RuntimeError("The backpropagation model has not been trained yet.")
