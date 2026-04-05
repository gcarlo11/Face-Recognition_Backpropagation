from __future__ import annotations

from pathlib import Path
import urllib.request
import zipfile

import cv2
import numpy as np

DATASET_URL = "https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip"


def _resolve_dataset_root(base_dir: Path) -> Path:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return base_dir

    subject_dirs = [base_dir / f"s{i}" for i in range(1, 41)]
    if all(subject_dir.exists() for subject_dir in subject_dirs):
        return base_dir

    for child_dir in base_dir.iterdir():
        if not child_dir.is_dir():
            continue
        nested_subject_dirs = [child_dir / f"s{i}" for i in range(1, 41)]
        if all(subject_dir.exists() for subject_dir in nested_subject_dirs):
            return child_dir

    return base_dir


def download_att_faces(target_dir: Path) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = _resolve_dataset_root(target_dir)
    if dataset_root != target_dir:
        return dataset_root

    subject_dirs = [target_dir / f"s{i}" for i in range(1, 41)]
    if all(subject_dir.exists() for subject_dir in subject_dirs):
        return target_dir

    zip_path = target_dir / "att_faces.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(DATASET_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)

    return _resolve_dataset_root(target_dir)


def load_att_faces(data_dir: Path, image_size: tuple[int, int] = (64, 64)) -> tuple[np.ndarray, np.ndarray]:
    data_dir = _resolve_dataset_root(Path(data_dir))
    subject_dirs = sorted(
        [path for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("s")],
        key=lambda path: int(path.name[1:]),
    )

    images: list[np.ndarray] = []
    labels: list[str] = []

    for subject_dir in subject_dirs:
        for image_path in sorted(subject_dir.iterdir()):
            if image_path.suffix.lower() not in {".pgm", ".png", ".jpg", ".jpeg"}:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
            images.append(resized)
            labels.append(subject_dir.name)

    if not images:
        raise FileNotFoundError(
            f"No AT&T face images were found in {data_dir}. Download the dataset first."
        )

    return np.stack(images).astype(np.uint8), np.asarray(labels)
