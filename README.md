# AT&T Face Recognition with OpenCV and Backpropagation

This project is a homework-friendly face recognition pipeline built around the AT&T face dataset. It uses OpenCV for image preprocessing and a simple neural network with manual backpropagation for classification.

## What is included

- `notebooks/face_recognition_backprop.ipynb` for the full exploratory pipeline
- `train.py` for reproducible model training and artifact export
- `app.py` for a Streamlit deployment demo
- `.github/workflows/ci.yml` for basic validation in CI
- `Dockerfile` for container deployment

## Project structure

- `src/att_faces.py` downloads and loads the AT&T dataset
- `src/backprop_model.py` contains the neural network with backpropagation
- `src/pipeline.py` connects preprocessing, training, saving, and prediction

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The notebook and training pipeline will download the AT&T dataset automatically into `data/att_faces` if it is missing.

If the download is blocked in your environment, manually place the extracted dataset in:

```text
data/att_faces/s1
...
data/att_faces/s40
```

## Run the notebook

Open `notebooks/face_recognition_backprop.ipynb` and run the cells from top to bottom.

## Train the model

From the project root:

```bash
python train.py --epochs 100 --hidden-size 128
```

This creates saved artifacts in `artifacts/`.

## Run the deployment app

After training, launch the Streamlit app:

```bash
streamlit run app.py
```

## Docker deployment

Build and run the container:

```bash
docker build -t att-face-recognition .
docker run -p 8501:8501 att-face-recognition
```

Then open `http://localhost:8501`.
