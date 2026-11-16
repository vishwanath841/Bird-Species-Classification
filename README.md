# Bird Species Classification Using Deep Learning

An image classification application that predicts the bird species shown in an uploaded image. The Streamlit interface uses a trained MobileNetV3 Small model and displays its most likely species, prediction confidence, and top three class predictions.

> **Model evaluation note:** The reported test accuracy is based on a 100-image test set containing only five images per species. This small test set does not guarantee real-world performance. The model has not been claimed as tested on broader real-world images.

## Features

- Upload bird images in JPG, JPEG, or PNG format.
- Preview the uploaded image before classification.
- Predict one of the 20 bird species in the model metadata.
- Show the predicted class confidence and top three predictions.
- Load the trained model once using Streamlit resource caching.
- Use a GPU when available and otherwise run on the CPU.

## Technologies

- Python
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/)
- [MobileNetV3 Small](https://pytorch.org/vision/stable/models/mobilenetv3.html)
- [Streamlit](https://streamlit.io/)
- [Pillow](https://python-pillow.org/)
- JSON metadata for class names and image size

## Model Details

| Property | Details |
|---|---|
| Architecture | MobileNetV3 Small |
| Training approach | Transfer learning |
| Number of classes | 20 |
| Input image size | 224 × 224 pixels |
| Validation accuracy | 98% (reported) |
| Test accuracy | 100% on the provided 100-image test set (reported) |

The test set contains just five images per species. Its reported 100% accuracy should be interpreted cautiously and should not be considered a guarantee of performance on new or real-world images.

The application constructs the MobileNetV3 Small architecture, replaces its final classifier layer with a 20-class output layer based on the metadata, and loads the trained weights from `models/best_bird_model.pth`. Uploaded images are converted to RGB, resized to 224 × 224, converted to tensors, and normalized using the standard ImageNet mean and standard deviation.

## Dataset

The model was trained using the Kaggle **Birds 20 Species – Image Classification** dataset:

[View the dataset on Kaggle](https://www.kaggle.com/datasets/umairshahpirzada/birds-20-species-image-classification)

The class names and expected image size used by the app are read from `models/metadata.json`.

## Project Structure

```text
Bird-Species-Classification/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── best_bird_model.pth
    └── metadata.json
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/vishwanath841/Bird-Species-Classification.git
cd Bird-Species-Classification
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents environment activation, you can invoke the environment's Python directly as shown in the run instructions below.

**macOS or Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements pin NumPy, PyTorch, and Torchvision to versions used to verify model inference. If you have a CUDA-enabled system and need a specific PyTorch CUDA build, follow the installation selector on the [official PyTorch website](https://pytorch.org/get-started/locally/) and retain compatible Torchvision and NumPy versions.

## Run the Application

From the project root, run:

```bash
python -m streamlit run app.py
```

On Windows, if you did not activate the environment, use:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## Example Prediction Workflow

1. Start the app and open its local URL in a browser.
2. Select a JPG, JPEG, or PNG image using the uploader.
3. Review the displayed image.
4. Click **Predict bird species**.
5. View the predicted species, confidence percentage, and top three predictions.

## Future Improvements

These are possible future extensions and are not currently implemented:

- Evaluate the model with a larger and more diverse test set.
- Assess performance on independently collected images and varied conditions.
- Add per-species evaluation metrics and a confusion matrix.
- Improve accessibility and provide optional descriptions for predicted species.

## Author

**[vishwanath841](https://github.com/vishwanath841)**
