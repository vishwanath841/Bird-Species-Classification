import json
import pickle
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError


PROJECT_DIR = Path(__file__).resolve().parent
METADATA_PATH = PROJECT_DIR / "models" / "metadata.json"
MODEL_PATH = PROJECT_DIR / "models" / "best_bird_model.pth"
EXPECTED_IMAGE_SIZE = 224


def load_metadata():
    """Load and validate the model metadata used by the app."""
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object.")

    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("metadata.json must contain a non-empty 'class_names' list.")
    if not all(isinstance(name, str) and name.strip() for name in class_names):
        raise ValueError("Every class name in metadata.json must be a non-empty string.")
    if len(class_names) < 3:
        raise ValueError("At least three classes are required to show the top 3 predictions.")
    if metadata.get("image_size") != EXPECTED_IMAGE_SIZE:
        raise ValueError(
            f"Expected metadata image_size to be {EXPECTED_IMAGE_SIZE}, "
            f"but found {metadata.get('image_size')!r}."
        )

    return class_names


@st.cache_resource(show_spinner="Loading the trained bird classification model...")
def load_model():
    """Build the model, load its trained weights, and cache it for this app session."""
    import torch
    from torchvision.models import mobilenet_v3_small

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model file not found: {MODEL_PATH}")

    class_names = load_metadata()
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(
        model.classifier[3].in_features,
        len(class_names),
    )

    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError("The model checkpoint does not contain a valid state dictionary.")

    # Strict loading also checks that the saved output layer matches metadata.json.
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device, class_names


def preprocess_image(image):
    """Apply the same resize, tensor conversion, and normalization as training."""
    from torchvision import transforms

    image_transform = transforms.Compose(
        [
            transforms.Resize((EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return image_transform(image).unsqueeze(0)


def main():
    st.set_page_config(page_title="Bird Species Classification", page_icon="🐦")
    st.title("Bird Species Classification")
    st.write(
        "Upload a bird image to classify it with a trained MobileNetV3 Small model. "
        "The results include the model's confidence and its top three predictions."
    )

    uploaded_file = st.file_uploader(
        "Choose a bird image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, and PNG.",
    )

    if uploaded_file is None:
        return

    try:
        with Image.open(uploaded_file) as uploaded_image:
            image = uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This image could not be opened. Please upload a valid JPG, JPEG, or PNG image.")
        return

    st.image(image, caption="Uploaded image", use_container_width=True)

    if not st.button("Predict bird species", type="primary"):
        return

    try:
        import torch

        model, device, class_names = load_model()
    except (
        FileNotFoundError,
        ImportError,
        json.JSONDecodeError,
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        st.error(
            "The trained model could not be loaded. Check that the model and metadata files "
            f"are valid and match. Details: {error}"
        )
        return

    try:
        input_tensor = preprocess_image(image).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(input_tensor), dim=1).squeeze(0)
        top_probabilities, top_indices = torch.topk(probabilities, k=3)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        st.error(f"The image could not be classified. Details: {error}")
        return

    predicted_index = top_indices[0].item()
    st.subheader(f"Predicted species: {class_names[predicted_index]}")
    st.metric("Prediction confidence", f"{top_probabilities[0].item() * 100:.2f}%")

    st.markdown("#### Top 3 predictions")
    for rank, (probability, class_index) in enumerate(
        zip(top_probabilities.tolist(), top_indices.tolist()),
        start=1,
    ):
        st.write(f"{rank}. **{class_names[class_index]}** — {probability * 100:.2f}%")


if __name__ == "__main__":
    main()