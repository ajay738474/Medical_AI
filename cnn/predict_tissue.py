import os

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from medmnist import INFO


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 64
NUM_CLASSES = 8

MODEL_PATH = "cnn/models/tissuemnist_fast_cnn_classifier.pth"

IMAGE_PATH = "cnn/tissue_test_image.png"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TissueMNIST - Single Image Prediction")
print("=" * 70)

print("Device:", device)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("\nERROR: Model not found.")
    print("Expected model:")
    print(MODEL_PATH)

    raise FileNotFoundError(MODEL_PATH)


# ============================================================
# TISSUE CNN
# SAME ARCHITECTURE USED DURING TRAINING
# ============================================================

class TissueCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),

            nn.Flatten(),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD CLASS NAMES
# ============================================================

info = INFO["tissuemnist"]

CLASS_NAMES = {
    int(key): value
    for key, value in info["label"].items()
}


print("\nTissueMNIST Classes:")
print("-" * 70)

for class_id in range(NUM_CLASSES):

    print(
        f"{class_id}: {CLASS_NAMES[class_id]}"
    )


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating model...")

model = TissueCNN()

model = model.to(device)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):

    print("\nERROR: Image not found.")

    print(
        "\nExpected image location:"
    )

    print(
        IMAGE_PATH
    )

    print(
        "\nPlace your image at:"
    )

    print(
        r"C:\Users\ajay1\OneDrive\OneDrive\Desktop\Medical_AI\cnn\tissue_test_image.png"
    )

    print(
        "\nOr change IMAGE_PATH at the top of this file."
    )

    raise FileNotFoundError(IMAGE_PATH)


# ============================================================
# IMAGE TRANSFORMATION
# SAME AS TRAINING
# ============================================================

transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image...")

image = Image.open(
    IMAGE_PATH
)

print(
    "Image path:",
    IMAGE_PATH
)

print(
    "Original image size:",
    image.size
)

print(
    "Image mode:",
    image.mode
)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

image_tensor = transform(image)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning prediction...")

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )


predicted_class = predicted_class.item()

confidence = confidence.item() * 100


# ============================================================
# GET PREDICTED CLASS NAME
# ============================================================

predicted_tissue = CLASS_NAMES[
    predicted_class
]


# ============================================================
# DISPLAY MAIN RESULT
# ============================================================

print("\n")
print("=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(
    f"Predicted class : {predicted_class}"
)

print(
    f"Predicted tissue: {predicted_tissue}"
)

print(
    f"Confidence      : {confidence:.2f}%"
)


# ============================================================
# DISPLAY ALL PROBABILITIES
# ============================================================

print("\n")
print("ALL CLASS PROBABILITIES")
print("-" * 70)

for class_id in range(NUM_CLASSES):

    probability = (
        probabilities[0][class_id].item()
        * 100
    )

    print(
        f"{class_id}. "
        f"{CLASS_NAMES[class_id]:40s}"
        f"{probability:7.2f}%"
    )


# ============================================================
# SORTED PREDICTIONS
# ============================================================

print("\n")
print("TOP PREDICTIONS")
print("-" * 70)

sorted_probabilities = torch.argsort(
    probabilities[0],
    descending=True
)

for rank, class_id in enumerate(
    sorted_probabilities,
    start=1
):

    class_id = class_id.item()

    probability = (
        probabilities[0][class_id].item()
        * 100
    )

    print(
        f"{rank}. "
        f"{CLASS_NAMES[class_id]:40s}"
        f"{probability:7.2f}%"
    )


# ============================================================
# IMPORTANT DISCLAIMER
# ============================================================

print("\n")
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "This model predicts TissueMNIST dataset classes."
)

print(
    "It is a prototype medical-image classification model."
)

print(
    "The prediction is NOT a clinical diagnosis."
)

print(
    "This model was trained using a 4,000-image "
    "balanced subset of the TissueMNIST training data."
)

print("=" * 70)
print("Prediction completed.")
print("=" * 70)