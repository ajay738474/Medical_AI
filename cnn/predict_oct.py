import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

import medmnist
from medmnist import INFO


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = r"C:\Users\ajay1\.medmnist"

MODEL_PATH = "cnn/models/octmnist_resnet18_full_classifier.pth"

DEFAULT_IMAGE = "cnn/oct_test_image.png"

IMAGE_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Choroidal neovascularization",
    1: "Diabetic macular edema",
    2: "Drusen",
    3: "Normal"
}


# ============================================================
# START
# ============================================================

print("=" * 70)
print("OCTMNIST PREDICTION")
print("=" * 70)

print(f"Device: {DEVICE}")
print()


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: OCTMNIST model not found.")
    print(f"Expected: {MODEL_PATH}")
    sys.exit(1)


# ============================================================
# IMAGE PATH
# ============================================================

if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = DEFAULT_IMAGE


# ============================================================
# CREATE TEST IMAGE IF NEEDED
# ============================================================

if not os.path.exists(image_path):

    print("Test image not found.")
    print("Creating one from OCTMNIST test dataset...")
    print()

    info = INFO["octmnist"]

    DatasetClass = getattr(
        medmnist,
        info["python_class"]
    )

    test_dataset = DatasetClass(
        split="test",
        root=DATA_ROOT,
        download=False
    )

    image, label = test_dataset[0]

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image.save(DEFAULT_IMAGE)

    image_path = DEFAULT_IMAGE

    print(
        f"Test image created: {image_path}"
    )

    print()


# ============================================================
# LOAD RESNET-18
# ============================================================

print("Loading ResNet-18...")

resnet = resnet18(
    weights=ResNet18_Weights.DEFAULT
)

for param in resnet.parameters():
    param.requires_grad = False

feature_size = resnet.fc.in_features

resnet.fc = nn.Identity()

resnet = resnet.to(DEVICE)

resnet.eval()

print("ResNet-18 loaded.")
print()


# ============================================================
# LOAD TRAINED CLASSIFIER
# ============================================================

print("Loading trained OCTMNIST classifier...")

classifier = nn.Linear(
    feature_size,
    4
)

classifier.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

classifier = classifier.to(DEVICE)

classifier.eval()

print("Classifier loaded.")
print(f"Feature size: {feature_size}")
print("Number of classes: 4")
print()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Lambda(
        lambda x:
        x.repeat(3, 1, 1)
        if x.shape[0] == 1
        else x
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD IMAGE
# ============================================================

print("Loading image...")
print(f"Image: {image_path}")

image = Image.open(
    image_path
).convert("L")

print(
    f"Original image size: {image.size}"
)

print()


# ============================================================
# TRANSFORM
# ============================================================

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

print("Extracting image features...")

with torch.no_grad():

    features = resnet(
        image_tensor
    )

    outputs = classifier(
        features
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    confidence = (
        probabilities[predicted_class].item()
        * 100
    )


# ============================================================
# RESULTS
# ============================================================

print()

print("=" * 70)
print("OCTMNIST PREDICTION RESULTS")
print("=" * 70)

print(
    f"Image: {image_path}"
)

print(
    f"Predicted class: {predicted_class}"
)

print(
    f"Predicted category: "
    f"{CLASS_NAMES[predicted_class]}"
)

print(
    f"Confidence: {confidence:.2f}%"
)

print("=" * 70)

print()

print("Class probabilities:")

for class_id, class_name in CLASS_NAMES.items():

    probability = (
        probabilities[class_id].item()
        * 100
    )

    print(
        f"{class_id} - "
        f"{class_name:<35} "
        f"{probability:.2f}%"
    )


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)

print()

print("Important:")
print(
    "This is an OCTMNIST dataset "
    "classification prediction."
)

print(
    "It should not be used as a "
    "standalone clinical diagnosis."
)

print("=" * 70)