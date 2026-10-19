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

MODEL_PATH = (
    "cnn/models/"
    "retinamnist_resnet18_full_classifier.pth"
)

DEFAULT_IMAGE = "cnn/retina_test_image.png"

IMAGE_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Normal",
    1: "Pathological Myopia",
    2: "Age-related Macular Degeneration",
    3: "Diabetic Retinopathy",
    4: "Other"
}


# ============================================================
# START
# ============================================================

print("=" * 70)
print("RETINAMNIST PREDICTION")
print("=" * 70)

print(f"Device: {DEVICE}")
print()


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: RetinaMNIST model not found.")
    print()
    print(
        f"Expected model: {MODEL_PATH}"
    )
    print()
    print(
        "Run cnn/train_retina.py first."
    )

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
    print(
        "Creating one from RetinaMNIST test dataset..."
    )
    print()

    info = INFO["retinamnist"]

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

print(
    "Loading trained RetinaMNIST classifier..."
)

classifier = nn.Linear(
    feature_size,
    len(CLASS_NAMES)
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
print(
    f"Feature size: {feature_size}"
)

print(
    f"Number of classes: {len(CLASS_NAMES)}"
)

print()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD IMAGE
# ============================================================

print("Loading image...")
print(
    f"Image: {image_path}"
)

try:

    image = Image.open(
        image_path
    ).convert("RGB")

except Exception as e:

    print()
    print(
        "ERROR: Could not load image."
    )

    print(e)

    sys.exit(1)


print(
    f"Original image size: {image.size}"
)

print()


# ============================================================
# TRANSFORM IMAGE
# ============================================================

image_tensor = transform(
    image
)

image_tensor = image_tensor.unsqueeze(
    0
)

image_tensor = image_tensor.to(
    DEVICE
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "Extracting image features..."
)

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
        probabilities[
            predicted_class
        ].item()
        * 100
    )


# ============================================================
# RESULTS
# ============================================================

print()

print("=" * 70)
print("RETINAMNIST PREDICTION RESULTS")
print("=" * 70)

print(
    f"Image: {image_path}"
)

print(
    f"Predicted class: "
    f"{predicted_class}"
)

print(
    f"Predicted category: "
    f"{CLASS_NAMES[predicted_class]}"
)

print(
    f"Confidence: "
    f"{confidence:.2f}%"
)

print("=" * 70)

print()

print(
    "Class probabilities:"
)

for class_id, class_name in CLASS_NAMES.items():

    probability = (
        probabilities[
            class_id
        ].item()
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
    "This is a RetinaMNIST dataset "
    "classification prediction."
)

print(
    "It should not be used as a "
    "standalone clinical diagnosis."
)

print("=" * 70)