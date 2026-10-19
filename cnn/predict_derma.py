import os
import sys
import torch

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from medmnist import DermaMNIST


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = (
    "cnn/models/"
    "dermamnist_resnet18_full_classifier.pth"
)

DATA_ROOT = r"C:\Users\ajay1\.medmnist"

DEFAULT_IMAGE_PATH = (
    "cnn/derma_test_image.png"
)

NUM_CLASSES = 7


# ============================================================
# 2. DERMAMNIST CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis-like lesions",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic nevi",
    6: "Vascular lesions"
}


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("DERMAMNIST PREDICTION")
print("=" * 70)

print(
    "Device:",
    device
)


# ============================================================
# 4. IMAGE PATH
# ============================================================

# You can provide an image manually:
#
# python cnn\predict_derma.py "C:\path\image.jpg"
#
# If no image path is provided, a sample from the
# DermaMNIST test dataset will be created automatically.

if len(sys.argv) > 1:

    IMAGE_PATH = sys.argv[1]

else:

    IMAGE_PATH = DEFAULT_IMAGE_PATH


# ============================================================
# 5. CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# ============================================================
# 6. CREATE TEST IMAGE IF NEEDED
# ============================================================

if not os.path.exists(IMAGE_PATH):

    print(
        "\nTest image not found."
    )

    print(
        "Creating one from DermaMNIST test dataset..."
    )

    dataset = DermaMNIST(
        root=DATA_ROOT,
        split="test",
        download=False
    )

    image = dataset.imgs[0]

    image = Image.fromarray(
        image
    )

    image.save(
        IMAGE_PATH
    )

    print(
        "Test image created:",
        IMAGE_PATH
    )


# ============================================================
# 7. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (128, 128)
    ),

    transforms.ToTensor(),

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
# 8. LOAD PRETRAINED RESNET-18
# ============================================================

print(
    "\nLoading ResNet-18..."
)

weights = ResNet18_Weights.DEFAULT

resnet = resnet18(
    weights=weights
)

resnet = resnet.to(
    device
)


# ============================================================
# 9. FREEZE RESNET
# ============================================================

for parameter in resnet.parameters():

    parameter.requires_grad = False


# ============================================================
# 10. CREATE FEATURE EXTRACTOR
# ============================================================

feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor = feature_extractor.to(
    device
)

feature_extractor.eval()

print(
    "ResNet-18 loaded."
)


# ============================================================
# 11. LOAD TRAINED DERMAMNIST CLASSIFIER
# ============================================================

print(
    "\nLoading trained DermaMNIST classifier..."
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

feature_size = checkpoint[
    "feature_size"
]

num_classes = checkpoint[
    "class_count"
]

classifier = torch.nn.Linear(
    feature_size,
    num_classes
)

classifier.load_state_dict(
    checkpoint[
        "classifier_state_dict"
    ]
)

classifier = classifier.to(
    device
)

classifier.eval()

print(
    "Classifier loaded."
)

print(
    "Feature size:",
    feature_size
)

print(
    "Number of classes:",
    num_classes
)


# ============================================================
# 12. LOAD IMAGE
# ============================================================

print(
    "\nLoading image..."
)

image = Image.open(
    IMAGE_PATH
).convert("RGB")

print(
    "Image:",
    IMAGE_PATH
)

print(
    "Original image size:",
    image.size
)


# ============================================================
# 13. PREPROCESS IMAGE
# ============================================================

image_tensor = transform(
    image
)

# Add batch dimension

image_tensor = image_tensor.unsqueeze(
    0
)

image_tensor = image_tensor.to(
    device
)


# ============================================================
# 14. EXTRACT FEATURES
# ============================================================

print(
    "\nExtracting image features..."
)

with torch.no_grad():

    features = feature_extractor(
        image_tensor
    )

    features = features.view(
        features.size(0),
        -1
    )


# ============================================================
# 15. MAKE PREDICTION
# ============================================================

with torch.no_grad():

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
# 16. GET CLASS NAME
# ============================================================

predicted_name = CLASS_NAMES.get(
    predicted_class,
    f"Unknown class {predicted_class}"
)


# ============================================================
# 17. DISPLAY MAIN RESULT
# ============================================================

print("\n")
print("=" * 70)

print(
    "DERMAMNIST PREDICTION RESULTS"
)

print("=" * 70)

print(
    "Image:",
    IMAGE_PATH
)

print(
    "Predicted class:",
    predicted_class
)

print(
    "Predicted category:",
    predicted_name
)

print(
    f"Confidence: {confidence:.2f}%"
)

print("=" * 70)


# ============================================================
# 18. DISPLAY ALL CLASS PROBABILITIES
# ============================================================

print(
    "\nClass probabilities:"
)

for class_id in range(
    NUM_CLASSES
):

    probability = (
        probabilities[class_id]
        .item()
        * 100
    )

    name = CLASS_NAMES[
        class_id
    ]

    print(
        f"{class_id} - "
        f"{name:<32} "
        f"{probability:.2f}%"
    )


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)

print(
    "PREDICTION COMPLETED"
)

print("=" * 70)

print(
    "\nImportant:"
)

print(
    "This is a DermaMNIST dataset classification "
    "prediction."
)

print(
    "It should not be used as a standalone "
    "clinical diagnosis."
)

print("=" * 70)