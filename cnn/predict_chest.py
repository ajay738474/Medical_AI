import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from medmnist import ChestMNIST


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = (
    "cnn/models/"
    "chestmnist_resnet18_full_classifier.pth"
)

DATA_ROOT = r"C:\Users\ajay1\.medmnist"

DEFAULT_IMAGE_PATH = (
    "cnn/chest_test_image.png"
)

NUM_CLASSES = 14


# ============================================================
# 2. CHESTMNIST LABEL NAMES
# ============================================================

CLASS_NAMES = {
    0: "Atelectasis",
    1: "Cardiomegaly",
    2: "Effusion",
    3: "Infiltration",
    4: "Mass",
    5: "Nodule",
    6: "Pneumonia",
    7: "Pneumothorax",
    8: "Consolidation",
    9: "Edema",
    10: "Emphysema",
    11: "Fibrosis",
    12: "Pleural Thickening",
    13: "Hernia"
}


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("CHESTMNIST PREDICTION")
print("=" * 70)

print("Device:", device)


# ============================================================
# 4. IMAGE PATH
# ============================================================

# You can provide an image from the command line:
#
# python cnn\predict_chest.py "C:\path\image.png"
#
# If no path is provided, the script creates a sample
# from the ChestMNIST test dataset.

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
        "Creating one from ChestMNIST test dataset..."
    )

    dataset = ChestMNIST(
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

    transforms.Grayscale(
        num_output_channels=3
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
# 8. LOAD RESNET-18
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
# 11. LOAD CHESTMNIST CLASSIFIER
# ============================================================

print(
    "\nLoading trained ChestMNIST classifier..."
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
    "Number of labels:",
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
).convert("L")

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
# 15. PREDICTION
# ============================================================

with torch.no_grad():

    outputs = classifier(
        features
    )

    probabilities = torch.sigmoid(
        outputs
    )[0]


# ============================================================
# 16. DISPLAY ALL PREDICTIONS
# ============================================================

print("\n")
print("=" * 70)
print("CHESTMNIST PREDICTION RESULTS")
print("=" * 70)

print(
    "\nPredicted probabilities:"
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
        f"{class_id:2d} - "
        f"{name:<25} "
        f"{probability:6.2f}%"
    )


# ============================================================
# 17. FIND HIGHEST-PROBABILITY FINDING
# ============================================================

top_class = torch.argmax(
    probabilities
).item()

top_probability = (
    probabilities[top_class]
    .item()
    * 100
)

top_name = CLASS_NAMES[
    top_class
]


print("\n")
print("=" * 70)

print(
    "Highest predicted finding:"
)

print(
    f"{top_name}"
)

print(
    f"Predicted probability: "
    f"{top_probability:.2f}%"
)

print("=" * 70)


# ============================================================
# 18. THRESHOLD-BASED FINDINGS
# ============================================================

THRESHOLD = 0.50

positive_findings = []

for class_id in range(
    NUM_CLASSES
):

    probability = probabilities[
        class_id
    ].item()

    if probability >= THRESHOLD:

        positive_findings.append(
            (
                CLASS_NAMES[class_id],
                probability * 100
            )
        )


print(
    "\nFindings above 50% probability:"
)

if len(positive_findings) == 0:

    print(
        "None"
    )

else:

    for name, probability in positive_findings:

        print(
            f"- {name}: "
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
    "These are model probabilities for "
    "ChestMNIST labels."
)

print(
    "They are not a standalone clinical diagnosis."
)

print("=" * 70)