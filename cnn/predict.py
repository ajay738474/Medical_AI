import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = "cnn/models/pathmnist_resnet18_full_classifier.pth"
IMAGE_PATH = "cnn/test_image.png"


# ============================================================
# 2. PATHMNIST CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Adipose",
    1: "Background",
    2: "Debris",
    3: "Lymphocytes",
    4: "Mucus",
    5: "Smooth Muscle",
    6: "Normal Colon Mucosa",
    7: "Cancer-associated Stroma",
    8: "Colorectal Adenocarcinoma Epithelium"
}


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 4. CHECK MODEL AND IMAGE
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================
# This must match the preprocessing used during training.
#
# Original PathMNIST image: 28 x 28
# Training input: 128 x 128
# ImageNet normalization is used because ResNet-18
# was pretrained on ImageNet.
# ============================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 6. LOAD PRETRAINED RESNET-18
# ============================================================

print("\nLoading ResNet-18...")

weights = ResNet18_Weights.DEFAULT

resnet = resnet18(
    weights=weights
)

resnet = resnet.to(device)


# ============================================================
# 7. FREEZE RESNET
# ============================================================

for param in resnet.parameters():
    param.requires_grad = False


# ============================================================
# 8. REMOVE ORIGINAL IMAGENET CLASSIFIER
# ============================================================
# We only need ResNet-18 as a feature extractor.
#
# ResNet-18 output:
# 512 features
# ============================================================

feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor = feature_extractor.to(device)

feature_extractor.eval()

print("ResNet-18 loaded.")


# ============================================================
# 9. LOAD OUR TRAINED CLASSIFIER
# ============================================================

print("\nLoading trained PathMNIST classifier...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

feature_size = checkpoint["feature_size"]
num_classes = checkpoint["class_count"]

classifier = torch.nn.Linear(
    feature_size,
    num_classes
)

classifier.load_state_dict(
    checkpoint["classifier_state_dict"]
)

classifier = classifier.to(device)

classifier.eval()

print("Classifier loaded.")

print(
    "Number of classes:",
    num_classes
)

print(
    "Training test accuracy:",
    f'{checkpoint["test_accuracy"]:.2f}%'
)


# ============================================================
# 10. LOAD IMAGE
# ============================================================

print("\nLoading image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

print(
    "Image size:",
    image.size
)


# ============================================================
# 11. PREPROCESS IMAGE
# ============================================================

image_tensor = transform(image)

# Add batch dimension
# From:
# [3, 128, 128]
#
# To:
# [1, 3, 128, 128]

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# ============================================================
# 12. EXTRACT FEATURES
# ============================================================

print("\nExtracting image features...")

with torch.no_grad():

    features = feature_extractor(
        image_tensor
    )

    # Convert:
    # [1, 512, 1, 1]
    #
    # Into:
    # [1, 512]

    features = features.view(
        features.size(0),
        -1
    )


# ============================================================
# 13. MAKE PREDICTION
# ============================================================

with torch.no_grad():

    outputs = classifier(
        features
    )

    # Convert raw model outputs
    # into probabilities.

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    # Find class with highest probability.

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    # Get confidence.

    confidence = (
        probabilities[
            0,
            predicted_class
        ].item()
        * 100
    )


# ============================================================
# 14. GET PREDICTED TISSUE NAME
# ============================================================

predicted_name = CLASS_NAMES.get(
    predicted_class,
    f"Unknown class {predicted_class}"
)


# ============================================================
# 15. DISPLAY MAIN RESULT
# ============================================================

print("\n" + "=" * 60)

print("PATHMNIST PREDICTION")

print("=" * 60)

print(
    "Image:",
    IMAGE_PATH
)

print(
    "Predicted class:",
    predicted_class
)

print(
    "Predicted tissue:",
    predicted_name
)

print(
    f"Confidence: {confidence:.2f}%"
)

print("=" * 60)


# ============================================================
# 16. DISPLAY ALL CLASS PROBABILITIES
# ============================================================

print("\nClass probabilities:")

for class_id, probability in enumerate(
    probabilities[0]
):

    class_name = CLASS_NAMES.get(
        class_id,
        f"Unknown class {class_id}"
    )

    percentage = (
        probability.item() * 100
    )

    print(
        f"{class_id} - "
        f"{class_name}: "
        f"{percentage:.2f}%"
    )


# ============================================================
# 17. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)

print("PREDICTION COMPLETED")

print("=" * 60)

print(
    "\nNote:"
)

print(
    "This prediction represents a PathMNIST "
    "dataset tissue category."
)

print(
    "It should not be used as a standalone "
    "clinical diagnosis."
)

print("=" * 60)