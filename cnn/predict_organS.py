import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from medmnist import INFO


# =========================
# CONFIG
# =========================

IMAGE_SIZE = 64
NUM_CLASSES = 11

MODEL_PATH = "cnn/models/organsmnist_fast_cnn_classifier.pth"
IMAGE_PATH = "cnn/organS_test_image.png"


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================
# CLASS NAMES
# =========================

info = INFO["organsmnist"]

CLASS_NAMES = {
    int(k): v
    for k, v in info["label"].items()
}

print("\nOrganSMNIST Classes:")

for index, name in CLASS_NAMES.items():
    print(index, ":", name)


# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# =========================
# MODEL
# =========================

class OrganCNN(nn.Module):

    def __init__(self, num_classes=11):

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

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# =========================
# CHECK IMAGE
# =========================

if not os.path.exists(IMAGE_PATH):

    print("\nTest image not found.")

    print("Creating test image from OrganSMNIST...")

    from medmnist import OrganSMNIST

    dataset = OrganSMNIST(
        split="test",
        download=True
    )

    os.makedirs("cnn", exist_ok=True)

    img = Image.fromarray(
        dataset.imgs[0]
    )

    img.save(IMAGE_PATH)

    print("Test image created:")
    print(IMAGE_PATH)


# =========================
# LOAD MODEL
# =========================

print("\nLoading trained model...")

model = OrganCNN(NUM_CLASSES)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model loaded successfully.")


# =========================
# LOAD IMAGE
# =========================

print("\nLoading image:")

print(IMAGE_PATH)

image = Image.open(
    IMAGE_PATH
).convert("L")


# =========================
# PREPROCESS IMAGE
# =========================

image_tensor = transform(
    image
).unsqueeze(0)

image_tensor = image_tensor.to(device)


# =========================
# PREDICTION
# =========================

with torch.no_grad():

    output = model(
        image_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    confidence = probabilities[
        predicted_class
    ].item() * 100


# =========================
# RESULT
# =========================

predicted_name = CLASS_NAMES[
    predicted_class
]

print("\n===================================")
print("       OrganSMNIST Prediction")
print("===================================")

print(
    f"Predicted class : {predicted_class}"
)

print(
    f"Predicted organ : {predicted_name}"
)

print(
    f"Confidence      : {confidence:.2f}%"
)


# =========================
# ALL PROBABILITIES
# =========================

print("\nClass probabilities:")

for index, probability in enumerate(
    probabilities
):

    print(
        f"{CLASS_NAMES[index]:30s} "
        f"{probability.item() * 100:.2f}%"
    )

print("\nPrediction completed.")