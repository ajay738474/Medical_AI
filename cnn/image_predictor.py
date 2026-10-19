# ============================================================
# MEDICAL AI - IMAGE PREDICTOR
# CNN Image Prediction Module
# ============================================================

import os

import torch
import torch.nn as nn

from torchvision import models, transforms

from PIL import Image

from medmnist import INFO


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "cnn",
    "models"
)


# ============================================================
# FAST CNN
# Used for:
# TissueMNIST
# OrganAMNIST
# OrganCMNIST
# OrganSMNIST
# ============================================================

class FastCNN(nn.Module):

    def __init__(self, num_classes):

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


# ============================================================
# RESNET18 CLASSIFIER
# ============================================================

def create_resnet18(
    num_classes,
    grayscale=False
):

    model = models.resnet18(
        weights=None
    )

    if grayscale:

        model.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_CONFIG = {

    "pathmnist": {
        "model": "resnet",
        "model_file":
            "pathmnist_resnet18_full_classifier.pth",
        "classes": 9,
        "grayscale": False
    },

    "chestmnist": {
        "model": "resnet",
        "model_file":
            "chestmnist_resnet18_full_classifier.pth",
        "classes": 14,
        "grayscale": True
    },

    "dermamnist": {
        "model": "resnet",
        "model_file":
            "dermamnist_resnet18_full_classifier.pth",
        "classes": 7,
        "grayscale": False
    },

    "octmnist": {
        "model": "resnet",
        "model_file":
            "octmnist_resnet18_full_classifier.pth",
        "classes": 4,
        "grayscale": True
    },

    "pneumoniamnist": {
        "model": "resnet",
        "model_file":
            "pneumoniamnist_resnet18_full_classifier.pth",
        "classes": 2,
        "grayscale": True
    },

    "retinamnist": {
        "model": "resnet",
        "model_file":
            "retinamnist_resnet18_full_classifier.pth",
        "classes": 5,
        "grayscale": False
    },

    "breastmnist": {
        "model": "resnet",
        "model_file":
            "breastmnist_resnet18_full_classifier.pth",
        "classes": 2,
        "grayscale": True
    },

    "bloodmnist": {
        "model": "resnet",
        "model_file":
            "bloodmnist_resnet18_full_classifier.pth",
        "classes": 8,
        "grayscale": False
    },

    "tissuemnist": {
        "model": "fast",
        "model_file":
            "tissuemnist_fast_cnn_classifier.pth",
        "classes": 8,
        "grayscale": True
    },

    "organamnist": {
        "model": "fast",
        "model_file":
            "organamnist_fast_cnn_classifier.pth",
        "classes": 11,
        "grayscale": True
    },

    "organcmnist": {
        "model": "fast",
        "model_file":
            "organcmnist_fast_cnn_classifier.pth",
        "classes": 11,
        "grayscale": True
    },

    "organsmnist": {
        "model": "fast",
        "model_file":
            "organsmnist_fast_cnn_classifier.pth",
        "classes": 11,
        "grayscale": True
    }
}


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def get_class_names(dataset_name):

    info = INFO[dataset_name]

    return {
        int(key): value
        for key, value in info["label"].items()
    }


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(dataset_name):

    dataset_name = dataset_name.lower().strip()

    if dataset_name not in DATASET_CONFIG:

        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )


    config = DATASET_CONFIG[
        dataset_name
    ]


    model_path = os.path.join(
        MODEL_DIR,
        config["model_file"]
    )


    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )


    # --------------------------------------------------------
    # Fast CNN
    # --------------------------------------------------------

    if config["model"] == "fast":

        model = FastCNN(
            config["classes"]
        )


    # --------------------------------------------------------
    # ResNet18
    # --------------------------------------------------------

    else:

        model = create_resnet18(
            config["classes"],
            config["grayscale"]
        )


    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    state_dict = torch.load(
        model_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)

    model.eval()


    return model


# ============================================================
# CREATE TRANSFORM
# ============================================================

def create_transform(dataset_name):

    config = DATASET_CONFIG[
        dataset_name
    ]


    if config["model"] == "fast":

        return transforms.Compose([

            transforms.Resize(
                (64, 64)
            ),

            transforms.Grayscale(
                num_output_channels=1
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.5],
                [0.5]
            )
        ])


    # --------------------------------------------------------
    # ResNet transform
    # --------------------------------------------------------

    if config["grayscale"]:

        return transforms.Compose([

            transforms.Resize(
                (128, 128)
            ),

            transforms.Grayscale(
                num_output_channels=1
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.5],
                [0.5]
            )
        ])


    return transforms.Compose([

        transforms.Resize(
            (128, 128)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(
    image_path,
    dataset_name
):

    dataset_name = dataset_name.lower().strip()


    if dataset_name not in DATASET_CONFIG:

        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        dataset_name
    )


    # --------------------------------------------------------
    # Load class names
    # --------------------------------------------------------

    class_names = get_class_names(
        dataset_name
    )


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    )


    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    transform = create_transform(
        dataset_name
    )


    image_tensor = transform(
        image
    )


    image_tensor = image_tensor.unsqueeze(
        0
    )


    image_tensor = image_tensor.to(
        DEVICE
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

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


        confidence = (
            probabilities[
                predicted_class
            ].item()
            * 100
        )


    predicted_label = class_names[
        predicted_class
    ]


    # --------------------------------------------------------
    # All probabilities
    # --------------------------------------------------------

    all_probabilities = {}

    for class_id, probability in enumerate(
        probabilities
    ):

        all_probabilities[
            class_names[class_id]
        ] = round(
            probability.item() * 100,
            2
        )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "dataset":
            dataset_name,

        "class_id":
            predicted_class,

        "label":
            predicted_label,

        "confidence":
            round(
                confidence,
                2
            ),

        "probabilities":
            all_probabilities
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("        MEDICAL AI IMAGE PREDICTOR")
    print("=" * 60)


    image_path = os.path.join(
        BASE_DIR,
        "cnn",
        "organS_test_image.png"
    )


    result = predict_image(
        image_path,
        "organsmnist"
    )


    print()
    print("Dataset:")
    print(result["dataset"])

    print()
    print("Prediction:")
    print(result["label"])

    print()
    print("Confidence:")
    print(
        f'{result["confidence"]:.2f}%'
    )

    print()
    print("All probabilities:")

    for label, probability in result[
        "probabilities"
    ].items():

        print(
            f"{label:30s} "
            f"{probability:.2f}%"
        )

    print()
    print("=" * 60)
    print("          PREDICTION COMPLETED")
    print("=" * 60)