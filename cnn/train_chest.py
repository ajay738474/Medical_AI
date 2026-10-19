import os
import torch
import numpy as np

from PIL import Image

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from medmnist import ChestMNIST


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_ROOT = r"C:\Users\ajay1\.medmnist"

FEATURE_DIR = "cnn/features/chest"

MODEL_DIR = "cnn/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "chestmnist_resnet18_full_classifier.pth"
)

BATCH_SIZE = 64

FEATURE_BATCH_SIZE = 512

EPOCHS = 5

NUM_CLASSES = 14

IMAGE_SIZE = 128


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs(
    FEATURE_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("CHESTMNIST TRAINING")
print("=" * 70)

print(
    "Device:",
    device
)


# ============================================================
# 4. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. CUSTOM DATASET
# ============================================================

class ChestDataset(Dataset):

    def __init__(
        self,
        split
    ):

        self.dataset = ChestMNIST(
            root=DATA_ROOT,
            split=split,
            download=False
        )

        self.images = self.dataset.imgs

        self.labels = self.dataset.labels

    def __len__(self):

        return len(
            self.images
        )

    def __getitem__(
        self,
        index
    ):

        image = self.images[index]

        label = self.labels[index]

        # Convert NumPy image to PIL
        image = Image.fromarray(
            image
        )

        # Apply preprocessing
        image = transform(
            image
        )

        # Convert labels to float
        # because BCEWithLogitsLoss
        # requires floating-point targets.

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return image, label


# ============================================================
# 6. LOAD DATASETS
# ============================================================

print("\nLoading ChestMNIST datasets...")

train_dataset = ChestDataset(
    "train"
)

val_dataset = ChestDataset(
    "val"
)

test_dataset = ChestDataset(
    "test"
)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Testing images:",
    len(test_dataset)
)


# ============================================================
# 7. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 8. LOAD PRETRAINED RESNET-18
# ============================================================

print("\nLoading pretrained ResNet-18...")

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
# 10. REMOVE ORIGINAL CLASSIFIER
# ============================================================

feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor = feature_extractor.to(
    device
)

feature_extractor.eval()

print(
    "ResNet-18 feature extractor ready."
)

print(
    "Feature size: 512"
)


# ============================================================
# 11. FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(
    loader,
    feature_file,
    label_file,
    split_name
):

    # --------------------------------------------------------
    # If cached features already exist, load them.
    # --------------------------------------------------------

    if (
        os.path.exists(feature_file)
        and os.path.exists(label_file)
    ):

        print(
            f"\nLoading cached {split_name} features..."
        )

        features = torch.load(
            feature_file,
            map_location="cpu",
            weights_only=True
        )

        labels = torch.load(
            label_file,
            map_location="cpu",
            weights_only=True
        )

        print(
            f"{split_name} features loaded:",
            features.shape
        )

        return features, labels


    # --------------------------------------------------------
    # Otherwise extract features.
    # --------------------------------------------------------

    print(
        f"\nExtracting {split_name} features..."
    )

    all_features = []

    all_labels = []

    with torch.no_grad():

        for batch_index, (
            images,
            labels
        ) in enumerate(loader):

            images = images.to(
                device
            )

            features = feature_extractor(
                images
            )

            features = features.view(
                features.size(0),
                -1
            )

            all_features.append(
                features.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

            if (
                batch_index + 1
            ) % 50 == 0:

                print(
                    f"{split_name}: "
                    f"{batch_index + 1} batches processed"
                )


    features = torch.cat(
        all_features,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )


    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    torch.save(
        features,
        feature_file
    )

    torch.save(
        labels,
        label_file
    )

    print(
        f"{split_name} features saved:"
    )

    print(
        features.shape
    )

    print(
        labels.shape
    )

    return features, labels


# ============================================================
# 12. FEATURE FILE PATHS
# ============================================================

train_feature_file = os.path.join(
    FEATURE_DIR,
    "train_features.pt"
)

train_label_file = os.path.join(
    FEATURE_DIR,
    "train_labels.pt"
)

val_feature_file = os.path.join(
    FEATURE_DIR,
    "val_features.pt"
)

val_label_file = os.path.join(
    FEATURE_DIR,
    "val_labels.pt"
)

test_feature_file = os.path.join(
    FEATURE_DIR,
    "test_features.pt"
)

test_label_file = os.path.join(
    FEATURE_DIR,
    "test_labels.pt"
)


# ============================================================
# 13. EXTRACT / LOAD FEATURES
# ============================================================

train_features, train_labels = extract_features(
    train_loader,
    train_feature_file,
    train_label_file,
    "Training"
)

val_features, val_labels = extract_features(
    val_loader,
    val_feature_file,
    val_label_file,
    "Validation"
)

test_features, test_labels = extract_features(
    test_loader,
    test_feature_file,
    test_label_file,
    "Testing"
)


# ============================================================
# 14. CREATE FEATURE DATASETS
# ============================================================

train_feature_dataset = TensorDataset(
    train_features,
    train_labels
)

val_feature_dataset = TensorDataset(
    val_features,
    val_labels
)

train_feature_loader = DataLoader(
    train_feature_dataset,
    batch_size=FEATURE_BATCH_SIZE,
    shuffle=True
)

val_feature_loader = DataLoader(
    val_feature_dataset,
    batch_size=FEATURE_BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 15. CALCULATE CLASS POSITIVE WEIGHTS
# ============================================================
# ChestMNIST is highly imbalanced.
#
# Positive weight:
#
# negative samples / positive samples
#
# This helps the classifier pay attention to
# less frequent findings.
# ============================================================

positive_count = train_labels.sum(
    dim=0
)

negative_count = (
    train_labels.shape[0]
    - positive_count
)

positive_count = torch.clamp(
    positive_count,
    min=1
)

pos_weight = (
    negative_count
    / positive_count
)

pos_weight = pos_weight.to(
    device
)

print("\nPositive class counts:")

print(
    positive_count
)

print("\nUsing class weights for training.")


# ============================================================
# 16. CREATE 14-LABEL CLASSIFIER
# ============================================================

classifier = torch.nn.Linear(
    512,
    NUM_CLASSES
)

classifier = classifier.to(
    device
)


# ============================================================
# 17. LOSS FUNCTION
# ============================================================

criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


# ============================================================
# 18. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=0.001
)


# ============================================================
# 19. TRAIN CLASSIFIER
# ============================================================

print("\n")
print("=" * 70)
print("STARTING CLASSIFIER TRAINING")
print("=" * 70)


for epoch in range(
    EPOCHS
):

    classifier.train()

    total_loss = 0.0

    total_samples = 0

    correct_predictions = 0


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for features, labels in train_feature_loader:

        features = features.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        outputs = classifier(
            features
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
            * features.size(0)
        )

        total_samples += (
            features.size(0)
        )


        # ----------------------------------------------------
        # Exact-match accuracy
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities >= 0.5
        ).float()

        correct_predictions += (
            (predictions == labels)
            .all(dim=1)
            .sum()
            .item()
        )


    train_loss = (
        total_loss
        / total_samples
    )

    train_exact_accuracy = (
        correct_predictions
        / total_samples
        * 100
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    classifier.eval()

    val_loss_total = 0.0

    val_samples = 0

    val_correct = 0


    with torch.no_grad():

        for features, labels in val_feature_loader:

            features = features.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = classifier(
                features
            )

            loss = criterion(
                outputs,
                labels
            )


            val_loss_total += (
                loss.item()
                * features.size(0)
            )

            val_samples += (
                features.size(0)
            )


            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            val_correct += (
                (predictions == labels)
                .all(dim=1)
                .sum()
                .item()
            )


    val_loss = (
        val_loss_total
        / val_samples
    )

    val_exact_accuracy = (
        val_correct
        / val_samples
        * 100
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training Loss     : "
        f"{train_loss:.4f}"
    )

    print(
        f"Training Exact Acc: "
        f"{train_exact_accuracy:.2f}%"
    )

    print(
        f"Validation Loss   : "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Exact Acc: "
        f"{val_exact_accuracy:.2f}%"
    )


# ============================================================
# 20. SAVE MODEL
# ============================================================

classifier.eval()

checkpoint = {

    "classifier_state_dict":
        classifier.state_dict(),

    "feature_size":
        512,

    "class_count":
        NUM_CLASSES,

    "test_dataset":
        "ChestMNIST",

    "model":
        "ResNet18",

    "task":
        "multi-label classification"
}


torch.save(
    checkpoint,
    MODEL_PATH
)


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)

print(
    "CHESTMNIST TRAINING COMPLETED"
)

print("=" * 70)

print(
    "Model saved:"
)

print(
    MODEL_PATH
)

print(
    "\nImportant:"
)

print(
    "ChestMNIST uses 14-label multi-label classification."
)

print(
    "The model outputs 14 independent probabilities."
)

print("=" * 70)