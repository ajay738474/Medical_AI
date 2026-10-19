import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 2. PATHS
# ============================================================

FEATURE_DIR = "cnn/features"

MODEL_PATH = (
    "cnn/models/"
    "pathmnist_resnet18_full_classifier.pth"
)


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")

test_features_path = os.path.join(
    FEATURE_DIR,
    "test_features.pt"
)

test_labels_path = os.path.join(
    FEATURE_DIR,
    "test_labels.pt"
)

if not os.path.exists(test_features_path):
    raise FileNotFoundError(
        f"Missing file: {test_features_path}"
    )

if not os.path.exists(test_labels_path):
    raise FileNotFoundError(
        f"Missing file: {test_labels_path}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing model: {MODEL_PATH}"
    )

print("All required files found.")


# ============================================================
# 4. LOAD TEST FEATURES
# ============================================================

print("\nLoading saved test features...")

test_features = torch.load(
    test_features_path,
    map_location="cpu",
    weights_only=True
)

test_labels = torch.load(
    test_labels_path,
    map_location="cpu",
    weights_only=True
)

print(
    "Test features:",
    test_features.shape
)

print(
    "Test labels:",
    test_labels.shape
)

print(
    "Number of test images:",
    len(test_labels)
)


# ============================================================
# 5. CREATE TEST DATALOADER
# ============================================================

test_dataset = TensorDataset(
    test_features,
    test_labels
)

test_loader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False
)


# ============================================================
# 6. LOAD TRAINED CLASSIFIER
# ============================================================

print("\nLoading trained classifier...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

feature_size = checkpoint["feature_size"]

class_count = checkpoint["class_count"]

saved_accuracy = checkpoint["test_accuracy"]


# ============================================================
# 7. CREATE CLASSIFIER
# ============================================================

classifier = torch.nn.Linear(
    feature_size,
    class_count
)


# ============================================================
# 8. LOAD TRAINED WEIGHTS
# ============================================================

classifier.load_state_dict(
    checkpoint["classifier_state_dict"]
)

classifier = classifier.to(device)

classifier.eval()

print("Classifier loaded successfully.")

print(
    "Feature size:",
    feature_size
)

print(
    "Number of classes:",
    class_count
)

print(
    "Saved test accuracy:",
    f"{saved_accuracy:.2f}%"
)


# ============================================================
# 9. GENERATE PREDICTIONS
# ============================================================

print("\nRunning full test evaluation...")

all_predictions = []

all_labels = []

with torch.no_grad():

    for batch_number, (features, labels) in enumerate(
        test_loader
    ):

        features = features.to(device)

        outputs = classifier(
            features
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        if (batch_number + 1) % 5 == 0:

            print(
                f"Processed test batches: "
                f"{batch_number + 1}"
            )


# ============================================================
# 10. OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# 11. PRINT OVERALL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("FULL PATHMNIST EVALUATION")
print("=" * 65)

print(
    f"Test Images : {len(all_labels)}"
)

print(
    f"Accuracy    : {accuracy * 100:.2f}%"
)

print(
    f"Precision   : {precision * 100:.2f}%"
)

print(
    f"Recall      : {recall * 100:.2f}%"
)

print(
    f"F1 Score    : {f1 * 100:.2f}%"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

report = classification_report(
    all_labels,
    all_predictions,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

matrix = confusion_matrix(
    all_labels,
    all_predictions
)

print(matrix)


# ============================================================
# 14. CLASS-BY-CLASS RESULTS
# ============================================================

print("\n" + "=" * 65)
print("CLASS-BY-CLASS SUMMARY")
print("=" * 65)

class_precision = precision_score(
    all_labels,
    all_predictions,
    average=None,
    zero_division=0
)

class_recall = recall_score(
    all_labels,
    all_predictions,
    average=None,
    zero_division=0
)

class_f1 = f1_score(
    all_labels,
    all_predictions,
    average=None,
    zero_division=0
)


for class_id in range(class_count):

    class_total = sum(
        1 for label in all_labels
        if label == class_id
    )

    print(
        f"\nClass {class_id}"
    )

    print(
        f"  Images    : {class_total}"
    )

    print(
        f"  Precision : "
        f"{class_precision[class_id] * 100:.2f}%"
    )

    print(
        f"  Recall    : "
        f"{class_recall[class_id] * 100:.2f}%"
    )

    print(
        f"  F1 Score  : "
        f"{class_f1[class_id] * 100:.2f}%"
    )


# ============================================================
# 15. SAVE EVALUATION RESULTS
# ============================================================

results_dir = "cnn/results"

os.makedirs(
    results_dir,
    exist_ok=True
)


results_file = os.path.join(
    results_dir,
    "pathmnist_evaluation.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PATHMNIST RESNET-18 EVALUATION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Test Images: {len(all_labels)}\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall: {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score: {f1 * 100:.2f}%\n\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 60 + "\n"
    )

    file.write(
        report
    )

    file.write(
        "\n\nCONFUSION MATRIX\n"
    )

    file.write(
        "=" * 60 + "\n"
    )

    file.write(
        str(matrix)
    )


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 65)
print("EVALUATION COMPLETED")
print("=" * 65)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Images evaluated:",
    len(all_labels)
)

print(
    "Evaluation report saved:",
    results_file
)

print("=" * 65)