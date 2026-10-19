def create_rag_query(
    dataset_name,
    predicted_class,
    predicted_label,
    confidence
):

    dataset_name = str(dataset_name).lower().strip()

    if dataset_name in [
        "organamnist",
        "organcmnist",
        "organsmnist"
    ]:

        subject = "Anatomy"

        query = (
            f"Explain the anatomy of the {predicted_label}. "
            f"Describe its location, structure, function, "
            f"blood supply, and important clinical information."
        )

    elif dataset_name == "tissuemnist":

        subject = "Pathology"

        query = (
            f"Explain the tissue type {predicted_label}. "
            f"Describe its microscopic structure, location, "
            f"normal function, and important pathological features."
        )

    elif dataset_name == "pathmnist":

        subject = "Pathology"

        query = (
            f"Explain the histological tissue finding {predicted_label}. "
            f"Describe its microscopic features and clinical significance."
        )

    elif dataset_name == "dermamnist":

        subject = "Dermatology"

        query = (
            f"Explain the dermatological lesion {predicted_label}. "
            f"Describe its appearance, causes, important features, "
            f"and clinical significance."
        )

    elif dataset_name == "chestmnist":

        subject = "Radiology"

        query = (
            f"Explain the chest imaging finding {predicted_label}. "
            f"Describe what it means, common causes, imaging features, "
            f"and relevant clinical information."
        )

    elif dataset_name == "pneumoniamnist":

        subject = "Pulmonology"

        query = (
            f"Explain the medical condition {predicted_label}. "
            f"Describe its causes, symptoms, imaging features, "
            f"and important medical information."
        )

    elif dataset_name in [
        "octmnist",
        "retinamnist"
    ]:

        subject = "Ophthalmology"

        query = (
            f"Explain the retinal finding {predicted_label}. "
            f"Describe its features, causes, and clinical significance."
        )

    elif dataset_name == "breastmnist":

        subject = "Oncology"

        query = (
            f"Explain the breast imaging finding {predicted_label}. "
            f"Describe its characteristics and clinical significance."
        )

    elif dataset_name == "bloodmnist":

        subject = "Pathology"

        query = (
            f"Explain the blood cell type {predicted_label}. "
            f"Describe its morphology, normal function, "
            f"and clinical importance."
        )

    else:

        subject = "General Medicine"

        query = (
            f"Explain the medical finding {predicted_label}. "
            f"Describe its structure, function, characteristics, "
            f"and relevant clinical information."
        )

    return {
        "dataset": dataset_name,
        "class_id": int(predicted_class),
        "prediction": predicted_label,
        "confidence": float(confidence),
        "rag_subject": subject,
        "rag_query": query
    }


def main():

    print()
    print("=" * 60)
    print("              CNN -> RAG BRIDGE")
    print("=" * 60)

    result = create_rag_query(
        dataset_name="organsmnist",
        predicted_class=5,
        predicted_label="kidney-right",
        confidence=35.50
    )

    print()
    print("CNN RESULT")
    print("-" * 60)

    print("Dataset    :", result["dataset"])
    print("Class ID   :", result["class_id"])
    print("Prediction :", result["prediction"])
    print("Confidence :", f'{result["confidence"]:.2f}%')

    print()
    print("RAG INFORMATION")
    print("-" * 60)

    print("RAG Subject:", result["rag_subject"])

    print()
    print("RAG Query:")
    print(result["rag_query"])

    print()
    print("=" * 60)
    print("       CNN -> RAG BRIDGE SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()