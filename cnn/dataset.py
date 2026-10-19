import medmnist
from medmnist import PathMNIST
from torchvision import transforms
import matplotlib.pyplot as plt


transform = transforms.ToTensor()

train_dataset = PathMNIST(
    split="train",
    transform=transform,
    download=True
)

image, label = train_dataset[0]

print("Image shape :", image.shape)
print("Label       :", label)
print("Dataset size:", len(train_dataset))

plt.imshow(image.permute(1, 2, 0))
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()