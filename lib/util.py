from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn


CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


NUM_CLASSES = len(CLASSES)


class COCOTrainImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotations_dir, transform=None, maximum_images=None):
        self.annotations_dir = Path(annotations_dir)
        self.img_labels = sorted(self.annotations_dir.glob("*.cls"))
        if maximum_images is not None:
            self.img_labels = self.img_labels[:maximum_images]
        self.img_dir = Path(img_dir)
        self.transform = transform

        missing = [
            image.stem
            for image in self.img_labels
            if not (self.img_dir / f"{image.stem}.jpg").exists()
        ]
        if missing:
            print(f"{len(missing)} Images not found ; examples : {missing[:5]}")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        label_path = self.img_labels[index]
        image_path = self.img_dir / f"{label_path.stem}.jpg"
        image = Image.open(image_path).convert("RGB")
        with open(label_path, "r", encoding="utf-8") as f:
            labels = [int(label) for label in f.readlines()]
        if any(i < 0 or i >= NUM_CLASSES for i in labels):
            raise ValueError(f"Class identifier invalid : {label_path}")
        if self.transform:
            image = self.transform(image)
        labels = torch.zeros(NUM_CLASSES).scatter_(0, torch.tensor(labels), value=1)
        return image, labels


class COCOTestImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.image_files = sorted(self.img_dir.glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path.stem


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset: torch.utils.data.dataset.Subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        label_path = self.subset.dataset.img_labels[self.subset.indices[index]]
        image = Image.open(
            self.subset.dataset.img_dir / f"{label_path.stem}.jpg"
        ).convert("RGB")
        labels = [
            int(x.strip()) for x in label_path.read_text().splitlines() if x.strip()
        ]
        labels = torch.zeros(80).scatter_(0, torch.tensor(labels), value=1)
        return self.transform(image), labels


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for imbalanced multi-label classification."""

    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        probabilities_neg = 1.0 - probabilities
        if self.clip > 0:
            probabilities_neg = (probabilities_neg + self.clip).clamp(max=1.0)

        loss = targets * torch.log(probabilities.clamp(min=self.eps))
        loss += (1.0 - targets) * torch.log(probabilities_neg.clamp(min=self.eps))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            probabilities_t = probabilities * targets + probabilities_neg * (
                1.0 - targets
            )
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss *= (1.0 - probabilities_t).pow(gamma)
        return -loss.mean()
