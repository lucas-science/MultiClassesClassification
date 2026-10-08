from pathlib import Path

from PIL import Image
import torch


CLASSES = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
           "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
           "hair drier", "toothbrush")


NUM_CLASSES = len(CLASSES)


class COCOTrainImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotations_dir, transform=None, maximum_images=None):
        self.annotations_dir = Path(annotations_dir)
        self.img_labels = sorted(self.annotations_dir.glob("*.cls"))
        if maximum_images is not None:
            self.img_labels = self.img_labels[:maximum_images]
        self.img_dir = Path(img_dir)
        self.transform = transform

        missing = [image.stem for image in self.img_labels if not (self.img_dir / f"{image.stem}.jpg").exists()]
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


def train_loop(train_loader, net, criterion, optimizer, device,
               mbatch_loss_group=-1):
    net.train()
    running_loss = 0.0
    mbatch_losses = []
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % mbatch_loss_group == mbatch_loss_group - 1:
            mbatch_losses.append(running_loss / mbatch_loss_group)
            running_loss = 0.0
    if mbatch_loss_group > 0:
        return mbatch_losses


def validation_loop(val_loader, net, criterion, num_classes, device,
                    multi_label=False, th_multi_label=0.5, one_hot=False, class_metrics=False):
    net.eval()
    loss = 0
    correct = 0
    size = len(val_loader.dataset)
    class_total = {label: 0 for label in range(num_classes)}
    class_tp = {label: 0 for label in range(num_classes)}
    class_fp = {label: 0 for label in range(num_classes)}
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            if not multi_label:
                predictions = torch.zeros_like(outputs)
                predictions[torch.arange(outputs.shape[0]), torch.argmax(outputs, dim=1)] = 1.0
            else:
                predictions = torch.where(outputs > th_multi_label, 1.0, 0.0)
            if not one_hot:
                labels_mat = torch.zeros_like(outputs)
                labels_mat[torch.arange(outputs.shape[0]), labels] = 1.0
                labels = labels_mat

            tps = predictions * labels
            fps = predictions - tps

            tps = tps.sum(dim=0)
            fps = fps.sum(dim=0)
            lbls = labels.sum(dim=0)

            for c in range(num_classes):
                class_tp[c] += tps[c]
                class_fp[c] += fps[c]
                class_total[c] += lbls[c]

            correct += tps.sum()

    class_prec = []
    class_recall = []
    freqs = []
    for c in range(num_classes):
        class_prec.append(0 if class_tp[c] == 0 else
                          class_tp[c] / (class_tp[c] + class_fp[c]))
        class_recall.append(0 if class_tp[c] == 0 else
                            class_tp[c] / class_total[c])
        freqs.append(class_total[c])

    freqs = torch.tensor(freqs)
    class_weights = 1. / freqs
    class_weights /= class_weights.sum()
    class_prec = torch.tensor(class_prec)
    class_recall = torch.tensor(class_recall)
    prec = (class_prec * class_weights).sum()
    recall = (class_recall * class_weights).sum()
    f1 = 2. / (1/prec + 1/recall)
    val_loss = loss / size
    accuracy = correct / freqs.sum()
    results = {"loss": val_loss, "accuracy": accuracy, "f1": f1,
               "precision": prec, "recall": recall}

    if class_metrics:
        class_results = []
        for p, r in zip(class_prec, class_recall):
            f1 = (0 if p == r == 0 else 2. / (1/p + 1/r))
            class_results.append({"f1": f1, "precision": p, "recall": r})
        results = results, class_results

    return results


def update_graphs(summary_writer, epoch, train_results, test_results,
                  train_class_results=None, test_class_results=None,
                  class_names=None, mbatch_group=-1, mbatch_count=0, mbatch_losses=None):
    step = (epoch + 1) if not mbatch_group > 0 else (epoch + 1) * mbatch_count

    if mbatch_group > 0:
        for i in range(len(mbatch_losses)):
            summary_writer.add_scalar("Losses/Train mini-batches",
                                      mbatch_losses[i],
                                      epoch * mbatch_count + (i+1)*mbatch_group)

    summary_writer.add_scalars("Losses/Train Loss vs Test Loss",
                               {"Train Loss": train_results["loss"],
                                "Test Loss": test_results["loss"]}, step)

    for name, key in (("Accuracy", "accuracy"), ("F1", "f1"),
                      ("Precision", "precision"), ("Recall", "recall")):
        summary_writer.add_scalars(f"Metrics/Train {name} vs Test {name}",
                                   {f"Train {name}": train_results[key],
                                    f"Test {name}": test_results[key]}, step)

    if train_class_results and test_class_results:
        for i in range(len(train_class_results)):
            for name, key in (("F1", "f1"), ("Precision", "precision"), ("Recall", "recall")):
                summary_writer.add_scalars(f"Class Metrics/{class_names[i]}/Train {name} vs Test {name}",
                                           {f"Train {name}": train_class_results[i][key],
                                            f"Test {name}": test_class_results[i][key]}, step)
    summary_writer.flush()
