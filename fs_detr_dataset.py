import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from selective_search import selective_search
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F

class PretrainingDataset(Dataset):
    def __init__(self, root_dir, transform=None, top_k=200, threshold=0.5, device='cpu'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            top_k (int): Maximum number of region proposals to keep per image.
            threshold (float): Confidence threshold for classifying a region as object (1) or not (0).
            device (str): Device to run classification model on ('cpu' or 'cuda').
        """
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.JPEG')]
        self.transform = transform
        self.top_k = top_k
        self.threshold = threshold
        self.device = device

        # Load pretrained ImageNet model
        self.classifier = models.resnet50(pretrained=True).eval().to(device)
        self.patch_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        img_array = np.array(image)
        height, width, _ = img_array.shape

        # Run selective search
        regions = selective_search(img_array, mode='single', random_sort=False)

        boxes = []
        labels = []
        seen = set()

        for region in regions:
            x, y, w, h = region
            if w <= 0 or h <= 0:
                continue
            if w * h < 500:
                continue
            rect = (x, y, w, h)
            if rect in seen:
                continue
            seen.add(rect)

            # Normalize box
            x1 = x / width
            y1 = y / height
            x2 = (x + w) / width
            y2 = (y + h) / height
            boxes.append([x1, y1, x2, y2])

            # Crop and classify patch
            patch = image.crop((x, y, x + w, y + h))
            input_tensor = self.patch_transform(patch).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.classifier(input_tensor)
                probs = F.softmax(logits, dim=1)
                max_prob = probs.max().item()

            # Label: 1 if max class prob ≥ threshold, else 0
            label = 1 if max_prob >= self.threshold else 0
            labels.append(label)

            # Stop early if we hit top_k
            if len(boxes) >= self.top_k:
                break

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, boxes, labels
