import os

import torch
from torchvision import datasets
import hashlib
from PIL import Image


def hash_string(input_string, hash_size=16):
    hash_obj = hashlib.sha256(input_string.encode())
    hash_digest = hash_obj.digest()

    truncated_hash = hash_digest[:hash_size]
    return truncated_hash


class KermanyXRayImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(KermanyXRayImageFolder, self).__init__(root, transform)
        self.data = []
        self.targets = []
        self.sample_ids = []

        for i, (img_path, target) in enumerate(self.samples):
            if i % 500 == 0:
                print(f"sample {i} of {len(self.samples)}")
            img = Image.open(img_path).convert('RGB')

            # Crop the image for labels in the image
            width, height = img.size
            if height > 300 and width > 300:
                img = img.crop((100, 100, width-100, height - 100))

            if self.transform is not None:
                img = self.transform(img)
            self.data.append(img)
            self.targets.append(target)
            sample_id = hash_string(os.path.basename(img_path))
            self.sample_ids.append(sample_id)

        self.data = torch.stack(self.data)
        self.targets = torch.tensor(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index], self.sample_ids[index]
