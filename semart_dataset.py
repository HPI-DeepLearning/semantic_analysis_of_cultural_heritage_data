import os
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd

from utils import get_image_transforms


class SemArtDataset(Dataset):
    def __init__(self, semart_root, split, img_input_size, desired_length=None):
        self.images, self.captions = self._load_data(semart_root, split)
        self.root_dir = semart_root
        self.transform = get_image_transforms(img_input_size)
        self.actual_length = len(self.images)
        self.desired_length = desired_length if desired_length else self.actual_length

    def _load_data(self, semart_root, split):
        location_df = pd.read_csv(f'{semart_root}/semart_{split}.csv', sep='\t', header=0, encoding="latin")
        images = []
        captions = []

        for idx, row in location_df.iterrows():
            captions.append(row["DESCRIPTION"])
            images.append(row['IMAGE_FILE'])

        return images, captions

    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer

    def __getitem__(self, index):
        index = index % self.actual_length
        image_name = os.path.join(self.root_dir,
                                  "Images",
                                  self.images[index])

        image = Image.open(image_name)
        image = image.convert(mode="RGB")
        image = self.transform(image)

        caption = self.captions[index]
        caption_array = self.vectorizer.transform([caption]).toarray()
        caption_tensor = torch.from_numpy(caption_array).float()
        caption_tensor = torch.squeeze(caption_tensor)

        return image, caption_tensor

    def __len__(self):
        return self.desired_length
