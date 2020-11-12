import collections

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import CocoCaptions
from tqdm import tqdm

from utils import get_image_transforms


class CoCoDataset(Dataset):
    def __init__(self, coco_root, split, img_input_size):
        self.coco, self.captions = self._load_data(coco_root, split, img_input_size)
        self.length = len(self.captions)

    def _load_data(self, root, split, img_input_size):
        coco = CocoCaptions(root=f'{root}/{split}2014', annFile=f'{root}/annotations/captions_{split}2014.json',
                            transform=get_image_transforms(img_input_size))
        captions = self._extract_captions(coco)

        return coco, captions

    def _extract_captions(self, coco):
        """
        Each img in coco has multiple assigned captions. We currently only choose the first one as label.
        This method extracts this first caption.

        :param coco: CoCo Dataset object
        :return: list of captions (first for each image)
        """
        anns = coco.coco.anns

        print("Extracting tokenized captions from CoCo")

        distinct_anns = collections.OrderedDict()

        for key in tqdm(anns):
            if not anns[key]['image_id'] in distinct_anns:
                distinct_anns[anns[key]['image_id']] = anns[key]['caption']

        return list(distinct_anns.values())

    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer

    def __getitem__(self, index):
        image, captions = self.coco[index]  # index 0 => image, index 1 => captions

        caption = captions[0]
        caption_array = self.vectorizer.transform([caption]).toarray()
        caption_tensor = torch.from_numpy(caption_array).float()
        caption_tensor = torch.squeeze(caption_tensor)

        return image, caption_tensor

    def __len__(self):
        return self.length
