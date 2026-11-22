"""Portable CT-IQA loading with the original image and caption preprocessing."""
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def pre_caption(caption, max_words=120):
    caption = re.sub(r"([,'!?\"()*#:;~])", '', caption.lower())
    caption = caption.replace('/', ' ').replace('<person>', 'person').replace('[sep]', '[SEP]')
    caption = re.sub(r'\s{2,}', ' ', caption).rstrip('\n').strip(' ')
    return ' '.join(caption.split(' ')[:max_words])


class CTIQADataset(Dataset):
    def __init__(self, root, split='train', annotations='iqagpt_caption.json', augment=False):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        path = Path(annotations)
        if not path.is_absolute():
            path = self.root / path
        self.rows = json.loads(path.read_text(encoding='utf-8'))[split]
        ids = [row['image_id'] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate image IDs in annotations')
        if any(Path(i).name != i or not i.startswith(split + '-') for i in ids):
            raise ValueError('Invalid image ID or split')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_id = row['image_id']
        image = np.load(self.root / self.split / 'npy' / (image_id + '.npy'), allow_pickle=False)
        if image.shape != (512, 512) or not np.isfinite(image).all():
            raise ValueError('Expected a finite 512x512 array: ' + image_id)
        image = np.clip(image.astype(np.float32), 0, 1).reshape(1, 512, 512)
        if self.augment:
            # Same flip axes and 90-degree rotation family as the archived loader.
            for axis in (0, 1, 2):
                if np.random.uniform() > 0.5:
                    image = np.flip(image, axis)
            image = np.rot90(image, np.random.randint(0, 4), axes=(1, 2))
        tensor = torch.from_numpy(image.copy()).repeat(3, 1, 1)
        return tensor, pre_caption(row['caption']), image_id
