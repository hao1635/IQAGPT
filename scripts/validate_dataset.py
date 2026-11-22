"""Validate CT-IQA file content, annotation IDs, splits and optional checksums."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np


def validate(root):
    root = Path(root)
    summaries = {}
    expected = {'train': 800, 'test': 200}
    seen = set()
    captions = json.loads((root / 'iqagpt_caption.json').read_text())
    labels = json.loads((root / 'labels.json').read_text())
    for split, size in expected.items():
        rows = captions[split]
        ids = [row['image_id'] for row in rows]
        if len(ids) != size or len(set(ids)) != size or seen.intersection(ids):
            raise ValueError('Invalid ID count or overlap: ' + split)
        seen.update(ids)
        if ids != [row['image_id'] for row in labels[split]]:
            raise ValueError('Caption/label order mismatch')
        for row in labels[split]:
            values = row['labels']
            if len(values) != 4 or any(not isinstance(v, int) for v in values):
                raise ValueError('Invalid label row')
            if any(not 0 <= values[i] <= (4 if i == 2 else 3) for i in range(4)):
                raise ValueError('Invalid label range')
        counts = Counter(i[-2:] for i in ids)
        if counts != {str(i).zfill(2): size // 8 for i in range(8)}:
            raise ValueError('Quality-level counts do not match')
        for image_id in ids:
            if Path(image_id).name != image_id or not image_id.startswith(split + '-'):
                raise ValueError('Invalid image ID')
            image = np.load(root / split / 'npy' / (image_id + '.npy'), allow_pickle=False)
            if image.shape != (512, 512) or image.dtype != np.float32 or not np.isfinite(image).all():
                raise ValueError('Invalid NPY content: ' + image_id)
        summaries[split] = {'count': size, 'shape': [512, 512], 'dtype': 'float32', 'quality_counts': dict(counts)}
    manifest = root / 'checksums.sha256'
    checked = 0
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            digest, filename = line.split('  ', 1)
            path = root / filename
            if root.resolve() not in path.resolve().parents:
                raise ValueError('Invalid checksum path')
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Checksum mismatch: ' + filename)
            checked += 1
    return {'splits': summaries, 'checksums_verified': checked}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    args = p.parse_args()
    print(json.dumps(validate(args.data_root), indent=2))
