"""Check the released loader's image/caption contract without private data."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from iqagpt.data import CTIQADataset, pre_caption


class DataTests(unittest.TestCase):
    def test_load_and_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train/npy').mkdir(parents=True)
            image = np.linspace(-0.5, 1.5, 512 * 512, dtype=np.float32).reshape(512, 512)
            np.save(root / 'train/npy/train-00000.npy', image)
            annotation = {'train': [{'image_id': 'train-00000', 'caption': 'Image Noise: Better than usual;'}]}
            (root / 'iqagpt_caption.json').write_text(json.dumps(annotation))
            tensor, caption, image_id = CTIQADataset(root)[0]
            self.assertEqual(tuple(tensor.shape), (3, 512, 512))
            self.assertEqual(image_id, 'train-00000')
            self.assertEqual(caption, 'image noise better than usual')
            self.assertTrue(np.array_equal(tensor[0].numpy(), np.clip(image, 0, 1)))
            augmented, _, _ = CTIQADataset(root, augment=True)[0]
            self.assertTrue(np.array_equal(np.sort(tensor[0].numpy(), axis=None), np.sort(augmented[0].numpy(), axis=None)))

    def test_invalid_image_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'iqagpt_caption.json').write_text(json.dumps({'train': [{'image_id': '../outside', 'caption': ''}]}))
            with self.assertRaises(ValueError):
                CTIQADataset(root)

    def test_caption_truncation(self):
        self.assertEqual(pre_caption('A, B; C', 2), 'a b')


if __name__ == '__main__':
    unittest.main()
