# IQAGPT

Code associated with **IQAGPT: computed tomography image quality assessment with vision-language and ChatGPT models** (Chen et al., 2024).

[Paper](https://doi.org/10.1186/s42492-024-00171-w) · [Dataset description](docs/DATASET_CARD.md) · [Release notes](docs/RELEASE_NOTES.md)

IQAGPT first generates a structured description of CT image quality with a vision-language captioning model. The description can then be converted into quality scores or used as context for a language model to write an image-quality report. The four attributes are image noise and structural fidelity, visibility of small structures, lesion conspicuity, and diagnostic confidence.

## Release status

This initial release extracts the captioning model from the archived Medical X-VL research project and adds portable data loading, training and inference entry points. It does not include a hosted ChatGPT service or require an OpenAI API key.

The core model and a historical checkpoint were smoke-tested on CPU. The exact checkpoint and annotation version behind the final published tables are still being reconciled. This release therefore makes no claim of reproducing those tables. See [release notes](docs/RELEASE_NOTES.md) for the specific differences and validation scope.

## Installation

Use Python 3.8 or 3.9. Install PyTorch and torchvision for your platform first. The existing environment used for the initial checks contains PyTorch 1.10.1 and torchvision 0.11.2; its full dependency versions are documented in the release notes.

```bash
git clone https://github.com/hao1635/IQAGPT.git
cd IQAGPT
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the commands from the repository root. Initial tokenizer loading uses the public Hugging Face model `microsoft/BiomedVLP-CXR-BERT-specialized`; use `--local-files-only` once its tokenizer has been cached. No remote Python code is requested by these entry points.

## CT-IQA dataset

CT-IQA contains 1,000 abdominal CT slices: **800 training images and 200 test images**. Each split contains eight quality levels: NDCT, LDCT, five MAP-NN denoising levels, and RED-CNN. Images are 512 × 512 float32 NumPy arrays. The released image representation is already windowed and normalized; the loader clips it to [0, 1] and repeats it across three channels.

The author has confirmed redistribution authorization for the prepared dataset. The Google Drive download link will be added once upload verification is complete. Dataset reuse terms are separate from the MIT code license; see the [dataset card](docs/DATASET_CARD.md).

```text
CT-IQA/
  train/npy/train-00000.npy
  test/npy/test-00000.npy
  iqagpt_caption.json
  labels.json
  annotation_versions/
    iqagpt_labels.json
    report3.json
  checksums.sha256
  DATASET_CARD.md
```

```bash
python scripts/validate_dataset.py --data-root /path/to/CT-IQA
```

`labels.json` preserves the scores read by the archived image-only loader. `annotation_versions/iqagpt_labels.json` preserves an alternate score file: its training labels agree with `labels.json`, while 115 of the 200 test records differ in at least one score. Neither is silently substituted for the other or asserted to be the final publication labels.

## Checkpoints

The original model was initialized from Medical X-VL. Obtain its abdominal VLP checkpoint through the [upstream project](https://github.com/sangjoon-park/Medical_X-VL) or its authors and save it locally. Initialization weights are distinct from CT-IQA fine-tuned weights. Third-party weights are not redistributed in this code repository.

A confirmed final IQAGPT checkpoint is not yet included. Inference requires a compatible, complete fine-tuned checkpoint; the loader checks shapes and rejects missing model parameters. Merely renaming the Medical X-VL initialization file is not sufficient for inference.

## Training

```bash
python train.py \
  --data-root /path/to/CT-IQA \
  --checkpoint /path/to/VLP_abdomen.pth \
  --config configs/iqagpt.yaml \
  --output-dir runs/iqagpt \
  --device cuda:0
```

The provided configuration follows the published training hyperparameters: 50 epochs, batch size 8, AdamW, initial learning rate 2e-4, weight decay 0.02, two warmup epochs from 1e-5, and cosine decay to 1e-6. This configuration was reconstructed from the paper; it is not a recovered configuration for a confirmed final checkpoint. The portable training loop saves a checkpoint each epoch and does not select checkpoints on test data. Full retraining has not been performed for this release.

## Caption inference

```bash
python infer.py \
  --data-root /path/to/CT-IQA \
  --checkpoint /path/to/finetuned_checkpoint.pth \
  --output runs/predictions.json \
  --device cuda:0
```

Use `--device cpu --limit 1` for a small smoke run. Outputs map each image ID to its predicted and reference captions. Converting captions into scores should use an explicitly chosen annotation version and disambiguate overlapping lesion-description phrases. Historical notebook substring matching is not treated as a verified evaluation implementation here.

## Citation

```bibtex
@article{chen2024iqagpt,
  title={IQAGPT: computed tomography image quality assessment with vision-language and ChatGPT models},
  author={Chen, Zhihao and Hu, Bin and Niu, Chuang and Chen, Tao and Li, Yuxin and Shan, Hongming and Wang, Ge},
  journal={Visual Computing for Industry, Biomedicine, and Art},
  volume={7},
  pages={20},
  year={2024},
  doi={10.1186/s42492-024-00171-w}
}
```

## Acknowledgments and licenses

The research code builds on [Medical X-VL](https://github.com/sangjoon-park/Medical_X-VL), [ALBEF](https://github.com/salesforce/ALBEF), [iBOT](https://github.com/bytedance/ibot), and Hugging Face Transformers. The upstream MIT license and file-level notices are preserved; additional license texts and attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset and checkpoint rights are separate from the code license.
