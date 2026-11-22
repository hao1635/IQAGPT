# Initial code release

Prepared on 2026-11-22 from the author's archived research project.

## Included

- The ViT-S/16 image encoder, modified BERT cross-modal captioning network, and original beam-search implementation.
- Portable single-device training/inference scripts with configurable dataset, checkpoint and output paths.
- CT-IQA loading, input validation, a deterministic archive exporter and data-loader tests.
- Paper-based hyperparameters and retained upstream license notices.

## Changes from the research workspace

The core caption-model constructor, non-distilled forward pass and beam search are extracted from the archived code. Unused task heads, debugger imports, unused LoRA imports and unrelated VQA/retrieval dependencies are omitted. Checkpoint key normalization and positional-embedding interpolation happen before loading. Inference rejects checkpoints with missing model parameters. The standard BertTokenizer reads the original CXR-BERT vocabulary; the archived custom tokenizer is an unchanged BertTokenizer subclass, so remote Python code is unnecessary. Data paths are command-line arguments. No server paths, credentials, notebook outputs, patient directories or internal training logs are included.

The training entry point is new release tooling. It uses explicit per-step warmup/cosine scheduling, saves model weights each epoch, and avoids automatic test-set evaluation during training. The original experimental scheduler and checkpoint-selection behavior have not been reproduced exactly. This tooling should not be mistaken for a recovered final experiment script.

## Validation

Checked in the existing Python 3.8.13 research environment with torch 1.10.1+cu102, torchvision 0.11.2+cu102, transformers 4.28.0, timm 0.6.12, numpy 1.24.2, and PyYAML 6.0. No GPU was used for release checks.

- The standard BERT tokenizer loaded offline, the Medical X-VL initialization loaded all model parameters, and a training-mode forward pass returned a finite loss. No optimizer update or retraining was performed.
- Three data-loader tests passed: clipping/channel replication and augmentation, invalid-ID rejection, and caption preprocessing.
- A historical caption-model checkpoint loaded with no missing or unexpected model keys after image-position interpolation.
- A short CPU beam-generation run on one CT-IQA sample completed. This establishes a smoke run, not inference accuracy.
- All 800 train and 200 test arrays passed dtype, shape, finiteness and annotation-reference checks.

A clean installation, full training run, full inference/evaluation pass and GPU run have not been validated for this release.

## Unresolved publication correspondence

1. The exact final-publication checkpoint has not been identified.
2. Two score annotation versions differ on 115 test records. Both are preserved without merging.
3. Existing notebook phrase matching does not fully recover the published table and contains overlapping lesion phrases. A new score evaluation protocol is not silently substituted here.
4. The extracted current model has 73,101,882 parameters, whereas the paper reports approximately 210M. This is an unresolved discrepancy between the archived model/configuration and the paper, not a claim about the final model.
5. ViT-C, ViT-R, CLIP-IQA, MiniGPT-4 and ChatGPT-interface experiments have not been consolidated into this initial captioning-code release. They remain distinct experiments in the author's archive.
6. The author confirmed dataset redistribution authorization on 2026-11-22. The specific authorization document or applicable dataset-license identifier has not been included; dataset rights remain separate from the MIT code license.
