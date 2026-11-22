# CT-IQA dataset card

## Contents and origin

CT-IQA was constructed for the 2024 IQAGPT study from abdominal CT scans described in the paper as originating from the 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge. Selected NDCT/LDCT pairs were supplemented with five MAP-NN denoising levels and one RED-CNN result. Some lesions were simulated during dataset construction, as described in the paper.

There are 800 training and 200 test arrays. All 1,000 current files were checked for finite numeric content, shape (512, 512), and float32 dtype. Each quality level has 100 training and 25 test images. No DICOM files, raw acquisition headers, patient directories, or original clinical records are included in the prepared archive. Numeric arrays do not by themselves establish de-identification of image pixels.

The paper describes an eight-patient/two-patient split. The current release checks image-ID separation; the complete mapping back to source patients has not been independently reconstructed.

## File conventions

An image ID such as `test-00705` contains a split prefix, a three-digit base slice index, and a two-digit quality index. Quality indices follow the archived notebook convention: 00 NDCT; 01 LDCT; 02–06 MAP-NN denoising levels 1–5; 07 RED-CNN.

`iqagpt_caption.json` has `train` and `test` arrays of `{ "image_id": "...", "caption": "..." }` records. `labels.json` has the same split and ID order, with `{ "image_id": "...", "labels": [a, b, c, d] }`. The label order is noise/fidelity, small structures, lesions, diagnostic confidence. For attributes a, b and d, stored values 0–3 correspond to displayed scores 1–4. For lesions, 0 means no lesion and 1–4 are lesion-conspicuity categories.

The archived loader clips the existing arrays to [0, 1]; do not apply HU normalization a second time. Duplicate PNG renderings are omitted from the prepared download because the numerical image arrays are the model inputs.

## Annotation versions

The prepared archive preserves current input files exactly. It includes `iqagpt_caption.json` and `labels.json`, plus `annotation_versions/iqagpt_labels.json` and `annotation_versions/report3.json`. The two score files agree on every training record, but 115/200 test records differ in at least one field. The provenance and final-publication status of those changes are unresolved. No claim is made that swapping files recovers the published metrics.

## Access and terms

The author confirmed on 2026-11-22 that redistribution of this prepared CT-IQA dataset is covered by existing authorization or an applicable open license. The specific authorization document or license identifier has not been supplied with this release; contact the authors for the applicable reuse terms.

The repository's MIT code license does not apply to source CT images, image derivatives, annotations, or third-party checkpoints. This dataset card does not assign a new blanket data license. Source-data attribution and any applicable conditions remain separate from the code license.

## Attribution

Cite the IQAGPT paper (DOI: 10.1186/s42492-024-00171-w), the original AAPM/Mayo dataset and relevant source-data publications. The historic challenge agreement requests acknowledgment of Cynthia McCollough, Mayo Clinic, AAPM, and NIBIB grants EB017095 and EB017185. References to those organizations describe dataset provenance, not endorsement.

## Intended use and limitations

This is a small research dataset for subjective CT quality assessment, focused on abdominal images and the denoising methods used in the paper. It is not validated for clinical deployment, other anatomies, or use as diagnostic ground truth. Lesion scores include a no-lesion category and require explicit handling in quantitative evaluation.
