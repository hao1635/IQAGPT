# Third-party notices

The root `LICENSE` preserves the MIT license from the archived Medical X-VL project, including its uta-smile copyright notice. The root code license does not override more specific notices in individual files.

| Component | Origin | Notice |
| --- | --- | --- |
| Medical X-VL adaptation | https://github.com/sangjoon-park/Medical_X-VL | Root MIT license |
| Captioning architecture | https://github.com/salesforce/ALBEF | Salesforce BSD-3-Clause header retained; `licenses/ALBEF-BSD-3-Clause.txt` |
| ViT implementation | https://github.com/bytedance/ibot | ByteDance header retained; `licenses/iBOT-Apache-2.0.txt`; source also credits DINO and timm |
| Modified BERT | Hugging Face Transformers | Google/Hugging Face/NVIDIA Apache-2.0 notice retained in `models/xbert.py`; `licenses/Apache-2.0.txt` |
| Beam search | Archived Medical X-VL `models/predictor.py` | File retained under the upstream project license; original comments retained |

`docs/SOURCE_PROVENANCE.json` records hashes of the original files and the mechanical extraction changes. External packages keep their own licenses. Third-party pretrained weights are obtained separately and are not covered by the code license. CT images and annotations have separate distribution terms described in the dataset card.
