"""Generate CT-IQA captions using a complete fine-tuned checkpoint."""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from iqagpt.checkpoints import load_checkpoint
from iqagpt.data import CTIQADataset
from models.model_generation import XVLModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--config', default='configs/iqagpt.yaml')
    p.add_argument('--annotations', default='iqagpt_caption.json')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--split', choices=['train', 'test'], default='test')
    p.add_argument('--limit', type=int)
    p.add_argument('--local-files-only', action='store_true')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(str(out))
    torch.set_num_threads(4)
    config = yaml.safe_load(Path(args.config).read_text())
    config['bert_config'] = str(Path(args.config).resolve().parent / Path(config['bert_config']).name)
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], local_files_only=args.local_files_only)
    model = XVLModel(config=config, tokenizer=tokenizer)
    print(json.dumps(load_checkpoint(model, args.checkpoint)))
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = CTIQADataset(args.data_root, args.split, args.annotations)
    if args.limit is not None:
        if args.limit < 1:
            p.error('--limit must be positive')
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    results = {}
    with torch.no_grad():
        for images, captions, ids in DataLoader(dataset, batch_size=1):
            tokens, _ = model(images.to(device), train=False)
            predicted = tokenizer.decode(tokens[0][0], skip_special_tokens=True).strip()
            results[ids[0]] = {'predicted': predicted, 'caption': captions[0]}
            print(ids[0], predicted, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
