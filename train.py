"""Single-device caption-model training; all outputs go to --output-dir."""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from iqagpt.checkpoints import load_checkpoint
from iqagpt.data import CTIQADataset
from models.model_generation import XVLModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--config', default='configs/iqagpt.yaml')
    p.add_argument('--annotations', default='iqagpt_caption.json')
    p.add_argument('--checkpoint', required=True, help='Medical X-VL initialization checkpoint')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--local-files-only', action='store_true')
    args = p.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    config['bert_config'] = str(Path(args.config).resolve().parent / Path(config['bert_config']).name)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'checkpoint.pth').exists():
        raise FileExistsError('Choose a new output directory; a checkpoint already exists')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer'], local_files_only=args.local_files_only)
    model = XVLModel(config=config, tokenizer=tokenizer)
    print(json.dumps(load_checkpoint(model, args.checkpoint, initialization=True)))
    device = torch.device(args.device)
    model.to(device)
    dataset = CTIQADataset(args.data_root, 'train', args.annotations, augment=True)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=args.workers)
    if not len(loader):
        raise ValueError('Empty training dataset')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    total = config['epochs'] * len(loader)
    warmup = config['warmup_epochs'] * len(loader)
    step = 0
    (out / 'config.yaml').write_text(yaml.safe_dump(config))
    for epoch in range(config['epochs']):
        model.train()
        loss_sum = 0.0
        for images, captions, _ in loader:
            if step < warmup:
                fraction = step / max(1, warmup)
                lr = config['warmup_start_lr'] + fraction * (config['learning_rate'] - config['warmup_start_lr'])
            else:
                fraction = (step - warmup) / max(1, total - warmup - 1)
                lr = config['min_lr'] + 0.5 * (config['learning_rate'] - config['min_lr']) * (1 + math.cos(math.pi * fraction))
            for group in optimizer.param_groups:
                group['lr'] = lr
            tokens = tokenizer(list(captions), padding='longest', truncation=True, max_length=180, return_tensors='pt').to(device)
            optimizer.zero_grad()
            loss = model(images.to(device), tokens, train=True)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.visual_encoder.parameters(), 3.0)
            optimizer.step()
            loss_sum += loss.item()
            step += 1
        stats = {'epoch': epoch, 'loss': loss_sum / len(loader), 'lr': lr}
        print(json.dumps(stats), flush=True)
        with (out / 'log.jsonl').open('a') as stream:
            stream.write(json.dumps(stats) + '\n')
        temporary = out / 'checkpoint.tmp'
        torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch}, temporary)
        temporary.replace(out / 'checkpoint.pth')


if __name__ == '__main__':
    main()
