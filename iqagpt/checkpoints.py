"""Checkpoint normalization; inference rejects partially loaded models."""
import inspect
import torch
from models.ibot_vit import interpolate_pos_embed


def load_checkpoint(model, path, initialization=False):
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = True
    checkpoint = torch.load(path, **options)
    state = checkpoint.get('model', checkpoint)
    state = {(key[7:] if key.startswith('module.') else key).replace('visual_encoder.backbone.', 'visual_encoder.'): value
             for key, value in state.items() if isinstance(value, torch.Tensor)}
    if 'visual_encoder.pos_embed' in state:
        state['visual_encoder.pos_embed'] = interpolate_pos_embed(state['visual_encoder.pos_embed'], model.visual_encoder)
    expected = model.state_dict()
    mismatch = [key for key in state if key in expected and state[key].shape != expected[key].shape]
    if mismatch:
        raise ValueError('Checkpoint/model shape mismatch: ' + ', '.join(mismatch[:12]))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing and not initialization:
        raise ValueError('Incomplete inference checkpoint; missing keys: ' + ', '.join(missing[:12]))
    if initialization:
        for prefix in ['visual_encoder.', 'text_encoder.']:
            matched = sum(key.startswith(prefix) and key in expected for key in state)
            if not matched:
                raise ValueError('Initialization lacks ' + prefix)
    return {'missing_keys': missing, 'unexpected_keys': unexpected}
