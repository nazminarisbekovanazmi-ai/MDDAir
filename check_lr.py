import torch, glob, os, re

def _parse_avr(path):
    m = re.search(r'_avr([\d.]+)-', os.path.basename(path))
    return float(m.group(1)) if m else 0.0

candidates = [p for p in glob.glob('FT-checkpoint_*.pth.tar') if os.path.getsize(p) > 1_000_000]
best = max(candidates, key=_parse_avr)
print(f'Checkpoint: {os.path.basename(best)}')
ckpt = torch.load(best, map_location='cpu')
print(f"Saved LR:   {ckpt['optimizer']['param_groups'][0]['lr']}")
print(f"Saved epoch: {ckpt.get('epoch', '?')}")
