import os
import glob
import re
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from utils.dataset_utils import PromptTrainDataset
from net.model import ChannelShuffle_skip_textguaid
from utils.val_utils import AverageMeter
from utils.eval_utils import eval_all_3D
from utils.pytorch_ssim import SSIM
import clip
import time
start = time.time()

torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Train 3D')
parser.add_argument('--save_epoch', type=int, default=1, help='save model every N epochs')
parser.add_argument('--save_item', type=int, default=2000, help='save model every N iterations')
parser.add_argument('--init_epoch', type=int, default=1, help='initial epoch for fine-tuning')
parser.add_argument('--save_dir', type=str, default=None, help='checkpoint save directory (auto-set per ablation variant if not specified)')
parser.add_argument('--gpu', type=str, default="0", help='GPUs')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--pretrained_1', type=str, default='./', help='path to pretrained checkpoint')
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'])
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=20, help='number of DataLoader workers')
parser.add_argument('--denoise_dir', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\WEB", help='training clean images for denoising')
parser.add_argument('--denoise_dir2', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\BSD400", help='optional second clean image directory for denoising training')
parser.add_argument('--derain_dir', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\Rain\Rain200L_train\rain", help='training rainy images')
parser.add_argument('--dehaze_dir', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\Reside_haze\training_haze_input", help='training hazy images')
parser.add_argument('--denoise_path', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\BSD68")
parser.add_argument('--denoise_path2', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\Urban100\image_SRF_4")
parser.add_argument('--derain_path', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\Rain\Rain100L_test\rain_test")
parser.add_argument('--dehaze_path', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\hazy_test\hazy_outdoor")
parser.add_argument('--output_path', type=str, default="output/")
parser.add_argument('--psnr_max', type=float, default=0, help='override checkpoint psnr_max threshold (0 = always save)')
parser.add_argument('--reset_scheduler', action='store_true', help='ignore saved scheduler state and start fresh cosine decay from --lr')
parser.add_argument('--pretrain_estimator_epochs', type=int, default=0, help='pre-train only DegradationEstimator with BCE for N epochs before joint training')
parser.add_argument('--ablation', type=str, default='full',
                    choices=['full', 'no_deg_estimator', 'no_spatial_attn', 'no_film'],
                    help='ablation variant to train')
args = parser.parse_args()

if args.save_dir is None:
    _BASE = r'C:\Users\Nazmi\Desktop\실험\MDDAir'
    _ABLATION_DIRS = {
        'full':             'new_3-task_checkpoints',
        'no_deg_estimator': os.path.join('ablation', 'no_deg_est'),
        'no_film':          os.path.join('ablation', 'no_film'),
        'no_spatial_attn':  os.path.join('ablation', 'no_spatial_attn'),
    }
    args.save_dir = os.path.join(_BASE, _ABLATION_DIRS[args.ablation])

psnr_max = 10
last_good_ckpt = None
_nan_streak = 0
NAN_RECOVERY_STREAK = 10  


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
pin_memory = device == "cuda"


def _init_log(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('\t'.join([
                'epoch', 'step', 'trigger',
                'psnr_n15', 'ssim_n15', 'psnr_n25', 'ssim_n25', 'psnr_n50', 'ssim_n50',
                'psnr_rain', 'ssim_rain', 'psnr_haze', 'ssim_haze',
                'psnr_avr',  'ssim_avr',  'loss'
            ]) + '\n')


def _write_log(log_path, epoch, step, trigger,
               psnr_n15, ssim_n15, psnr_n25, ssim_n25, psnr_n50, ssim_n50,
               psnr_rain, ssim_rain, psnr_haze, ssim_haze,
               psnr_avr, ssim_avr, loss=None):
    loss_str = f'{loss:.5f}' if loss is not None else '-'
    with open(log_path, 'a') as f:
        f.write('\t'.join([
            str(epoch), str(step), trigger,
            f'{psnr_n15:.4f}', f'{ssim_n15:.4f}',
            f'{psnr_n25:.4f}', f'{ssim_n25:.4f}',
            f'{psnr_n50:.4f}', f'{ssim_n50:.4f}',
            f'{psnr_rain:.4f}', f'{ssim_rain:.4f}',
            f'{psnr_haze:.4f}', f'{ssim_haze:.4f}',
            f'{psnr_avr:.4f}',  f'{ssim_avr:.4f}',
            loss_str
        ]) + '\n')


ssim_loss_fn = SSIM(window_size=11)


def train(train_loader, model, optimizer, epoch, criterionL1, scaler, writer):
    loss_sum = 0
    l1_sum = 0
    ssim_sum = 0
    type_sum = 0
    losses = AverageMeter()
    psnr_tqdm = 10
    ssim_tqdm = 0.009
    loss_tqdm = 0.0
    model.train()
    global psnr_max, last_good_ckpt, _nan_streak

    loop_train = tqdm(train_loader, total=len(train_loader), leave=False, colour="magenta")
    for i, ([_, de_id], degrad_patch, clean_patch) in enumerate(loop_train):
        input_var = degrad_patch.to(device)
        target_var = clean_patch.to(device)
        img_id = de_id.tolist()

        true_type = torch.zeros(len(img_id), 7).to(device)
        for b, idx in enumerate(img_id):
            true_type[b, idx] = 1.0

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output, _, type_probs_pred = model(input_var)
            # rain (idx 3) appears 4x less than haze (18k vs 72k) — upweight to fix bias
            type_pos_w = torch.ones(7, device=input_var.device)
            type_pos_w[3] = 2.0  # rain vs haze: 4.0 was too aggressive, hurt haze detection
            type_loss = F.binary_cross_entropy_with_logits(type_probs_pred.float(), true_type, pos_weight=type_pos_w)
            l1_loss   = criterionL1(output, target_var)
            with torch.amp.autocast('cuda', enabled=False):
                ssim_loss = 1 - ssim_loss_fn(output.float(), target_var.float())
            # Start conservative (0.1, same convention as type_loss's weight) — measure
            # actual l1 vs ssim magnitudes after a few epochs before trusting this;
            # the 5D run showed an unweighted ssim term can dominate and cost PSNR.
            loss = l1_loss + 0.1 * ssim_loss + 0.1 * type_loss

        if not torch.isfinite(loss):
            _nan_streak += 1
            optimizer.zero_grad()
            print(f'  [warn] non-finite loss at epoch {epoch} item {i} (consecutive={_nan_streak}) '
                  f'l1={l1_loss.item():.5f} ssim={ssim_loss.item():.5f} type={type_loss.item():.5f}')
            try:
                with torch.autograd.detect_anomaly():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        out_dbg, _, tp_dbg = model(input_var)
                        tl_dbg = F.binary_cross_entropy_with_logits(tp_dbg.float(), true_type)
                        ll_dbg = criterionL1(out_dbg, target_var)
                        with torch.amp.autocast('cuda', enabled=False):
                            sl_dbg = 1 - ssim_loss_fn(out_dbg.float(), target_var.float())
                        (ll_dbg + 0.1 * sl_dbg + 0.1 * tl_dbg).backward()
            except RuntimeError as e:
                print(f'  [anomaly] {e}')
            optimizer.zero_grad()
            if _nan_streak >= NAN_RECOVERY_STREAK and last_good_ckpt is not None:
                print(f'  [recover] {_nan_streak} consecutive non-finite losses -> reloading last good checkpoint: {os.path.basename(last_good_ckpt)}')
                ckpt = torch.load(last_good_ckpt, map_location=device)
                model.load_state_dict(ckpt['state_dict'])
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                except (ValueError, KeyError):
                    pass
                _nan_streak = 0
            continue

        _nan_streak = 0
        loss_sum += loss.item()
        l1_sum   += l1_loss.item()
        ssim_sum += ssim_loss.item()
        type_sum += type_loss.item()
        losses.update(loss.item())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if (i % 10 == 0) and (i != 0):
            loss_tqdm = loss_sum / 10
            step = (epoch - 1) * len(train_loader) + i
            writer.add_scalar("loss/total", loss_sum / 10, step)
            writer.add_scalar("loss/l1",    l1_sum   / 10, step)
            writer.add_scalar("loss/ssim",  ssim_sum / 10, step)
            writer.add_scalar("loss/type",  type_sum / 10, step)
            loss_sum = l1_sum = ssim_sum = type_sum = 0.0

        if (i % args.save_item == 0) and (i != 0):
            m = eval_all_3D(model, args, device)
            psnr_n15,  ssim_n15  = m['psnr_n15'],   m['ssim_n15']
            psnr_n25,  ssim_n25  = m['psnr_n25'],   m['ssim_n25']
            psnr_n50,  ssim_n50  = m['psnr_n50'],   m['ssim_n50']
            psnr_rain, ssim_rain = m['psnr_rain'],  m['ssim_rain']
            psnr_haze, ssim_haze = m['psnr_haze'],  m['ssim_haze']
            psnr_avr,  ssim_avr  = m['psnr_avr'],   m['ssim_avr']
            psnr_tqdm = psnr_avr
            ssim_tqdm = ssim_avr
            _write_log(log_path, epoch, i, 'iter',
                       psnr_n15, ssim_n15, psnr_n25, ssim_n25, psnr_n50, ssim_n50,
                       psnr_rain, ssim_rain, psnr_haze, ssim_haze, psnr_avr, ssim_avr,
                       loss=loss_tqdm)
            ckpt_name = 'checkpoint_epoch_V1_{:0>4}_{}_p_n{:.2f}-{:.4f}_{:.2f}-{:.4f}_{:.2f}-{:.4f}_p_r{:.2f}-{:.4f}_p_h{:.2f}-{:.4f}avr{:.2f}-{:.4f}.pth.tar'.format(
                epoch, i // args.save_item,
                psnr_n15, ssim_n15, psnr_n25, ssim_n25, psnr_n50, ssim_n50,
                psnr_rain, ssim_rain, psnr_haze, ssim_haze, psnr_avr, ssim_avr)
            if psnr_avr > psnr_max - 0.0001:
                psnr_max = max(psnr_avr, psnr_max)
                last_good_ckpt = os.path.join(args.save_dir, ckpt_name)
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
                           last_good_ckpt)
            else:
                torch.save({}, os.path.join(args.save_dir, ckpt_name))

        loop_train.set_description(f'training->epoch:[{epoch}/{args.epochs}],item:[{i}/{len(train_loader)}]')
        loop_train.set_postfix(loss=loss_tqdm, psnr=f'{psnr_tqdm:.4f}', ssim=f'{ssim_tqdm:.4f}')

    return losses.avg


def pretrain_estimator(train_loader, model, scaler, num_epochs):
    """Stage 1: train only DegradationEstimator with BCE. Freezes all other params."""
    DE_NAMES = ['Noise-15', 'Noise-25', 'Noise-50', 'Rain', 'Haze', 'Blur', 'Low']

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('deg_estimator.')

    est_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    print('\n=== Stage 1: DegradationEstimator pre-training ===')
    for ep in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = [0] * 7
        counts  = [0] * 7

        loop = tqdm(train_loader, total=len(train_loader), leave=False, colour="cyan")
        for ([_, de_id], degrad_patch, _) in loop:
            input_var = degrad_patch.to(device)
            img_id = de_id.tolist()

            true_type = torch.zeros(len(img_id), 7, device=device)
            for b, idx in enumerate(img_id):
                true_type[b, idx] = 1.0

            est_optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, _, type_logits = model(input_var)
                type_pos_w = torch.ones(7, device=device)
                type_pos_w[3] = 2.0
                loss = F.binary_cross_entropy_with_logits(
                    type_logits.float(), true_type, pos_weight=type_pos_w)

            scaler.scale(loss).backward()
            scaler.unscale_(est_optimizer)
            torch.nn.utils.clip_grad_norm_(model.deg_estimator.parameters(), max_norm=1.0)
            scaler.step(est_optimizer)
            scaler.update()
            total_loss += loss.item()

            with torch.no_grad():
                preds = torch.sigmoid(type_logits.float()).argmax(dim=1).cpu().tolist()
                for b, (pred, true_idx) in enumerate(zip(preds, img_id)):
                    counts[true_idx] += 1
                    if pred == true_idx:
                        correct[true_idx] += 1

            loop.set_description(f'[EstPretrain] epoch {ep}/{num_epochs}')
            loop.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'\n[EstPretrain] Epoch {ep}/{num_epochs}  loss={avg_loss:.4f}')
        for i, name in enumerate(DE_NAMES):
            if counts[i] > 0:
                acc = 100.0 * correct[i] / counts[i]
                print(f'  {name:<10}: {correct[i]:>5}/{counts[i]:<5}  ({acc:.1f}%)')

    # unfreeze all parameters for joint training
    for param in model.parameters():
        param.requires_grad = True
    print('=== Stage 1 done — all parameters unfrozen for joint training ===\n')


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)}) | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"Using device: {device}")

    clip_model, _ = clip.load("ViT-B/32", device=device)
    for param in clip_model.parameters():
        param.requires_grad = False

    inputext = [
        "Gaussian noise with a standard deviation of 15",
        "Gaussian noise with a standard deviation of 25",
        "Gaussian noise with a standard deviation of 50",
        "Rain degradation with rain lines",
        "Hazy degradation with normal haze",
        "Blur degradation with motion blur",
        "Lowlight degradation",
    ]

    with torch.no_grad():
        _text_tokens_all = clip.tokenize(inputext).to(device)
        text_features_all = clip_model.encode_text(_text_tokens_all).to(dtype=torch.float32)
        text_features_all = F.normalize(text_features_all, dim=-1)

    ablation_kwargs = {
        'full':             {},
        'no_deg_estimator': {'use_deg_estimator': False},
        'no_spatial_attn':  {'use_spatial_attn': False},
        'no_film':          {'use_film': False},
    }[args.ablation]
    print(f"==> ablation variant: {args.ablation} {ablation_kwargs}")
    model = ChannelShuffle_skip_textguaid(text_features_all=text_features_all, **ablation_kwargs)
    criterionL1 = nn.L1Loss()
    model.to(device)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    log_path = os.path.join(args.save_dir, 'training_log.txt')
    _init_log(log_path)

    def _parse_avr(path):
        m = re.search(r'avr(\d+\.\d+)-', os.path.basename(path))
        return float(m.group(1)) if m else -1.0

    # Prefer latest.pth.tar (most recent epoch) for resuming; fall back to best named checkpoint
    _latest = os.path.join(args.save_dir, 'latest.pth.tar')
    candidates = [
        p for p in glob.glob(os.path.join(args.save_dir, 'checkpoint_*.pth.tar'))
        if os.path.getsize(p) > 1_000_000
    ]
    best_ckpt = _latest if os.path.isfile(_latest) else (max(candidates, key=_parse_avr) if candidates else None)

    if best_ckpt is not None:
        last_good_ckpt = best_ckpt
        print(f'==> resuming from: {os.path.basename(best_ckpt)}')
        model_info = torch.load(best_ckpt, map_location=device)
        model_state = model.state_dict()
        compatible_dict = {
            k: v for k, v in model_info['state_dict'].items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = set(model_info['state_dict']) - set(compatible_dict)
        if skipped:
            print(f"    skipping {len(skipped)} incompatible key(s) (architecture changed): {sorted(skipped)}")
        model.load_state_dict(compatible_dict, strict=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        try:
            optimizer.load_state_dict(model_info['optimizer'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            if args.reset_scheduler:
                print('    reset_scheduler=True — starting fresh cosine decay from lr={}'.format(args.lr))
            else:
                scheduler.load_state_dict(model_info['scheduler'])
        except (ValueError, KeyError):
            print('    optimizer/scheduler not restored (architecture changed) — starting fresh')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        # always enforce --lr after loading optimizer state
        for g in optimizer.param_groups:
            g['lr'] = args.lr
        cur_epoch = model_info['epoch']
        # latest.pth.tar has no PSNR in name → _parse_avr returns -1 → psnr_max stays 0
        psnr_max = max(_parse_avr(best_ckpt), 0.0)
        if args.psnr_max is not None:
            psnr_max = args.psnr_max
        print(f'    epoch={cur_epoch}, psnr_max={psnr_max:.2f}')
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        cur_epoch = args.init_epoch

    if args.pretrained_1 and os.path.isfile(args.pretrained_1):
        print("=> loading pretrained '{}'".format(args.pretrained_1))
        model_pretrained = torch.load(args.pretrained_1, map_location=device)
        model_state = model.state_dict()
        pretrained_dict = {
            k: v for k, v in model_pretrained['state_dict'].items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = set(model_pretrained['state_dict']) - set(pretrained_dict)
        if skipped:
            print(f"    skipping {len(skipped)} incompatible key(s) (architecture changed): {sorted(skipped)}")
        model.load_state_dict(pretrained_dict, strict=False)

    train_dataset = PromptTrainDataset(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=True, persistent_workers=args.num_workers > 0)

    writer = SummaryWriter("./logs_train")
    scaler = torch.amp.GradScaler('cuda')

    print('load dataset ok')
    print(f"  batch_size={args.batch_size}  patch_size={args.patch_size}  num_workers={args.num_workers}  epochs={args.epochs}")

    if args.pretrain_estimator_epochs > 0:
        pretrain_estimator(train_loader, model, scaler, args.pretrain_estimator_epochs)

    m = eval_all_3D(model, args, device)
    psnr_n15,  ssim_n15  = m['psnr_n15'],   m['ssim_n15']
    psnr_n25,  ssim_n25  = m['psnr_n25'],   m['ssim_n25']
    psnr_n50,  ssim_n50  = m['psnr_n50'],   m['ssim_n50']
    psnr_rain, ssim_rain = m['psnr_rain'],  m['ssim_rain']
    psnr_haze, ssim_haze = m['psnr_haze'],  m['ssim_haze']
    psnr_avr,  ssim_avr  = m['psnr_avr'],   m['ssim_avr']
    print('initial test: psnr_noise:{:.4f}-{:.4f}_{:.4f}-{:.4f}_{:.4f}-{:.4f} rain:{:.4f}-{:.4f} haze:{:.4f}-{:.4f} avr{:.4f}-{:.4f}'
          .format(psnr_n15, ssim_n15, psnr_n25, ssim_n25, psnr_n50, ssim_n50,
                  psnr_rain, ssim_rain, psnr_haze, ssim_haze, psnr_avr, ssim_avr))
    _write_log(log_path, 0, 0, 'init',
               psnr_n15, ssim_n15, psnr_n25, ssim_n25, psnr_n50, ssim_n50,
               psnr_rain, ssim_rain, psnr_haze, ssim_haze, psnr_avr, ssim_avr)

    for epoch in range(cur_epoch, args.epochs + 1):
        loss = train(train_loader, model, optimizer, epoch, criterionL1, scaler, writer)
        scheduler.step()

        if epoch % args.save_epoch == 0:
            m = eval_all_3D(model, args, device)
            psnr_n15,  ssim_n15  = m['psnr_n15'],   m['ssim_n15']
            psnr_n25,  ssim_n25  = m['psnr_n25'],   m['ssim_n25']
            psnr_n50,  ssim_n50  = m['psnr_n50'],   m['ssim_n50']
            psnr_rain, ssim_rain = m['psnr_rain'],  m['ssim_rain']
            psnr_haze, ssim_haze = m['psnr_haze'],  m['ssim_haze']
            psnr_avr,  ssim_avr  = m['psnr_avr'],   m['ssim_avr']
            _write_log(log_path, epoch, '-', 'epoch',
                       psnr_n15, ssim_n15, psnr_n25, ssim_n25, psnr_n50, ssim_n50,
                       psnr_rain, ssim_rain, psnr_haze, ssim_haze, psnr_avr, ssim_avr,
                       loss=loss)
            ckpt_name = 'checkpoint_epoch_V1_{:0>4}_p_n{:.2f}-{:.4f}_{:.2f}-{:.4f}_{:.2f}-{:.4f}_p_r{:.2f}-{:.4f}_p_h{:.2f}-{:.4f}avr{:.2f}-{:.4f}.pth.tar'.format(
                epoch, psnr_n15, ssim_n15, psnr_n25, ssim_n25, psnr_n50, ssim_n50,
                psnr_rain, ssim_rain, psnr_haze, ssim_haze, psnr_avr, ssim_avr)
            # Save full checkpoint every epoch (overwrite latest + unique named file)
            full_ckpt = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(full_ckpt, os.path.join(args.save_dir, 'latest.pth.tar'))
            torch.save(full_ckpt, os.path.join(args.save_dir, ckpt_name))
            if psnr_avr > psnr_max - 0.01:
                psnr_max = max(psnr_avr, psnr_max)
                last_good_ckpt = os.path.join(args.save_dir, ckpt_name)

        print('Epoch [{0}]\tlr: {lr:.6f}\tLoss: {loss:.5f}'.format(
            epoch, lr=optimizer.param_groups[-1]['lr'], loss=loss))
        # ... run one epoch ...
        print(f"Epoch time: {time.time() - start:.1f}s")

    writer.close()
