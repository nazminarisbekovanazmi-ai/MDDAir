import os, glob, re
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from utils.dataset_utils import PromptTrainDataset5D
from utils.eval_utils import eval_all
from net.model import ChannelShuffle_skip_textguaid
from utils.val_utils import AverageMeter
from utils.pytorch_ssim import SSIM
import clip


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--save_epoch', type=int, default=1,
                    help='save model per every N epochs')
parser.add_argument('--save_item', type=int, default=3000,
                    help='save model per every N item')
parser.add_argument('--init_epoch', type=int, default=1,
                    help='if finetune model, set the initial epoch')
parser.add_argument('--save_dir', type=str, default=r'C:\Users\Nazmi\Desktop\실험\MDDAir\new_5-deg_checkpoints',
                     help='save parameter dir')

parser.add_argument('--gpu', type=str, default="0", help='GPUs')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--pretrained_1', type=str, default='./',
        help='training loss')

parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int, default=20, help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'lowlight'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--ablation', type=str, default='full',
                    choices=['full', 'no_deg_estimator', 'no_spatial_attn', 'no_film'],
                    help='which architectural component to ablate for this run.')

parser.add_argument('--patch_size', type=int, default=192, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# train datasets path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\WEB",
                    help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir2', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\BSD400",
                    help='optional second clean image directory for denoising training.')
parser.add_argument('--derain_dir', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\Rain\Rain200L_train\rain",
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\Reside_haze\training_haze_input",
                    help='where training images of dehazing saves.')
parser.add_argument('--deblur_path', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\Gopro\test", help='path to GoPro test set; dataset_utils appends /blur/ for input and /sharp/ for GT.')
parser.add_argument('--deblur_dir', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\Gopro\train\blur", help='path to GoPro training blur images; GT resolved from ../sharp/')
parser.add_argument('--lowlight_path', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\lol_dataset\eval15", help='path to LOL test set (eval15); dataset_utils appends /low/ and /high/')
parser.add_argument('--lowlight_dir', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\lol_dataset\our485\low", help='path to LOL training low-light images; GT resolved from ../high/')

parser.add_argument('--denoise_path', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\BSD68", help='save path of test noisy images')
parser.add_argument('--denoise_path2', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\Urban100\image_SRF_4", help='optional second test set for denoising (Urban100)')
parser.add_argument('--derain_path', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\Rain\Rain100L_test\rain_test", help='save path of test raining images')
parser.add_argument('--dehaze_path', type=str, default=r"C:\Users\Nazmi\Desktop\실험\Degradation_datasets\hazy_test\hazy_outdoor", help='save path of test hazy images')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_name', type=str, default="model.ckpt", help='checkpoint save path')

args = parser.parse_args()

psnr_max = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
pin_memory = device == "cuda"

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

# severity_lookup = [15/75, 25/75, 50/75, 1.0, 1.0, 1.0, 1.0]

with torch.no_grad():
    _text_tokens_all = clip.tokenize(inputext).to(device)
    text_features_all = clip_model.encode_text(_text_tokens_all).to(dtype=torch.float32)
    text_features_all = F.normalize(text_features_all, dim=-1)


ssim_loss_fn = SSIM(window_size=11)


_LOG_COLUMNS = [
    'epoch', 'step', 'trigger',
    'psnr_noise', 'ssim_noise', 'psnr_rain', 'ssim_rain', 'psnr_haze', 'ssim_haze',
    'psnr_blur', 'ssim_blur', 'psnr_lol', 'ssim_lol',
    'psnr_avr', 'ssim_avr', 'loss', 'type_loss',
]


def _init_log(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('\t'.join(_LOG_COLUMNS) + '\n')


def _write_log(log_path, epoch, step, trigger, m=None, loss=None, type_loss=None):
    """m is the dict returned by eval_all(), or None if no eval ran this step."""
    def fmt(key):
        return f'{m[key]:.4f}' if m is not None else '-1.0000'
    loss_str = f'{loss:.5f}' if loss is not None else '-'
    type_loss_str = f'{type_loss:.5f}' if type_loss is not None else '-'
    with open(log_path, 'a') as f:
        f.write('\t'.join([
            str(epoch), str(step), trigger,
            fmt('psnr_noise'), fmt('ssim_noise'),
            fmt('psnr_rain'),  fmt('ssim_rain'),
            fmt('psnr_haze'),  fmt('ssim_haze'),
            fmt('psnr_blur'),  fmt('ssim_blur'),
            fmt('psnr_lol'),   fmt('ssim_lol'),
            fmt('psnr_avr'),   fmt('ssim_avr'),
            loss_str, type_loss_str,
        ]) + '\n')


def train(train_loader, model, optimizer, epoch, criterionL1, scaler, writer, log_path):
    loss_sum = 0
    l1_sum = 0
    ssim_sum = 0
    type_sum = 0
    losses    = AverageMeter()
    type_losses = AverageMeter()
    psnr_tqdm = 10
    ssim_tqdm = 0.009
    loss_tqdm = 0.0
    type_loss_tqdm = 0.0

    model.train()
    global psnr_max

    global_step = (epoch - 1) * len(train_loader)

    loop_train = tqdm(train_loader, total=len(train_loader), leave=False)
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
            type_pos_w = torch.ones(7, device=input_var.device)
            type_pos_w[3] = 2.0  # rain vs haze: 4.0 was too aggressive, hurt haze detection
            type_loss = F.binary_cross_entropy_with_logits(type_probs_pred.float(), true_type, pos_weight=type_pos_w)
            l1_loss   = criterionL1(output, target_var)
            with torch.amp.autocast('cuda', enabled=False):
                ssim_loss = 1 - ssim_loss_fn(output.float(), target_var.float())
            loss = l1_loss + 0.1 * ssim_loss + 0.1 * type_loss

        if not torch.isfinite(loss):
            # A non-finite forward pass must not reach the real backward — it
            # would poison the optimizer's running moments permanently instead
            # of just spoiling this one batch. Re-run *only this bad batch*
            # under detect_anomaly() to pinpoint the offending forward op,
            # rather than paying that overhead on every iteration.
            print(f"    skipping batch {i}: non-finite loss (l1={l1_loss.item()}, ssim={ssim_loss.item()}, type={type_loss.item()})")
            try:
                with torch.autograd.detect_anomaly():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        output_dbg, _, type_probs_dbg = model(input_var)
                        l1_dbg = criterionL1(output_dbg, target_var)
                        with torch.amp.autocast('cuda', enabled=False):
                            ssim_dbg = 1 - ssim_loss_fn(output_dbg.float(), target_var.float())
                        type_dbg = F.binary_cross_entropy_with_logits(type_probs_dbg.float(), true_type)
                        (l1_dbg + 0.1 * ssim_dbg + 0.1 * type_dbg).backward()
            except RuntimeError as e:
                print(f"    [anomaly] {e}")
            optimizer.zero_grad()
            continue

        loss_sum += loss.item()
        l1_sum   += l1_loss.item()
        ssim_sum += ssim_loss.item()
        type_sum += type_loss.item()
        losses.update(loss.item())
        type_losses.update(type_loss.item())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if (i % 10 == 0) and (i != 0):
            loss_tqdm      = loss_sum / 10
            type_loss_tqdm = type_sum / 10
            step = global_step + i
            writer.add_scalar("loss/total", loss_sum / 10, step)
            writer.add_scalar("loss/l1",    l1_sum   / 10, step)
            writer.add_scalar("loss/ssim",  ssim_sum / 10, step)
            writer.add_scalar("loss/type",  type_sum / 10, step)
            loss_sum = l1_sum = ssim_sum = type_sum = 0.0

        if (i % args.save_item == 0) and (i != 0):
            m = eval_all(model, args, device)
            psnr_noise, ssim_noise = m['psnr_noise'], m['ssim_noise']
            psnr_rain,  ssim_rain  = m['psnr_rain'],  m['ssim_rain']
            psnr_haze,  ssim_haze  = m['psnr_haze'],  m['ssim_haze']
            psnr_blur,  ssim_blur  = m['psnr_blur'],  m['ssim_blur']
            psnr_lol,   ssim_lol   = m['psnr_lol'],   m['ssim_lol']
            psnr_avr,   ssim_avr   = m['psnr_avr'],   m['ssim_avr']
            psnr_tqdm = psnr_avr
            ssim_tqdm = ssim_avr
            _write_log(log_path, epoch, i, 'iter', m=m, loss=loss_tqdm, type_loss=type_loss_tqdm)
            if psnr_avr > psnr_max - 0.00001:
                psnr_max = max(psnr_avr, psnr_max)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(args.save_dir,
                                 'FT-checkpoint_epoch_V1_{:0>4}_{}_pn{:.2f}-{:.4f}_pr{:.2f}-{:.4f}_ph{:.2f}-{:.4f}_pb{:.2f}-{:.4f}_pl{:.2f}-{:.4f}_avr{:.2f}-{:.4f}.pth.tar'.format(
                                     epoch, i // args.save_item, psnr_noise, ssim_noise, psnr_rain, ssim_rain, psnr_haze, ssim_haze, psnr_blur, ssim_blur, psnr_lol, ssim_lol, psnr_avr, ssim_avr)))
            else:
                torch.save({}, os.path.join(args.save_dir,
                                            'FT-checkpoint_epoch_V1_{:0>4}_{}_pn{:.2f}-{:.4f}_pr{:.2f}-{:.4f}_ph{:.2f}-{:.4f}_pb{:.2f}-{:.4f}_pl{:.2f}-{:.4f}_avr{:.2f}-{:.4f}.pth.tar'.format(
                                                epoch, i // args.save_item, psnr_noise, ssim_noise, psnr_rain, ssim_rain, psnr_haze, ssim_haze, psnr_blur, ssim_blur, psnr_lol, ssim_lol, psnr_avr, ssim_avr)))

        loop_train.set_description(f'training->epoch:[{epoch}/{args.epochs}],item:[{i}/{len(train_loader)}]')
        loop_train.set_postfix(loss=loss_tqdm, tloss=f'{type_loss_tqdm:.4f}', psnr=f'{psnr_tqdm:.4f}', ssim=f'{ssim_tqdm:.4f}')

    return losses.avg, type_losses.avg


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("Using device:", device)
    np.random.seed(0)
    torch.manual_seed(0)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=min(args.epochs, 100))
    cur_epoch = args.init_epoch

    def _parse_avr(path):
        m = re.search(r'_avr([\d.]+)-', os.path.basename(path))
        return float(m.group(1)) if m else 0.0

    ft_candidates = [
        p for p in glob.glob(os.path.join(args.save_dir, 'FT-checkpoint_*.pth.tar'))
        if os.path.getsize(p) > 1_000_000
    ]
    best_ckpt = max(ft_candidates, key=_parse_avr) if ft_candidates else None

    if best_ckpt is not None:
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        try:
            optimizer.load_state_dict(model_info['optimizer'])
            scheduler.load_state_dict(model_info['scheduler'])
            cur_epoch = model_info['epoch']
            # force-override LR so --lr argument takes effect even when resuming
            for g in optimizer.param_groups:
                g['lr'] = args.lr
        except (ValueError, KeyError):
            print('    optimizer/scheduler not restored — starting fresh')
            cur_epoch = args.init_epoch
    elif args.pretrained_1 and os.path.isfile(args.pretrained_1):
        print(f"=> loading pretrained '{args.pretrained_1}'")
        model_pretrained = torch.load(args.pretrained_1, map_location=device)
        pretrained_dict = {k: v for k, v in model_pretrained['state_dict'].items() if k in model.state_dict()}
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        print("=> no checkpoint found, starting from scratch")

    train_dataset = PromptTrainDataset5D(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=True, persistent_workers=args.num_workers > 0)

    writer = SummaryWriter("./logs_train")
    scaler = torch.amp.GradScaler('cuda')

    print('load dataset ok')
    print(f"  batch_size={args.batch_size}  patch_size={args.patch_size}  num_workers={args.num_workers}  epochs={args.epochs}")

    log_path = os.path.join(args.save_dir, 'training_log.txt')
    _init_log(log_path)

    for epoch in range(cur_epoch, args.epochs + 1):
        loss, type_loss_avg = train(train_loader, model, optimizer, epoch, criterionL1, scaler, writer, log_path)
        scheduler.step()
        m = None
        psnr_avr, ssim_avr = -1.0, -1.0
        if epoch % args.save_epoch == 0:
            m = eval_all(model, args, device)
            psnr_noise, ssim_noise = m['psnr_noise'], m['ssim_noise']
            psnr_rain,  ssim_rain  = m['psnr_rain'],  m['ssim_rain']
            psnr_haze,  ssim_haze  = m['psnr_haze'],  m['ssim_haze']
            psnr_blur,  ssim_blur  = m['psnr_blur'],  m['ssim_blur']
            psnr_lol,   ssim_lol   = m['psnr_lol'],   m['ssim_lol']
            psnr_avr,   ssim_avr   = m['psnr_avr'],   m['ssim_avr']
            # Always overwrite latest.pth.tar so any interruption still has the most recent epoch
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
                os.path.join(args.save_dir, 'latest.pth.tar'))
            if psnr_avr > psnr_max - 0.001:
                psnr_max = max(psnr_avr, psnr_max)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(args.save_dir,
                                 'FT-checkpoint_epoch_V1_{:0>4}__pn{:.2f}-{:.4f}_pr{:.2f}-{:.4f}_ph{:.2f}-{:.4f}_pb{:.2f}-{:.4f}_pl{:.2f}-{:.4f}_avr{:.2f}-{:.4f}.pth.tar'.format(
                                     epoch, psnr_noise, ssim_noise, psnr_rain, ssim_rain, psnr_haze, ssim_haze, psnr_blur, ssim_blur, psnr_lol, ssim_lol, psnr_avr, ssim_avr)))
            else:
                torch.save({}, os.path.join(args.save_dir,
                                            'FT-checkpoint_epoch_V1_{:0>4}__pn{:.2f}-{:.4f}_pr{:.2f}-{:.4f}_ph{:.2f}-{:.4f}_pb{:.2f}-{:.4f}_pl{:.2f}-{:.4f}_avr{:.2f}-{:.4f}.pth.tar'.format(
                                                epoch, psnr_noise, ssim_noise, psnr_rain, ssim_rain, psnr_haze, ssim_haze, psnr_blur, ssim_blur, psnr_lol, ssim_lol, psnr_avr, ssim_avr)))

        _write_log(log_path, epoch, '-', 'epoch', m=m, loss=loss, type_loss=type_loss_avg)

        print('Epoch [{0}]\t'
              'lr: {lr:.6f}\t'
              'Loss: {loss:.5f}\t'
              'Type Loss: {tloss:.5f}'
              .format(epoch, lr=optimizer.param_groups[-1]['lr'],
                      loss=loss, tloss=type_loss_avg))

    writer.close()
