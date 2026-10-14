"""
Phase 2: Train CompletionFormer FROM SCRATCH on AURORA weather-corrupted data,
mirroring the Sparse2Dense (self-supervised-depth-completion) modality recipe:
  - data root: AuroraKITTI/data_try, KITTI-DC layout, glob-discovered
  - inputs: RGB + sparse velodyne (px/256 -> metres); GT (px/256 -> metres)
  - bottom-crop 352x1216, random hflip (p=0.5 shared), color jitter 0.1 on RGB
  - CLEAN da3 teacher (precomputed) -> teacher disp -> depth proxy (per-image
    inv-depth affine fit vs sparse GT) -> SSI + multi-scale gradient distillation
  - supervised: CompletionFormer native 1.0*L1 + 1.0*L2 on final pred
  - total = sup + 0.1*SSI + 0.05*grad
  - optim Adam, lr 1e-5, step decay x0.1 every 5 epochs, batch 4, 20 epochs
Distillation losses imported verbatim from self-supervised-depth-completion/distill.py
"""
import os, sys, glob, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))

from model.completionformer import CompletionFormer

# Load distill.py by file path (avoid adding the whole self-supervised repo to
# sys.path, which has a top-level model.py that shadows CF's model package).
import importlib.util as _ilu
_distill_path = '/path/to/aurora-root/PhD_Project_2025/self-supervised-depth-completion/distill.py'
_spec = _ilu.spec_from_file_location('cf_distill', _distill_path)
_distill = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_distill)
load_da3_as_disp = _distill.load_da3_as_disp
disp_to_depth_proxy = _distill.disp_to_depth_proxy
ScaleShiftInvariantLoss = _distill.ScaleShiftInvariantLoss
MultiScaleResidualGradientLoss = _distill.MultiScaleResidualGradientLoss

DATA_ROOT = '/path/to/aurora-root/Dataset/AuroraKITTI/data_try'
DA3_ROOT = '/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti_depth_DC/raw'
OHEIGHT, OWIDTH = 352, 1216
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ModelArgs:
    model_name = 'CompletionFormer'
    prop_time = 6
    prop_kernel = 3
    conf_prop = True
    affinity = 'TGASS'
    affinity_gamma = 0.5
    preserve_input = False
    legacy = False


def gt_to_velo(p):
    return p.replace('data_depth_annotated', 'data_depth_velodyne').replace('groundtruth', 'velodyne_raw')


def gt_to_rgb(p):
    # .../data_depth_annotated/train/<drive>/proj_depth/groundtruth/<cam>/<frame>.png
    parts = p.split('/')
    frame = parts[-1]; cam = parts[-2]; drive = parts[-5]
    return os.path.join(DATA_ROOT, 'data_rgb', 'train', drive, cam, 'data', frame)


def gt_to_da3(p):
    parts = p.split('/')
    frame = parts[-1]; cam = parts[-2]; drive = parts[-5]
    date = '_'.join(drive.split('_')[:3])
    return os.path.join(DA3_ROOT, date, drive, cam, 'da3', frame)


def depth_read(path):
    d = np.array(Image.open(path), dtype=np.float32) / 256.0
    return d


class AuroraDataset(Dataset):
    def __init__(self, jitter=0.1, crop_h=0):
        self.jitter = jitter
        self.crop_h = crop_h   # >0: random vertical crop to crop_h x OWIDTH (train)
        gts = sorted(glob.glob(os.path.join(
            DATA_ROOT, 'data_depth_annotated', 'train', '*_sync',
            'proj_depth', 'groundtruth', 'image_0[23]', '*.png')))
        # keep only frames whose rgb + velo + da3 all exist
        self.items = []
        for g in gts:
            r = gt_to_rgb(g); v = gt_to_velo(g); t = gt_to_da3(g)
            if os.path.exists(r) and os.path.exists(v) and os.path.exists(t):
                self.items.append((r, v, g, t))
        print('AuroraDataset frames:', len(self.items), '/', len(gts))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rgb_p, velo_p, gt_p, da3_p = self.items[idx]
        rgb = Image.open(rgb_p).convert('RGB')
        W, H = rgb.size
        rgb = np.asarray(rgb)                      # HWC uint8
        sparse = depth_read(velo_p)                # HW metres
        gt = depth_read(gt_p)                      # HW metres
        teacher_disp = load_da3_as_disp(da3_p)     # HW [0,255]
        if teacher_disp.shape != gt.shape:
            teacher_disp = cv2.resize(teacher_disp, (gt.shape[1], gt.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

        # bottom-centre crop to 352x1216
        i = rgb.shape[0] - OHEIGHT
        j = int(round((rgb.shape[1] - OWIDTH) / 2.0))
        sl = (slice(i, i + OHEIGHT), slice(j, j + OWIDTH))
        rgb = rgb[sl]; sparse = sparse[sl]; gt = gt[sl]; teacher_disp = teacher_disp[sl]

        # optional random vertical crop (full width) to reduce memory
        if self.crop_h and self.crop_h < OHEIGHT:
            top = random.randint(0, OHEIGHT - self.crop_h)
            cs = (slice(top, top + self.crop_h), slice(0, OWIDTH))
            rgb = rgb[cs]; sparse = sparse[cs]; gt = gt[cs]; teacher_disp = teacher_disp[cs]

        # shared horizontal flip
        if random.random() < 0.5:
            rgb = rgb[:, ::-1].copy(); sparse = sparse[:, ::-1].copy()
            gt = gt[:, ::-1].copy(); teacher_disp = teacher_disp[:, ::-1].copy()

        # rgb to tensor + color jitter (on rgb only) + imagenet norm
        rgb_t = TF.to_tensor(rgb)                  # 3HW float [0,1]
        if self.jitter > 0:
            j0 = self.jitter
            rgb_t = TF.adjust_brightness(rgb_t, random.uniform(1 - j0, 1 + j0))
            rgb_t = TF.adjust_contrast(rgb_t, random.uniform(1 - j0, 1 + j0))
            rgb_t = TF.adjust_saturation(rgb_t, random.uniform(1 - j0, 1 + j0))
        rgb_t = TF.normalize(rgb_t, MEAN, STD)

        return {
            'rgb': rgb_t,
            'dep': torch.from_numpy(sparse)[None],
            'gt': torch.from_numpy(gt)[None],
            'tdisp': torch.from_numpy(teacher_disp)[None],
        }


def l1l2_loss(pred, gt, max_depth=90.0, t_valid=1e-4):
    # Mirror CompletionFormer native L1+L2 (weights 1.0 each):
    # clamp to [0,max_depth], per-sample sum/num_valid, then sum over batch.
    gt_c = torch.clamp(gt, 0, max_depth)
    pred_c = torch.clamp(pred, 0, max_depth)
    mask = (gt_c > t_valid).float().detach()
    nv = mask.sum(dim=[1, 2, 3]) + 1e-8
    l1 = ((pred_c - gt_c).abs() * mask).sum(dim=[1, 2, 3]) / nv
    l2 = (((pred_c - gt_c) ** 2) * mask).sum(dim=[1, 2, 3]) / nv
    l1 = l1.sum(); l2 = l2.sum()
    return l1 + l2, l1, l2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--accum', type=int, default=1, help='grad accumulation steps')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--ssi', type=float, default=0.1)
    ap.add_argument('--grad', type=float, default=0.05)
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'experiments_aurora'))
    ap.add_argument('--max_steps', type=int, default=0, help='debug: limit steps/epoch')
    ap.add_argument('--mem_frac', type=float, default=0.72,
                    help='cap GPU memory fraction (~17GB of 24GB) for shared GPU')
    ap.add_argument('--subset', type=int, default=0,
                    help='per-epoch random subset size (0=use full set); fresh resample each epoch')
    ap.add_argument('--crop_h', type=int, default=0,
                    help='random vertical crop height for training (0=full 352)')
    ap.add_argument('--seed', type=int, default=43)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)

    use_wandb = bool(os.environ.get('WANDB_API_KEY'))
    if use_wandb:
        import wandb
        wandb.init(project='IF__CompletionFormer_AURORA', config=vars(args))

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    ds = AuroraDataset(jitter=0.1, crop_h=args.crop_h)
    N = len(ds)

    def make_loader(epoch):
        # fresh random subset each epoch (seeded by base seed + epoch) so the
        # model sees varied data over training but only ~subset frames per epoch
        if args.subset and args.subset < N:
            g = torch.Generator(); g.manual_seed(args.seed + epoch)
            idx = torch.randperm(N, generator=g)[:args.subset].tolist()
            sampler = torch.utils.data.SubsetRandomSampler(idx)
            return DataLoader(ds, batch_size=args.batch, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
        return DataLoader(ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)

    model = CompletionFormer(ModelArgs()).cuda()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ssi_crit = ScaleShiftInvariantLoss().cuda()
    grad_crit = MultiScaleResidualGradientLoss(K=4).cuda()

    per_epoch = args.subset if (args.subset and args.subset < N) else N
    nsteps = per_epoch // args.batch if args.max_steps == 0 else args.max_steps
    print(f'steps/epoch={nsteps} epochs={args.epochs} subset={args.subset} '
          f'crop_h={args.crop_h} batch={args.batch} accum={args.accum} '
          f'eff_batch={args.batch*args.accum}', flush=True)

    for epoch in range(args.epochs):
        lr = args.lr * (0.1 ** (epoch // 5))
        for g in opt.param_groups:
            g['lr'] = lr
        dl = make_loader(epoch)
        t0 = time.time()
        running = 0.0
        for it, sample in enumerate(dl):
            if args.max_steps and it >= args.max_steps:
                break
            rgb = sample['rgb'].cuda(non_blocking=True)
            dep = sample['dep'].cuda(non_blocking=True)
            gt = sample['gt'].cuda(non_blocking=True)
            tdisp = sample['tdisp'].cuda(non_blocking=True)

            out = model({'rgb': rgb, 'dep': dep})
            pred = out['pred']

            sup, l1, l2 = l1l2_loss(pred, gt)
            with torch.no_grad():
                teacher_depth = disp_to_depth_proxy(tdisp, gt, (gt > 0))
            ssi = sum(ssi_crit(pred, teacher_depth))
            grd = sum(grad_crit(pred, teacher_depth))
            loss = sup + args.ssi * ssi + args.grad * grd

            (loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()

            running += loss.item()
            if it % 50 == 0:
                el = time.time() - t0
                ips = (it + 1) / el
                eta_ep = (nsteps - it - 1) / max(ips, 1e-6)
                msg = (f'ep{epoch} it{it}/{nsteps} lr{lr:.1e} loss{loss.item():.4f} '
                       f'sup{sup.item():.4f} ssi{float(ssi):.4f} grd{float(grd):.4f} '
                       f'{ips:.2f}it/s eta_ep{eta_ep/60:.1f}m')
                print(msg, flush=True)
                if use_wandb:
                    wandb.log({'loss': loss.item(), 'sup': sup.item(),
                               'ssi': float(ssi), 'grad': float(grd), 'lr': lr})
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f'  peak_alloc {peak:.2f} GB', flush=True)
        ckpt = os.path.join(args.out, f'model_{epoch+1:02d}.pt')
        torch.save({'net': model.state_dict(), 'epoch': epoch + 1, 'args': vars(args)}, ckpt)
        print(f'==EPOCH {epoch} done in {dt/60:.1f}m avg_loss {running/nsteps:.4f} '
              f'saved {ckpt}', flush=True)
    print('TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
