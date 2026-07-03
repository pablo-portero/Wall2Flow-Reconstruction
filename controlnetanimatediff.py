

import os
import math
from pathlib import Path

import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR


######################################################
##              i) Dataset Definition               ##
######################################################
# Reads each HDF5 snapshot, drops the solver ghost cells, and min-max
# normalises every physical field to roughly [0, 1] using the global stats
# precomputed over the training set.
class ChannelFlowDataset(Dataset):
    def __init__(self, file_paths, minsmaxs, dtype=np.float32):
        self.file_paths = list(file_paths)
        self.mm    = minsmaxs
        self.dtype = dtype

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def _norm(arr, lo, hi):
        return (arr - lo) / ((hi - lo) + 1e-12)

    def __getitem__(self, idx):
        with h5py.File(self.file_paths[idx], "r", swmr=True) as f:
            X = f["X_features"][...]      # (3, H+2, W+2)        wall map  with ghosts
            Y = f["Y_features"][...]      # (3, H+2, Ny, W+2)    velocity  with ghosts

        # Strip ghost cells.
        X = X[:,    1:-1, 1:-1].astype(self.dtype, copy=False)        # (3, H, W)
        Y = Y[:, 1:-1,    :,    1:-1].astype(self.dtype, copy=False)  # (3, H, Ny, W)

        # Wall data: pressure, tau_wx, tau_wz
        X[0] = self._norm(X[0], self.mm["P_min"],       self.mm["P_max"])
        X[1] = self._norm(X[1], self.mm["tau_w_x_min"], self.mm["tau_w_x_max"])
        X[2] = self._norm(X[2], self.mm["tau_w_z_min"], self.mm["tau_w_z_max"])

        # Velocity components: u, v, w
        Y[0] = self._norm(Y[0], self.mm["u_min"], self.mm["u_max"])
        Y[1] = self._norm(Y[1], self.mm["v_min"], self.mm["v_max"])
        Y[2] = self._norm(Y[2], self.mm["w_min"], self.mm["w_max"])

        return torch.from_numpy(X), torch.from_numpy(Y)


######################################################
##           ii) U-Net Building Blocks              ##
######################################################
# Stable-Diffusion-style blocks: conv -> GroupNorm -> SiLU, twice, with
# optional stride-2 downsampling on the first conv of an encoder stage.

class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        s = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, stride=s, padding=1)
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act   = nn.SiLU()

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2,
                                        padding=1, output_padding=1)
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.conv  = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act   = nn.SiLU()

    def forward(self, x, skip):
        x = self.act(self.norm1(self.up(x)))
        # Cover (H, W) not divisible by 8.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm2(self.conv(x)))
        return x


class FinalBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act  = nn.SiLU()

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        return self.act(self.norm(self.conv(x)))


# 1x1 conv with zero-init weights and bias.  Identity injection at t=0;
# the control branch only contributes once gradients flow through it.
class ZeroConv1x1(nn.Conv2d):
    def __init__(self, ch_in, ch_out):
        super().__init__(ch_in, ch_out, 1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


######################################################
##        iii) Wall-Normal Positional Encoding      ##
######################################################
# Sinusoidal PE evaluated on continuous wall-normal coordinates (e.g.
# y_plus normalised to [0, 1]) rather than integer slice indices.  Sees the
# stretched DNS grid as it really is.
def sinusoidal_pe_continuous(positions, dim):
    # positions : (N,) tensor of continuous coords in roughly [0, 1].
    # Returns   : (1, N, dim).
    half = dim // 2
    device = positions.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
    )
    angles = positions[:, None] * freqs[None, :] * (2.0 * math.pi)
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)        # (N, dim)
    return pe.unsqueeze(0)                                               # (1, N, dim)


######################################################
##     iv) AnimateDiff Motion Transformer (mid)     ##
######################################################
# Per-pixel temporal Transformer along the wall-normal axis.  Operates on
# the U-Net bottleneck features only (matches the preprint description).
# Pre-LN, GELU activations, zero-initialised output projection so the module
# starts as the identity and learns to add cross-slice corrections.
class MotionTransformer(nn.Module):
    def __init__(self, ch, num_layers=4, num_heads=8, ff_mult=4):
        super().__init__()
        self.norm_in = nn.LayerNorm(ch)
        self.proj_in = nn.Linear(ch, ch)

        layer = nn.TransformerEncoderLayer(
            d_model=ch,
            nhead=num_heads,
            dim_feedforward=ch * ff_mult,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.proj_out = nn.Linear(ch, ch)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, z, positions=None):
        # z         : (B, N, C, H, W).
        # positions : (N,) continuous wall-normal coords; falls back to uniform [0, 1].
        # Returns   : (B, N, C, H, W) residual to be added to z by the caller.
        B, N, C, H, W = z.shape
        x = z.permute(0, 3, 4, 1, 2).reshape(B * H * W, N, C)            # (B*H*W, N, C)

        x = self.norm_in(x)
        x = self.proj_in(x)

        if positions is None:
            positions = torch.arange(N, device=z.device).float() / max(N - 1, 1)
        x = x + sinusoidal_pe_continuous(positions, C)

        x = self.transformer(x)
        x = self.proj_out(x)
        return x.reshape(B, H, W, N, C).permute(0, 3, 4, 1, 2).contiguous()


######################################################
##      v) Generator: ControlNet + Motion @ mid     ##
######################################################
# - Base branch: encodes the previous slice, decodes the next one.
# - Control branch: encodes the wall data (3-channel) to feature maps that
#   match every base-branch resolution.
# - Zero-conv injections from control into base at every encoder and decoder
#   stage (A, B, C, mid, dD, dC, dB, dA).
# - Motion Transformer at the mid bottleneck along the wall-normal axis.
#
#   prev_uv : (B, 3, H, W)
#   cond2d  : (B, 3, H, W)   wall data (p, tau_wx, tau_wz)
#   output  : (B, 3, H, W)   next slice
class W2FGenerator(nn.Module):
    def __init__(self, in_ch=3, cond_ch=3, base=32):
        super().__init__()

        # Base branch
        self.b_A   = EncBlock(in_ch,   base,   downsample=False)   # H
        self.b_B   = EncBlock(base,    2*base, downsample=True)    # H/2
        self.b_C   = EncBlock(2*base,  4*base, downsample=True)    # H/4
        self.b_D   = EncBlock(4*base,  8*base, downsample=True)    # H/8
        self.b_mid = EncBlock(8*base,  8*base, downsample=False)
        self.b_dD  = DecBlock(8*base,  4*base, 4*base)
        self.b_dC  = DecBlock(4*base,  2*base, 2*base)
        self.b_dB  = DecBlock(2*base,  base,   base)
        self.b_out = FinalBlock(base,  base,   3)

        # Control branch (3-channel wall data)
        self.c_A   = EncBlock(cond_ch, base,   downsample=False)
        self.c_B   = EncBlock(base,    2*base, downsample=True)
        self.c_C   = EncBlock(2*base,  4*base, downsample=True)
        self.c_D   = EncBlock(4*base,  8*base, downsample=True)
        self.c_mid = EncBlock(8*base,  8*base, downsample=False)

        # Zero-conv injections (one per stage in S = {A,B,C,D,mid,dD,dC,dB,dA})
        self.zc_A   = ZeroConv1x1(base,   base)
        self.zc_B   = ZeroConv1x1(2*base, 2*base)
        self.zc_C   = ZeroConv1x1(4*base, 4*base)
        self.zc_D   = ZeroConv1x1(8*base, 8*base)
        self.zc_mid = ZeroConv1x1(8*base, 8*base)
        self.zc_dD  = ZeroConv1x1(4*base, 4*base)
        self.zc_dC  = ZeroConv1x1(2*base, 2*base)
        self.zc_dB  = ZeroConv1x1(base,   base)
        self.zc_out = ZeroConv1x1(base,   3)

        # Motion Transformer at mid only (preprint baseline).
        self.motion_mid = MotionTransformer(ch=8*base, num_layers=4,
                                            num_heads=8, ff_mult=4)

    # Encode control once per sample (wall data is constant across slices).
    def encode_control(self, cond2d):
        cA = self.c_A(cond2d)
        cB = self.c_B(cA)
        cC = self.c_C(cB)
        cD = self.c_D(cC)
        cm = self.c_mid(cD)
        return cA, cB, cC, cD, cm

    # Encode one slice with the base branch.
    def encode_slice(self, slice_uvw):
        xA = self.b_A(slice_uvw)
        xB = self.b_B(xA)
        xC = self.b_C(xB)
        xD = self.b_D(xC)
        xm = self.b_mid(xD)
        return xA, xB, xC, xD, xm

    # Step forward used by the cached free-run rollout.
    #   xA_cache..xm_cache : lists of length L_eff with base encodes of the
    #                       last L_eff slices.
    #   cA..cm             : control encodes (cached, constant across slices).
    #   y_positions        : (L_eff,) wall-normal coords matching the cache.
    # Returns the prediction for the next slice.
    def step_forward(self, xA_cache, xB_cache, xC_cache, xD_cache, xm_cache,
                     cA, cB, cC, cD, cm,
                     gs=1.0, use_motion=True, y_positions=None):
        L = len(xm_cache)

        # Motion @ mid only, over the full clip in the cache.
        if use_motion and L > 1:
            xm_seq = torch.stack(xm_cache, dim=1)                           # (B,L,C,h,w)
            xm_seq = xm_seq + self.motion_mid(xm_seq, positions=y_positions)
            xm_last = xm_seq[:, -1]
        else:
            xm_last = xm_cache[-1]

        # Inject control on the last slice's encodes only — decoder is per-slice.
        xA = xA_cache[-1] + self.zc_A(cA)   * gs
        xB = xB_cache[-1] + self.zc_B(cB)   * gs
        xC = xC_cache[-1] + self.zc_C(cC)   * gs
        xD = xD_cache[-1] + self.zc_D(cD)   * gs
        xm = xm_last       + self.zc_mid(cm) * gs

        d = self.b_dD(xm, xC); d = d + self.zc_dD(cC) * gs
        d = self.b_dC(d,  xB); d = d + self.zc_dC(cB) * gs
        d = self.b_dB(d,  xA); d = d + self.zc_dB(cA) * gs
        out = self.b_out(d, xA) + self.zc_out(cA) * gs
        return torch.sigmoid(out)

    # Single-slice forward (sanity check / inference helper, no motion).
    def forward(self, prev_uv, cond2d, gs=1.0):
        xA, xB, xC, xD, xm = self.encode_slice(prev_uv)
        cA, cB, cC, cD, cm = self.encode_control(cond2d)

        xA = xA + self.zc_A(cA)   * gs
        xB = xB + self.zc_B(cB)   * gs
        xC = xC + self.zc_C(cC)   * gs
        xD = xD + self.zc_D(cD)   * gs
        xm = xm + self.zc_mid(cm) * gs

        d = self.b_dD(xm, xC); d = d + self.zc_dD(cC) * gs
        d = self.b_dC(d,  xB); d = d + self.zc_dC(cB) * gs
        d = self.b_dB(d,  xA); d = d + self.zc_dB(cA) * gs
        out = self.b_out(d, xA) + self.zc_out(cA) * gs
        return torch.sigmoid(out)


######################################################
##           vi) PatchGAN Discriminator             ##
######################################################
# Receives (target/fake slice (3) + wall data (3) + previous slice (3)) = 9 ch.
# Strides chosen so that the output is roughly H/8 x W/8 (matches preprint).
class PatchDiscriminator2D(nn.Module):
    def __init__(self, in_ch=9, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,    base,    4, 2, 1), nn.LeakyReLU(0.2, inplace=True),  # /2
            nn.Conv2d(base,     2*base,  4, 2, 1), nn.LeakyReLU(0.2, inplace=True),  # /4
            nn.Conv2d(2*base,   4*base,  4, 2, 1), nn.LeakyReLU(0.2, inplace=True),  # /8
            nn.Conv2d(4*base,   8*base,  4, 1, 1), nn.LeakyReLU(0.2, inplace=True),  # ~/8
            nn.Conv2d(8*base,   1,       4, 1, 1),                                   # logit map
        )

    def forward(self, x):
        return self.net(x)


######################################################
##         vii) Finite-Difference Operators         ##
######################################################
# Conventions for the velocity tensor in the rollout:
#   field : (B, Ny_eff, 3, H, W)
#   axis -1 (W) = streamwise (x), periodic, uniform spacing dx.
#   axis -2 (H) = spanwise   (z), periodic, uniform spacing dz.
#   axis -4 (Ny_eff) = wall-normal (y), non-periodic, NON-uniform spacing.
#
# If your dataset stores W=z and H=x instead, just swap dx <-> dz at the
# call sites in main().

def d_dx_periodic(f, dx):
    # ∂f/∂x with centered FD and periodic wrap along the last axis.
    f_p = torch.roll(f, shifts=-1, dims=-1)
    f_m = torch.roll(f, shifts=+1, dims=-1)
    return (f_p - f_m) / (2.0 * dx)

def d2_dx2_periodic(f, dx):
    f_p = torch.roll(f, shifts=-1, dims=-1)
    f_m = torch.roll(f, shifts=+1, dims=-1)
    return (f_p - 2.0 * f + f_m) / (dx ** 2)

def d_dz_periodic(f, dz):
    # Spanwise axis is -2 (the "H" axis of a 2D slice).
    f_p = torch.roll(f, shifts=-1, dims=-2)
    f_m = torch.roll(f, shifts=+1, dims=-2)
    return (f_p - f_m) / (2.0 * dz)

def d2_dz2_periodic(f, dz):
    f_p = torch.roll(f, shifts=-1, dims=-2)
    f_m = torch.roll(f, shifts=+1, dims=-2)
    return (f_p - 2.0 * f + f_m) / (dz ** 2)


def precompute_y_fd_coeffs(y_coords, device):
    y = y_coords.to(device).float()
    h_lo = (y[1:-1] - y[:-2])           # (Ny-2,)   h_j-
    h_hi = (y[2:]   - y[1:-1])          # (Ny-2,)   h_j+
    eps  = 1e-12

    c1_lo  = -1.0 / (h_lo + h_hi + eps)
    c1_hi  =  1.0 / (h_lo + h_hi + eps)

    c2_lo  =  2.0 / (h_lo * (h_lo + h_hi) + eps)
    c2_mid = -2.0 / (h_lo *  h_hi          + eps)
    c2_hi  =  2.0 / (h_hi * (h_lo + h_hi) + eps)

    return (c1_lo, c1_hi), (c2_lo, c2_mid, c2_hi)


def d_dy_nonuniform(f, c1):
    c1_lo, c1_hi = c1
    Ny_eff = f.shape[1]
    # Slice coefficients to match interior points of f.
    c1_lo_use = c1_lo[: Ny_eff - 2].view(1, -1, 1, 1, 1)
    c1_hi_use = c1_hi[: Ny_eff - 2].view(1, -1, 1, 1, 1)
    interior = c1_lo_use * f[:, :-2] + c1_hi_use * f[:,  2:]
    # Pad with zeros at the two ends so the returned tensor has the same Ny_eff.
    pad = torch.zeros_like(f[:, :1])
    return torch.cat([pad, interior, pad], dim=1)


def d2_dy2_nonuniform(f, c2):
    c2_lo, c2_mid, c2_hi = c2
    Ny_eff = f.shape[1]
    c2_lo_use  = c2_lo[: Ny_eff - 2].view(1, -1, 1, 1, 1)
    c2_mid_use = c2_mid[: Ny_eff - 2].view(1, -1, 1, 1, 1)
    c2_hi_use  = c2_hi[: Ny_eff - 2].view(1, -1, 1, 1, 1)
    interior = (c2_lo_use  * f[:,  :-2]
              + c2_mid_use * f[:, 1:-1]
              + c2_hi_use  * f[:,  2:])
    pad = torch.zeros_like(f[:, :1])
    return torch.cat([pad, interior, pad], dim=1)


######################################################
##              viii) Physics Losses                ##
######################################################
def denormalise_uvw(field, mm):
    a = field.new_tensor([mm["u_max"] - mm["u_min"],
                          mm["v_max"] - mm["v_min"],
                          mm["w_max"] - mm["w_min"]]).view(1, 1, 3, 1, 1)
    b = field.new_tensor([mm["u_min"], mm["v_min"], mm["w_min"]]).view(1, 1, 3, 1, 1)
    return field * a + b


def divergence(u, v, w, dx, dz, c1y):
    v5 = v.unsqueeze(2)                                                  # (B, Ny, 1, H, W)
    dvdy = d_dy_nonuniform(v5, c1y).squeeze(2)                           # (B, Ny, H, W)
    dudx = d_dx_periodic(u, dx)
    dwdz = d_dz_periodic(w, dz)
    return dudx + dvdy + dwdz


def momentum_residual(u, v, w, rho, mu, dx, dz, c1y, c2y):
    def lap(f):
        # ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
        f5 = f.unsqueeze(2)
        d2y = d2_dy2_nonuniform(f5, c2y).squeeze(2)
        return d2_dx2_periodic(f, dx) + d2y + d2_dz2_periodic(f, dz)

    def d_dy(f):
        return d_dy_nonuniform(f.unsqueeze(2), c1y).squeeze(2)

    # Pre-form the products once.
    uu = u * u; vv = v * v; ww = w * w
    uv = u * v; uw = u * w; vw = v * w

    # x-momentum
    conv_x = rho * (d_dx_periodic(uu, dx) + d_dy(uv) + d_dz_periodic(uw, dz))
    diff_x = mu  * lap(u)
    R_x    = conv_x - diff_x

    # y-momentum
    conv_y = rho * (d_dx_periodic(uv, dx) + d_dy(vv) + d_dz_periodic(vw, dz))
    diff_y = mu  * lap(v)
    R_y    = conv_y - diff_y

    # z-momentum
    conv_z = rho * (d_dx_periodic(uw, dx) + d_dy(vw) + d_dz_periodic(ww, dz))
    diff_z = mu  * lap(w)
    R_z    = conv_z - diff_z

    return R_x, R_y, R_z


def mass_loss(pred_field, gt_field, mm, dx, dz, c1y):
    pred_d = denormalise_uvw(pred_field, mm)                # (B, Ny_eff, 3, H, W)
    gt_d   = denormalise_uvw(gt_field,   mm)
    u_p, v_p, w_p = pred_d[:, :, 0], pred_d[:, :, 1], pred_d[:, :, 2]
    u_g, v_g, w_g = gt_d[:, :, 0],   gt_d[:, :, 1],   gt_d[:, :, 2]
    div_p = divergence(u_p, v_p, w_p, dx, dz, c1y)
    div_g = divergence(u_g, v_g, w_g, dx, dz, c1y)
    return (div_p - div_g).abs().mean()


def momentum_loss(pred_field, gt_field, mm, rho, mu, dx, dz, c1y, c2y):
    pred_d = denormalise_uvw(pred_field, mm)
    gt_d   = denormalise_uvw(gt_field,   mm)
    u_p, v_p, w_p = pred_d[:, :, 0], pred_d[:, :, 1], pred_d[:, :, 2]
    u_g, v_g, w_g = gt_d[:, :, 0],   gt_d[:, :, 1],   gt_d[:, :, 2]
    Rxp, Ryp, Rzp = momentum_residual(u_p, v_p, w_p, rho, mu, dx, dz, c1y, c2y)
    Rxg, Ryg, Rzg = momentum_residual(u_g, v_g, w_g, rho, mu, dx, dz, c1y, c2y)
    return ((Rxp - Rxg).abs().mean()
          + (Ryp - Ryg).abs().mean()
          + (Rzp - Rzg).abs().mean()) / 3.0


def periodic_loss(field):
    diff_x = (field[..., :,  0]   - field[..., :, -1]).abs().mean()          # x periodicity (W axis)
    diff_z = (field[..., 0,    :] - field[..., -1, :]).abs().mean()          # z periodicity (H axis)
    return 0.5 * (diff_x + diff_z)


######################################################
##              ix) Training Step                   ##
######################################################
def train_one_epoch(model, D, loader, optG, optD, scheduler, device,
                    gs, weights, mm, rho, mu, dx, dz, c1y, c2y,
                    L_ctx, y_positions_norm, y_coords_phys,
                    use_amp=True, tbptt_K=8, lambda_adv_now=0.0):
    
    model.train(); D.train()
    data_m = adv_m = d_m = mass_m = mom_m = per_m = 0.0
    n_chunks_total = 0

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    for X, Y in loader:
        X = X.to(device, non_blocking=True)             # (B, 3, H, W)
        Y = Y.to(device, non_blocking=True)             # (B, 3, H, Ny, W)
        Y_seq = Y.permute(0, 3, 1, 2, 4).contiguous()   # (B, Ny, 3, H, W)
        B, Ny, _, H, W = Y_seq.shape

        K = tbptt_K if tbptt_K is not None else (Ny - 1)

        ######################################################
        ##   Seed: encode the first slice and prime caches  ##
        ######################################################
        with amp_ctx, torch.no_grad():
            xA_buf, xB_buf, xC_buf, xD_buf, xm_buf = [], [], [], [], []
            a, b, c, d_, mb = model.encode_slice(Y_seq[:, 0])
            xA_buf.append(a); xB_buf.append(b); xC_buf.append(c)
            xD_buf.append(d_); xm_buf.append(mb)
            y_pos_cache = [y_positions_norm[0:1]]

        # y_pred_list holds DETACHED predictions of the rollout so far.
        # The seed slice is GT, so it is the first entry.
        y_pred_list = [Y_seq[:, 0].detach()]
        prev_for_D  = Y_seq[:, 0]

        chunk_data = chunk_adv = chunk_d = chunk_mass = chunk_mom = chunk_per = 0.0
        chunk_count = 0

        ######################################################
        ##         Roll through the wall-normal axis        ##
        ######################################################
        for start in range(1, Ny, K):
            end     = min(start + K, Ny)
            n_steps = end - start

            ##############################################
            ##  Forward: produce n_steps grad-enabled slices
            ##############################################
            with amp_ctx:
                cA, cB, cC, cD, cm = model.encode_control(X)
                preds = []
                for s in range(n_steps):
                    y_pos_t = torch.cat(y_pos_cache)
                    pred_t = model.step_forward(
                        xA_buf, xB_buf, xC_buf, xD_buf, xm_buf,
                        cA, cB, cC, cD, cm,
                        gs=gs, use_motion=True, y_positions=y_pos_t,
                    )
                    preds.append(pred_t)

                    # Re-encode the new slice to extend the cache.
                    a, b, c, d_, mb = model.encode_slice(pred_t)
                    xA_buf.append(a); xB_buf.append(b); xC_buf.append(c)
                    xD_buf.append(d_); xm_buf.append(mb)
                    y_pos_cache.append(y_positions_norm[start + s : start + s + 1])

                    if len(xm_buf) > L_ctx:
                        xA_buf.pop(0); xB_buf.pop(0); xC_buf.pop(0)
                        xD_buf.pop(0); xm_buf.pop(0)
                        y_pos_cache.pop(0)

                pred_chunk = torch.stack(preds, dim=1)            # (B, n_steps, 3, H, W)
                targ_chunk = Y_seq[:, start:end]
                prev_chunk = torch.cat(
                    [prev_for_D.unsqueeze(1), pred_chunk[:, :-1]], dim=1
                ).detach()

            ##############################################
            ##           Discriminator update           ##
            ##############################################
            X_rep = X.unsqueeze(1).expand(-1, n_steps, -1, -1, -1)

            def flat(t, n=n_steps):
                return t.reshape(B * n, t.shape[2], H, W)

            if lambda_adv_now > 0:
                pred_det = pred_chunk.detach()
                with amp_ctx:
                    real_in = flat(torch.cat([targ_chunk, X_rep, prev_chunk], dim=2))
                    fake_in = flat(torch.cat([pred_det,   X_rep, prev_chunk], dim=2))
                    d_loss = (F.relu(1 - D(real_in)).mean()
                              + F.relu(1 + D(fake_in)).mean())
                optD.zero_grad(set_to_none=True)
                d_loss.backward()
                optD.step()
            else:
                d_loss = torch.zeros((), device=device)

            ##############################################
            ##  Build partial field for physics losses   ##
            ##############################################
            # Past detached + current grad-enabled, concatenated along wall-normal.
            # Past tensors are detached -> no compute graph -> ~no memory.
            # IMPORTANT: physics is computed in fp32 (FDs hate bf16).
            past_det = torch.stack(y_pred_list, dim=1)            # (B, len_so_far, 3, H, W)
            partial  = torch.cat([past_det, pred_chunk], dim=1)   # (B, len_so_far + n_steps, 3, H, W)
            partial_gt = Y_seq[:, : start + n_steps]              # (B, len_so_far + n_steps, 3, H, W)

            ##############################################
            ##              Generator update            ##
            ##############################################
            with amp_ctx:
                l_data = F.l1_loss(pred_chunk, targ_chunk)
                if lambda_adv_now > 0:
                    fake_in_g = flat(torch.cat([pred_chunk, X_rep, prev_chunk], dim=2))
                    g_adv     = -D(fake_in_g).mean()
                else:
                    g_adv = torch.zeros((), device=device)

            # Physics terms in fp32.
            partial_fp32    = partial.float()
            partial_gt_fp32 = partial_gt.float()

            l_mass = mass_loss(partial_fp32, partial_gt_fp32, mm, dx, dz, c1y)
            l_mom  = momentum_loss(partial_fp32, partial_gt_fp32, mm,
                                   rho, mu, dx, dz, c1y, c2y)
            l_per  = periodic_loss(pred_chunk.float())   # per-slice, only over this chunk

            g_loss = (l_data
                      + lambda_adv_now      * g_adv
                      + weights["mass"]  * 0.1   * l_mass
                      + weights["mom"]   *0.1   * l_mom
                      + weights["per"]     *0.1  * l_per)

            optG.zero_grad(set_to_none=True)
            g_loss.backward()
            optG.step()

            ##############################################
            ##  Append chunk to y_pred_list (detached)  ##
            ##############################################
            for i in range(n_steps):
                y_pred_list.append(pred_chunk[:, i].detach())

            # Detach the encoder cache so the next chunk does not back-prop into this one.
            xA_buf = [t.detach() for t in xA_buf]
            xB_buf = [t.detach() for t in xB_buf]
            xC_buf = [t.detach() for t in xC_buf]
            xD_buf = [t.detach() for t in xD_buf]
            xm_buf = [t.detach() for t in xm_buf]
            prev_for_D = pred_chunk[:, -1].detach()

            chunk_data += l_data.item()
            chunk_adv  += (lambda_adv_now * g_adv).item()
            chunk_d    += d_loss.item()
            chunk_mass += (weights["mass"] * l_mass).item()
            chunk_mom  += (weights["mom"]  * l_mom).item()
            chunk_per  += (weights["per"]  * l_per).item()
            chunk_count += 1

        if scheduler is not None:
            scheduler.step()

        nb = max(chunk_count, 1)
        data_m += chunk_data / nb
        adv_m  += chunk_adv  / nb
        d_m    += chunk_d    / nb
        mass_m += chunk_mass / nb
        mom_m  += chunk_mom  / nb
        per_m  += chunk_per  / nb
        n_chunks_total += 1

    n = max(n_chunks_total, 1)
    return {
        "data": data_m / n, "adv": adv_m / n, "d": d_m / n,
        "mass": mass_m / n, "mom": mom_m / n, "per": per_m / n,
    }


######################################################
##                 x) Validation                    ##
######################################################
@torch.no_grad()
def validate(model, loader, device, gs, L_ctx, y_positions_norm, use_amp=True):

    model.eval()
    tot_loss = 0.0; tot_count = 0
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        Y_seq = Y.permute(0, 3, 1, 2, 4).contiguous()
        Ny    = Y_seq.shape[1]

        with amp_ctx:
            cA, cB, cC, cD, cm = model.encode_control(X)

            xA_buf, xB_buf, xC_buf, xD_buf, xm_buf = [], [], [], [], []
            a, b, c, d_, mb = model.encode_slice(Y_seq[:, 0])
            xA_buf.append(a); xB_buf.append(b); xC_buf.append(c)
            xD_buf.append(d_); xm_buf.append(mb)
            y_pos_cache = [y_positions_norm[0:1]]

            preds = []
            for t_idx in range(1, Ny):
                y_pos_t = torch.cat(y_pos_cache)
                pred_t = model.step_forward(
                    xA_buf, xB_buf, xC_buf, xD_buf, xm_buf,
                    cA, cB, cC, cD, cm,
                    gs=gs, use_motion=True, y_positions=y_pos_t,
                )
                preds.append(pred_t)

                a, b, c, d_, mb = model.encode_slice(pred_t)
                xA_buf.append(a); xB_buf.append(b); xC_buf.append(c)
                xD_buf.append(d_); xm_buf.append(mb)
                y_pos_cache.append(y_positions_norm[t_idx : t_idx + 1])

                if len(xm_buf) > L_ctx:
                    xA_buf.pop(0); xB_buf.pop(0); xC_buf.pop(0)
                    xD_buf.pop(0); xm_buf.pop(0)
                    y_pos_cache.pop(0)

            pred_seq = torch.stack(preds, dim=1)
            targ_seq = Y_seq[:, 1:]

        tot_loss  += F.mse_loss(pred_seq.float(), targ_seq.float(), reduction="sum").item()
        tot_count += targ_seq.numel()

    return tot_loss / max(tot_count, 1)


######################################################
##                 xi) Fit Loop                     ##
######################################################
def fit(model, D, train_loader, val_loader, optG, optD, scheduler, device,
        gs, num_epochs, output_dir,
        weights, mm, rho, mu, dx, dz, c1y, c2y,
        L_ctx, y_positions_norm, y_coords_phys,
        lambda_adv, adv_warmup_epochs=200,
        use_amp=True, tbptt_K=8):

    os.makedirs(output_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        if epoch <= adv_warmup_epochs:
            lambda_adv_now = 0.0
        elif epoch <= 2 * adv_warmup_epochs:
            ramp = (epoch - adv_warmup_epochs) / max(adv_warmup_epochs, 1)
            lambda_adv_now = lambda_adv * ramp
        else:
            lambda_adv_now = lambda_adv

        tr  = train_one_epoch(model, D, train_loader, optG, optD, scheduler, device,
                              gs, weights, mm, rho, mu, dx, dz, c1y, c2y,
                              L_ctx, y_positions_norm, y_coords_phys,
                              use_amp=use_amp, tbptt_K=tbptt_K,
                              lambda_adv_now=lambda_adv_now)
        val = validate(model, val_loader, device, gs, L_ctx, y_positions_norm, use_amp=use_amp)

        print(f"[{epoch}/{num_epochs}]  "
              f"L1:{tr['data']:.4e}  adv:{tr['adv']:.4e}  D:{tr['d']:.4e}  "
              f"mass:{tr['mass']:.4e}  mom:{tr['mom']:.4e}  per:{tr['per']:.4e}  "
              f"lam_adv:{lambda_adv_now:.4f}  |  ValMSE:{val:.4e}")

        if val < best_val:
            best_val = val
            ckpt = {
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "D_state":      D.state_dict(),
                "optG_state":   optG.state_dict(),
                "optD_state":   optD.state_dict(),
                "best_val_mse": best_val,
            }
            path = os.path.join(output_dir, f"w2f_best_epoch{epoch}.pt")
            torch.save(ckpt, path)
            print(f"   ↪ Saved: {path}")

        torch.cuda.empty_cache()

    print(f"Training done. Best Val MSE = {best_val:.6e}")


######################################################
##                  xii) Main                       ##
######################################################
if __name__ == "__main__":

    ##############################################
    ##  Paths and hyperparameters (per preprint) ##
    ##############################################
    samples_dir     = Path("...")
    mesh_dir        = Path("...")
    checkpoints_dir = Path("...")
    mesh_path       = mesh_dir / "3d_turbulent_channel_flow-MESH.h5"

    # If you already saved global mins/maxs from a previous run, point here.
    # If None, they will be recomputed by scanning the dataset (slower).
    minsmax_file    = None    # e.g. Path("/home/jofre/.../minsmaxs.h5")

    # Per Table 3 of the preprint
    guidance_scale  = 3.0
    num_epochs      = 600
    batch_size      = 2
    lr              = 1e-4
    lambda_adv      = 1e-3
    lambda_mass     = 0.2
    lambda_mom      = 0.075
    lambda_per      = 0.025
    L               = 8                  # slice-stack window (= TBPTT chunk length)

    # Memory / training behaviour
    L_ctx              = L               # context window for the Motion Transformer
    tbptt_K            = L               # backprop through chunks of K slices
    use_amp            = True            # bfloat16 autocast
    adv_warmup_epochs  = 0             # L1+physics first, GAN after warmup

    # Physical constants (channel flow at Re_tau = 180, matches RHEA setup)
    Re_tau   = 180.0
    delta    = 1.0
    rho_0    = 1.0
    u_tau    = 1.0
    mu_phys  = rho_0 * u_tau * delta / Re_tau          # dynamic viscosity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ##############################################
    ##           Build the file list             ##
    ##############################################
    # Adjust this range to your actual training set.
    file_indices  = list(range(280000,  43150001, 10000)) 
    data_files    = [f"datapost0_3d_turbulent_channel_flow_{i}.h5" for i in file_indices]
    file_paths    = [str(samples_dir / f) for f in data_files]
    print(f"Number of samples available: {len(file_paths)}")

    ##############################################
    ##  Load (or compute) min/max statistics    ##
    ##############################################
    if minsmax_file is not None and minsmax_file.exists():
        with h5py.File(minsmax_file, "r") as f:
            minsmaxs = {k: float(f["minsmaxs"][k][()]) for k in f["minsmaxs"].keys()}
    else:
        print("Computing global min/max statistics over the training set (slow first run)...")
        keys = ["P_min", "P_max", "tau_w_x_min", "tau_w_x_max",
                "tau_w_z_min", "tau_w_z_max",
                "u_min", "u_max", "v_min", "v_max", "w_min", "w_max"]
        minsmaxs = {k: None for k in keys}
        for p in file_paths:
            with h5py.File(p, "r") as f:
                Xi = f["X_features"][:][:, 1:-1, 1:-1]
                Yi = f["Y_features"][:][:, 1:-1, :, 1:-1]
            updates = {
                "P_min":       Xi[0].min(), "P_max":       Xi[0].max(),
                "tau_w_x_min": Xi[1].min(), "tau_w_x_max": Xi[1].max(),
                "tau_w_z_min": Xi[2].min(), "tau_w_z_max": Xi[2].max(),
                "u_min":       Yi[0].min(), "u_max":       Yi[0].max(),
                "v_min":       Yi[1].min(), "v_max":       Yi[1].max(),
                "w_min":       Yi[2].min(), "w_max":       Yi[2].max(),
            }
            for k, v in updates.items():
                if minsmaxs[k] is None:
                    minsmaxs[k] = float(v)
                elif k.endswith("_min"):
                    minsmaxs[k] = min(minsmaxs[k], float(v))
                else:
                    minsmaxs[k] = max(minsmaxs[k], float(v))
        print("Stats:", minsmaxs)

    ##############################################
    ##           Build dataset/loaders          ##
    ##############################################
    dataset = ChannelFlowDataset(file_paths, minsmaxs)
    n_total = len(dataset)
    n_train = int(0.74 * n_total)        # matches preprint partition (74 / 19 / 7)
    n_test  = max(int(0.07 * n_total), 1)
    n_val   = n_total - n_train - n_test
    train_set, val_set, _ = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=False)

    ##############################################
    ##  Load mesh, compute spacings and FD coeffs
    ##############################################
    with h5py.File(mesh_path, "r") as f:
        x_data = f["x"][:]
        y_data = f["y"][:]
        z_data = f["z"][:]

    # Strip ghosts (1 cell each side, matching the dataset).
    x_phys = np.array(x_data[0, 0, :])[1:-1]
    y_phys = np.array(y_data[0, :, 0])[1:-1]
    z_phys = np.array(z_data[:, 0, 0])[1:-1]

    # Uniform spacings in x and z (channel flow grid is uniform there).
    dx = float(x_phys[1] - x_phys[0])
    dz = float(z_phys[1] - z_phys[0])

    Ny = len(y_phys)
    y_phys_T = torch.from_numpy(y_phys.astype(np.float32)).to(device)
    c1y, c2y = precompute_y_fd_coeffs(y_phys_T, device)

    # Normalised positions for the Motion Transformer's positional encoding.
    y_pos_np  = (y_phys - y_phys.min()) / max(y_phys.max() - y_phys.min(), 1e-12)
    y_pos_T   = torch.from_numpy(y_pos_np.astype(np.float32)).to(device)

    print(f"Grid: Nx={len(x_phys)}, Ny={Ny}, Nz={len(z_phys)}  |  dx={dx:.4e}, dz={dz:.4e}")
    print(f"y_phys range: [{y_phys.min():.4e}, {y_phys.max():.4e}]")

    ##############################################
    ##         Build G and D + optimisers       ##
    ##############################################
    model = W2FGenerator(in_ch=3, cond_ch=3, base=32).to(
        device, memory_format=torch.channels_last
    )
    D = PatchDiscriminator2D(in_ch=9, base=64).to(
        device, memory_format=torch.channels_last
    )

    optG = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-6)
    optD = optim.Adam(D.parameters(),     lr=lr, betas=(0.5, 0.999))
    schedulerG = OneCycleLR(optG, max_lr=lr,
                            steps_per_epoch=len(train_loader), epochs=num_epochs)

    weights = {"mass": lambda_mass, "mom": lambda_mom, "per": lambda_per}

    ##############################################
    ##                Launch fit                ##
    ##############################################
    fit(model, D, train_loader, val_loader, optG, optD, schedulerG, device,
        guidance_scale, num_epochs, checkpoints_dir,
        weights, minsmaxs, rho_0, mu_phys, dx, dz, c1y, c2y,
        L_ctx, y_pos_T, y_phys_T,
        lambda_adv=lambda_adv, adv_warmup_epochs=adv_warmup_epochs,
        use_amp=use_amp, tbptt_K=tbptt_K)

    print("ALL DONE")