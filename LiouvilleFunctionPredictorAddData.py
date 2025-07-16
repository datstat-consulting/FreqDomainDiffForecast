import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.fft import rfft, irfft, rfftfreq

# -------------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------------
N_POINTS   = 1_000_000    # length of training series
WINDOW_SZ  = 4096         # forecast window length
STRIDE     = 512          # overlap stride for training windows
GAMMA      = 0.5          # fractional integration order
EPS        = 1e-3         # low‑freq stabilization
TIMESTEPS  = 500          # diffusion steps
EPOCHS     = 100          # training epochs
BASE_CH    = 32           # UNet base channels
DEPTH      = 4            # UNet depth
BATCH_SIZE = 8            # windows per batch
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE      = torch.float32

# -------------------------------------------------------------------------
# 0. LOAD ATTACHED CSV FEATURES
# -------------------------------------------------------------------------
# Assumes each CSV has a single column "value" of length >= N_POINTS + WINDOW_SZ
def load_series(path):
    df = pd.read_csv(path)
    # use 'value' if it exists, otherwise fall back to the first column
    col = 'value' if 'value' in df.columns else df.columns[0]
    return df[col].to_numpy(dtype=np.float32)

binary        = load_series('/content/data/binary.csv')
divisors_sum  = load_series('/content/data/divisors_sum.csv')
mobius        = load_series('/content/data/mobius_results.csv')
primes        = load_series('/content/data/primes.csv')
squares       = load_series('/content/data/squares.csv')
# stack into shape (N+W, n_feat)
features_full = np.stack([binary,
                          divisors_sum,
                          mobius,
                          primes,
                          squares], axis=1)

# -------------------------------------------------------------------------
# 1. Liouville sequence generator (same as before)
# -------------------------------------------------------------------------
def liouville_sequence(n: int) -> np.ndarray:
    spf   = np.zeros(n+1, dtype=int)
    for p in range(2, n+1):
        if spf[p] == 0:
            spf[p::p] = np.where(spf[p::p] == 0, p, spf[p::p])
    spf[1] = 1
    omega = np.zeros(n+1, dtype=int)
    for k in range(2, n+1):
        omega[k] = omega[k // spf[k]] + 1
    return ((-1) ** omega[1:]).astype(np.float32)

liouville_full = liouville_sequence(N_POINTS + WINDOW_SZ)

# -------------------------------------------------------------------------
# 2. Windowed conditional dataset with extra features
# -------------------------------------------------------------------------
class WindowedFreqConditionalDataset(Dataset):
    def __init__(self,
                 series: np.ndarray,
                 features: np.ndarray,
                 gamma: float, eps: float,
                 window_size: int, stride: int,
                 device: str):
        self.device = device
        self.win    = window_size
        self.stride = stride
        # slice Liouville windows
        self.windows = [
            series[i:i+window_size].astype(np.float32)
            for i in range(0, N_POINTS - window_size + 1, stride)
        ]
        # slice feature windows (shape each: (window_size, n_feat))
        self.features = [
            features[i:i+window_size].astype(np.float32).T
            for i in range(0, N_POINTS - window_size + 1, stride)
        ]
        # fractional‑integration filter
        f      = rfftfreq(window_size, d=1.0)
        H_int  = ((1j*2*np.pi*f)**2 + eps**2)**(-gamma/2)
        self.Hr = torch.from_numpy(H_int.real).to(device, DTYPE)
        self.Hi = torch.from_numpy(H_int.imag).to(device, DTYPE)
        # mask for “past” vs “future” freq bins
        Nf      = window_size//2 + 1
        mask_np = np.zeros((1, Nf), dtype=np.float32)
        mask_np[0, :Nf//2] = 1.0
        self.mask = torch.from_numpy(mask_np).to(device, DTYPE)
        # number of extra features
        self.n_feat = features.shape[1]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # --- Liouville part ---
        x_win = self.windows[idx]                # (W,)
        Xc    = rfft(x_win)                      # (Nf,)
        Xr    = torch.from_numpy(Xc.real).to(self.device, DTYPE)
        Xi    = torch.from_numpy(Xc.imag).to(self.device, DTYPE)
        # fractionally integrate main series
        Yr    = Xr * self.Hr - Xi * self.Hi
        Yi    = Xr * self.Hi + Xi * self.Hr
        X_int = torch.stack([Yr, Yi], dim=0)     # (2, Nf)
        X_obs = X_int * self.mask                # (2, Nf)

        # --- Extra features part ---
        feat_win = self.features[idx]            # (n_feat, W)
        # FFT each feature => (n_feat, Nf) complex
        Fc      = np.fft.rfft(feat_win, axis=1)
        Fr      = torch.from_numpy(Fc.real).to(self.device, DTYPE)
        Fi      = torch.from_numpy(Fc.imag).to(self.device, DTYPE)
        # stack real+imag ==> (2*n_feat, Nf)
        F_stack = torch.cat([Fr, Fi], dim=0)

        return X_int, X_obs, self.mask, F_stack

# build dataset & loader
dataset = WindowedFreqConditionalDataset(
    series=liouville_full,
    features=features_full,
    gamma=GAMMA, eps=EPS,
    window_size=WINDOW_SZ,
    stride=STRIDE,
    device=DEVICE
)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# -------------------------------------------------------------------------
# 3. Precompute fractional differentiation filter for forecast
# -------------------------------------------------------------------------
f_win   = rfftfreq(WINDOW_SZ, d=1.0)
H_diff  = ((1j*2*np.pi*f_win)**2 + EPS**2)**( GAMMA/2)
Hdiff_r = torch.from_numpy(H_diff.real).to(DEVICE, DTYPE)
Hdiff_i = torch.from_numpy(H_diff.imag).to(DEVICE, DTYPE)

# -------------------------------------------------------------------------
# 4. Extend UNet to accept extra feature channels
# -------------------------------------------------------------------------
def conv_block(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv1d(ch_in,  ch_out, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(ch_out, ch_out, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet1D(nn.Module):
    def __init__(self, in_ch, base_ch=BASE_CH, depth=DEPTH):
        super().__init__()
        self.downs = nn.ModuleList()
        ch = in_ch
        for d in range(depth):
            ch_out = base_ch * (2**d)
            self.downs.append(nn.ModuleDict({
                'conv': conv_block(ch,   ch_out),
                'down': nn.Conv1d(ch_out, ch_out, 3, stride=2, padding=1)
            }))
            ch = ch_out
        self.bottleneck = conv_block(ch, ch*2)
        ch *= 2
        self.ups = nn.ModuleList()
        for d in reversed(range(depth)):
            ch_out = base_ch * (2**d)
            self.ups.append(nn.ModuleDict({
                'conv': conv_block(ch + ch_out, ch_out)
            }))
            ch = ch_out
        self.final = nn.Conv1d(ch, 2, 1)  # predict noise on main series only

    def forward(self, x):
        skips = []
        for layer in self.downs:
            x = layer['conv'](x); skips.append(x)
            x = layer['down'](x)
        x = self.bottleneck(x)
        for layer in self.ups:
            skip = skips.pop()
            x    = F.interpolate(x, size=skip.shape[-1],
                                 mode='linear', align_corners=False)
            x    = torch.cat([x, skip], dim=1)
            x    = layer['conv'](x)
        return self.final(x)

# compute in_ch = 2(noisy main) + 2(obs main) +1(mask) + 2*n_feat
in_ch = 2 + 2 + 1 + 2 * dataset.n_feat
unet  = UNet1D(in_ch=in_ch, base_ch=BASE_CH, depth=DEPTH)

# -------------------------------------------------------------------------
# 5. Conditional DDPM (pass feature channels through unchanged)
# -------------------------------------------------------------------------
class ConditionalDiffusion1D:
    def __init__(self, model, T, device):
        self.model     = model.to(device)
        betas          = torch.linspace(1e-4, 2e-2, T,
                                       device=device, dtype=DTYPE)
        self.alpha     = 1 - betas
        self.alpha_cum = torch.cumprod(self.alpha, dim=0)
        self.T         = T
        self.device    = device

    def q_sample(self, x0, t, noise):
        a = self.alpha_cum[t].sqrt().view(-1,1,1)
        b = (1 - self.alpha_cum[t]).sqrt().view(-1,1,1)
        return a * x0 + b * noise

    def p_losses(self, X_int, X_obs, mask, F_cond, t):
        noise = torch.randn_like(X_int)
        x0    = X_obs * mask + noise * (1 - mask)
        x_t   = self.q_sample(x0, t, noise)
        # concat main noisy, observed, mask, plus feature FFT cond
        inp   = torch.cat([x_t, X_obs, mask.expand_as(X_obs), F_cond], dim=1)
        pred  = self.model(inp)
        return F.mse_loss(pred * (1-mask), noise * (1-mask))

    def train(self, loader, epochs, lr=1e-4):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for ep in range(1, epochs+1):
            for X_int, X_obs, mask, F_cond in loader:
                X_int, X_obs, mask, F_cond = [
                    t.to(self.device) for t in (X_int, X_obs, mask, F_cond)
                ]
                B = X_int.size(0)
                t = torch.randint(0, self.T, (B,), device=self.device)
                loss = self.p_losses(X_int, X_obs, mask, F_cond, t)
                opt.zero_grad(); loss.backward(); opt.step()
            print(f"Epoch {ep}/{epochs}  loss={loss.item():.6f}")

    @torch.no_grad()
    def sample(self, X_obs, mask, F_cond):
        x = X_obs * mask + torch.randn_like(X_obs) * (1 - mask)
        for i in reversed(range(self.T)):
            t   = torch.full((1,), i,
                             device=self.device, dtype=torch.long)
            inp = torch.cat([x, X_obs,
                             mask.expand_as(X_obs),
                             F_cond], dim=1)
            eps = self.model(inp)
            b   = 1 - self.alpha[i]; a = self.alpha[i]
            ac  = self.alpha_cum[i]
            coef= b / (1 - ac).sqrt()
            x   = (x - coef * eps) / a.sqrt()
            x   = X_obs * mask + x * (1 - mask)
            if i > 0:
                x = x + (1 - mask) * b.sqrt() * torch.randn_like(x)
        return x

cond_diff = ConditionalDiffusion1D(unet, TIMESTEPS, DEVICE)
cond_diff.train(loader, EPOCHS)

# -------------------------------------------------------------------------
# 6. FORECAST NEXT WINDOW
# -------------------------------------------------------------------------
# last observed Liouville window
last_win = liouville_full[N_POINTS-WINDOW_SZ:N_POINTS]
Xc       = rfft(last_win)
Xr       = torch.from_numpy(Xc.real).to(DEVICE,DTYPE)
Xi       = torch.from_numpy(Xc.imag).to(DEVICE,DTYPE)
Yr       = Xr * dataset.Hr - Xi * dataset.Hi
Yi       = Xr * dataset.Hi + Xi * dataset.Hr
X_obs    = torch.stack([Yr, Yi], dim=0).unsqueeze(0)  # (1,2,Nf)
mask_win = dataset.mask.unsqueeze(0)                 # (1,1,Nf)

# last observed feature FFTs
feat_last = features_full[N_POINTS-WINDOW_SZ:N_POINTS].T  # (n_feat, W)
Fc        = np.fft.rfft(feat_last, axis=1)
Fr        = torch.from_numpy(Fc.real).to(DEVICE,DTYPE)
Fi        = torch.from_numpy(Fc.imag).to(DEVICE,DTYPE)
F_cond    = torch.cat([Fr, Fi], dim=0).unsqueeze(0)       # (1,2*n_feat,Nf)

X_int_pred = cond_diff.sample(X_obs, mask_win, F_cond)    # (1,2,Nf)
R_pred, I_pred = X_int_pred[0]
Xr_p = R_pred * Hdiff_r - I_pred * Hdiff_i
Xi_p = R_pred * Hdiff_i + I_pred * Hdiff_r
spec_pred = Xr_p.cpu().numpy() + 1j * Xi_p.cpu().numpy()
y_pred = irfft(spec_pred, n=WINDOW_SZ)                    # (WINDOW_SZ,)

# -------------------------------------------------------------------------
# 7. EVALUATE & PLOT (same as before)
# -------------------------------------------------------------------------
true_next = liouville_full[N_POINTS:N_POINTS+WINDOW_SZ]
pred_sign = np.sign(y_pred); pred_sign[pred_sign==0] = 1

accuracy = (pred_sign == true_next).mean()
mse_val  = ((y_pred - true_next)**2).mean()
tp = np.sum((pred_sign== 1)&(true_next== 1))
tn = np.sum((pred_sign==-1)&(true_next==-1))
fp = np.sum((pred_sign== 1)&(true_next==-1))
fn = np.sum((pred_sign==-1)&(true_next== 1))

print(f"Forecast Accuracy: {accuracy*100:.2f}%  MSE: {mse_val:.6f}")
print("TP, TN, FP, FN =", tp, tn, fp, fn)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(true_next, label='True λ')
plt.plot(y_pred,    label='Predicted', alpha=0.7)
plt.title('Forecast λ(N+1…N+W)'); plt.legend(); plt.show()

# Directional accuracy: predicting up/down relative to the last known observation
import numpy as np

# last observed value at time N
last_obs = liouville_full[N_POINTS-1]

# true and predicted directions relative to last_obs
dir_true_rel = np.sign(true_next - last_obs)
dir_pred_rel = np.sign(y_pred    - last_obs)
# map zeros (no change) to +1 (arbitrary)
dir_true_rel[dir_true_rel == 0] = 1
dir_pred_rel[dir_pred_rel == 0] = 1
dir_acc_rel = (dir_pred_rel == dir_true_rel).mean()
print(f"Directional accuracy (relative to last obs): {dir_acc_rel*100:.2f}%")

# Directional accuracy: predicting up/down between successive points in the forecast window
dir_true_seq = np.sign(true_next[1:] - true_next[:-1])
dir_pred_seq = np.sign(y_pred[1:]      - y_pred[:-1])
dir_true_seq[dir_true_seq == 0] = 1
dir_pred_seq[dir_pred_seq == 0] = 1
dir_acc_seq = (dir_pred_seq == dir_true_seq).mean()
print(f"Directional accuracy (successive in window): {dir_acc_seq*100:.2f}%")

# Directional accuracy and bar plot
import numpy as np
import matplotlib.pyplot as plt

# last observed value at time N
last_obs = liouville_full[N_POINTS-1]

# 1) Directional accuracy relative to last observation
dir_true_rel = np.sign(true_next - last_obs)
dir_pred_rel = np.sign(y_pred    - last_obs)
dir_true_rel[dir_true_rel == 0] = 1
dir_pred_rel[dir_pred_rel == 0] = 1
dir_acc_rel  = (dir_pred_rel == dir_true_rel).mean()

# 2) Directional accuracy between successive points in the forecast window
dir_true_seq = np.sign(true_next[1:] - true_next[:-1])
dir_pred_seq = np.sign(y_pred[1:]      - y_pred[:-1])
dir_true_seq[dir_true_seq == 0] = 1
dir_pred_seq[dir_pred_seq == 0] = 1
dir_acc_seq  = (dir_pred_seq == dir_true_seq).mean()

print(f"Directional accuracy (relative to last obs): {dir_acc_rel*100:.2f}%")
print(f"Directional accuracy (successive in window) : {dir_acc_seq*100:.2f}%")

# Bar plot of the two directional accuracies
labels = ['Relative to last obs', 'Successive in window']
accuracies = [dir_acc_rel, dir_acc_seq]

plt.figure(figsize=(6,4))
plt.bar(labels, accuracies, edgecolor='k')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Directional Accuracy Comparison')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()
