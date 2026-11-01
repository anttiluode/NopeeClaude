"""Adapters that try to give back what a cheaper Wan sampling run loses.

Shape of the idea (TBC's public description of their "hypercolumn adapter", Oasis prototype):
at every spatial location, a local centre-surround spatial filter (difference of Gaussians,
space constant lambda) followed by a leaky excitatory/inhibitory recurrence in time (time
constant tau); a small residual branch that starts with no effect. TBC fit lambda and tau to
recordings of living neurons. WE HAVE NO RECORDINGS: every constant here is either hand-chosen
("prior"), drawn at random ("random"), or learned. Nothing in this file is measured biology.

Tensors are video latents (B, C, F, H, W) or transformer tokens (B, F*h*w, D).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hand-chosen prior: a spread of centre sizes (latent pixels) and time constants (latent frames).
# Surround 3x the centre (retina-like ratio). These are NOT fitted to anything.
PRIOR_LAMBDA = (0.5, 0.8, 1.2, 1.8)
PRIOR_KAPPA = 3.0
PRIOR_TAU_E = (0.7, 1.5, 3.0, 6.0)
PRIOR_TAU_I_RATIO = 0.5
KSIZE = 7


def _spread(values, n):
    """Repeat a short tuple of constants over n channels (channel groups)."""
    v = torch.tensor(values, dtype=torch.float32)
    return v.repeat_interleave(math.ceil(n / len(v)))[:n]


def gaussian_2d(sigma, k=KSIZE):
    """sigma: (n,) -> (n, k, k) kernels, each normalised to sum 1."""
    r = torch.arange(k, dtype=sigma.dtype, device=sigma.device) - k // 2
    d2 = (r[:, None] ** 2 + r[None, :] ** 2)[None]
    g = torch.exp(-d2 / (2 * sigma[:, None, None] ** 2))
    return g / g.sum((1, 2), keepdim=True)


class DoG(nn.Module):
    """Per-channel centre-surround kernel a_c*G(lambda) - a_s*G(kappa*lambda), applied per frame."""

    def __init__(self, n, mode="learned", seed=0):
        super().__init__()
        lam = _spread(PRIOR_LAMBDA, n)
        kap = torch.full((n,), PRIOR_KAPPA)
        if mode == "random":
            g = torch.Generator().manual_seed(seed)
            lam = torch.exp(torch.empty(n).uniform_(math.log(0.3), math.log(3.0), generator=g))
            kap = torch.exp(torch.empty(n).uniform_(math.log(1.2), math.log(6.0), generator=g))
        trainable = mode == "learned"
        self.log_lam = nn.Parameter(lam.log(), requires_grad=trainable)
        self.log_kap1 = nn.Parameter((kap - 1).log(), requires_grad=trainable)
        # gains are always trainable (they set how much centre and surround count)
        self.a_c = nn.Parameter(torch.ones(n))
        self.a_s = nn.Parameter(torch.full((n,), 0.5))

    def kernel(self):
        lam = self.log_lam.exp().clamp(0.25, 4.0)
        kap = 1 + self.log_kap1.exp()
        return self.a_c[:, None, None] * gaussian_2d(lam) - self.a_s[:, None, None] * gaussian_2d((lam * kap).clamp(max=6.0))

    def forward(self, x):  # x (B, n, F, H, W)
        B, n, Fr, H, W = x.shape
        k = self.kernel().to(x.dtype)[:, None]
        y = x.permute(0, 2, 1, 3, 4).reshape(B * Fr, n, H, W)
        y = F.conv2d(F.pad(y, (KSIZE // 2,) * 4, mode="replicate"), k, groups=n)
        return y.reshape(B, Fr, n, H, W).permute(0, 2, 1, 3, 4)


class LeakyEI(nn.Module):
    """Causal leaky E/I pair per channel, scanned over frames:
        E_f = (1-a_E) E_{f-1} + a_E (u_f - w_EI * I_{f-1})
        I_f = (1-a_I) I_{f-1} + a_I (w_IE * relu(E_f)),   a = 1 - exp(-1/tau)
    Output is E. An impulse gives a damped (possibly ringing) response."""

    def __init__(self, n, mode="learned", seed=0):
        super().__init__()
        tE = _spread(PRIOR_TAU_E, n)
        tI = tE * PRIOR_TAU_I_RATIO
        if mode == "random":
            g = torch.Generator().manual_seed(seed + 1)
            tE = torch.exp(torch.empty(n).uniform_(math.log(0.3), math.log(10.0), generator=g))
            tI = torch.exp(torch.empty(n).uniform_(math.log(0.2), math.log(10.0), generator=g))
        trainable = mode == "learned"
        self.log_tE = nn.Parameter(tE.log(), requires_grad=trainable)
        self.log_tI = nn.Parameter(tI.log(), requires_grad=trainable)
        self.w_ei = nn.Parameter(torch.full((n,), 0.5))   # through softplus below
        self.w_ie = nn.Parameter(torch.full((n,), 0.5))

    def forward(self, u):  # u (B, n, F, H, W)
        aE = (1 - torch.exp(-1 / self.log_tE.exp()))[None, :, None, None].to(u.dtype)
        aI = (1 - torch.exp(-1 / self.log_tI.exp()))[None, :, None, None].to(u.dtype)
        wei = F.softplus(self.w_ei)[None, :, None, None].to(u.dtype)
        wie = F.softplus(self.w_ie)[None, :, None, None].to(u.dtype)
        E = torch.zeros_like(u[:, :, 0])
        I = torch.zeros_like(E)
        out = []
        for f in range(u.shape[2]):
            E = (1 - aE) * E + aE * (u[:, :, f] - wei * I)
            I = (1 - aI) * I + aI * wie * F.relu(E)
            out.append(E)
        return torch.stack(out, 2)


class FreeKernels(nn.Module):
    """Control: the same slots, unconstrained. Free depthwise k x k spatial kernel per channel
    and a free causal temporal kernel (length T) per channel."""

    def __init__(self, n, T=4):
        super().__init__()
        self.sp = nn.Parameter(torch.zeros(n, 1, KSIZE, KSIZE))
        with torch.no_grad():
            self.sp[:, 0, KSIZE // 2, KSIZE // 2] = 1.0
        self.T = T
        self.tm = nn.Parameter(torch.zeros(n, T))
        with torch.no_grad():
            self.tm[:, -1] = 1.0

    def forward(self, x):
        B, n, Fr, H, W = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(B * Fr, n, H, W)
        y = F.conv2d(F.pad(y, (KSIZE // 2,) * 4, mode="replicate"), self.sp.to(x.dtype), groups=n)
        y = y.reshape(B, Fr, n, H, W).permute(0, 3, 4, 1, 2).reshape(B * H * W, n, Fr)
        y = F.conv1d(F.pad(y, (self.T - 1, 0)), self.tm.to(x.dtype)[:, None], groups=n)   # causal
        return y.reshape(B, H, W, n, Fr).permute(0, 3, 4, 1, 2)


class Branch(nn.Module):
    """in-projection -> spatial -> GELU -> temporal -> out-projection (zero-init), as a residual."""

    def __init__(self, c_in, r, kind="hc_learned", seed=0):
        super().__init__()
        self.kind = kind
        self.proj_in = nn.Linear(c_in, r)
        if kind == "free":
            self.filt = FreeKernels(r)
            self.spatial = self.temporal = None
        else:
            mode = kind.split("_", 1)[1]           # learned | fixed | random
            mode = "prior" if mode == "fixed" else mode
            self.spatial = DoG(r, "learned" if mode == "learned" else mode, seed)
            self.temporal = LeakyEI(r, "learned" if mode == "learned" else mode, seed)
            self.filt = None
        self.proj_out = nn.Linear(r, c_in)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def core(self, z):  # z (B, r, F, H, W)
        if self.filt is not None:
            # same nonlinearity position as the hypercolumn: spatial part, GELU, temporal part
            B, n, Fr, H, W = z.shape
            y = z.permute(0, 2, 1, 3, 4).reshape(B * Fr, n, H, W)
            y = F.conv2d(F.pad(y, (KSIZE // 2,) * 4, mode="replicate"), self.filt.sp.to(z.dtype), groups=n)
            y = F.gelu(y.reshape(B, Fr, n, H, W).permute(0, 2, 1, 3, 4))
            y = y.permute(0, 3, 4, 1, 2).reshape(B * H * W, n, Fr)
            y = F.conv1d(F.pad(y, (self.filt.T - 1, 0)), self.filt.tm.to(z.dtype)[:, None], groups=n)
            return y.reshape(B, H, W, n, Fr).permute(0, 3, 4, 1, 2)
        return self.temporal(F.gelu(self.spatial(z)))

    def forward_grid(self, x):  # x (B, C, F, H, W) channels-first video
        z = self.proj_in(x.movedim(1, -1)).movedim(-1, 1)
        return self.proj_out(self.core(z).movedim(1, -1)).movedim(-1, 1)


class LatentAdapter(nn.Module):
    """Acts once on the final latent of a cheap run: x -> x + branch(x)."""

    def __init__(self, channels, kind="hc_learned", r=None, seed=0):
        super().__init__()
        self.branch = Branch(channels, r or channels, kind, seed)

    def forward(self, x):
        return x + self.branch.forward_grid(x.float()).to(x.dtype)


class Unsharp(nn.Module):
    """Training-free baseline: x + g * (x - G_sigma * x) per frame. Two scalars, grid-searched."""

    def __init__(self, sigma=1.0, gain=0.0):
        super().__init__()
        self.sigma, self.gain = float(sigma), float(gain)

    def forward(self, x):
        if self.gain == 0.0:
            return x
        B, C, Fr, H, W = x.shape
        k = gaussian_2d(torch.tensor([self.sigma]))[0].to(x)[None, None].expand(C, 1, KSIZE, KSIZE)
        y = x.permute(0, 2, 1, 3, 4).reshape(B * Fr, C, H, W)
        blur = F.conv2d(F.pad(y, (KSIZE // 2,) * 4, mode="replicate"), k, groups=C)
        out = y + self.gain * (y - blur)
        return out.reshape(B, Fr, C, H, W).permute(0, 2, 1, 3, 4)


# ---------------------------------------------------------------- in-block (TBC's actual shape)

class BlockAdapters(nn.Module):
    """Parallel residual branches on the outputs of chosen transformer blocks.
    Tokens (B, F*h*w, D) are viewed as a (B, D, F, h, w) grid; the grid size is captured from the
    transformer's input by a pre-hook, so the adapter follows any resolution."""

    def __init__(self, dim, blocks=(0, 1, 2, 3), r=32, kind="hc_learned", seed=0):
        super().__init__()
        self.blocks = tuple(blocks)
        self.branches = nn.ModuleDict({str(b): Branch(dim, r, kind, seed + b) for b in self.blocks})
        self.grid = None
        self._handles = []

    def install(self, transformer):
        self.remove()
        p_t, p_h, p_w = transformer.config.patch_size

        def pre(module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            _, _, Fr, H, W = hs.shape
            self.grid = (Fr // p_t, H // p_h, W // p_w)

        self._handles.append(transformer.register_forward_pre_hook(pre, with_kwargs=True))
        for b in self.blocks:
            self._handles.append(transformer.blocks[b].register_forward_hook(self._hook(str(b))))
        return self

    def _hook(self, key):
        def fn(module, args, out):
            Fr, h, w = self.grid
            B, L, D = out.shape
            g = out.float().reshape(B, Fr, h, w, D).permute(0, 4, 1, 2, 3)
            d = self.branches[key].forward_grid(g)
            return out + d.permute(0, 2, 3, 4, 1).reshape(B, L, D).to(out.dtype)
        return fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def n_params(m, trainable_only=True):
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)
