"""R^3 diffusion methods."""
import numpy as np
from scipy.special import gamma
import torch


class R3Diffuser:
    """VP-SDE diffuser class for translations."""

    def __init__(self, r3_conf):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b

    def _scale(self, x):
        return x * self._r3_conf.coordinate_scaling

    def _unscale(self, x):
        return x / self._r3_conf.coordinate_scaling

    def b_t(self, t):
        return self.min_b + t * (self.max_b - self.min_b)

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return np.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t) * x

    def sample_ref(self, n_samples: float=1):
        return np.random.normal(size=(n_samples, 3))
    
    def marginal_b_t(self, t):
        return t * self.min_b + 0.5 * (t**2) * (self.max_b - self.min_b)

    def calc_trans_0(self, score_t, x_t, t, use_torch=True):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1/2*beta_t)

    def forward(self, x_t_1: np.ndarray, t: float, num_t: int):
        """Samples marginal p(x(t) | x(t-1)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t).to(x_t_1.device)
        z_t_1 = torch.tensor(np.random.normal(size=x_t_1.shape)).to(x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * np.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std

    def forward_marginal(self, x_0: np.ndarray, t: float):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_0 = self._scale(x_0)
        x_t = np.random.normal(
            loc=np.exp(-1/2*self.marginal_b_t(t)) * x_0,
            scale=np.sqrt(1 - np.exp(-self.marginal_b_t(t)))
        )
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        return x_t, score_t

    # def score_scaling(self, t: float):
    #     return 1 / np.sqrt(self.conditional_var(t))
    
    def score_scaling(self, t):
        return 1 / torch.sqrt(self.conditional_var(t, use_torch=True))

    def reverse(self, *, x_t, score_t, t, dt, mask=None, center=True, noise_scale=1.0):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        if not torch.is_tensor(x_t):
            x_t = torch.tensor(x_t)
        device = x_t.device
        
        x_t = x_t * self._r3_conf.coordinate_scaling
        
        # 预先在外面存好 sqrt_dt_tensor，不要每一步都创建
        if not hasattr(self, '_sqrt_dt_t'):
            self._sqrt_dt_t = torch.sqrt(torch.tensor(dt, device=device))
        
        b_t = self.min_b + t*(self.max_b - self.min_b)
        g_t = torch.sqrt(b_t)
        f_t = -1/2 * b_t * x_t
        
        # 生成噪声（直接在 GPU 上生成）
        z = noise_scale * torch.randn_like(score_t)
        
        # 计算扰动
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * self._sqrt_dt_t * z

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = torch.ones(x_t.shape[:-1], device=device)
            
        x_t_1 = x_t - perturb   #(11,135,3)
        
        if center:
            mask_sum = torch.sum(mask, dim=-1, keepdim=True)
            com = torch.sum(x_t_1, dim=-2) / (mask_sum + 1e-10)
            x_t_1 -= com.unsqueeze(-2) # 自动对齐维度
    
        # 反缩放并返回
        return x_t_1 / self._r3_conf.coordinate_scaling

    def conditional_var(self, t, use_torch=True):
        """xt|x0 的条件方差"""
        # 默认设为 True，强制走 GPU 路径
        if use_torch or torch.is_tensor(t):
            return 1.0 - torch.exp(-self.marginal_b_t(t))
        return 1.0 - np.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, use_torch=False, scale=False):
        if use_torch:
            exp_fn = torch.exp
        else:
            exp_fn = np.exp
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return -(x_t - exp_fn(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t, use_torch=use_torch)
