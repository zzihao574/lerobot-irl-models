import math
import torch
from functools import partial


class ExponentialMovingAverage:
    """Lightweight EMA helper for model parameters."""

    def __init__(self, parameters, decay: float):
        if not (0.0 < decay <= 1.0):
            raise ValueError(f"EMA decay must be in (0, 1], got {decay}")
        self.decay = decay
        self.shadow_params = [
            p.detach().clone()
            for p in parameters
            if p.requires_grad
        ]
        self.collected_params = None

    def _iter_params(self, parameters):
        return [p for p in parameters if p.requires_grad]

    @torch.no_grad()
    def update(self, parameters):
        params = self._iter_params(parameters)
        if len(params) != len(self.shadow_params):
            raise ValueError("Parameter set changed after EMA initialization.")
        one_minus_decay = 1.0 - self.decay
        for shadow, param in zip(self.shadow_params, params, strict=True):
            shadow.mul_(self.decay).add_(param.detach(), alpha=one_minus_decay)

    @torch.no_grad()
    def store(self, parameters):
        self.collected_params = [
            p.detach().clone()
            for p in self._iter_params(parameters)
        ]

    @torch.no_grad()
    def copy_to(self, parameters):
        params = self._iter_params(parameters)
        if len(params) != len(self.shadow_params):
            raise ValueError("Parameter set changed after EMA initialization.")
        for param, shadow in zip(params, self.shadow_params, strict=True):
            param.copy_(shadow)

    @torch.no_grad()
    def restore(self, parameters):
        if self.collected_params is None:
            return
        params = self._iter_params(parameters)
        if len(params) != len(self.collected_params):
            raise ValueError("Parameter set changed after EMA store.")
        for param, saved in zip(params, self.collected_params, strict=True):
            param.copy_(saved)
        self.collected_params = None


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def make_sample_density(
    sigma_sample_density_type, sigma_max=80, sigma_min=0.001, sigma_data=0.5
):
    if sigma_sample_density_type == "loglogistic":
        loc = math.log(sigma_data)
        scale = 0.5
        min_value = sigma_min
        max_value = sigma_max
        return partial(
            rand_log_logistic,
            loc=loc,
            scale=scale,
            min_value=min_value,
            max_value=max_value,
        )


def rand_log_logistic(
    shape,
    loc=0.0,
    scale=1.0,
    min_value=0.0,
    max_value=float("inf"),
    device="cpu",
    dtype=torch.float32,
):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = (
        torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf)
        + min_cdf
    )
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(
        math.log(sigma_max), math.log(sigma_min), n, device=device
    ).exp()
    return torch.cat([sigmas, sigmas.new_zeros([1])])


@torch.no_grad()
def sample_ddim(
    model,
    state,
    action,
    goal,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,  # Reserved for progress-bar compatibility (currently unused).
):
    """
    DPM-Solver 1( or DDIM sampler"""
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])

    def sigma_fn(t):
        return t.neg().exp()

    def t_fn(sigma):
        return sigma.log().neg()

    for i in range(len(sigmas) - 1):
        # predict the next action
        if isinstance(state, tuple):
            denoised = model(
                state[0], state[1], action, goal, sigmas[i] * s_in, **extra_args
            )
        else:
            denoised = model(state, action, goal, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "action": action,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
    return action
