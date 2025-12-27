from .denoiser_head import DenoiserHead
from .identity_head import IdentityHead

def build_denoiser_head(cfg):

    head_type = cfg.get("head_type", "supervise").lower()
    if head_type == "supervise":
        return DenoiserHead(**cfg)
    elif head_type == "identity":         # ★ 新增
        return IdentityHead()
    else:
        raise ValueError(f"Unknown head_type {head_type}")