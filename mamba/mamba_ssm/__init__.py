__version__ = "2.3.2.post1"

# [VIM MODIFIED] 新增 bimamba_inner_fn / mamba_inner_fn_no_out_proj 导出
from mamba_ssm.ops.selective_scan_interface import (
    selective_scan_fn,
    mamba_inner_fn,
    bimamba_inner_fn,            # [VIM ADDED]
    mamba_inner_fn_no_out_proj,  # [VIM ADDED]
)
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
