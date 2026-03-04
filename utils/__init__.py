# Re-export everything from submodules so existing imports keep working.
# e.g. `from utils import get_nonlinearity` still resolves.

from utils.helpers import *  # noqa: F401,F403
from utils.helpers import (
    COLOR,
    CtxCollator,
    aggregate_score_dict,
    compute_masked_statistics,
    compute_masked_statistics_with_var,
    concat_and_pad,
    convert_to_dict_of_lists,
    empty_gpu_cache,
    get_all_submodules,
    get_first_value_split,
    get_nonlinearity,
    is_oom_exception,
    merge_statistics,
    merge_statistics_from_var,
    pack_attn_mxs,
    pack_kv_cache,
    pad_and_concat_buffered_attn_mxs,
    pop_stats_from_dict_of_lists,
    safe_tensor_print,
)

# Note: utils.hydra_helpers and utils.longbench are NOT imported here to
# avoid circular imports (hydra_helpers imports from main which imports
# from namm.trainer which imports from utils). Hydra _target_ references
# use the full path e.g. utils.hydra_helpers.LlamaCompatModel.
