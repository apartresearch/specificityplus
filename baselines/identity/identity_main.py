from typing import Any, Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from .identity_hparams import IDENTITYHyperParams


def apply_identity_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: IDENTITYHyperParams,
        copy=False,
        return_orig_weights=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    return model, {}
