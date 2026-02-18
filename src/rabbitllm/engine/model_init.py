"""Model skeleton creation with attention implementation fallback."""

import logging
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def create_model_with_attn_fallback(
    fallback_chain: List[str],
    create_fn: Callable[[str], object],
    clean_memory_fn: Callable[[], None],
) -> Tuple[Optional[object], Optional[str]]:
    """Try creating the model with each attention implementation until one succeeds.

    Parameters
    ----------
    fallback_chain : list of str
        Order of attn implementations to try (e.g. ["flash_attention_2", "sdpa", "eager"]).
    create_fn : callable
        create_fn(attn_implementation) returns the model (on meta device).
    clean_memory_fn : callable
        Called after each failed attempt to free memory.

    Returns
    -------
    tuple (model, active_attn_implementation)
        Model and the impl that worked, or (None, None) if all failed.
    """
    model = None
    active_impl = None
    for impl in fallback_chain:
        try:
            model = create_fn(impl)
            active_impl = impl
            logger.info("Model initialized with attn_implementation='%s'", impl)
            return (model, active_impl)
        except (ValueError, TypeError):
            logger.info(
                "attn_implementation='%s' not supported for this model, trying next.",
                impl,
            )
            if model is not None:
                del model
            clean_memory_fn()
            model = None
    return (None, None)
