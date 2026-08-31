"""M4 macro-action codebooks and receding-horizon latent planning.

Codebook fitting intentionally remains usable without the optional PyTorch
training dependencies. Tensor planning symbols are exported when torch exists.
"""

from .macro_codebook import (
    BASIC_CODE_NAMES,
    MacroCode,
    MacroCodebook,
    MacroCodebookFitConfig,
    build_macro_codebook,
    fit_macro_codebook_from_episodes,
    resample_actions_to_model_ticks,
)

__all__ = [
    "BASIC_CODE_NAMES",
    "MacroCode",
    "MacroCodebook",
    "MacroCodebookFitConfig",
    "build_macro_codebook",
    "fit_macro_codebook_from_episodes",
    "resample_actions_to_model_ticks",
]

try:
    from .cem import HybridCEMConfig, HybridCEMPlanner, PlanResult
    from .legality import LegalityContext, expand_macro_codes, legal_code_mask
    from .mpc import RecedingHorizonMPC, canonical_to_minerl_action, first_macro_actions
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "HybridCEMConfig",
            "HybridCEMPlanner",
            "LegalityContext",
            "PlanResult",
            "RecedingHorizonMPC",
            "canonical_to_minerl_action",
            "expand_macro_codes",
            "first_macro_actions",
            "legal_code_mask",
        ]
    )
