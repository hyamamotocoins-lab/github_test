"""Campaign B (S2) autonomous exploration package."""

from .errors import (
    CampaignFatalError,
    CandidateRejected,
    NeedCanonicalM2,
    TimeBudgetClosed,
)

__all__ = [
    'CampaignBConfig',
    'CampaignFatalError',
    'CandidateRejected',
    'NeedCanonicalM2',
    'TimeBudgetClosed',
    'load_campaign_b_config',
    'run_campaign_b',
    'run_mass_explore',
    'run_advance_selected',
    'run_gpu_m3_batch',
]


def __getattr__(name: str):
    if name == 'CampaignBConfig':
        from .config import CampaignBConfig
        return CampaignBConfig
    if name == 'load_campaign_b_config':
        from .config import load_campaign_b_config
        return load_campaign_b_config
    if name == 'run_campaign_b':
        from .driver import run_campaign_b
        return run_campaign_b
    if name == 'run_mass_explore':
        from .mass_explore import run_mass_explore
        return run_mass_explore
    if name == 'run_advance_selected':
        from .advance_selected import run_advance_selected
        return run_advance_selected
    if name == 'run_gpu_m3_batch':
        from .gpu_m3_batch import run_gpu_m3_batch
        return run_gpu_m3_batch
    raise AttributeError(name)
