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
    'run_pre_m6_batch',
    'run_close_obligations_batch',
    'run_m6_batch',
    'run_pipeline_to_m6',
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
    if name == 'run_pre_m6_batch':
        from .pre_m6_batch import run_pre_m6_batch
        return run_pre_m6_batch
    if name == 'run_close_obligations_batch':
        from .close_obligations import run_close_obligations_batch
        return run_close_obligations_batch
    if name == 'run_m6_batch':
        from .m6_batch import run_m6_batch
        return run_m6_batch
    if name == 'run_pipeline_to_m6':
        from .pipeline_to_m6 import run_pipeline_to_m6
        return run_pipeline_to_m6
    raise AttributeError(name)
