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
    'run_end_to_end',
    'EndToEndConfig',
    'load_end_to_end_config',
    'run_post_m2_pipeline',
    'collect_pipeline_status',
    'scan_and_update_catalog',
    'recover_interrupted_work',
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
    if name == 'run_end_to_end':
        from .end_to_end import run_end_to_end
        return run_end_to_end
    if name == 'EndToEndConfig':
        from .end_to_end import EndToEndConfig
        return EndToEndConfig
    if name == 'load_end_to_end_config':
        from .end_to_end import load_end_to_end_config
        return load_end_to_end_config
    if name == 'run_post_m2_pipeline':
        from .post_m2_pipeline import run_post_m2_pipeline
        return run_post_m2_pipeline
    if name == 'collect_pipeline_status':
        from .pipeline_status import collect_pipeline_status
        return collect_pipeline_status
    if name == 'scan_and_update_catalog':
        from .m6_certified_catalog import scan_and_update_catalog
        return scan_and_update_catalog
    if name == 'recover_interrupted_work':
        from .pipeline_recovery import recover_interrupted_work
        return recover_interrupted_work
    raise AttributeError(name)
