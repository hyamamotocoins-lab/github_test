"""Campaign B exception hierarchy (fail-closed vs candidate-local)."""

from __future__ import annotations


class CampaignFatalError(RuntimeError):
    """Stop the entire campaign; do not archive as ordinary rejection."""


class CandidateRejected(RuntimeError):
    """Reject one candidate and continue the campaign."""


class NeedCanonicalM2(CampaignFatalError):
    """Required shared M2 is missing and generation is not allowed."""


class TimeBudgetClosed(RuntimeError):
    """Wall-clock budget forbids starting a new heavy stage."""


class InvariantViolation(CampaignFatalError):
    """Certification / staged-only / M6 / Campaign C invariant broken."""
