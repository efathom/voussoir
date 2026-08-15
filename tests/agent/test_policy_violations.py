"""Locks the PolicyViolation StrEnum: Phase 5 variants are present with the
correct lowercase snake_case string values, and pre-existing variants have not
been renamed or removed.
"""

from voussoir.agent import PolicyViolation


def test_new_phase5_variants_present():
    assert PolicyViolation.CAPABILITY_DENIED == "capability_denied"
    assert PolicyViolation.TAINT_EXFILTRATION == "taint_exfiltration"
    assert PolicyViolation.CAPABILITY_CLAMPED_EMPTY == "capability_clamped_empty"


def test_existing_variants_still_present():
    # regression: don't accidentally rename earlier variants
    assert PolicyViolation.MAX_STEPS == "max_steps"
    assert PolicyViolation.DELEGATE_NOT_FOUND == "delegate_not_found"
    assert PolicyViolation.STREAMING_NOT_SUPPORTED == "streaming_not_supported"


def test_authz_denied_variant():
    assert PolicyViolation.AUTHZ_DENIED == "authz_denied"
