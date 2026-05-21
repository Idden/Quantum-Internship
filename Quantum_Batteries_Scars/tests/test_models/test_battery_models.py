"""Tests for quantum battery model implementations."""

from data_utils import save_metadata


def test_placeholder(test_subdir):
    """Placeholder test - replace when models are implemented."""
    # --- Artifacts ---
    save_metadata(test_subdir / "metadata.json", {
        'test_name': 'test_placeholder',
        'status': 'placeholder — replace when battery models are implemented',
        'artifacts': ['metadata.json'],
    })
