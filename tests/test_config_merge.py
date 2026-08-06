import pytest
from icare_risk.utils import deep_merge


def test_deep_merge_duplicate_warning(capsys):
    """Verify that deep_merge prints a warning when overwriting defaults."""

    default_config = {
        'custom_scores': {
            'increment_score': {'points': 10}
        }
    }

    user_config = {
        'custom_scores': {
            'increment_score': {'points': 99},  # This overlaps!
            'new_score': {'points': 5}  # This is safely new
        }
    }

    # Perform the merge
    merged = deep_merge(default_config, user_config)

    # Capture the terminal output
    captured = capsys.readouterr()

    # Assertions
    assert "⚠️ Override Alert" in captured.out
    assert "custom_scores -> increment_score -> points" in captured.out
    assert merged['custom_scores']['increment_score']['points'] == 99
    assert merged['custom_scores']['new_score']['points'] == 5


def test_deep_merge_strict_mode_failure():
    """Verify that strict mode raises a hard error on collision."""

    default_config = {'feature': True}
    user_config = {'feature': False}

    with pytest.raises(ValueError, match="Strict Mode Error"):
        deep_merge(default_config, user_config, strict=True)