import pytest

from scripts.validate_prediction import load_trial, paired_outcome_change


@pytest.mark.parametrize('f,g,added,lost,censored', [
    (False, True, True, False, False), (True, False, False, True, False),
    (True, True, False, False, False), (False, False, False, False, False),
    (None, True, False, False, True), (True, None, False, False, True),
    (None, False, False, False, True), (False, None, False, False, True),
    (None, None, False, False, True),
])
def test_censored_pairs_never_count_as_gain_or_loss(f, g, added, lost, censored):
    assert paired_outcome_change(f, g) == {
        'added_over_F': added, 'lost_F_success': lost, 'comparison_censored': censored}


def test_missing_archived_trial_fails_without_creating_files(tmp_path):
    path = tmp_path/'missing_archive'
    with pytest.raises(FileNotFoundError):
        load_trial(path)
    assert not path.exists()
