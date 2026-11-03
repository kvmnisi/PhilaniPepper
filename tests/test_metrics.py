import math

import pytest

from backend.evaluation.metrics import (
    cer,
    classification_report,
    corpus_error_rate,
    latency_summary,
    mcnemar_exact,
    wer,
)


def test_wer_identical_after_normalisation():
    assert wer("Ngiyabonga kakhulu.", "ngiyabonga kakhulu")["rate"] == 0.0


def test_wer_counts_each_operation():
    result = wer("a b c d", "a x c d e")
    assert (result["substitutions"], result["deletions"], result["insertions"]) == (1, 0, 1)
    assert result["rate"] == pytest.approx(0.5)


def test_wer_can_exceed_one_through_insertions():
    # The AST baseline reported 134.71% WER for this reason.
    assert wer("yebo", "yeah bongeya south")["rate"] == pytest.approx(3.0)


def test_digits_versus_words_is_not_normalised_away():
    result = wer("eight oh one", "801")
    assert result["errors"] == 3


def test_cer_ignores_spaces():
    assert cer("ngeviki elilandelayo", "ngevikielilandelayo")["rate"] == 0.0
    assert cer("abcd", "abed")["rate"] == pytest.approx(0.25)


def test_empty_reference():
    assert wer("", "")["rate"] == 0.0
    assert math.isinf(wer("", "something")["rate"])


def test_corpus_wer_weights_by_reference_length():
    pairs = [("a b c d", "a b c d"), ("a b", "x b")]
    assert corpus_error_rate(pairs) == pytest.approx(1 / 6)


def test_macro_f1_matches_hand_computation():
    y_true = ["a", "a", "b", "b", "c"]
    y_pred = ["a", "b", "b", "b", "a"]
    report = classification_report(y_true, y_pred)
    # a: P=1/2 R=1/2 F=0.5 ; b: P=2/3 R=1 F=0.8 ; c: P=0 R=0 F=0
    assert report["macro_f1"] == pytest.approx((0.5 + 0.8 + 0.0) / 3)
    assert report["accuracy"] == pytest.approx(3 / 5)


def test_mcnemar_exact_values():
    a = [True] * 10 + [False] * 2
    b = [False] * 10 + [True] * 2
    result = mcnemar_exact(a, b)
    assert (result["only_a_correct"], result["only_b_correct"]) == (10, 2)
    # two-sided exact binomial, n=12, k=2: 2 * (1 + 12 + 66) / 4096
    assert result["p_value"] == pytest.approx(2 * 79 / 4096)


def test_mcnemar_no_discordant_pairs():
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0


def test_latency_summary():
    summary = latency_summary([10, 20, 30, None, 40])
    assert summary["n"] == 4
    assert summary["median_ms"] == 20
    assert summary["max_ms"] == 40
    assert latency_summary([None]) is None
