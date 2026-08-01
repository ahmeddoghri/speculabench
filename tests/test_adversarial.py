"""Tests for the i.i.d.-agreement assumption and the bursty-model fix."""

from __future__ import annotations

import pytest

from speculabench.decode import SpeculativeDecoder
from speculabench.eval_v2 import DRAFT_LENGTHS, build_report
from speculabench.model import DraftModel, TargetModel, make_stream
from speculabench.model_v2 import BurstyDraftModel

# --- the finding: i.i.d. agreement misreports what a bursty draft supports --

def test_bursty_model_matches_nominal_agreement_rate():
    """burstiness must not be a backdoor way of cheating the marginal rate."""
    truth = make_stream(20_000, seed=0)
    draft = BurstyDraftModel(truth, agreement=0.8, burstiness=0.4, seed=0)
    matches = sum(
        1 for i in range(len(truth)) if draft.propose(i) == truth.tokens[i]
    )
    assert matches / len(truth) == pytest.approx(0.8, abs=0.01)


def test_burstiness_zero_reduces_to_iid_speedup():
    truth = make_stream(20_000, seed=0)
    iid = SpeculativeDecoder(
        DraftModel(truth, agreement=0.8, seed=0), TargetModel(truth),
        draft_length=4, draft_cost_ratio=0.1,
    ).run(20_000)
    bursty = SpeculativeDecoder(
        BurstyDraftModel(truth, agreement=0.8, burstiness=0.0, seed=0),
        TargetModel(truth), draft_length=4, draft_cost_ratio=0.1,
    ).run(20_000)
    assert bursty.speedup == pytest.approx(iid.speedup, rel=0.05)


def test_bursty_draft_beats_iid_at_the_same_nominal_agreement():
    """Same marginal accuracy, clustered instead of independent misses:
    longer streaks land more of the window, so speedup is higher."""
    report = build_report(agreement=0.8, burstiness=0.4, total_tokens=20_000, seed=0)
    for iid_row, bursty_row in zip(report["iid"], report["bursty"]):
        if iid_row["draft_length"] >= 4:
            assert bursty_row["speedup"] >= iid_row["speedup"]


def test_recommended_draft_length_shifts_with_burstiness():
    report = build_report(agreement=0.8, burstiness=0.4, total_tokens=20_000, seed=0)
    assert report["bursty_best"]["draft_length"] > report["iid_best"]["draft_length"]


# --- held out seed, evaluated once ------------------------------------------

def test_holdout_seed_reproduces_the_same_qualitative_shift():
    report = build_report(agreement=0.8, burstiness=0.4, total_tokens=20_000, seed=99)
    assert report["bursty_best"]["draft_length"] >= report["iid_best"]["draft_length"]
    assert report["bursty_best"]["speedup"] > report["iid_best"]["speedup"]


# --- the original benchmark is unaffected -----------------------------------

def test_original_eval_module_untouched():
    import speculabench.model as model
    assert not hasattr(model, "BurstyDraftModel")


def test_original_benchmark_still_reproduces():
    """python -m speculabench.eval must still print the exact published numbers."""
    truth = make_stream(2000, seed=0)
    target = TargetModel(truth)
    draft = DraftModel(truth, agreement=0.8, seed=0)
    dec = SpeculativeDecoder(draft, target, draft_length=4, draft_cost_ratio=0.1)
    r = dec.run(2000)
    assert r.acceptance_rate == pytest.approx(0.572, abs=0.01)
    assert r.speedup == pytest.approx(2.35, abs=0.02)


# --- report shape / reproducibility -----------------------------------------

def test_report_is_reproducible():
    a = build_report(seed=0)
    b = build_report(seed=0)
    assert a == b


def test_report_covers_all_draft_lengths():
    report = build_report()
    assert [r["draft_length"] for r in report["iid"]] == list(DRAFT_LENGTHS)
    assert [r["draft_length"] for r in report["bursty"]] == list(DRAFT_LENGTHS)
