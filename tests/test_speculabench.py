from speculabench.decode import SpeculativeDecoder
from speculabench.model import DraftModel, TargetModel, make_stream


def _decoder(agreement, draft_length=4, ratio=0.1, n=500, seed=0):
    truth = make_stream(n, seed=seed)
    draft = DraftModel(truth, agreement=agreement, seed=seed)
    return SpeculativeDecoder(draft, TargetModel(truth), draft_length, ratio), n


def test_output_length_is_exact():
    dec, n = _decoder(0.8)
    r = dec.run(n)
    assert r.tokens_generated == n


def test_perfect_draft_accepts_everything():
    dec, n = _decoder(agreement=1.0)
    r = dec.run(n)
    # a perfect draft should be accepted essentially all the time
    assert r.acceptance_rate > 0.99


def test_useless_draft_still_correct_but_slow():
    # agreement 0.0 means every proposal is wrong; speedup should be poor
    dec, n = _decoder(agreement=0.0)
    r = dec.run(n)
    assert r.acceptance_rate == 0.0
    assert r.speedup < 1.0  # the draft overhead is pure loss here


def test_higher_agreement_gives_higher_speedup():
    lo, n = _decoder(agreement=0.5)
    hi, _ = _decoder(agreement=0.9)
    assert hi.run(n).speedup > lo.run(n).speedup


def test_speedup_beats_baseline_for_good_draft():
    dec, n = _decoder(agreement=0.85)
    assert dec.run(n).speedup > 1.0


def test_deterministic():
    a, n = _decoder(agreement=0.8)
    b, _ = _decoder(agreement=0.8)
    assert a.run(n).speedup == b.run(n).speedup


def test_draft_length_must_be_positive():
    truth = make_stream(10)
    import pytest
    with pytest.raises(ValueError):
        SpeculativeDecoder(DraftModel(truth), TargetModel(truth), draft_length=0)
