"""Benchmark: how does speculative decoding speedup move with the two knobs
that actually matter, draft acceptance rate and draft length?

Run it:

    python -m speculabench.eval

Everything is deterministic (fixed seeds), so the numbers reproduce exactly on
your machine. No GPU, no weights, no API keys.
"""
from __future__ import annotations

from speculabench.decode import SpeculativeDecoder
from speculabench.model import DraftModel, TargetModel, make_stream


def run(total_tokens: int = 2000, seed: int = 0) -> None:
    truth = make_stream(total_tokens, seed=seed)
    target = TargetModel(truth)

    print(f"workload: generate {total_tokens} tokens, draft is 10x cheaper than target\n")

    print("acceptance rate vs speedup (draft_length=4):")
    print(f"  {'agreement':>10}  {'accept rate':>12}  {'speedup':>9}  {'tokens/pass':>12}")
    for agreement in (0.5, 0.6, 0.7, 0.8, 0.9):
        draft = DraftModel(truth, agreement=agreement, seed=seed)
        dec = SpeculativeDecoder(draft, target, draft_length=4, draft_cost_ratio=0.1)
        r = dec.run(total_tokens)
        print(f"  {agreement:>10.2f}  {r.acceptance_rate:>11.1%}  "
              f"{r.speedup:>8.2f}x  {r.avg_tokens_per_target_call:>12.2f}")

    print("\ndraft length vs speedup (agreement=0.8):")
    print(f"  {'draft_len':>10}  {'accept rate':>12}  {'speedup':>9}  {'tokens/pass':>12}")
    best_len, best_speedup = 1, 0.0
    for draft_length in (1, 2, 4, 6, 8, 12):
        draft = DraftModel(truth, agreement=0.8, seed=seed)
        dec = SpeculativeDecoder(draft, target, draft_length=draft_length, draft_cost_ratio=0.1)
        r = dec.run(total_tokens)
        if r.speedup > best_speedup:
            best_len, best_speedup = draft_length, r.speedup
        print(f"  {draft_length:>10}  {r.acceptance_rate:>11.1%}  "
              f"{r.speedup:>8.2f}x  {r.avg_tokens_per_target_call:>12.2f}")

    print(f"\nsweet spot: draft_length={best_len} gives the best speedup "
          f"({best_speedup:.2f}x) at 80% agreement.")
    print("longer drafts help until the extra rejected tokens stop paying for "
          "themselves. that turnover point is the whole game.")


if __name__ == "__main__":
    run()
