"""Does the i.i.d. Bernoulli agreement assumption change the recommendation?

``speculabench.eval`` sweeps draft length against speedup and reports a
"sweet spot," the draft length past which longer speculation stops paying
for itself. That number depends entirely on how the draft's matches and
misses are distributed over positions. The original model draws
match/mismatch independently at each position; real drafts are bursty
(see :mod:`speculabench.model_v2`).

This holds the *marginal* agreement rate fixed and compares the i.i.d.
model against a bursty one with the same nominal agreement, at the same
seeds, to show how much the reported speedup and the recommended draft
length shift once matches are allowed to cluster the way they do in
practice.

    python -m speculabench.eval_v2
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

from .decode import SpeculativeDecoder
from .model import DraftModel, TargetModel, make_stream
from .model_v2 import BurstyDraftModel

DRAFT_LENGTHS = (1, 2, 4, 6, 8, 12)


def _sweep(model_cls, agreement: float, total_tokens: int, seed: int, **kw) -> List[Dict]:
    truth = make_stream(total_tokens, seed=seed)
    target = TargetModel(truth)
    rows = []
    for dl in DRAFT_LENGTHS:
        draft = model_cls(truth, agreement=agreement, seed=seed, **kw)
        dec = SpeculativeDecoder(draft, target, draft_length=dl, draft_cost_ratio=0.1)
        r = dec.run(total_tokens)
        rows.append({
            "draft_length": dl,
            "acceptance_rate": round(r.acceptance_rate, 4),
            "speedup": round(r.speedup, 4),
        })
    return rows


def _best(rows: List[Dict]) -> Dict:
    return max(rows, key=lambda r: r["speedup"])


def build_report(agreement: float = 0.8, burstiness: float = 0.4,
                  total_tokens: int = 20_000, seed: int = 0) -> Dict:
    iid = _sweep(DraftModel, agreement, total_tokens, seed)
    bursty = _sweep(BurstyDraftModel, agreement, total_tokens, seed, burstiness=burstiness)
    return {
        "agreement": agreement,
        "burstiness": burstiness,
        "iid": iid,
        "bursty": bursty,
        "iid_best": _best(iid),
        "bursty_best": _best(bursty),
    }


def format_report(report: Dict) -> str:
    lines = [
        f"draft length vs speedup at matched agreement={report['agreement']:.2f} "
        f"(iid vs bursty, burstiness={report['burstiness']:.2f})",
        "=" * 78,
        f"{'draft_len':>10}{'iid accept':>12}{'iid speedup':>13}"
        f"{'bursty accept':>15}{'bursty speedup':>16}",
        "-" * 78,
    ]
    for i, dl in enumerate(DRAFT_LENGTHS):
        iid_row = report["iid"][i]
        b_row = report["bursty"][i]
        lines.append(
            f"{dl:>10}{iid_row['acceptance_rate']:>12.1%}{iid_row['speedup']:>12.2f}x"
            f"{b_row['acceptance_rate']:>15.1%}{b_row['speedup']:>15.2f}x"
        )
    lines.append("")
    ib, bb = report["iid_best"], report["bursty_best"]
    lines.append(
        f"iid sweet spot:    draft_length={ib['draft_length']:<3} ({ib['speedup']:.2f}x)"
    )
    lines.append(
        f"bursty sweet spot: draft_length={bb['draft_length']:<3} ({bb['speedup']:.2f}x)"
    )
    if ib["draft_length"] != bb["draft_length"]:
        lines.append(
            "\nSame nominal agreement rate, different recommended draft length. "
            "The i.i.d. model tells you to stop speculating earlier than a "
            "bursty draft with the identical marginal accuracy actually "
            "supports."
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agreement", type=float, default=0.8)
    parser.add_argument("--burstiness", type=float, default=0.4)
    parser.add_argument("--tokens", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    report = build_report(args.agreement, args.burstiness, args.tokens, args.seed)
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
