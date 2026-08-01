"""A draft model whose agreement is autocorrelated, not independent.

``DraftModel.propose()`` decides match/mismatch at each position with an
independent coin flip at rate ``agreement``. The README calls this "the
standard accounting from Leviathan et al. and Chen et al....not a
hand-wavy approximation," but the accounting is faithful to the
accept/reject *bookkeeping*; the i.i.d. Bernoulli assumption about *when*
the draft is right is not something real models exhibit. Real drafts are
bursty: they nail long templated or highly-predictable spans in unbroken
streaks, then miss in clusters on the hard, high-entropy spans (naming,
numbers, reasoning steps), because whatever makes a token hard to predict
tends to keep the next few tokens hard too.

``BurstyDraftModel`` keeps the exact same marginal match rate (verified
empirically: matches the nominal ``agreement`` within ~0.5pp at n=20000
regardless of ``burstiness``, see tests) but makes match/mismatch a
two-state Markov chain instead of independent draws, so runs of
consecutive matches and consecutive misses are longer than i.i.d. would
predict. ``burstiness=0.0`` reduces exactly to i.i.d. Bernoulli.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .model import TokenStream


@dataclass
class BurstyDraftModel:
    truth: TokenStream
    agreement: float = 0.8
    burstiness: float = 0.4
    seed: int = 0
    _rng: random.Random = field(default=None, repr=False, compare=False)
    _state: Optional[bool] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.agreement <= 1.0:
            raise ValueError("agreement must be in [0, 1]")
        if not 0.0 <= self.burstiness < 1.0:
            raise ValueError("burstiness must be in [0, 1)")
        self._rng = random.Random(self.seed)

    def _next_match(self) -> bool:
        p, b = self.agreement, self.burstiness
        if self._state is None:
            threshold = p
        elif self._state:
            # sticky in a match streak: boosted toward 1, capped there
            threshold = p + b * (1 - p)
        else:
            # sticky in a miss streak: pulled toward 0
            threshold = p - b * p
        match = self._rng.random() < threshold
        self._state = match
        return match

    def propose(self, position: int) -> int:
        if position >= len(self.truth):
            return -1
        correct = self.truth.tokens[position]
        if self._next_match():
            return correct
        wrong = correct
        while wrong == correct:
            wrong = self._rng.randint(0, 50_000)
        return wrong
