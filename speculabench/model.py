"""Deterministic stand-in models so the whole simulation runs with no GPU,
no weights, and no API keys.

The point of speculabench is not to run a real transformer. It is to model the
accept/reject dynamics of speculative decoding faithfully enough that the
speedup math is honest. So a "model" here is just something that emits a token
distribution, and agreement between draft and target is controlled by a single
knob: how often they agree at each position.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class TokenStream:
    """The ground-truth sequence the target model would produce on its own.

    Speculative decoding must produce exactly this sequence (that is the whole
    guarantee: same output as the target, just faster). We generate it once and
    treat it as truth.
    """

    tokens: list[int]

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class TargetModel:
    """The big, slow, authoritative model.

    It knows the true continuation. Its job in the loop is to verify draft
    proposals: it accepts a proposed token only if it matches what it would
    have generated itself at that position.
    """

    truth: TokenStream

    def verify(self, position: int, proposed: int) -> bool:
        if position >= len(self.truth):
            return False
        return self.truth.tokens[position] == proposed

    def token_at(self, position: int) -> int:
        return self.truth.tokens[position]


@dataclass
class DraftModel:
    """The small, fast model that proposes tokens.

    Its quality is summarized by ``agreement``: the probability that a proposed
    token matches the target at any given position. A strong, well-aligned draft
    model has agreement near 0.9; a mismatched one might sit at 0.5. This is the
    single most important input to the whole speedup calculation.
    """

    truth: TokenStream
    agreement: float = 0.8
    seed: int = 0
    _rng: random.Random = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.agreement <= 1.0:
            raise ValueError("agreement must be in [0, 1]")
        self._rng = random.Random(self.seed)

    def propose(self, position: int) -> int:
        """Propose a token for ``position``.

        With probability ``agreement`` we propose the correct token. Otherwise
        we propose a wrong one, which the target will reject.
        """
        if position >= len(self.truth):
            return -1
        correct = self.truth.tokens[position]
        if self._rng.random() < self.agreement:
            return correct
        # deliberately wrong: any token other than the correct one
        wrong = correct
        while wrong == correct:
            wrong = self._rng.randint(0, 50_000)
        return wrong


def make_stream(length: int, seed: int = 0) -> TokenStream:
    """Build a deterministic ground-truth token stream of a given length."""
    rng = random.Random(seed)
    return TokenStream([rng.randint(0, 50_000) for _ in range(length)])
