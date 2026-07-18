"""The speculative decoding loop itself, plus the cost model that turns
accept/reject counts into a real speedup number.

The loop:
  1. The draft model proposes ``draft_length`` tokens ahead.
  2. The target verifies them in one pass, accepting the longest correct prefix.
  3. On the first rejection, the target supplies the correct token itself
     (a "bonus" token that comes for free with the verification pass).
  4. Repeat from the new position.

This is the standard speculative decoding accounting from Leviathan et al.
and Chen et al. (2023). The guarantee is that the output is identical to what
the target would have produced alone; the only thing that changes is speed.
"""
from __future__ import annotations

from dataclasses import dataclass

from speculabench.model import DraftModel, TargetModel


@dataclass
class DecodeResult:
    """Everything you need to judge whether speculation paid off."""

    tokens_generated: int
    draft_calls: int
    target_calls: int
    tokens_accepted: int
    tokens_proposed: int
    # cost is expressed in "target forward passes", the expensive unit
    baseline_cost: float
    speculative_cost: float

    @property
    def acceptance_rate(self) -> float:
        if self.tokens_proposed == 0:
            return 0.0
        return self.tokens_accepted / self.tokens_proposed

    @property
    def speedup(self) -> float:
        if self.speculative_cost == 0:
            return 0.0
        return self.baseline_cost / self.speculative_cost

    @property
    def avg_tokens_per_target_call(self) -> float:
        if self.target_calls == 0:
            return 0.0
        return self.tokens_generated / self.target_calls


class SpeculativeDecoder:
    """Run the speculative decoding loop and account for its cost.

    ``draft_cost_ratio`` is how expensive one draft-model forward pass is
    relative to one target forward pass. A 7B draft in front of a 70B target
    is roughly 0.1; a tiny draft can be 0.02. It matters: a draft that is too
    slow eats the savings from the tokens it accepts.
    """

    def __init__(self, draft: DraftModel, target: TargetModel,
                 draft_length: int = 4, draft_cost_ratio: float = 0.1) -> None:
        if draft_length < 1:
            raise ValueError("draft_length must be at least 1")
        if draft_cost_ratio < 0:
            raise ValueError("draft_cost_ratio must be non-negative")
        self.draft = draft
        self.target = target
        self.draft_length = draft_length
        self.draft_cost_ratio = draft_cost_ratio

    def run(self, total_tokens: int) -> DecodeResult:
        pos = 0
        draft_calls = 0
        target_calls = 0
        accepted = 0
        proposed = 0

        while pos < total_tokens:
            window = min(self.draft_length, total_tokens - pos)

            # 1. Draft proposes `window` tokens, one forward pass per token.
            proposals = [self.draft.propose(pos + i) for i in range(window)]
            draft_calls += window
            proposed += window

            # 2. Target verifies the whole window in ONE forward pass and
            #    accepts the longest correct prefix.
            target_calls += 1
            n_accepted = 0
            for i, tok in enumerate(proposals):
                if self.target.verify(pos + i, tok):
                    n_accepted += 1
                else:
                    break
            accepted += n_accepted

            # 3. Move past the accepted prefix. If the whole window was
            #    accepted, the verification pass also yields one bonus token
            #    (the target's own next token) for free. If some token was
            #    rejected, the target supplies the correct token at the first
            #    mismatch, also for free from the same pass.
            pos += n_accepted
            if pos < total_tokens:
                pos += 1  # the free target-supplied token

        # ---- cost accounting, in units of one target forward pass ----
        # Baseline autoregressive decoding: one target pass per token.
        baseline_cost = float(total_tokens)
        # Speculative: every target verification pass, plus the draft passes.
        speculative_cost = target_calls + draft_calls * self.draft_cost_ratio

        return DecodeResult(
            tokens_generated=total_tokens,
            draft_calls=draft_calls,
            target_calls=target_calls,
            tokens_accepted=accepted,
            tokens_proposed=proposed,
            baseline_cost=baseline_cost,
            speculative_cost=speculative_cost,
        )
