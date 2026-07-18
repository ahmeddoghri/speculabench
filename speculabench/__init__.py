"""speculabench: measure what speculative decoding actually buys you.

Speculative decoding runs a small, fast draft model to propose several tokens
at once, then a big target model verifies them in a single forward pass. Every
token the target agrees with is free. Every token it rejects costs you a
resync. The whole thing lives or dies on one number: the acceptance rate.

This package models that loop without needing a GPU or a real model, so you can
reason about the tradeoff (draft length, acceptance rate, cost ratio) before
you spend a week wiring up real weights.
"""
from speculabench.decode import DecodeResult, SpeculativeDecoder
from speculabench.model import DraftModel, TargetModel, TokenStream

__all__ = [
    "DraftModel",
    "TargetModel",
    "TokenStream",
    "SpeculativeDecoder",
    "DecodeResult",
]

__version__ = "0.1.0"
