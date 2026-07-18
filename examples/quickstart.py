"""Sixty-second tour of speculabench.

    python examples/quickstart.py
"""
from speculabench.decode import SpeculativeDecoder
from speculabench.model import DraftModel, TargetModel, make_stream

# The target model's true output. Speculative decoding must reproduce it exactly.
truth = make_stream(1000, seed=0)
target = TargetModel(truth)

# A draft model that agrees with the target 80% of the time.
draft = DraftModel(truth, agreement=0.8, seed=0)

# Propose 4 tokens ahead; the draft is 10x cheaper than the target.
decoder = SpeculativeDecoder(draft, target, draft_length=4, draft_cost_ratio=0.1)
result = decoder.run(total_tokens=1000)

print(f"acceptance rate : {result.acceptance_rate:.1%}")
print(f"speedup         : {result.speedup:.2f}x")
print(f"tokens/pass     : {result.avg_tokens_per_target_call:.2f}")
print(f"target passes   : {result.target_calls} (vs {result.tokens_generated} for plain decoding)")
