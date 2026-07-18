# ⚡ speculabench

**Model what speculative decoding actually buys you, before you spend a week wiring up real weights.**

![CI](https://github.com/ahmeddoghri/speculabench/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-7%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Speculative decoding lives or dies on one number: the acceptance rate.**
> This models the whole accept/reject loop with no GPU and no weights, so you
> can find your speedup sweet spot in a second: `python -m speculabench.eval`.

Everybody wants speculative decoding because the pitch is irresistible: a tiny
draft model guesses several tokens ahead, the big model checks them all in one
pass, and every correct guess is basically free. Papers quote 2x to 3x speedups
and you nod along.

Then you go to implement it and hit the questions nobody put on the slide. How
long should the draft be? What acceptance rate do I actually need before this
pays off? At what point does a longer draft start losing money because the big
model keeps rejecting the tail?

speculabench answers those without a single GPU. It models the accept/reject
dynamics faithfully (this is the standard accounting from Leviathan et al. and
Chen et al., 2023) and reports the one thing you care about: real speedup, in
units of expensive forward passes. You turn the knobs, you watch the number
move, you stop guessing.

---

## The result in one command

```bash
python -m speculabench.eval
```
```
acceptance rate vs speedup (draft_length=4):
   agreement   accept rate    speedup   tokens/pass
        0.50        24.5%      1.41x          1.98
        0.60        34.2%      1.69x          2.37
        0.70        42.2%      1.92x          2.68
        0.80        57.2%      2.35x          3.29
        0.90        75.3%      2.86x          4.01

draft length vs speedup (agreement=0.8):
   draft_len   accept rate    speedup   tokens/pass
           1        79.9%      1.64x          1.80
           2        72.2%      2.04x          2.44
           4        57.2%      2.35x          3.29
           6        47.6%      2.41x          3.85
           8        38.6%      2.27x          4.08
          12        29.9%      2.09x          4.59
```

Read the second table top to bottom and you can see the whole tradeoff. Longer
drafts pack more tokens into each verification pass, but the acceptance rate
falls as you reach further ahead, and past a point the rejected tail stops
paying for itself. Here the turnover is at length 6. That turnover point is the
entire engineering decision, and now you can see it instead of guessing it.

## Install

```bash
git clone https://github.com/ahmeddoghri/speculabench
cd speculabench && pip install -e .
python examples/quickstart.py
```

## Use it

```python
from speculabench.decode import SpeculativeDecoder
from speculabench.model import DraftModel, TargetModel, make_stream

truth = make_stream(1000, seed=0)          # what the target would generate alone
target = TargetModel(truth)
draft = DraftModel(truth, agreement=0.8)   # a draft that agrees 80% of the time

decoder = SpeculativeDecoder(draft, target, draft_length=4, draft_cost_ratio=0.1)
result = decoder.run(total_tokens=1000)

print(result.acceptance_rate)   # 0.57
print(result.speedup)           # 2.35x fewer expensive passes than plain decoding
```

## How the accounting works

```
loop until the output is complete:
  1. draft proposes `draft_length` tokens (one cheap pass each)
  2. target verifies the whole batch in ONE expensive pass, and
     accepts the longest correct prefix
  3. the same pass hands back one correct token for free
     (the target's own next token, or the fix at the first mismatch)
  4. jump past everything accepted plus that free token

cost, measured in expensive target passes:
  baseline    = one target pass per token
  speculative = target passes + draft passes * draft_cost_ratio
```

The guarantee, straight from the papers: the output is byte-for-byte identical
to what the target would have produced on its own. Speculation only changes
speed, never the answer. That is why it is safe to turn on in production.

## The knobs that matter

| Knob | What it is | Why it matters |
|---|---|---|
| `agreement` | how often the draft matches the target | the single biggest lever on speedup |
| `draft_length` | how many tokens to propose per round | too short wastes the pass, too long over-reaches |
| `draft_cost_ratio` | draft pass cost / target pass cost | a slow draft eats the savings from the tokens it lands |

Point these at your real draft/target pair (measure agreement on a sample of
your own traffic) and the speedup number this reports is the one you should
expect in production.

## Tests

```bash
pip install pytest && pytest -q      # 7 passing
```

## License

MIT © Ahmed Doghri
