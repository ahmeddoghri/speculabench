# ⚡ speculabench

**Model what speculative decoding actually buys you, before you spend a week wiring up real weights.**

![CI](https://github.com/ahmeddoghri/speculabench/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-16%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Speculative decoding lives or dies on one number: the acceptance rate.**
> Everything else in the paper is decoration. This models the whole accept/reject
> loop with no GPU and no weights, so you find your speedup sweet spot before
> your cloud bill finds you: `python -m speculabench.eval`.
>
> **Update:** the original model draws match/mismatch independently at every
> position. Real drafts are bursty, long streaks of agreement on easy spans,
> clustered misses on hard ones, and that changes the answer: at the *same*
> nominal 80% agreement rate, a bursty draft's real sweet spot is
> `draft_length=8` at 2.89x, not `draft_length=6` at 2.48x. Modeled and
> measured below. `python -m speculabench.eval_v2`.

Every speculative decoding pitch sounds the same. Tiny draft model guesses
ahead, big model checks the guesses in one pass, correct guesses are basically
free, 2 to 3x speedup, slide advances, everyone claps. Nobody claps at the part
where you actually have to pick a draft length and an acceptance rate and find
out the hard way that you picked wrong.

I got tired of nodding along to speedup numbers I couldn't reproduce, so I
built the spreadsheet nobody hands you. How long should the draft be? What
acceptance rate do you need before this is worth the code review? At what
draft length does the model start proposing tokens that just get thrown away,
turning your speedup into a very expensive coin flip?

speculabench answers all of that with zero GPUs and zero weights. It models
the accept/reject dynamics faithfully (this is the standard accounting from
Leviathan et al. and Chen et al., 2023), not a hand-wavy approximation, and
reports the one number that actually pays your AWS bill: real speedup, in
units of expensive forward passes you didn't have to run. Turn the knobs,
watch the number move, stop guessing, stop nodding.

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

Read the second table top to bottom and the whole tradeoff falls out for free.
Longer drafts pack more tokens into each verification pass, but the acceptance
rate drops as you reach further ahead, and past a point the rejected tail is
just you paying compute to be told no. Here the turnover is at length 6. That
one number is the entire engineering decision everyone argues about in
standup, and now you can see it instead of vibing it.

## The i.i.d. assumption is quietly load-bearing

"Not a hand-wavy approximation" is true of the accept/reject *bookkeeping*
(that part is exactly the Leviathan/Chen accounting). It is not true of
*where* the draft is right and wrong. `DraftModel` decides match or
mismatch at each position with an independent coin flip. Nothing about how
language models actually fail looks like that: a draft nails long
templated or highly predictable spans in unbroken streaks, then misses in
clusters on the hard spans, because whatever makes one token hard to
predict usually makes the next few hard too. That's autocorrelation, and
i.i.d. Bernoulli has none of it by construction.

It matters because the whole point of this tool is picking a draft length,
and the ideal draft length depends on the *shape* of accept/reject runs,
not just their average rate. `speculabench/model_v2.py` adds
`BurstyDraftModel`: a two-state Markov chain with a `burstiness` knob,
calibrated so its marginal match rate matches the nominal `agreement`
within half a percentage point at n=20,000 regardless of burstiness
(`burstiness=0.0` reduces exactly to i.i.d.). Sweep both models at the
identical 80% agreement rate and identical seed:

```bash
python -m speculabench.eval_v2
```
```
draft length vs speedup at matched agreement=0.80 (iid vs bursty, burstiness=0.40)
==============================================================================
 draft_len  iid accept  iid speedup  bursty accept  bursty speedup
------------------------------------------------------------------------------
         1       79.9%        1.64x          80.2%           1.64x
         2       72.1%        2.04x          75.3%           2.09x
         4       59.7%        2.42x          66.6%           2.62x
         6       49.4%        2.48x          59.2%           2.84x
         8       41.6%        2.41x          52.6%           2.89x
        12       30.6%        2.12x          43.2%           2.81x

iid sweet spot:    draft_length=6   (2.48x)
bursty sweet spot: draft_length=8   (2.89x)
```

Same marginal accuracy, a different recommended draft length and a
meaningfully higher achievable speedup, because long agreement streaks pack
more accepted tokens into each verification pass than the i.i.d. model ever
lets them. This holds up on a second, held-out seed evaluated exactly once
(sweet spot again shifts from `draft_length=6` to `draft_length=8`, 2.49x
to 2.94x). The original `model.py`/`eval.py` are untouched, so the
published i.i.d. numbers above still reproduce exactly; `BurstyDraftModel`
is opt-in, not a silent default change. Measure your own draft's real
run-length distribution (not just its average agreement) before trusting
either model's recommended draft length in production.

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
to what the target would have produced alone. Speculation only changes speed,
never the answer, which is the one property that lets you turn this on in
production without your on-call engineer developing a facial tic.

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
pip install pytest && pytest -q      # 16 passing
```

## License

MIT © Ahmed Doghri
