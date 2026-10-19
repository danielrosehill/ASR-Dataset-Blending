# ASR Dataset Blending

Research notes on **blending datasets for ASR fine-tunes**: mixing a small domain or
personal-voice dataset with general-domain speech ("normal sentences") so the fine-tune
learns the new material without overfitting or forgetting what the base model could do.

Written with small personal fine-tunes in mind — NVIDIA Parakeet TDT 0.6B v2 on one
speaker, English-Hebrew code-switching, Whisper — but most of it is model-agnostic.

## Notes

| File | Question it answers |
|---|---|
| [notes/01-mixing-tools.md](notes/01-mixing-tools.md) | What tools exist for weighted dataset mixing, and what are the exact knobs? (HF `datasets`, Lhotse, NeMo, icefall, ESPnet, SpeechBrain, MosaicML Streaming, Data-Juicer) |
| [notes/02-forgetting-best-practice.md](notes/02-forgetting-best-practice.md) | What does the evidence say about replay ratios, layer freezing, LoRA, weight averaging and evaluation? |
| [notes/03-replay-corpora.md](notes/03-replay-corpora.md) | Which general corpora to blend in, with HF ids, licences and punctuation/casing status |

## Working conclusions (2026-10-19)

1. **Start at ~10% general replay per batch**; it is the only ratio with controlled ASR
   evidence (arXiv 2307.07280). Go to 20%+ for bigger shifts such as code-switching.
2. **Replay data should match the base model's training distribution and transcript
   style.** Parakeet v2 was trained entirely on punctuated, cased text, so replay must be
   PnC too — LibriSpeech as shipped is not.
3. **Your own voice reading general sentences** (Common Voice sentence corpus, CMU
   ARCTIC) is the cheapest high-value addition.
4. **No tool picks the ratio.** Every stack accepts explicit weights; choose them with a
   short sweep scored on two dev sets (personal + general).
5. **Stack cheap protections:** partial-layer tuning or LLRD, then base/fine-tuned weight
   averaging at the end.
6. **Watch the traps:** HF `interleave_datasets` defaults to `first_exhausted`; NeMo's two
   "temperature" knobs point in opposite directions; mixing weights are per utterance,
   not per hour.

## Open questions

- No published replay-ratio sweep exists for healthy single-speaker personalisation —
  a candidate experiment for this repo.
- No existing tool treats weighted blending + duration matching + PnC consistency checks
  as one step — a candidate small tool.

## Conventions

Notes tag claims as confirmed from a source on the check date, measured in a paper, or
inferred/folklore. All dates are absolute.
