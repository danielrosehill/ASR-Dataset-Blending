# Preventing overfitting and forgetting in ASR fine-tunes

Researched 2026-10-19. Full text read for arXiv 2305.09681, 2406.04240, 2606.07608 and
2307.07280; the rest from abstracts or summaries (marked). **[M]** measured result from a
paper; **[F]** forum advice, blog or vendor heuristic.

## Bottom line

- **10% general-data replay is the only ratio with controlled ASR evidence**, and it
  removes a large part of the forgetting at almost no in-domain cost. 20% buys more
  retention for a measurable in-domain cost. Even <1% helps when well chosen.
- Replay **stacks** with partial-layer tuning, layer-wise LR decay, LoRA and
  base/fine-tuned weight averaging. Weight averaging is free at inference — always try it.
- **Always evaluate on two test sets** (in-domain + general) and pick checkpoints on both.
- For Whisper, short clips without silence cause memorised-phrase hallucinations on
  silence; keep silence and no-speech clips in the mix.

## 1. Replay ratios that have been measured

### "Replay to Remember" — arXiv 2307.07280 [M]

Whisper-small, XLSR-53, XLS-R-300M fine-tuned on Swiss German (SVC-de); forgetting
measured on Common Voice German. Experience replay (ER) = share of original-domain data
in every batch.

| Model / layers tuned | ER | New-domain WER | General WER |
|---|---|---|---|
| Whisper-small, all layers | 0 | 2.2 | 29.1 |
| Whisper-small, last 6 enc+dec | 0 | 3.2 | 24.5 |
| Whisper-small, last 6 enc+dec | 10% | 3.0 | 18.1 |
| Whisper-small, last 6 decoder | 20% | 4.9 | 16.8 |
| XLSR-53, all layers | 0 / 10% / 20% | 5.5 / 5.6 / 5.9 | 55.0 / 42.7 / 31.6 |

LRs: Whisper 3e-5, XLS-R 3e-4. A 10× higher LR improved the new domain but worsened
forgetting and destabilised training.

### Other evidence

- **Chang, Lee & Lee, arXiv 2104.01616 [M]:** WSJ → LibriSpeech-100 → Switchboard with
  GEM rehearsal. ~30 min stored audio (<1% per corpus) cut average WER 30.9 → 26.8;
  220 min → 17.6 (joint training 16.7). **EWC was no better than plain fine-tuning.**
  Replay utterances near the median length slightly beat random.
- **Vander Eeckt & Van hamme, arXiv 2112.09427 [M, abstract]:** 0.6% of original data
  closes >40% of the fine-tune-vs-joint gap.
- **Vander Eeckt & Van hamme, arXiv 2601.18266 [M, abstract]:** works with one utterance
  per previous task — fine-tune, SVD the weight delta, learn gates over singular values.
- **Liu et al., arXiv 2509.24310 [M]:** Whisper-small fine-tuned only on Mandarin-English
  code-switched data: LibriSpeech test-clean **3.4 → 13.3 WER** while the CS test went
  24.9 → 8.8. Joint CS + monolingual training kept test-clean at 5.7. No ratio given.
- **Simmons, Tira/English Whisper large-v3, COMPUTEL 2025 [M, abstract]:** English forgot
  after little training; adding **pseudo-labelled English** (base model as labeller)
  greatly reduced it. Ratio unknown (full text 404 on 2026-10-19).
- **arXiv 2607.11163 [M, summary]:** on multilingual Whisper a 2,000-utterance replay
  buffer alone gave limited protection; replay + gradient projection ≈ zero forgetting.
- **arXiv 2505.17496 [M]:** for spoken language models, experience replay beat model
  merging and LoRA-scale discounting; combining helps further.
- LLM folklore [F]: "mix in 5–20% general data"; one practitioner found 50% best
  (<https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting>). Not ASR-specific.

**No original data available?** Pseudo-label general audio with the base model and
replay that — the documented substitute.

## 2. Complements to replay

**Partial-layer tuning**
- 2307.07280: tuning the last 6 encoder + 6 decoder layers kept most of the in-domain gain
  with much less forgetting [M].
- Euphonia (arXiv 1907.13511): tuning a subset of layers often beats tuning everything [M].
- Tomanek et al. (arXiv 2110.04612): RNN-T, first 5 encoder layers only [M].

**Layer-wise LR decay (LLRD)** — Plantinga et al., arXiv 2305.09681 [M]
- Whisper small.en on CORAAL: LLRD 1.0 → test-clean 3.35 → 5.86; LLRD 0.8 → 3.32,
  *and* better in-domain WER (13.2 vs 14.4).
- NeMo Conformer-CTC on SPGI: test-clean 2.15 → 2.98, test-other 4.48 → 6.52.
- Freezing non-encoder layers ≈ lowering LR. Learning-without-Forgetting KD added nothing
  on top of LLRD. Whisper still forgot on the full domain set; they subsampled to ~10%.

**Weight averaging / merging**
- arXiv 2210.15282: averaging old and adapted models does well on both tasks, beat all
  baselines; better still with a distillation loss [M].
- 2305.09681: averaging domain experts brought the NeMo Conformer back to LibriSpeech
  2.18 / 4.86 (base 2.15 / 4.48) [M].
- **Recipe:** sweep α between base and fine-tuned weights (WiSE-FT style), plot both test
  sets, pick the knee. Refinement: arXiv 2601.14751.

**LoRA / adapters**
- NVIDIA "Damage Control During Domain Adaptation", arXiv 2210.03255
  ([blog](https://developer.nvidia.com/blog/controlled-adaptation-of-speech-recognition-models-to-new-domains/)):
  full fine-tuning a 120M Conformer-Transducer on commands pushed general WER **5% →
  ~30–50%**; adapters held it with minor loss. Tolerable general-WER loss capped at
  **3% absolute** [M/F].
- Whisper Uyghur (arXiv 2408.10680): Aishell-1 CER 8.82 → 30.87 full FT, → 24.31 LoRA.
  **LoRA forgets less but still forgets** [M].
- arXiv 2405.09673 (LLM): "LoRA learns less and forgets less" [M].
- **Whisper LoRA caveat (arXiv 2606.07608) [M]:** α = 2r destabilised the decoder after
  ~1 epoch (hallucinations, repetition loops, lost EOS). **α/r ≤ 0.2 fixed it.**

**EWC** — no help in 2104.01616; reported to work for adding a Whisper language in
arXiv 2407.06800. Weaker than replay or averaging where compared [M].

**KD from the base model** — preserved Whisper's `<|nospeech|>` behaviour (zero
hallucinations on 5,750 samples) without improving WER (2606.07608); no gain over LLRD
(2305.09681) [M].

## 3. Speaker personalisation

- **Euphonia, arXiv 1907.13511 [M]:** 71% of the improvement from **5 minutes** of data.
- **Tomanek et al., arXiv 2110.04612 [M]:** biggest gains in the first **50 utterances
  (~3.6 min)**, knee before 100; ~250 utterances (~18 min) got 79% of speakers to
  WER ≤ 15%. Steps: 5k (<100 utt), 10k (100), 15k (250). 80/10/10 splits with **no
  phrase overlap**. Mild speakers saturate early.
- **Green et al. 2021 [M]:** 432 speakers; personalised median WER 4.6% vs 31%.
- **Apple, arXiv 2406.04240 [M]:** tune the whole encoder or whole decoder; early-encoder-
  only or head-only was unstable. A **small, repetitive vocabulary overfit under full
  fine-tuning**; LoRA kept original performance. LoRA rank 2 on decoder MLP W1 competitive.
- **Counter-evidence, arXiv 2606.31722 [M]:** for a single dysarthric speaker on Whisper
  large-v3, LoRA r=8 was 15–39% worse than full FT. Far-from-typical speech needs
  capacity; near-typical speech (a healthy single speaker) → LoRA/partial tuning is safer.

## 4. Evaluation practice

- **Two test sets, always:** in-domain held-out + a general set (LibriSpeech test-clean/
  other, Common Voice, FLEURS).
- A **random 2,000-utterance subset** tracked the full test set within 3% relative
  (2305.09681) — cheap enough to evaluate every checkpoint [M].
- **Checkpoint selection on both sets.** The HF Whisper blog's `load_best_model_at_end`
  uses in-domain WER only (<https://huggingface.co/blog/fine-tune-whisper>) — add the
  general set yourself [F].
- **Normalisation:** NeMo and Whisper needed different normalisers in 2305.09681. The HF
  blog's WER applies none, so casing/punctuation drift shows up as WER. 2606.07608
  argues for convention-aware WER and warns about benchmark contamination.
- **Phrase-disjoint splits** — otherwise in-domain WER measures memorised transcripts.

## 5. Text-side and hallucination concerns

- Overfitting tracks **small, repetitive vocabulary** (2406.04240); hallucinations were
  **memorised high-frequency phrases emitted on silence** (2606.07608) [M]. Counter with
  a large, non-repeating sentence bank.
- **Silence mismatch (2606.07608) [M]:** median leading silence 0.04 s in training vs
  1.92 s at test; `<|nospeech|>` probability fell from >90% to 8.8%. Fixes by effect:
  α/r ≤ 0.2, KD from base, trimming silence to 0.3 s at inference. → Keep leading/
  trailing silence and some no-speech clips in training.
- **Clip-length shift [M]:** NVIDIA attributes much of its 5% → 30–50% collapse to 1 s
  command clips replacing 15–20 s utterances. Very short personal clips are a forgetting
  risk on their own.
- **Timestamps [F]:** fine-tuning only on `<|notimestamps|>` data can erode timestamp
  ability; LoRA reportedly preserves it
  (<https://discuss.huggingface.co/t/timestamps-reduce-whisper-hallucinations/76034>).
  Keeping some timestamped replay is the logical hedge — unmeasured.

## 6. Rules of thumb

| Rule | Value | Source | Type |
|---|---|---|---|
| Replay share | 10% (20% for more retention) | 2307.07280 | M |
| Minimal useful replay | <1% / ~30 min, near median length | 2104.01616, 2112.09427 | M |
| Whisper LR | ~40× below pretraining LR, linear decay (large-v2 5e-6, tiny 3.75e-5) | J.W. Kim via vasistalodagala/whisper-finetune | F |
| Whisper LR in papers | 1e-5, 3e-5, 5e-6 | HF blog, 2307.07280, 2606.07608 | M/F |
| NeMo fine-tune LR | 1e-4 to 1e-5; constant or cosine instead of Noam | [NeMo docs](https://docs.nvidia.com/nemo/speech/latest/asr/fine_tuning.html) | F |
| LLRD factor | 0.8–0.9 per layer | 2305.09681 | M |
| Whisper LoRA scale | α/r ≤ 0.2 | 2606.07608 | M (one study) |
| Personal data | 5 min ≈ 70% of gain; knee ~50 utt; ~250 utt (~18 min) near saturation | 1907.13511, 2110.04612 | M |
| Tolerable general-WER loss | 3% absolute | NVIDIA 2210.03255 | F |

## Gaps

- No ASR replay sweep beyond 0/10/20%; nothing measured for **healthy single-speaker**
  personalisation — evidence is extrapolated from domain, dialect and atypical speech.
- No NVIDIA/NeMo guidance on Parakeet forgetting or replay found.
- Prior own result: the 70/20/10 mix in the English-Hebrew Parakeet fine-tune moved
  LibriSpeech test-clean 1.2 → 1.1
  (<https://dev.to/danielrosehill/parakeet-that-knows-what-a-makolet-is-an-english-hebrew-code-switching-fine-tune-1p9d>).
