# What general data to blend in ("normal sentences")

Researched 2026-10-19. Tags: **[C]** confirmed on a source page that day; **[M]** from
model knowledge, not re-fetched; **[I]** inferred.

## Bottom line

- **English replay should come from the base model's own training sources, with
  punctuation and capitalisation (PnC).** For Parakeet TDT 0.6B v2 that is the Granary
  English subsets plus VoxPopuli raw text and Common Voice. LibriSpeech and AMI are only
  usable after PnC is restored.
- **Hebrew replay:** ivrit.ai (`crowd-transcribe-v5`, Knesset) plus FLEURS `he_il`.
  Common Voice Hebrew is tiny (5.26 validated hours) and no longer on Hugging Face.
- **Your own voice reading general sentences is the most useful single addition.** It
  keeps the speaker adaptation and replays the base model's text distribution.
- **Synthetic TTS speech is a weak replay source.** Use it to inject vocabulary, never as
  the anchor for general performance.

## What Parakeet TDT 0.6B v2 was trained on [C]

Source: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2>

- ~120,000 h English ("Granary" on the card).
- ~10,000 h human-transcribed (NeMo ASR Set 3.0): LibriSpeech 960 h, Fisher, National
  Speech Corpus Part 1, VCTK, VoxPopuli EN, Europarl-ASR EN, Multilingual LibriSpeech EN
  (2,000 h subset), Mozilla Common Voice v7.0, AMI.
- ~110,000 h pseudo-labelled: YouTube-Commons, YODAS, LibriLight.
- Stage 2 fine-tune: ~500 h human-transcribed NeMo ASR Set 3.0.
- "All transcriptions preserve punctuation and capitalization."
- Model licence CC-BY-4.0.

The v3 card lists ~660,000 pseudo-labelled hours across 25 languages and ~7,500 h of NeMo
ASR Set 3.0 in stage 2 [C].

`nvidia/Granary` on HF [C]: 76 subsets named language + source (`en_yodas`, `en_ytc`,
`en_librilight`, `en_voxpopuli`, …), CC-BY-4.0, splits `asr` and `ast`. Rows are JSON
manifests with relative `audio_filepath`, so audio must be fetched from the upstream
sources [I]. Sample transcripts carry PnC (pipeline: arXiv 2505.13404). The HF dataset
viewer failed with a CastError on 2026-10-19. No English-only hour count is published.

## Corpora

| Corpus | HF id | Hours | Licence | PnC | Notes |
|---|---|---|---|---|---|
| Granary (EN subsets) | `nvidia/Granary` | ~110k h pseudo-labels in v2's mix | CC-BY-4.0 [C] | Yes (pseudo) | Closest match to v2's distribution. Manifests only [I]. |
| LibriSpeech | `openslr/librispeech_asr` [M] | 960 | CC-BY-4.0 [M] | **No** | In v2's mix. Read audiobooks. Restore PnC or weight lightly. |
| Common Voice EN | `mozilla-foundation/common_voice_*` is **empty** [C] | ESB train cut 1,409 h [C] | CC0 | Yes | Since Oct 2025 only via Mozilla Data Collective (account + ToS) [C]. Widest accent spread. |
| VoxPopuli EN | `facebook/voxpopuli` [M] | ~543 h [M] | CC0 [C] | `raw_text` has casing [M] | In v2's mix. EU parliament speech. |
| People's Speech | `MLCommons/peoples_speech` | 30k+ h [C] | CC-BY / CC-BY-SA [C] | **No** [C] | Mostly American; some labels ASR-generated. |
| GigaSpeech | `speechcolab/gigaspeech` (gated) | XS 10 … XL 10k [C] | **Audio non-commercial** [C] | Tags like `<COMMA>` [C] | Needs conversion to PnC. |
| FLEURS | `google/fleurs` (`en_us`, `he_il`) [M] | ~10–12 h/lang [M] | CC-BY-4.0 [M] | `raw_transcription` | Good eval set; light `he_il` replay. |
| TED-LIUM 3 | `LIUM/tedlium` [M] | 452 [M] | CC-BY-NC-ND-3.0 [M] | No | Skip for public releases. |
| SPGISpeech | `kensho/spgispeech` (gated) | 4,900 [C] | Non-commercial [C] | Yes | Finance vocabulary. |
| Earnings-22 | `distil-whisper/earnings22` [M] | 105 h ESB train [C] | CC-BY-SA-4.0 [C] | Yes | Accented business English; mainly a test set. |
| AMI | `edinburghcstr/ami` [M] | 78 h ESB train [C] | CC-BY-4.0 [C] | ESB version yes | In v2's mix. Meetings. |
| ESB bundle | `esb/datasets` | various | per corpus | Orthographic [C] | One loader for PnC versions of CV, SPGI, Earnings-22, AMI. |
| ivrit.ai crowd-transcribe-v5 | `ivrit-ai/crowd-transcribe-v5` (gated) | ~295 h [C] | ivrit.ai License (research + commercial) [C] | Not stated | Volunteer-transcribed podcasts. |
| ivrit.ai Knesset plenums | `ivrit-ai/knesset-plenums` (gated) | 8,825 h; ~4,700 h quality > 0.7 [C] | ivrit.ai License [C] | Punctuated, **non-verbatim** [C] | Filter on quality score. `knesset-plenums-whisper-training` also exists. |
| ivrit.ai other | `ivrit-ai/audio-v2`, `crowd-transcribe-v4`, `VoxKnesset`, `knesset-committees` [C] | — | varies | varies | `audio-v2` mostly unlabelled. |
| HebDB | HUJI project page, not confirmed on HF | ~2,500 h weak labels [C] | check | weak | arXiv 2407.07566. |
| Common Voice Hebrew | Mozilla Data Collective only | v25.0: 5.26 h validated, 58 speakers, ~90% male [C] | CC0 | Yes | Too small for replay; its 11,024-sentence text corpus is useful as prompts. |

## Principle: replay should match the base model's original distribution

- Vander Eeckt & Van hamme, arXiv 2112.09427: best method closes >40% of the gap between
  fine-tuning and joint training using **0.6% of the original data** as replay [C].
- Plantinga et al., arXiv 2305.09681: Whisper forgets more than a NeMo model after domain
  fine-tuning; averaging base and adapted weights reduces it [C].
- Ibrahim et al., arXiv 2403.08763 (LLMs): 1% replay already cuts forgetting
  substantially; ~5% for weak shifts, ~25% for strong (secondary summary) [C/I]; 50%
  replay hurt adaptation [C].
- Mix-cd, arXiv 2402.08096: prioritise replay samples the base model got right and the
  new model gets wrong [C].

**Working default [I]:** 5–20% replay per batch for a personal-voice fine-tune; more for
code-switching (a larger shift). Tune against a held-out general eval set.

## Own voice reading general sentences

Not literal replay (new audio), but it replays the text distribution and stops the model
linking your voice to domain vocabulary [I]. Prompt sources:

- **Common Voice sentence corpus** — CC0, in `common-voice/common-voice` under
  `server/data` [C]; check per-file licensing (some Europarl text) [C]. Hebrew: 11,024
  sentences [C].
- **CMU ARCTIC** — 1,132 phonetically balanced prompts from Project Gutenberg text [C].
- **Harvard / IEEE sentences** — 720 sentences [M].
- **FLEURS `raw_transcription`** — reading these gives a matched eval set in your voice [I].

## Synthetic TTS speech

- SynthASR, arXiv 2106.07803: tunes sampling weights rather than a fixed ratio; adds
  encoder freezing and regularisation against forgetting [C].
- arXiv 2410.12279: diminishing returns with scale; best synthetic-to-real WER ratio 1.46 [C].
- arXiv 2407.17997: synthetic-only training underperforms real across architectures [C].
- arXiv 2601.00935 (code-switching, Whisper-small): gains flatten at **~2–3× real data** [C].
- arXiv 2606.17662 (Indic): synthetic helps, less than the same amount of extra real data [C].
- arXiv 2606.29031: RIR augmentation narrows the synthetic gap [C].

Use synthetic speech for vocabulary and code-switched text only, cap it at or below the
real-data volume, add RIR/noise augmentation [I].

## Practical selection [I unless marked]

- **Duration matching:** histogram the domain clips, sample replay to the same
  distribution, cap at the training `max_duration` (~20–40 s in NeMo configs), floor ~1 s.
  Knesset segments averaged ~1.85 s in a sample [C].
- **PnC consistency:** Parakeet v2 was trained on PnC text only. Mixing lowercase,
  unpunctuated corpora teaches it to drop punctuation. Either use PnC-native corpora or
  restore PnC with a punctuation model first.
- **Conventions:** match number formatting, apostrophes and quotes between replay and
  your own transcripts.
- **Licences for public releases:** exclude GigaSpeech audio, SPGISpeech, TED-LIUM.

## Gaps

- No English-only Granary hour count.
- [M] IDs/licences not re-fetched on 2026-10-19.
- No NVIDIA-official replay-ratio guidance for Parakeet found.

## Sources

<https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2> ·
<https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3> ·
<https://huggingface.co/datasets/nvidia/Granary> · <https://arxiv.org/abs/2505.13404> ·
<https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0> ·
<https://huggingface.co/datasets/esb/datasets> ·
<https://huggingface.co/datasets/MLCommons/peoples_speech> ·
<https://huggingface.co/datasets/speechcolab/gigaspeech> · <https://huggingface.co/ivrit-ai> ·
<https://huggingface.co/datasets/ivrit-ai/knesset-plenums> ·
<https://arxiv.org/abs/2407.07566> · <https://arxiv.org/abs/2112.09427> ·
<https://arxiv.org/abs/2305.09681> · <https://arxiv.org/abs/2403.08763> ·
<https://arxiv.org/abs/2402.08096> · <https://arxiv.org/abs/2106.07803> ·
<https://arxiv.org/abs/2410.12279> · <https://arxiv.org/abs/2407.17997> ·
<https://arxiv.org/abs/2601.00935> · <https://arxiv.org/abs/2606.17662> ·
<https://huggingface.co/papers/2606.29031> · <https://github.com/common-voice/common-voice>
