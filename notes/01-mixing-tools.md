# Tools for blending speech datasets in ASR fine-tuning

Researched 2026-10-19. **[C]** confirmed from docs or current source on GitHub
master/main that day; **[I]** inferred or from memory.

**Headline:** no tool chooses the blend ratio for you. Every stack lets you state
weights; none optimises them for speech. Pick the ratio with a small sweep against two
dev sets (see [02-forgetting-best-practice.md](02-forgetting-best-practice.md)).

## Trap: "temperature" means opposite things

| Knob | Formula | Lower value | Higher value |
|---|---|---|---|
| NeMo Lhotse `reweight_temperature` | `w_i^T / Σ w_j^T` | T<1 oversamples small sets; T=0 equal | T>1 amplifies differences |
| NeMo legacy `concat_sampling_temperature` (default 5) | `len_i^(1/T)` normalised | close to raw size ratios | flattens toward uniform |
| ESPnet `upsampling_factor` β | `(n_l/N)^β` | β→0 upsamples small categories | β=1 proportional |

So legacy NeMo "temperature 5" *flattens* the mix while Lhotse `reweight_temperature: 5`
*skews* it **[C]**. For a tiny personal set plus a large general set, write the target
proportions directly as weights and leave temperature at 1.

## Weights are per utterance, not per hour

HF `interleave_datasets`, Lhotse `mux` and NeMo `input_cfg` weights all select per
example **[C/I]**. If personal clips are much shorter or longer than general clips, the
share of audio hours differs from the share of utterances. Compute the ratio you
actually want in both units.

---

## 1. Hugging Face `datasets` (docs v4.8.4) [C]

- `concatenate_datasets([d1, d2])` — stack, no weighting.
- `interleave_datasets(datasets, probabilities=[...], seed=..., stopping_strategy=...)`.
- `stopping_strategy`: `"first_exhausted"` (default, stops when any source runs out),
  `"all_exhausted"` (restarts finished sources), `"all_exhausted_without_replacement"`.
- Works on streaming `IterableDataset`s.

```python
from datasets import load_dataset, interleave_datasets, Audio
general  = load_dataset("...general...", split="train", streaming=True)
personal = load_dataset("me/my-voice", split="train", streaming=True)
mix = interleave_datasets([personal, general], probabilities=[0.3, 0.7],
                          seed=42, stopping_strategy="all_exhausted")
mix = mix.cast_column("audio", Audio(sampling_rate=16000))
```

**Trap:** with the default `first_exhausted`, the epoch ends when the small personal set
runs out, silently discarding most of the general data. Use `all_exhausted`.

No temperature, no duration bucketing. <https://huggingface.co/docs/datasets/process#interleave>

## 2. Lhotse [C] (`lhotse/lazy.py`, `lhotse/cut/set.py`)

- `CutSet.mux(*manifests, stop_early=False, weights=None, seed=0)`.
- `CutSet.infinite_mux(*manifests, weights=None, seed=0, max_open_streams=None)` — caps
  open sub-iterators; the docstring warns against it for short runs.
- `.repeat(times=None, preserve_id=False)` — makes a small set an infinite stream.

```python
from lhotse import CutSet
personal = CutSet.from_file("personal_cuts.jsonl.gz").repeat()
general  = CutSet.from_file("general_cuts.jsonl.gz").repeat()
train = CutSet.mux(personal, general, weights=[0.3, 0.7], seed=0)
# then DynamicBucketingSampler(train, max_duration=..., shuffle=True)
```

Augmentation: `CutSet.mix(..., snr=20, mix_prob=1.0)`,
`lhotse.dataset.cut_transforms.CutMix(cuts, snr=(10,20), p=0.5)`,
`Recording.reverb_rir(rir_recordings=None)` (synthetic RIRs if none given).

## 3. NVIDIA NeMo (Parakeet, Canary, FastConformer)

### 3a. Lhotse path — recommended [C]

Set `model.train_ds.use_lhotse=True`; describe the mix in `train_ds.input_cfg` (inline
or a YAML path).

- Each entry: `type` (`nemo`, `nemo_tarred`, `lhotse`, `lhotse_shar`, `group`), paths,
  `weight`, optional `tags`. Groups nest; effective weight = outer × inner.
- If any sibling has a weight, all must ("Missing dataset weight" assertion). Weights may
  be hours, counts or probabilities; they are normalised. 0 disables a source.
- `reweight_temperature`: scalar or one value per nesting level (length mismatch →
  `ValueError`). Docstring example: hours `[197, 2159]` → T=1 `[0.0836, 0.9164]`,
  T=0 `[0.5, 0.5]`.
- `max_open_streams` switches to `infinite_mux`. The stream is infinite unless
  `force_finite`, so train by `max_steps`.
- `multi_config: True` + `sampler_fusion` (`round_robin` default,
  `randomized_round_robin` with `sampler_weights`, or `zip`). `zip` takes a mini-batch
  from each sampler every step — a fixed share of every batch.
- Augmentation in the same config: `noise_path`, `noise_snr` (default `(10.0, 20.0)`),
  `noise_mix_prob` (0.5), `rir_enabled` (False), `rir_path` (Lhotse RecordingSet),
  `rir_prob` (0.5). These apply to the whole mixed stream, not per source [I].

```yaml
model:
  train_ds:
    use_lhotse: true
    reweight_temperature: 1.0
    input_cfg:
      - type: nemo
        manifest_filepath: /data/personal/train_manifest.json
        weight: 0.3
        tags: {source: personal}
      - type: nemo_tarred
        manifest_filepath: /data/general/manifest__OP_0..63_CL_.json
        tarred_audio_filepath: /data/general/audio__OP_0..63_CL_.tar
        weight: 0.7
        tags: {source: general}
    noise_path: /data/musan/noise_manifest.json
    noise_mix_prob: 0.5
```

**Trap:** one group cannot mix tarred and non-tarred sources without
`force_map_dataset` or `force_iterable_dataset` → `ValueError`
(`nemo/collections/common/data/lhotse/cutset.py`). Tar the personal set or set a flag.

The rendered NeMo docs do not document `reweight_temperature`, `max_open_streams` or the
`concat_*` keys; the above is from source.

### 3b. Legacy path [C]

`train_ds.is_concat: true`; `concat_sampling_technique` (`temperature` default,
`random`, `round-robin`); `concat_sampling_temperature` (5); `concat_sampling_probabilities`
(only with `random`); `concat_sampling_scale`; `concat_sampling_seed`
(`audio_to_text_dataset.py` → `ConcatDataset`).

Tarred bucketing: `manifest_filepath` as a list of lists = one bucket each;
`bucketing_strategy` (`fixed_order`, `synced_randomized` default, `fully_randomized`);
`bucketing_weights=[2,1,1,3]` = reads per epoch (crude upsampling).

Sources: <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html> ·
<https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/data/lhotse/cutset.py> ·
<https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/data/lhotse/dataloader.py> ·
<https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/data/audio_to_text_dataset.py>

## 4. k2 / icefall [C]

- **Batch-level:** `egs/librispeech/ASR/pruned_transducer_stateless3/train.py`,
  `--giga-prob` (0.5) = chance the next batch comes from GigaSpeech. Each batch is pure;
  stops when either loader is exhausted.
- **Cut-level:** `multi_dataset.py` (e.g. `egs/speech_llm/ASR_LLM/whisper_llm_zh/`) uses
  `CutSet.mux(weights=[len(c) for c in cutsets])` — size-proportional across 13 corpora.
  Replace with explicit ratios for a personal blend.
- `--enable-musan` → `CutMix(cuts=cuts_musan, p=0.5, snr=(10,20))`.

## 5. ESPnet (espnet2) [C]

`batch_type:` `catbel` (balanced), `catpow` (MMS, arXiv 2305.13516: `P(l)=(n_l/N)^β`,
`upsampling_factor`, `dataset_scaling_factor` ≥1, needs `category2utt`),
`catpow_balance_dataset` (two-level: `category_upsampling_factor`,
`dataset_upsampling_factor`; needs `category2utt`, `dataset2utt`, `utt2dataset`).
Only a power law — no explicit target ratios [I].

## 6. SpeechBrain [C] (`speechbrain/dataio/sampler.py`)

- `ConcatDatasetBatchSampler(samplers, batch_sizes)` — fixed items per batch from each
  source, e.g. `[8, 16]`. The only tool here that **guarantees** batch composition
  (NeMo `sampler_fusion: zip` is closest). Epoch ends with the smallest source.
- `ReproducibleWeightedRandomSampler`, `BalancingDataSampler(dataset, key)`.
- `speechbrain.augment.time_domain.AddNoise`, `AddReverb`.

## 7. Mixture optimisation from the LLM world

- **MosaicML Streaming** [C]: `Stream(proportion=|repeat=|choose=)`, one scheme for all
  streams; any modality in MDS shards; no speech recipes found.
  <https://docs.mosaicml.com/projects/streaming/en/stable/dataset_configuration/mixing_data_sources.html>
- **Data-Juicer** [C]: `dataset.configs[*].weight` + `max_sample_num`; CLI
  `dj-process --dataset_path "0.5 a.json 0.5 b.json"`; audio ops `audio_duration_filter`,
  `audio_nmf_snr_filter`, `audio_size_filter`, `audio_add_gaussian_noise_mapper`. Static
  (offline) mixing. <https://github.com/datajuicer/data-juicer/blob/main/docs/DatasetCfg.md>
- **DoReMi, RegMix, Data Mixing Laws, AutoMixer**: no published application to ASR
  mixture weights found (absence of search results, not proof) [I]. Nearest:
  arXiv 2510.20860 (speech-language pretraining mix). Speech practice remains hand-set
  weights + temperature/power-law upsampling.

**Cheap RegMix-style sweep at personal scale [I]:** 4–6 short fine-tunes at personal
weight {0.1, 0.2, 0.3, 0.5}; score a personal dev set and a general dev set; take the knee.

## 8. Augmentation at mix time

`audiomentations`: `AddBackgroundNoise(sounds_path, min_snr_db=3.0, max_snr_db=30.0, p=0.5)` [C],
`ApplyImpulseResponse(ir_path, p=...)` [I]; per-example numpy transforms for HF `.map()` /
`set_transform()`. Lhotse, NeMo, icefall and SpeechBrain have MUSAN + RIR built in.

## 9. Existing "blending" projects

None makes weighted blending a first-class feature [C, searches empty].
`vasistalodagala/whisper-finetune` concatenates multiple dataset dirs, no weights [I];
`markcheeky/whisper-finetune` adds noise/music augmentation. HF forum
"Fine-tuning: Merge or chain?" (<https://discuss.huggingface.co/t/fine-tuning-merge-or-chain/41373>)
is unresolved. **This is the gap a small tool in this repo could fill.**

## Per-stack recommendation

- **Parakeet v2 (NeMo):** Lhotse `input_cfg`, explicit `weight`s,
  `reweight_temperature: 1.0`, `max_steps`. Tar the personal set or set
  `force_iterable_dataset`. Tag sources for per-source validation. Use `multi_config` +
  `sampler_fusion: zip` if you need a guaranteed personal share per batch.
- **Whisper / wav2vec2 (HF):** `interleave_datasets(probabilities=[p, 1-p],
  stopping_strategy="all_exhausted")`; for fixed batch composition, port SpeechBrain's
  `ConcatDatasetBatchSampler` pattern into a custom sampler.
