# Model Card — Gemma Tounsi 1.0 (27B)

> **STATUS: PLACEHOLDER — NO MODEL HAS BEEN TRAINED OR RELEASED.**
>
> This document is a skeleton. Every section below is a slot to be filled with
> **measured** values after a real training run and real evaluation runs. Nothing here
> may be filled in speculatively: no scores, no claims, no example outputs from an
> imagined model. `scripts/release.sh` is responsible for populating the results.

## Model details

| Field | Value |
| --- | --- |
| Model name | Gemma Tounsi 1.0 (27B) |
| Base model | Gemma 3 27B — *TODO: exact identifier + revision* |
| Adaptation method | QLoRA (performed with the external Soup training engine) |
| Languages | Tunisian Arabic (Derja, Arabic script), Tunisian Arabizi (Latin script) |
| Developed by | *TODO* |
| Release date | *TODO* |
| Artifact type | *TODO: LoRA adapter and/or merged weights* |
| License | *TODO — inherits the Gemma Terms of Use; see NOTICE* |
| Distribution | *TODO: registry link. Weights are never stored in Git.* |

## Intended use

*TODO after release.*

- **Intended uses:** *TODO — e.g. Tunisian-language assistance, Derja/Arabizi understanding
  and generation, research on dialectal adaptation of open-weight models.*
- **Out-of-scope / not intended:** *TODO — e.g. high-stakes advice (medical, legal, financial),
  authoritative factual claims about Tunisia, any use prohibited by the Gemma Prohibited Use
  Policy.*
- **Intended users:** *TODO.*

## Training

*TODO after training. Must record, at minimum:*

- [ ] Soup version / commit SHA used
- [ ] Frozen configuration reference (`configs/releases/v1.0.yaml`)
- [ ] Hyperparameters as actually run — **not decided yet** (LoRA rank/alpha, learning rate,
      batch size, sequence length, epochs, target modules, quantization settings)
- [ ] Compute used (GPU type, count, wall-clock time)
- [ ] Random seeds

See [`TRAINING.md`](TRAINING.md).

## Training data

*TODO after data preparation. Must record:*

- [ ] Tunisian adaptation sources with licenses and permitted use
- [ ] Replay/rehearsal (retention) sources with licenses
- [ ] Final mixture proportions as actually used
- [ ] Preprocessing and deduplication applied
- [ ] Confirmation that no evaluation data entered training

See [`DATA.md`](DATA.md) and `data/manifests/`.

## Evaluation

*TODO after evaluation. Results must come from real runs and always include the base
Gemma 3 27B baseline measured under identical conditions.*

| Track | Base Gemma 3 27B | Gemma Tounsi 1.0 | Δ |
| --- | --- | --- | --- |
| TounsiBench (Derja) | *TODO* | *TODO* | *TODO* |
| Arabizi | *TODO* | *TODO* | *TODO* |
| Retention | *TODO* | *TODO* | *TODO* |

See [`EVALUATION.md`](EVALUATION.md).

## Limitations

*TODO after evaluation — based on observed failures, not assumptions. Expect to document:*

- [ ] Dialect and regional coverage gaps within Tunisia
- [ ] Behaviour on Arabizi spellings unseen in training
- [ ] Any measured regression in general capabilities
- [ ] Known failure modes (e.g. MSA drift, code-switching errors)

## Bias, risks & safety

*TODO after evaluation. Must cover:*

- [ ] Biases inherited from the base model and from Tunisian web/social data
- [ ] Whether safety behaviour was measurably affected by adaptation
- [ ] Recommended mitigations for downstream deployment

## Environmental impact

*TODO: hardware, training duration, and estimated energy/emissions.*

## Citation

See [`CITATION.cff`](../CITATION.cff).
