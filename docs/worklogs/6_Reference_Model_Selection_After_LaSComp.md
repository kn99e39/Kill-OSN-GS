# 6 Reference Model Selection After LaS-Comp

**Date:** 2026-09-11
**Scope:** source-level selection only; no candidate was installed, cloned, patched, or executed, and Experiment 1 was not run.

## LaS-Comp remains closed and unmodified

The frozen LaS-Comp record is deliberately retained without a source, configuration, checkpoint, or environment change in this batch:

| Track | Final status |
| --- | --- |
| Pristine historical reproduction | `BLOCKED_PRE_INFERENCE` |
| Functional method reproduction | `FUNCTIONAL_REPRODUCTION_FAILED` |
| Runtime O/G suitability | `NOT_ADJUDICATED` |

The first result belongs to the historical-provenance gate; the second is the independently observed released-source/configuration incompatibility documented in Worklog #5. Neither result authorizes a LaS-Comp patch, a retry, O-influence tracing, candidate-G inventory, a structural intervention, or Experiment 1.

## Selection rules

The candidate must be judged qualitatively, without a synthetic numerical score:

1. Public reproducibility: author source, documented checkpoint route, environment, and runnable workflow.
2. `G` materialization: a geometry artifact can be written and frozen.
3. `O/G` separability: observed geometry `O` and generated geometry `G` can be named independently at the experimental boundary. `G = F(O)` is acceptable; independent generation is not required.
4. Region accessibility: a fixed external multi-region partition can be applied to frozen geometry.
5. GT/evaluation support: an author-designated ground-truth or evaluation subset exists.
6. Reuse: the same frozen `O/G` pair can support baseline, plausible-wrong, and oracle-A conditions rather than regenerating either input.
7. Scientific relevance: the model must give a non-trivial structural completion boundary.

All source revisions below were resolved read-only from the official Git remotes on 2026-09-11. The target machine is LabServer63: **NVIDIA RTX 3080 Ti, 12,288 MiB**, driver 595.71.05.

## Candidate assessment

### 1. 3D-Fixer — not selected

Official source: [HorizonRobotics/3D-Fixer](https://github.com/HorizonRobotics/3D-Fixer), revision `f6a60328b4646389b4dd4b2ecea9df31bab9a9bd`; paper: [3D-Fixer](https://arxiv.org/abs/2604.04406); model: [HorizonRobotics/3D-Fixer](https://huggingface.co/HorizonRobotics/3D-Fixer); code license: Apache-2.0.

The author evaluation source first writes MoGe-derived scene/instance PLYs, then passes the instance points to `pipeline.run`, and exports per-instance `GLB` outputs. It also ships ARSG-110K-oriented inference/evaluation scripts and object/scene IoU, Chamfer, precision, recall, and F1 calculations. That makes `O` visible geometry PLY and an output `G`/completion GLB materializable, while the generative integration happens internally; therefore the O/G boundary is only partial.

The decisive exclusion is hardware: the official README requires at least 24GB and names RTX 4090/L20 as verified. A 12GB RTX 3080 Ti is below the declared minimum. No reduced-memory replacement, patch, or installation is authorized.

| Classification | Result |
| --- | --- |
| `PUBLIC_REPRODUCIBILITY` | `STRONG` |
| `G_MATERIALIZATION` | `CLEAR` |
| `O_G_SEPARABILITY` | `PARTIAL` |
| `REGION_ACCESSIBILITY` | `STRONG` |
| `GT_EVALUATION_SUPPORT` | `STRONG` |
| `HARDWARE_FEASIBILITY` | `WEAK` |
| `EXPERIMENT_1_CAUSAL_SUITABILITY` | `PARTIAL` |

### 2. World Tracing — not selected

Official source: [haoz19/world-tracing](https://github.com/haoz19/world-tracing), revision `00335499389f10f25ceea39e1cda82fa4b9719cd`; paper: [World Tracing](https://arxiv.org/abs/2606.13652); checkpoints: [object](https://huggingface.co/haoz19/object-model-6layer), [scene](https://huggingface.co/haoz19/scene-model-6layer-840), and [dynamic](https://huggingface.co/haoz19/dynamic-model-16frame); license: CC BY-NC-ND 4.0.

This is a well-materialized image-to-layered-geometry release: a forward pass predicts registered `xyz_pred` maps, the sample workflow writes an `.rrd`, and its optional TRELLIS.2 path produces a GLB. It has public weights and packaged demo images. It is, however, an inference-only release with no author GT-completion evaluator. More importantly, its input is RGB/RGBA, not a separately supplied observed geometry `O`; treating derived visible layers as `O` would create a new, unvalidated boundary. The authors tested on 80GB A100/H100 and only state that smaller GPUs may reduce steps/resolution, so 12GB remains unverified.

| Classification | Result |
| --- | --- |
| `PUBLIC_REPRODUCIBILITY` | `STRONG` |
| `G_MATERIALIZATION` | `CLEAR` |
| `O_G_SEPARABILITY` | `WEAK` |
| `REGION_ACCESSIBILITY` | `PARTIAL` |
| `GT_EVALUATION_SUPPORT` | `WEAK` |
| `HARDWARE_FEASIBILITY` | `PARTIAL` |
| `EXPERIMENT_1_CAUSAL_SUITABILITY` | `WEAK` |

### 3. PGNet (Completion-by-Correction) — selected primary, pending hard gate

Official source: [RobWonn/PGNet](https://github.com/RobWonn/PGNet), revision `cd2b368e960d6e733259c0c9f78c4ea7b91f9480`; paper: [AAAI record](https://doi.org/10.1609/aaai.v40i9.37710); prior dataset: [Wang131/ShapeNetViPC-Gen](https://huggingface.co/datasets/Wang131/ShapeNetViPC-Gen); license: Apache-2.0.

PGNet exposes the desired experimental boundary in its own evaluator:

```text
O = ShapeNetViPC-Partial/.../<view>.dat
G = ShapeNetViPC-Gen/trellis/.../<view>.pt
final = model(gen_pc=G, partial_pc=O)["all_pcds"][-1]
```

Its generator script writes the TRELLIS-derived prior point cloud as `.pt`; the evaluation loader independently reads generated prior, partial point cloud, and GT, then calls `model(gen_pc, partial_pc)`. The same frozen `O/G` can consequently be used outside PGNet for a deterministic region partition and then reused for baseline, plausible-wrong, and oracle-A conditions. ShapeNetViPC supplies test lists, partials, GT, and rendered views; the official evaluator specifies Chamfer-L2, F-score at 0.001, and EMD.

This is selection evidence, not readiness evidence. The official repository explicitly needs a trained `-M ...pth` checkpoint but does not publish an author-controlled PGNet trained-checkpoint link in the README, repository, releases, or author-linked Hugging Face material inspected here. The authors report Ubuntu 24.04, Python 3.10, PyTorch 2.4/CUDA 12.1, CUDA extensions, and an RTX 4090 test machine, with no 12GB minimum. Missing checkpoint provenance and 12GB runtime are hard gates.

| Classification | Result |
| --- | --- |
| `PUBLIC_REPRODUCIBILITY` | `PARTIAL` |
| `G_MATERIALIZATION` | `CLEAR` |
| `O_G_SEPARABILITY` | `STRONG` |
| `REGION_ACCESSIBILITY` | `STRONG` |
| `GT_EVALUATION_SUPPORT` | `STRONG` |
| `HARDWARE_FEASIBILITY` | `PARTIAL` |
| `EXPERIMENT_1_CAUSAL_SUITABILITY` | `STRONG` |

### 4. GenPC — selected fallback, not installed

Official source: [liannuaa/GenPC](https://github.com/liannuaa/GenPC), revision `bac9e7b59f4fea0eaaa936f24ef60f342f349829`; paper: [GenPC](https://openaccess.thecvf.com/content/CVPR2025/html/Li_GenPC_Zero-shot_Point_Cloud_Completion_via_3D_Generative_Priors_CVPR_2025_paper.html); license: MIT.

GenPC starts with a partial `.ply`, uses depth prompting and image-to-3D generation, then performs geometric registration/fusion. Its configuration requests intermediate saving and the evaluator reads `workspace/<id>/<id>_fused.ply` against `data/GT/<id>.ply`. This makes final geometry externally accessible and suggests a pre-fusion generated artifact, but the exact generator-only artifact contract is not stated in the README and requires post-checkout inspection. Its supplied path depends on a sizeable multi-upstream stack (TRELLIS.2, Nunchaku Qwen Image Edit, CUDA extensions, Kaolin, PyTorch3D); it states CUDA >=12/Python 3.10/PyTorch 2.6 cu126, warns of a stage-to-stage memory leak, and gives neither a VRAM minimum nor a packaged canonical test split.

| Classification | Result |
| --- | --- |
| `PUBLIC_REPRODUCIBILITY` | `PARTIAL` |
| `G_MATERIALIZATION` | `POSSIBLE` |
| `O_G_SEPARABILITY` | `PARTIAL` |
| `REGION_ACCESSIBILITY` | `PARTIAL` |
| `GT_EVALUATION_SUPPORT` | `PARTIAL` |
| `HARDWARE_FEASIBILITY` | `PARTIAL` |
| `EXPERIMENT_1_CAUSAL_SUITABILITY` | `PARTIAL` |

It is the fallback because its prior/fusion design is genuinely related, but it loses to PGNet on the explicitly file-separated O/G boundary and documented evaluation subset. **Do not install it in the PGNet gate or this selection batch.**

### 5. Point-based Instance Completion with Scene Constraints — not selected

Official source: [wkhademi/point_based_instance_completion](https://github.com/wkhademi/point_based_instance_completion), revision `a36e38f8a0a4983d11cbafa793b753a491638170`; paper: [ICLR/OpenReview](https://openreview.net/forum?id=llSiIJosDj); license: AGPL-3.0.

This candidate has author Box checkpoint links, a documented CUDA 11.8-era environment, test/visualization paths, and CD/IoU/LFD/PCR evaluation. Its ScanWCF task includes partial scans, scene meshes, free/occluded-space constraints, and aligned GT. But ScanWCF needs a terms-of-use request and excludes licensed ShapeNet meshes. More importantly for this experiment, the direct completion network cross-attends/fuses its observations and constraints before producing the completion: the release does not expose an independently generated `G` before fusion. It remains relevant context but is not a clean causal reference.

| Classification | Result |
| --- | --- |
| `PUBLIC_REPRODUCIBILITY` | `PARTIAL` |
| `G_MATERIALIZATION` | `CLEAR` |
| `O_G_SEPARABILITY` | `WEAK` |
| `REGION_ACCESSIBILITY` | `STRONG` |
| `GT_EVALUATION_SUPPORT` | `PARTIAL` |
| `HARDWARE_FEASIBILITY` | `PARTIAL` |
| `EXPERIMENT_1_CAUSAL_SUITABILITY` | `PARTIAL` |

## Decision and bounded next gate

**Primary: PGNet (Completion-by-Correction).**
**Fallback: GenPC — not installed.**

The decision prioritizes the scientific causal boundary before convenience: PGNet's official data layout and forward call expose `O` and `G` as separate point-cloud files, while preserving official GT evaluation. Its technical reproducibility is not yet strong because the model checkpoint route is unresolved. The primary is therefore **conditional**, not approved for Experiment 1.

The next task is one bounded **PGNet checkpoint-and-single-sample functional-reproduction gate**:

1. Locate an author-controlled trained PGNet category checkpoint matching the exact source/configuration; record URL, immutable revision or SHA-256, size, and license. A self-trained or third-party substitute is forbidden.
2. Only if that gate passes, create a fresh, immutable PGNet checkout at `cd2b368e960d6e733259c0c9f78c4ea7b91f9480`, and a separate Python 3.10/PyTorch 2.4 CUDA 12.1 environment. Compile only the four official extensions and prove import/model-load compatibility on the 3080 Ti.
3. Use exactly one stated official ShapeNetViPC test-list item. Freeze and hash the partial `.dat` (`O`), pre-generated prior `.pt` (`G`), GT, rendered view, configuration, normalization choices, and seed **before** `model(gen_pc, partial_pc)`.
4. Save the unmodified baseline fine output outside the pristine upstream checkout and run only the author Chamfer-L2, F-score 0.001, and EMD evaluator on that sample.
5. Stop immediately if the author checkpoint is absent/unverifiable, one complete official `O/G/GT` sample cannot be formed, extensions/model load fail, 12GB OOM occurs, or the claimed O/G boundary cannot be preserved. In that case, record the rejection and do not install the fallback in the same batch.

No external region partition, Structural Attachment, plausible-wrong condition, oracle-A condition, or Experiment 1 is authorized by this worklog.

The machine has **not reached Experiment 1 readiness**.

## Durable record

The machine-readable candidate inventory, classifications, source revisions, risks, decision, and exact next-gate stop conditions are in [REFERENCE_MODEL_SELECTION_20260911.json](../../reference_models/REFERENCE_MODEL_SELECTION_20260911.json).
