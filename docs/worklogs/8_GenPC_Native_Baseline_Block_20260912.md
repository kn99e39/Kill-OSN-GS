# Worklog #8 — Native Windows GenPC baseline continuation (2026-09-12)

## Scope and guardrails

- Continued Worklog #7 on the MAIN native Windows RTX 5080 workstation.
- Kept PyTorch 2.8.0+cu128, Python 3.11.9, CUDA capability 12.0 / sm_120.
- Did not move to WSL/Linux, downgrade PyTorch/CUDA, or run Experiment 1.
- Kept the pristine GenPC checkout at commit `bac9e7b59f4fea0eaaa936f24ef60f342f349829` untouched.

## Native toolchain gates closed

- CUDA Toolkit 12.8 installed side-by-side; driver 596.49 and CUDA 13.3 were not removed.
- `nvcc`: CUDA 12.8, V12.8.61.
- Visual Studio Build Tools 2022 17.14.35, MSVC 19.44.35228 x64, Windows SDK 10.0.26100.0.
- Official Kaolin 0.18.0 wheel installed for torch 2.8.0 / CUDA 12.8.
- Official NVlabs nvdiffrast pinned at `253ac4fcea7de5f396371124af597e6cc957bfae`; native wheel built and CUDA triangle smoke test passed.
- Official GenPC Chamfer3D and EMD extensions built with CUDA 12.8/MSVC; both CUDA smoke tests passed.
- Official PyTorch3D pinned at `0a7d4c1a171e8b768c63f15b17564f9ad495f49b`; rebuilt successfully using official NVIDIA CCCL v2.7.0 (`b5fe509fd11a925f90d6495176707cc1184eed9d`, the CUDA 12.8-matched CUB/Thrust pair). PyTorch3D CUDA renderer smoke test passed.

## Load gates

- InstantMesh official Zero123++ / InstantMesh-base / DINO / FlexiCubes load gate passed.
- ControlNet official base / depth ControlNet / VAE load gate passed.
- External, auditable compatibility shims remain outside pristine checkouts:
  - inject the missing `save_ply` API expected by the pinned GenPC adapter;
  - enable `trust_remote_code` for the current diffusers custom `zero123plus` pipeline.

## DDNM mapping

- Official DDNM checkout: `00b58eac7843a4c99114fd8fa42da7aa2b6808af`.
- Source and paper/supplement audit found no author-released GenPC DDNM adapter/checkpoint with the required `DepthPrompting` interface.
- Classification remains `PUBLISHED_INPAINTING_UNAVAILABLE`; exact published inpainting reproduction is blocked.
- The predeclared method-valid path uses the author-controlled `cv2` inpainting implementation and is explicitly classified `INPAINTING_COMPONENT_DEVIATION`.

## Single 07136 method-valid run

The predeclared manifest is `reference_models/genpc/METHOD_VALID_BASELINE_07136_20260912.json`. It fixes sample `07136` (sofa), seed 42, the published camera/registration settings, ControlNet scale 0.99 / 30 steps, and excludes Experiment 1.

Depth Prompting, cv2 inpainting, and the external correction that writes the downstream-required `img.png` completed. The current preserved output is:

`reference_models/genpc/runs/baseline_07136/07136/`

The generated `img.png` is present with SHA256 `5f294d735c1107cf887905f969659bf69a2f6a3788cd798a27a31ae37ee2bf1a`.

The run then stopped before ScaleAdapter because the author source requests gated `briaai/RMBG-2.0`; the native process received HTTP 401 and no local cache or HF token was available. The official model card states that access requires the non-commercial access form/login: https://huggingface.co/briaai/RMBG-2.0/tree/main.

No public segmentation substitute was introduced, because that would add an unplanned component deviation beyond the predeclared inpainting deviation. The runner supports resuming from the preserved completed depth/ControlNet stage once the user-authorized model access is available.

Earlier pre-inference launcher failures (wrong system Python, then UTF-8 locale, then an external flag/category lookup, then missing InstantMesh `src` path) were preserved in separate run records and corrected without changing the official source. The latest failure is the only current blocker.

## Not yet performed

- No registration/fusion output exists.
- No pre-fusion O/G export was attempted, per the instruction to instrument only after a successful baseline.
- No baseline-vs-instrumented equivalence check or frozen artifact set exists.
- Experiment 1 was not run.

## Current status

The native toolchain and progressive model load gates are ready. The single 07136 method-valid run is blocked at the author-required gated RMBG-2.0 access step. Resume the existing run after authenticating to Hugging Face and accepting the model’s stated access terms; do not substitute another background-removal model without re-declaring the experiment.

EXACT PUBLISHED BASELINE: BLOCKED
GENPC METHOD-VALID SUBSTRATE: NOT READY
EXPERIMENT 1: NOT RUN
