# AI4TB-Mobile — Research Prototype (TRL-3 → TRL-4/5)

This folder implements **AI4TB-Mobile**, the multimodal TB screening
platform proposed to ICMR (see `TB proposal sent to ICMR.pdf`): X-ray,
cough, and symptom models fused into a single risk score, a Streamlit
screening app + GIS surveillance dashboard, a federated learning
simulation, and a **native Android app that runs all four trained
models fully offline, on-device** (`android_app/`).

**Read `docs/PROJECT_CHECKLIST.md` before trusting any metric in this
project.** Several components (cough model, symptom model, the
X-ray/cough/symptom pairing) have important data-limitation caveats
documented there — high accuracy in some places reflects a data
limitation or a rule-recovery exercise, not validated clinical
performance. This is expected and explained, not a bug.

## Folder structure

```
TB screening/
├── 100_TBDataset/          Chest X-ray dataset (Normal / Tuberculosis), existing
├── dataset_200/cough/      Cough audio dataset (tb / notb), existing
├── Tb disease symptoms.csv Symptom checklist data (domain-knowledge label assigned)
├── data/
│   └── processed/          Generated manifests + train/val/test splits (all modalities)
├── src/
│   ├── common/             Shared config, paths, constants
│   ├── xray/                Chest X-ray data pipeline + model + Grad-CAM
│   ├── cough/                Cough audio pipeline + model
│   ├── symptom/              Symptom labeling (domain-knowledge rule) + model
│   ├── fusion/                Multimodal fusion (xray + cough + symptom)
│   ├── dashboard/             Mock GIS surveillance data generator
│   ├── federated/             FedAvg simulation (X-ray model, 3 simulated institutions)
│   └── export/                PyTorch->ONNX export + Kotlin weight generation for the native app
├── app/
│   └── app.py                Streamlit app: screening UI + surveillance dashboard
├── android_app/             Native Android app -- on-device X-ray/cough/symptom/fusion
│                             inference via ONNX Runtime Mobile, no server needed.
│                             Debug APK: android_app/app/build/outputs/apk/debug/app-debug.apk
├── models/                  Trained checkpoints (xray/, cough/, symptom/, fusion/, federated/, onnx/)
├── reports/                 Metrics, plots, evaluation outputs (generated)
├── docs/
│   └── PROJECT_CHECKLIST.md Running task checklist + all data-limitation caveats
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -r requirements.txt
```

## Running the app

```bash
.venv/Scripts/streamlit.exe run app/app.py
```

Or via the repo-root `.claude/launch.json` config named `tb-screening-app`.

## Native Android app

`android_app/app/build/outputs/apk/debug/app-debug.apk` is a debug build
that runs fully offline — install it on a phone, no server required.
The launcher screen does on-device symptom + X-ray + cough + fusion
scoring via ONNX Runtime Mobile; a secondary "Open web dashboard" screen
still exists for reaching the GIS dashboard (needs the Streamlit server
running on the same WiFi, see above). See
[docs/PROJECT_CHECKLIST.md](docs/PROJECT_CHECKLIST.md)'s "On-device
native app" section for install steps, what's been verified vs. not
(no physical Android device was available to test this build directly),
and how to rebuild after changing the Android or model code.

## Status

See [docs/PROJECT_CHECKLIST.md](docs/PROJECT_CHECKLIST.md) for what's done,
in progress, and pending, plus how to re-run/test each piece. Work
proceeds one task at a time with sign-off between tasks — do not assume
later-stage components exist until the checklist marks them done.
