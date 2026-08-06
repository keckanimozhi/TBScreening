# AI4TB-Mobile — Project Checklist

Source of truth for what's been built vs. what's pending. Updated after
every completed task. One task is worked at a time; the next task does not
start until the previous one is reviewed and approved.

## Scope decisions (2026-07-25)

- No pre-existing codebase — building the ML pipeline from scratch. The
  screenshots in the ICMR proposal are mockups, not a working prototype.
- Current phase targets **ML pipeline only**: X-ray model, cough model,
  fusion, explainability. Mobile/web app UI, GIS dashboard, and federated
  learning are deferred until the core models are validated.
- `Tb disease symptoms.csv` has no TB/non-TB label column (every row looks
  like a presumptive-TB patient's symptom checklist) — symptom-based
  modeling is deferred. If a labeled version becomes available, revisit.

## Task status

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Project scaffolding (folders, README, requirements.txt, checklist) | ✅ Done | |
| 2 | X-ray data pipeline (manifest + train/val/test split) | ✅ Done | `src/xray/manifest.py`; label cross-checked across folder/metadata/filename suffix — no mismatches |
| 3 | X-ray CNN baseline model (train + evaluate) | ✅ Done | ResNet18 (layer4+fc fine-tuned), test AUC 0.893, acc 0.833, F1 0.839 on n=30 held-out test set |
| 4 | X-ray explainability (Grad-CAM) | ✅ Done | manual hook-based Grad-CAM on `layer4`; 30 overlays in `reports/xray/gradcam/`, heatmaps concentrate on central/upper lung fields as expected |
| 5 | Cough audio data pipeline (features + split) | ✅ Done | all 200 clips mono/44.1kHz/1.0s; log-mel (64, 173) feature extractor + manifest/split in place |
| 6 | Cough classifier model (train + evaluate) | ⚠️ Done, caveat below | see "Cough data limitation" section — metrics are not a generalization estimate |
| 6b | Fix cough split leakage + build synthetic multimodal patient mapping | ✅ Done | `src/cough/manifest.py` fixed, `src/fusion/manifest.py` added |
| 7 | Multimodal fusion (X-ray + cough → risk score) | ✅ Done | late fusion (logistic regression on [p_xray, p_cough]); see caveats below — cough saturates the test set so fusion can't show lift here |
| 9 | Symptom-based labeling + model | ⚠️ Done, caveat below | domain-knowledge NTEP rule label + logistic regression; see "Symptom data limitation" |
| 10 | Extend fusion to 3 modalities | ✅ Done | see "3-modality fusion" section |
| 11 | Streamlit screening app UI | ✅ Done | `app/app.py`; verified interactively in browser (see notes below) |
| 12 | GIS surveillance dashboard | ✅ Done | `src/dashboard/mock_data.py` + dashboard page in `app/app.py`; verified in browser |
| 13 | Federated learning simulation | ✅ Done | FedAvg over 3 simulated institutions on the X-ray model; see "Federated learning" section |
| 14 | Android WebView wrapper APK | ✅ Done | `android_app/`; see "Android APK" section for install + rebuild instructions |
| 15 | Export PyTorch models to ONNX + verify parity | ✅ Done | max logit diff 0.00014 (X-ray), 0.000002 (cough) vs PyTorch |
| 16 | Port symptom + fusion logistic regression to native constants | ✅ Done | `ModelWeights.kt`, hand-verified against known Streamlit outputs |
| 17 | On-device log-mel spectrogram extraction (Kotlin) | ✅ Done | algorithm verified in Python to ~1e-6 vs librosa before porting; see "On-device native app" section |
| 18 | Native Android UI (camera/mic + on-device inference) | ✅ Done | replaces WebView as launcher; WebView kept as secondary screen |
| 19 | Build, verify, deliver native on-device APK | ✅ Done | 121.7MB debug APK; see "On-device native app" section for what could/couldn't be verified |

## Environment

A virtualenv lives at `TB screening/.venv/` (Windows). Created in Task 2
with: `python -m venv .venv`. Installed: `openpyxl`, `pandas`,
`scikit-learn`, `Pillow`, `matplotlib`, `torch`/`torchvision` (CPU-only
wheels, no GPU detected on this machine — training/inference runs on
CPU), `librosa`, `soundfile`.

## Caveats to remember

- X-ray test set is only 30 images (15 TB / 15 Normal) — metrics are
  indicative, not statistically robust. Treat as a sanity check that the
  pipeline works, not as a clinical performance claim. The proposal's
  planned 800-participant multicentric validation is what would produce
  real numbers.
- X-ray model checkpoint: `models/xray/resnet18_best.pt` (best val AUC =
  0.911, epoch 22). Training log: `reports/xray/train_history.json`. Test
  metrics: `reports/xray/test_metrics.json`.

### Cough data limitation (important — read before trusting cough metrics)

`dataset_200/cough` is 200 `.wav` clips but only **7 real subjects**: TB
subjects `10A`(53 clips)/`113A`(27)/`108A`(20); non-TB subjects
`2C`(51)/`6C`(23)/`3C`(15)/`4C`(11). The first Task 6 run used a random
file-level split and got a suspicious 1.0 val AUC — traced to clips from
the same subject appearing in both train and val/test, so the model was
recognizing individuals/microphones, not TB acoustics.

Fixed in `src/cough/manifest.py` (`group_split_by_subject`): splits are
now assigned per whole subject, so no subject's clips cross a split
boundary (train=`10A,2C,4C`, val=`113A,6C`, test=`108A,3C`). This removed
the leakage bug, but **even after the fix, test AUC is still 1.0**
(precision 0.91, recall 1.0, f1 0.95, n=35) — because each split still
only contains ONE TB subject and 1-2 non-TB subjects. Perfect separation
of "one person's cough recordings" vs "a different person's cough
recordings" is easy regardless of TB status (voice, room acoustics, mic
gain all differ trivially between two individuals) — it is not evidence
the model has learned generalizable TB-vs-non-TB acoustic markers.

**Bottom line:** the cough pipeline (manifest → features → model →
eval) is verified working end-to-end and ready for real data. Its
current metrics should not be quoted as classifier performance in any
report — they reflect a 2-person discrimination task, not a validated
7-subject or 200-patient result. This will remain true until the real
multicentric cough collection (proposal Phase II) provides more unique
subjects.

### Synthetic multimodal pairing (`src/fusion/manifest.py`)

Per user direction, X-ray and cough datasets (unrelated sources, no real
per-patient link) are paired by matching label only: each of the 200
X-ray images (200 independent real patients) is assigned to a same-label
cough clip, given a new synthetic `patient_id` (e.g. `TB-0001`,
`NORM-0001`). Split membership is driven by the cough side's
subject-grouped split (so a cough subject's clips never cross splits);
X-ray images are freely shuffled to fill the required per-split counts
since they carry no leakage risk. Output:
`data/processed/multimodal_manifest.csv` +
`multimodal_train/val/test.csv` (115/50/35 pairs).

**This pairing is synthetic, not a real patient correspondence.** Any
fusion model built on it demonstrates the fusion *architecture* works,
not a validated cross-modal relationship — that requires the proposal's
real paired data collection (Phase II, 800 participants with matched
X-ray + cough + symptoms per patient).

### Symptom data limitation (important — read before trusting symptom metrics)

`Tb disease symptoms.csv` has no TB/non-TB outcome column. Per direction,
a label was assigned using the standard NTEP/WHO presumptive pulmonary
TB screening rule (`src/symptom/labeling.py`): presumptive TB = cough
>=2 weeks, OR fever >=2 weeks plus at least one of weight loss / night
sweats / hemoptysis / blood-tinged sputum. This gives a 73%/27% split
(734/266 of 1000) — not degenerate, and it's the actual rule Indian TB
screening programmes use to decide who gets referred for confirmatory
testing, so it's a defensible proxy for "referral recommendation," which
is the role the proposal assigns to this module.

Two things to keep in mind:

1. **The source data itself looks synthetic.** All 13 symptoms are
   present at ~50% prevalence independently, with no correlation
   structure between them, and patient names are placeholder-looking
   (e.g. "Noe", "Genna"). IDs are unique (no duplicate-patient issue like
   cough had), so row-level splitting is valid — but the symptom pattern
   itself likely doesn't reflect a real population of TB/non-TB patients.
2. **The label is a deterministic function of the same features fed to
   the model**, so the classifier's job is to reconstruct the rule, not
   to discover new signal. It did so essentially perfectly: val AUC
   1.0, test accuracy/precision/recall/F1/AUC all 1.0
   (`reports/symptom/test_metrics.json`). The learned logistic-regression
   coefficients confirm this — `cough and phlegm continuously for two
   weeks to four weeks` (+6.28) and `fever for two weeks` (+5.28) dominate
   by an order of magnitude over every other symptom, exactly mirroring
   the rule's structure (`reports/symptom/feature_importance.json`).

**Bottom line:** this is a correctly-working rule-based referral-scoring
pipeline, not a validated diagnostic signal — same caveat class as the
cough model, for a different reason (label circularity here vs. subject
scarcity there). Re-derive the label/model once real presumptive-TB
patient symptom records (with an actual confirmed-TB outcome) exist.

### Fusion model (`src/fusion/`) — now 3 modalities

Late fusion: `p_xray`, `p_cough`, `p_symptom` come from the
already-trained, frozen unimodal models (Tasks 3, 6/6b, 9) run over
`multimodal_manifest.csv` (`src/fusion/build_features.py` →
`data/processed/fusion_{split}.csv`), then a logistic regression
(`src/fusion/train.py`) learns to combine the three probabilities. Chose
logistic regression over a heavier fusion network because there are only
3 input features and 115 training pairs — anything more expressive would
just overfit three numbers. Symptom pairing added in Task 10: each
synthetic patient also gets a same-label symptom row from
`symptom_manifest.csv` (see `symptom_id` column in
`multimodal_manifest.csv`), sampled without replacement, split membership
still driven by the cough side.

Test set (n=35, same 35 pairs used for all four so the comparison is
apples-to-apples):

| | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| X-ray only | 0.886 | 0.833 | 1.00 | 0.909 | 0.980 |
| Cough only | 0.943 | 0.909 | 1.00 | 0.952 | 1.000 |
| Symptom only | 0.971 | 0.952 | 1.00 | 0.976 | 1.000 |
| Fused (3-modality) | 0.971 | 0.952 | 1.00 | 0.976 | 1.000 |

Fusion matches (not exceeds) symptom-only, because symptom-only is
already saturated at AUC 1.0 on this split — no headroom to demonstrate
lift, for the reason explained in "Symptom data limitation" above (the
label is a deterministic function of the symptom features, so it's
almost perfectly learnable on its own). Same underlying issue as the
cough-only saturation: **none of these numbers should be read as "fusion
doesn't help" or "symptoms alone are enough"** — they reflect the data
limitations of both the cough and symptom sources, not the fusion
architecture. Re-run this comparison once real paired multicentric data
exists (proposal Phase II); that is the only way to get a trustworthy
answer to "does fusion beat unimodal."

### App UI (`app/app.py`)

Streamlit app, chosen (over FastAPI+HTML or native Android) for fastest
iteration in a research-prototype context. Single page today: symptom
checklist + optional X-ray/cough upload → per-modality probabilities +
fused risk score/band + Grad-CAM overlay + contributing-symptoms
explainer, with the same data caveats surfaced as a banner. Sidebar has
a placeholder second page ("Surveillance Dashboard") for Task 12 to fill
in. Missing modalities (no X-ray/cough uploaded) are imputed with their
training-set mean probability (`src/fusion/predict.py`) rather than an
arbitrary 0.5, and the UI says so explicitly under the result.

Verified interactively (browser automation via Claude_Browser): loaded
the page, confirmed all 13 symptom checkboxes + both uploaders render;
ran with nothing checked (correctly showed Symptoms 1%, Fused 17%, Low,
"routine follow-up"); checked "Fever for two weeks" and re-ran (Symptoms
jumped to 74%, Fused to 61%, Medium band, correct recommendation text,
and the contributing-symptoms panel correctly showed
`Fever for two weeks (weight +5.28)` matching the trained coefficient).

**Tooling note for whoever tests this next**: Streamlit's current
checkbox widget is React-Aria-based and does NOT keep the native
`<input type="checkbox">`'s `checked` DOM property in sync — reading
`input.checked` via JS always shows `false` even when the box is visibly
checked. The real signal is `label[data-selected="true"]` inside
`div.stCheckbox`. Cost some time to discover; don't re-debug this if it
comes up again, just check `data-selected`. X-ray/cough file upload
interactions were not exercised in this browser session (no file-upload
capability on this browser surface) — the underlying `predict_xray` /
`predict_cough` / `gradcam_overlay_for_image` functions are exercised
independently by the Task 3/4/6 CLI scripts, so risk is limited to the
Streamlit file-handling glue (temp file write + pass path), which is a
few lines of standard code.

### GIS surveillance dashboard (`src/dashboard/mock_data.py`)

No real geolocated screening events exist yet, so this generates a
seeded synthetic dataset (180 mock events) clustered around 4 points
near Perundurai/Erode (the proposal's primary clinical site) with random
risk bands and dates over the last 12 months — purely to build/demo the
dashboard UI ahead of real field deployment. Every number here is
fabricated; the dashboard shows a prominent banner saying so.

Replaces the Task 11 placeholder page in `app/app.py`
(`render_dashboard_page`): 4 KPI tiles (total/high/medium/low),
`st.map` with per-point risk-band coloring (red/orange/green), and a
monthly bar chart. Verified in browser: KPI counts (180 total, 21
high/94 medium/65 low) matched the standalone data-generation output
exactly, the map rendered with tile attribution (confirming the colored
scatter layer loaded without error), and all 12 months appeared in the
bar chart.

**Browser-testing note**: navigating between the "Screening" and
"Surveillance Dashboard" sidebar radio options via the browser
automation tool's coordinate-based click didn't register (same
React-Aria pattern as the checkboxes) even though the click coordinates
were confirmed correct via `getBoundingClientRect`. Had to dispatch a
full `pointerdown`→`mousedown`→`pointerup`→`mouseup`→`click` event
sequence via JS to get the radio to actually switch pages. If this
recurs when testing other pages/widgets, that's the fix — a single
synthetic `click` isn't enough for this component library.

### Federated learning simulation (`src/federated/`)

FedAvg over 3 simulated institutions, built on the X-ray model (chosen
over cough/symptom because it's the only track with a sizeable image
dataset from independent real patients — 200 distinct people; cough only
has 7 real subjects, too few to simulate multiple institutions
meaningfully). `src/federated/partition.py` splits the 140-image X-ray
train set into 3 stratified, disjoint client subsets (~46-47 images
each, ~23-24 TB). `src/federated/simulate.py` runs 15 rounds: each round,
every client trains locally for 2 epochs on its own private subset
(`layer4` + `fc` only, matching the centralized model's fine-tuning
scope — same reasoning as Task 3, and it means only the trained update
is "sent" to the server, not the whole network, per the proposal's "only
model updates are transmitted" design), then weights are averaged
(weighted by client size) into a new global model. No client's images
are ever combined or shared — only the trainable weight tensors are
aggregated.

Test set (same 30 images used for the Task 3 centralized model, so this
is apples-to-apples):

| | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Centralized (Task 3) | 0.833 | 0.813 | 0.867 | 0.839 | 0.893 |
| Federated (3 clients, 15 rounds) | 0.733 | 1.000 | 0.467 | 0.636 | 0.876 |

AUC is close (0.876 vs 0.893), but federated recall is notably lower
(missed more TB cases, though with perfect precision — zero false
positives). This gap is expected and mostly attributable to each
simulated client having only ~46 images to train on (vs. 140 for the
centralized model) — not evidence that FedAvg itself doesn't work. With
the proposal's real 800-participant multicentric data spread across
real hospitals, each site would have far more data than these synthetic
clients do, and the gap would be expected to narrow. Saved to
`models/federated/xray_fedavg_trainable.pt` (just the aggregated
`layer4`+`fc` weights, not a full checkpoint) and
`reports/federated/test_metrics.json`.

## How to test each completed task

- **Task 1**: run `ls` (or `dir`) inside `TB screening/` and confirm
  `src/`, `data/processed/`, `reports/`, `docs/`, `README.md`,
  `requirements.txt` exist. No runtime behavior to test yet.
- **Task 2**: from `TB screening/`, run
  `.venv/Scripts/python.exe -m src.xray.manifest`. Expect:
  `Manifest: 200 images (100 TB, 100 Normal)` and a 140/30/30 train/val/test
  split (70/15/15 TB per split). Check `data/processed/xray_manifest.csv`,
  `xray_train.csv`, `xray_val.csv`, `xray_test.csv` were created with
  columns `filename, filepath, label, gender, age, findings`.
- **Task 3**: run
  `.venv/Scripts/python.exe -m src.xray.train` (trains, prints per-epoch
  val AUC, saves `models/xray/resnet18_best.pt`), then
  `.venv/Scripts/python.exe -m src.xray.evaluate` (prints and saves test
  metrics to `reports/xray/test_metrics.json`). Expect test AUC roughly in
  the high 0.8s-0.9 range given the current 30-image test set; exact
  numbers vary slightly run to run since only inference (not training) is
  seeded deterministically end-to-end.
- **Task 4**: run
  `.venv/Scripts/python.exe -m src.xray.gradcam`. Expect
  `Saved 30 Grad-CAM overlays to .../reports/xray/gradcam`. Open a few
  `..._true-TB_pred-TB_...png` files and confirm the heatmap is
  concentrated on the lung fields, not background/shoulders.
- **Task 5**: run
  `.venv/Scripts/python.exe -m src.cough.manifest`. Expect
  `Manifest: 200 recordings (100 TB, 100 non-TB)` and (after the Task 6b
  fix) a subject-grouped 115/50/35 split, e.g.
  `train subjects: ['10A', '2C', '4C']`. Check
  `data/processed/cough_manifest.csv` and `cough_train/val/test.csv`
  (now include a `subject_id` column). Feature extractor sanity check:
  `extract_log_mel(<any wav path>)` should return a `(64, 173)` float32
  array with mean ~0, std ~1.
- **Task 6 / 6b**: run
  `.venv/Scripts/python.exe -m src.cough.train` then
  `.venv/Scripts/python.exe -m src.cough.evaluate`. See the "Cough data
  limitation" section above before interpreting the numbers — do not
  expect a lower/more-realistic AUC than 1.0; that itself would be
  surprising given only 2 test subjects.
- **Multimodal pairing**: run
  `.venv/Scripts/python.exe -m src.fusion.manifest`. Expect
  `train: 115 pairs (53 TB)`, `val: 50 pairs (27 TB)`,
  `test: 35 pairs (20 TB)`, and the reminder that pairing is synthetic.
  Check `data/processed/multimodal_manifest.csv` has no duplicate
  `xray_filename` or `cough_filename` values.
- **Task 7**: run, in order,
  `.venv/Scripts/python.exe -m src.fusion.build_features`,
  `.venv/Scripts/python.exe -m src.fusion.train`,
  `.venv/Scripts/python.exe -m src.fusion.evaluate`. Expect the
  xray_only/cough_only/fused comparison table above (or close to it —
  logistic regression fit is deterministic given the data, but the
  underlying unimodal checkpoints could differ slightly if retrained).
  Note: on first import each session, `sklearn.linear_model` occasionally
  raised a transient `DLL load failed ... Application Control policy`
  error on this machine (looked like an antivirus on-access scan lock);
  it succeeded on immediate retry both times it happened. If it recurs,
  just re-run the command.
- **Task 9**: run
  `.venv/Scripts/python.exe -m src.symptom.manifest`, then
  `.venv/Scripts/python.exe -m src.symptom.train`, then
  `.venv/Scripts/python.exe -m src.symptom.evaluate`. Expect
  `Manifest: 1000 patients (734 presumptive-TB, 266 not)`, then val/test
  AUC of 1.0 (see "Symptom data limitation" — this is expected, not a
  bug). Check `reports/symptom/feature_importance.json` shows `cough and
  phlegm continuously for two weeks to four weeks` and `fever for two
  weeks` as by far the largest-magnitude coefficients.
- **Task 10**: run, in order,
  `.venv/Scripts/python.exe -m src.fusion.manifest`,
  `.venv/Scripts/python.exe -m src.fusion.build_features`,
  `.venv/Scripts/python.exe -m src.fusion.train`,
  `.venv/Scripts/python.exe -m src.fusion.evaluate`. Expect the 4-row
  comparison table above (xray_only/cough_only/symptom_only/fused).
  Check `data/processed/multimodal_manifest.csv` has a `symptom_id`
  column with no duplicates.
- **Task 11**: run
  `.venv/Scripts/streamlit.exe run app/app.py` (or use the
  `tb-screening-app` launch.json config at the repo root). Opens at
  http://localhost:8501. Check some symptoms, click "Run screening",
  confirm the Symptoms/Fused metrics change and the risk band/
  recommendation update accordingly.
- **Task 12**: with the app running, switch to "Surveillance Dashboard"
  in the sidebar. Confirm 4 KPI tiles, a colored hotspot map, and a
  12-month bar chart all render with no errors. Or standalone:
  `.venv/Scripts/python.exe -c "from src.dashboard.mock_data import generate_mock_events; print(generate_mock_events().shape)"`
  should print `(180, 7)`.
- **Task 13**: run
  `.venv/Scripts/python.exe -m src.federated.partition` (expect 3 clients,
  ~46-47 images each), then
  `.venv/Scripts/python.exe -m src.federated.simulate` (expect 15 rounds
  of `val_auc=...` output, then the federated-vs-centralized comparison
  table above, saved to `reports/federated/test_metrics.json`). Takes
  several minutes on CPU (3 clients × 15 rounds × 2 local epochs).
- **Task 14**: install the APK on an Android phone on the same WiFi as
  this machine, start the Streamlit server, open the app, confirm the
  server URL field, tap "Go", and confirm the screening page loads
  inside the app. See "Android APK" section for full detail.
- **Tasks 15-19 (on-device native app)**: run
  `.venv/Scripts/python.exe -m src.export.export_xray` and
  `-m src.export.export_cough` (each prints a max-logit-diff and asserts
  it's small), `-m src.export.verify_mel_algorithm` (prints max abs diff
  vs librosa, should be ~1e-6, and writes the mel filterbank asset), and
  `-m src.export.export_logreg_kotlin` (writes `ModelWeights.kt`). Then
  rebuild per the "On-device native app" section and install on a phone:
  check symptoms + tap "Run screening" with no X-ray/cough (should show
  "not provided (used average)" for both and a plausible fused score);
  pick an X-ray image and confirm a probability appears; tap "Record
  cough" and confirm "Recorded (1.0s captured)" appears before running
  screening again. None of this on-device behavior has been physically
  verified — no Android device/emulator was available in this
  environment — so treat first real-device testing as the actual test
  of this feature, not a formality.

### Android APK (`android_app/`)

A **WebView wrapper**, not an offline native app: it's a thin Kotlin
shell that displays the existing Streamlit app (`app/app.py`) in a
WebView. The phone must be on the **same WiFi network** as whatever
machine is running the Streamlit server — there is no on-device ML,
no bundled models, and no offline mode. This was the explicit,
faster-to-build choice over a full native app with on-device
TFLite-converted models (which would need to redo Task 11's UI natively
plus convert all 4 trained models).

**Where the APK is**: `android_app/app/build/outputs/apk/debug/app-debug.apk`
(unsigned debug build — Android will warn about installing from an
unknown source; that's expected for a sideloaded debug build, not a
red flag).

**To install on a phone**:
1. Make sure the phone is on the same WiFi network as this machine.
2. Copy `app-debug.apk` to the phone (e.g. via USB, email to yourself, or a
   cloud drive) and tap it to install — you'll need to allow "install
   from unknown sources" for whichever app you use to open the file.
3. Start the Streamlit server on this machine:
   `.venv/Scripts/streamlit.exe run app/app.py` (from `TB screening/`).
4. Open the AI4TB-Mobile app on the phone. The server address field
   defaults to `http://192.168.1.3:8501` (this machine's LAN IP at build
   time). If that's no longer this machine's IP (check via `ipconfig`
   locally, or Streamlit prints "Network URL: ..." on startup), edit the
   field in-app and tap "Go" — it's saved for next launch.
5. If the app can't reach the server: Windows Firewall may be blocking
   inbound connections on port 8501. This needs a firewall rule change,
   which is a system-security setting — do this yourself rather than
   have it changed for you; Windows will typically prompt to allow it
   the first time a device tries to connect if the firewall is set to
   ask.

**Build environment note** (only matters if rebuilding): this project's
root folder path (`...\R&D\TB screening\...`) contains an `&`, which
breaks Android's `sdkmanager.bat`/`gradlew.bat` — they don't quote paths
internally, so cmd.exe splits commands at the `&`. Found this the hard
way (build failed with `'C:\Users\kanim\2026-27\R' is not recognized...`).
**Workaround**: all SDK/Gradle tooling and the actual build run from a
copy of this project at `C:\ai4tb_build\project\` (JDK 17 + Android SDK
cmdline-tools + Gradle 8.9, all under `C:\ai4tb_build\tools\`), not from
inside the `R&D` tree. The canonical source of truth for the Android
code is still `android_app/` in this repo — `C:\ai4tb_build\` is only a
build workspace, kept in sync by copying files across before a rebuild.

**To rebuild** (from Git Bash / this Bash tool, not PowerShell, so `export` works):
```bash
# sync source -> build workspace (only needed if android_app/ changed)
cp -r "TB screening/android_app/"* /c/ai4tb_build/project/

export JAVA_HOME="C:/ai4tb_build/tools/jdk-17.0.19+10"
cd /c/ai4tb_build/project
./gradlew.bat assembleDebug

# copy the result back into the repo
cp app/build/outputs/apk/debug/app-debug.apk \
   "../../TB screening/android_app/app/build/outputs/apk/debug/app-debug.apk"
```

**Bug found and fixed during this build**: `network_security_config.xml`
originally had an explanatory comment containing `--` (e.g. "not HTTPS --
there's no cert"), which is invalid inside XML comments per spec and
failed `mergeDebugResources`/`parseDebugLocalResources` with `The string
"--" is not permitted within comments`. Fixed by rewording to avoid the
double-hyphen. **This bug recurred** in `AndroidManifest.xml` during
Task 19 (a new comment used `--` again) — same fix, same lesson: never
use `--` inside an XML comment in this project, anywhere. Ran a project
-wide scan (`re.finditer(r'<!--(.*?)-->', ...)` over every `.xml` file)
after the second occurrence to confirm no other instances remain.

### On-device native app (Tasks 15-19) — a real offline app, not the wrapper

Per direction ("I have to build this app with an on device ML model"),
`android_app/` was converted from a WebView wrapper into a real native
app: **MainActivity is now a native screening screen that runs all four
trained models entirely on-device, with no server, no network calls.**
The original WebView wrapper (Task 14) still exists as `WebViewActivity`
(button: "Open web dashboard"), reachable from the native screen, since
it's the only way to reach the GIS surveillance dashboard from a phone —
the native screen does not attempt to replicate that or the X-ray
Grad-CAM visualization (see limitations below).

**Model conversion:**
- X-ray CNN and cough CNN: PyTorch → ONNX (`src/export/export_xray.py`,
  `src/export/export_cough.py`), each verified against the original
  PyTorch model's output on its own held-out test set before being
  trusted: max logit difference 0.00014 (X-ray, n=30) and 0.000002
  (cough, n=35) — both far below anything that would change a
  prediction. Bundled as `android_app/app/src/main/assets/{xray,cough}.onnx`
  (44.7MB / 93KB), run via `onnxruntime-android` in `OnnxModels.kt`.
- Symptom and fusion models: both are just a logistic regression (dot
  product + sigmoid), so no ML runtime needed — `src/export/export_logreg_kotlin.py`
  extracts the trained `coef_`/`intercept_` and generates
  `ModelWeights.kt` directly (not hand-written; regenerate via that
  script if the models are retrained). Hand-verified: an all-symptoms
  -unchecked input gives sigmoid(-4.248) ≈ 1.4%, and checking only
  "fever for two weeks" gives sigmoid(-4.248+5.28) ≈ 74% — both match
  what the Streamlit app showed for the same inputs during Task 11
  testing.

**On-device cough feature extraction** (`MelSpectrogram.kt`) is the
hardest piece: no librosa on Android, so log-mel spectrogram extraction
(FFT → mel filterbank → power_to_db → standardize) had to be
hand-implemented. There is no Android device/emulator in this
environment to test the Kotlin file directly, so the verification
strategy was: implement the *exact same algorithm* in Python
(`src/export/verify_mel_algorithm.py`, a manual radix-2 FFT + framing +
windowing, not calling librosa's stft/melspectrogram functions), compare
its output against real `librosa.feature.melspectrogram` +
`librosa.power_to_db` on 10 real cough clips, confirm it matches to
~1e-6, *then* port that verified algorithm into Kotlin unchanged. Found
one real bug this way: librosa >=0.10 defaults `pad_mode="constant"`
(zero-padding), not `"reflect"` as commonly assumed from older
tutorials/docs — using reflect gave a 1.35 max error, switching to
constant padding brought it to 0.000001. The mel filterbank matrix
itself (`librosa.filters.mel(...)`, Slaney scale) is precomputed in
Python and bundled as a binary asset
(`assets/mel_filterbank_64x513.bin`, 64×513 float32) rather than
re-derived on-device, removing a second independent source of numerical
risk.

**What could NOT be verified** (no physical Android device or emulator
available in this environment):
- The actual Kotlin `MelSpectrogram`/`OnnxModels`/`CoughRecorder` code
  running on-device end-to-end. The build compiling cleanly against the
  real `onnxruntime-android` AAR (not a mock) is meaningful evidence the
  API usage is correct, and the algorithm was verified in Python before
  porting, but this is not the same as having actually run it.
- `Bitmap.createScaledBitmap`'s resize filtering for the X-ray image is
  not guaranteed to numerically match PIL/torchvision's resize used
  during training. If on-device X-ray predictions look off compared to
  the same image run through `src/xray/evaluate.py`, this is the first
  place to check.
- Real microphone recording quality/levels (gain, noise floor) on an
  actual device, vs. the clean pre-recorded clips the cough model was
  trained on.

**Scope cut, explicit**: Grad-CAM is not implemented on-device.
`onnxruntime-android` is inference-only (no backprop), so the
hook-based Grad-CAM from `src/xray/gradcam.py` doesn't port as-is. A CAM
(not Grad-CAM) variant is feasible without backprop — ResNet18's
`avgpool → fc` structure means a class activation map can be computed
directly from `layer4`'s feature maps and the `fc` weight vector alone —
but this needs a second ONNX output (the feature maps) and on-device
heatmap rendering, which was out of scope for this pass. Worth a
follow-up task if on-device explainability matters.

**To rebuild after changing on-device model code**: same
`C:\ai4tb_build\` workflow as the wrapper (see above) — sync
`android_app/` into `/c/ai4tb_build/project/`, including
`app/src/main/assets/` if the ONNX models or filterbank changed, then
`./gradlew.bat assembleDebug` with `JAVA_HOME` set to the extracted JDK.
To regenerate the ONNX models or `ModelWeights.kt` after retraining:
```bash
.venv/Scripts/python.exe -m src.export.export_xray
.venv/Scripts/python.exe -m src.export.export_cough
.venv/Scripts/python.exe -m src.export.export_logreg_kotlin
# then copy models/onnx/*.onnx into android_app/app/src/main/assets/
```
