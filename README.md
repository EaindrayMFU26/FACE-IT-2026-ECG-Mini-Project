# FACE IT 2026 · Hands-on Mini Project — ECG Signal Processing & Arrhythmia Classification

**Student (GitHub):** `EaindrayMFU26`

ECG signal processing, feature extraction, heart-rate analysis and a simple machine-learning
classifier (Normal vs Arrhythmia) for **PhysioNet AHA Database record 0001, second channel,
fs = 250 Hz, first 1 minute**.

The complete analysis is in the notebook **`EaindrayMFU26_ECG_Mini_Project.ipynb`**.
The submission report (code, JupyterLab screenshots, plots, feature tables and ML results) is
**`report/ECG_Mini_Project_Report_EaindrayMFU26.pdf`**.

## Results at a glance

| Step | Result |
|---|---|
| 1. Data | Record 0001, channel 2, 250 Hz; 10-s and 60-s excerpts; files verified against PhysioNet SHA-256 sums |
| 2. FIR band-stop | 281-tap Kaiser FIR, stop band 48–52 Hz, zero-phase; 49–51 Hz power −59 dB; visible change negligible (weak 50-Hz interference) |
| 3. R-peaks (first 10 s) | 13 R-peaks (Pan–Tompkins), incl. 1 ectopic (PVC-like) beat |
| 4. Heart rhythm (1 min) | **76 R-peaks → 76 bpm (75.6 bpm from mean RR) → Normal** |
| 5. Features | Mean RR 0.78–0.81 s, mean QRS 0.061–0.068 s per 10-s segment |
| 6. ML (SVM, RBF) | Test set: confusion matrix [[21, 0], [0, 15]] → accuracy, sensitivity, specificity, precision = 100 % |

## Repository contents

```
EaindrayMFU26_ECG_Mini_Project.ipynb   main notebook (all steps, executed in JupyterLab)
data/ahadb/                            PhysioNet AHA DB record 0001 (.hea/.dat/.atr) + SHA256SUMS.txt
data/arrhythmia_training_data.csv      the 50 arrhythmia samples from the assignment
figures/                               all plots (PNG, 200 dpi)
results/                               feature tables, predictions and metrics (CSV / JSON)
screenshots/                           JupyterLab screenshots used in the report
report/                                PDF report (+ HTML source)
tools/capture_screenshots.py           runs the notebook in JupyterLab and captures screenshots
tools/build_report.py                  assembles the PDF report
```

## How to run

```bash
pip install -r requirements.txt
jupyter lab EaindrayMFU26_ECG_Mini_Project.ipynb      # then: Run ▸ Run All Cells
```

The notebook reads the record from `data/ahadb/`. To download it directly from PhysioNet
instead, use `wfdb.rdrecord("0001", pn_dir="ahadb", sampto=60 * 250)`.

## Method summary

1. **Acquisition:** `wfdb.rdrecord`, second channel (index 1), first 60 s (15 000 samples).
2. **Filtering:** FIR band-stop 48–52 Hz (Kaiser window, ≥ 40 dB, 2-Hz transition, 281 taps),
   applied forward–backward (`filtfilt`) for zero phase.
3. **R-peaks:** Pan–Tompkins (band-pass 5–15 Hz, derivative, squaring, 150-ms integration,
   adaptive thresholds, search-back) + RR/template plausibility checks.
4. **Heart rate:** count of R-peaks per minute and 60 / mean RR; < 60 Bradycardia, 60–100 Normal,
   > 100 Tachycardia.
5. **Features:** RR interval of every beat; Q = minimum 40 ms before R, S = minimum 80 ms after R,
   QRS = t_S − t_Q; segment value = **mean** of the normal (NN) beats.
6. **ML:** individual NN beats (Normal, n = 67) + 50 assignment beats (Arrhythmia);
   stratified 70/30 split; StandardScaler + SVM (RBF).

## Data source and references

* PhysioNet AHA Database (`ahadb`, v1.0.0), record 0001 — https://physionet.org/content/ahadb/1.0.0/
  (files obtained from PhysioNet's open-data mirror and verified with PhysioNet's SHA256SUMS).
* Goldberger A. L. et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 101(23):e215–e220, 2000.
* Pan J., Tompkins W. J. *A real-time QRS detection algorithm.* IEEE Trans. Biomed. Eng. 32(3):230–236, 1985.
