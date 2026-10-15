"""Assemble the PDF report from the executed notebook, figures, result tables and screenshots.

Usage:  python tools/build_report.py      (after tools/capture_screenshots.py)
Output: report/ECG_Mini_Project_Report_EaindrayMFU26.pdf  (+ report/report.html)
"""
import html
import json
from pathlib import Path

import nbformat
import pandas as pd
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
STUDENT = "EaindrayMFU26"
NB_NAME = f"{STUDENT}_ECG_Mini_Project.ipynb"
REPORT = ROOT / "report"
PDF = REPORT / f"ECG_Mini_Project_Report_{STUDENT}.pdf"
RES, FIG, SHOTS = ROOT / "results", ROOT / "figures", ROOT / "screenshots"
CHROMIUM = "/opt/pw-browsers/chromium"

S = json.loads((RES / "summary.json").read_text())
nb = nbformat.read(ROOT / NB_NAME, as_version=4)
CODE = {}
for cell in nb.cells:
    first = cell.source.split("\n", 1)[0]
    if cell.cell_type == "code" and first.startswith("# Step "):
        CODE[first.split()[2]] = cell.source
MANIFEST = json.loads((SHOTS / "manifest.json").read_text())
ENV = next(o["text"] for c in nb.cells if c.cell_type == "code"
           and c.source.startswith("# Step 0.1") for o in c.outputs if o.get("name") == "stdout")

counters = {"Figure": 0, "Table": 0, "Code": 0, "Screenshot": 0}


def nxt(kind):
    counters[kind] += 1
    return f"{kind} {counters[kind]}"


def esc(s):
    return html.escape(str(s))


def code(label):
    src = CODE[label]
    title = src.split("\n", 1)[0].split("—", 1)[1].split("|")[0].strip()
    body = highlight(src, PythonLexer(), HtmlFormatter(cssclass="hl", linenos="inline"))
    return (f'<div class="listing"><div class="cap">{nxt("Code")} · Step {label} — '
            f'{esc(title)}</div>{body}</div>')


def shot(fname, caption):
    m = next(e for e in MANIFEST if e["file"] == fname)
    part = f" (part {m['part']} of {m['parts']})" if m.get("parts", 1) > 1 else ""
    return (f'<figure class="shot"><img src="../screenshots/{fname}">'
            f'<figcaption><b>{nxt("Screenshot")}{part}.</b> {caption} '
            f'<span class="src">[{fname}]</span></figcaption></figure>')


def shots_for(*prefixes, caption):
    out = []
    for e in MANIFEST:
        steps = [s for s in e["steps"] if s]
        if any(str(s).startswith(prefixes) for s in steps) and e["file"] != "ss00_overview_environment.jpg":
            out.append(shot(e["file"], caption.format(cells=", ".join(
                s if s[0].isdigit() else "markdown" for s in steps))))
    return "\n".join(out)


def fig(fname, caption, width="100%"):
    return (f'<figure class="plot"><img src="../figures/{fname}" style="width:{width}">'
            f'<figcaption><b>{nxt("Figure")}.</b> {caption}</figcaption></figure>')


def table(df, caption, fmts=None, cls="data", index=False, note=""):
    fmts = fmts or {}
    d = df.reset_index() if index else df
    head = "".join(f"<th>{esc(c)}</th>" for c in d.columns)
    rows = []
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            if c in fmts and pd.notna(v) and not isinstance(v, str):
                v = fmts[c].format(v)
            elif isinstance(v, float) and pd.isna(v):
                v = "–"
            cells.append(f"<td>{esc(v)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    note = f'<div class="note">{note}</div>' if note else ""
    return (f'<div class="tbl"><div class="cap"><b>{nxt("Table")}.</b> {caption}</div>'
            f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}'
            f"</tbody></table>{note}</div>")


def halves(df):
    """Two side-by-side compact tables (saves pages for long listings)."""
    out, half = [], (len(df) + 1) // 2
    for part in (df.iloc[:half], df.iloc[half:]):
        head = "".join(f"<th>{esc(c)}</th>" for c in part.columns)
        rows = "".join("<tr>" + "".join(
            f"<td>{v:.4f}</td>" if isinstance(v, float) else f"<td>{esc(v)}</td>"
            for v in r) + "</tr>" for r in part.itertuples(index=False))
        out.append(f'<table class="data compact"><thead><tr>{head}</tr></thead>'
                   f"<tbody>{rows}</tbody></table>")
    return "".join(out)


def kv_table(series, caption, fmt=None):
    df = pd.DataFrame({"Quantity": series.index, "Value": series.values})
    if fmt:
        df["Value"] = [fmt(k, v) for k, v in zip(df["Quantity"], df["Value"])]
    return table(df, caption, cls="data kv")


# ---------------------------------------------------------------- data for the text
hr, ml, flt = S["heart_rate"], S["ml"], S["filter"]
seg = pd.read_csv(RES / "step5_segment_features.csv")
segd = pd.read_csv(RES / "step5_segment_features_detailed.csv")
beats = pd.read_csv(RES / "step5_beat_features.csv")
r10 = pd.read_csv(RES / "step3_rpeaks_first10s.csv")
pred = pd.read_csv(RES / "step6_test_predictions.csv")
dataset = pd.read_csv(RES / "step6_dataset.csv")
excl = pd.read_csv(RES / "step6_excluded_beats_predictions.csv")
metrics = pd.read_csv(RES / "step6_metrics.csv", index_col=0)["value"]
fstats = pd.read_csv(RES / "step2_filter_effect.csv", index_col=0)["value"]
hrs = pd.read_csv(RES / "step4_heart_rate.csv", index_col=0)["value"]
cm = ml["confusion_matrix"]
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
pct = lambda v: f"{100 * float(v):.1f} %"
nn = beats[beats["NN beat"]]
excluded = beats[~beats["NN beat"]].copy()
excluded["reason"] = [
    "first beat – no preceding R-peak" if b == 1 else
    "ectopic (PVC-like) beat" if t.startswith("Ectopic") else
    "low-confidence beat (noise burst)" if t.startswith("Low") else
    "RR = compensatory pause after an ectopic beat" if beats.loc[beats["beat"] == b - 1,
                                                                   "beat type"].iloc[0].startswith("Ectopic")
    else "RR adjacent to the low-confidence beat"
    for b, t in zip(excluded["beat"], excluded["beat type"])]

# ---------------------------------------------------------------- sections
H = []
H.append(f"""
<section class="cover">
  <div class="kicker">FACE IT 2026 · Hands-on Mini Project</div>
  <h1 class="title">ECG Signal Processing, Heart-Rate Analysis and Arrhythmia Classification</h1>
  <div class="subtitle">PhysioNet AHA Database — record 0001, second channel, fs = 250 Hz,
  1-minute recording of a recruitment-medical-examination candidate</div>
  <table class="meta">
    <tr><th>Student (GitHub)</th><td><b>{STUDENT}</b></td></tr>
    <tr><th>Notebook</th><td><code>{NB_NAME}</code></td></tr>
    <tr><th>Working environment</th><td>JupyterLab 4.6.4 · Python 3.13 (Linux) ·
        NumPy, SciPy, pandas, Matplotlib, scikit-learn, wfdb</td></tr>
    <tr><th>Date</th><td>15 October 2026</td></tr>
  </table>
  <div class="tiles">
    <div class="tile"><div class="lbl">Heart rate (1 min)</div>
      <div class="val">{hr['hr_count_bpm']:.0f} bpm</div><div class="sub">{hr['classification']}
      (60–100 bpm)</div></div>
    <div class="tile"><div class="lbl">R-peaks detected</div>
      <div class="val">{hr['n_peaks']}</div><div class="sub">{S['rpeaks_first10s']} in the
      first 10 s</div></div>
    <div class="tile"><div class="lbl">Ectopic (PVC-like) beats</div>
      <div class="val">{hr['n_ectopic']}</div><div class="sub">isolated, with compensatory
      pause</div></div>
    <div class="tile"><div class="lbl">SVM test accuracy</div>
      <div class="val">{100 * ml['Accuracy']:.0f} %</div><div class="sub">{tp + tn} / {tp + tn + fp + fn}
      test beats correct</div></div>
  </div>
  <h3>Results at a glance</h3>
  <table class="data glance">
    <thead><tr><th>Step</th><th>Task</th><th>Result</th></tr></thead><tbody>
    <tr><td>1</td><td>Data acquisition (first 10 s)</td><td>Record 0001, channel 2, 2 500 samples;
      files verified against PhysioNet SHA-256 sums</td></tr>
    <tr><td>2</td><td>FIR band-stop filtering</td><td>{flt['numtaps']}-tap Kaiser FIR,
      48–52 Hz; 49–51 Hz power −{flt['Power reduction 49-51 Hz (dB)']:.0f} dB; visible change
      negligible (little 50-Hz interference)</td></tr>
    <tr><td>3</td><td>R-peak detection (first 10 s)</td><td>{S['rpeaks_first10s']} R-peaks
      (Pan–Tompkins); beat 5 is ectopic</td></tr>
    <tr><td>4</td><td>Heart rhythm (1 min)</td><td><b>{hr['n_peaks']} R-peaks → Heart Rate =
      {hr['hr_count_bpm']:.0f} bpm → {hr['classification']}</b> ({hr['hr_mean_rr_bpm']} bpm from mean RR)</td></tr>
    <tr><td>5</td><td>Feature extraction (6 × 10 s)</td><td>Mean RR {segd['RR mean (s)'].min():.3f}–{segd['RR mean (s)'].max():.3f} s;
      mean QRS {segd['QRS mean (s)'].min():.3f}–{segd['QRS mean (s)'].max():.3f} s</td></tr>
    <tr><td>6</td><td>ML classification (SVM)</td><td>Confusion matrix [[{tn}, {fp}], [{fn}, {tp}]];
      accuracy, sensitivity, specificity, precision = 100 %</td></tr>
  </tbody></table>
</section>
""")

TOC = [("s1", "Case context, data and working environment"),
       ("s2", "Step 1 — Data acquisition and visualization"),
       ("s3", "Step 2 — ECG filtering (FIR band-stop)"),
       ("s4", "Step 3 — R-peak detection"),
       ("s5", "Step 4 — Heart-rhythm analysis"),
       ("s6", "Step 5 — ECG feature extraction"),
       ("s7", "Step 6 — Machine-learning classification"),
       ("s8", "Interpretation, conclusion and limitations"),
       ("a1", "Appendix A — Beat-level feature table"),
       ("a2", "Appendix B — Machine-learning data set"),
       ("a3", "Appendix C — Reproducibility"),
       ("a4", "References")]
toc_rows = "".join(f'<tr><td><a href="#{a}">{esc(t)}</a></td><td class="pg">@@{a}@@</td></tr>'
                   for a, t in TOC)
H.append(f"""
<section class="page">
<h2>Contents</h2>
<table class="toc"><tbody>{toc_rows}</tbody></table>
<p class="muted">Each step contains (a) the method, (b) the code used, (c) screenshots of the code
and its output taken in JupyterLab, and (d) the plots, tables and interpretation. Every screenshot
shows the JupyterLab menu bar, the notebook tab <code>{NB_NAME}</code>, the toolbar with the active
kernel and the status bar. All numbers in this report were produced by the notebook and read from its
saved result files.</p>
</section>
<section class="page">
<h2 id="s1">Case context, data and working environment</h2>
<p>As part of a recruitment medical examination, a candidate's ECG was recorded for one minute.
The signal is taken from the <b>PhysioNet AHA Database (ahadb v1.0.0), record 0001</b>. The record
contains two ECG channels sampled at <b>250 Hz</b> (12-bit, 400 adu/mV, 3 h long); as required, the
<b>second channel</b> is analysed, using the first 10 s (Steps 1–3) and the first 60 s (Steps 4–5).
The original PhysioNet files (<code>0001.hea</code>, <code>0001.dat</code>, <code>0001.atr</code>)
were obtained from PhysioNet's official open-data mirror and all three match PhysioNet's published
SHA-256 checksums (Step 1.1; see its output in the Step 1 screenshot).</p>
<p>All processing was done in a Jupyter notebook running in <b>JupyterLab</b>. Screenshot 1 shows
the working environment: the project folder in the file browser, the notebook header, and the first
cell, which prints the student identity, run time, Python version, working folder and package versions.</p>
{shot("ss00_overview_environment.jpg", "JupyterLab working environment: project folder, notebook "
      f"<code>{NB_NAME}</code> and the identity / environment cell (Step 0.1).")}
{code("0.1")}
{code("0.2")}
{shots_for("0.", caption="Notebook cells {cells}: identity, environment, imports, constants and plotting style.")}
</section>
""")

H.append(f"""
<section class="page">
<h2 id="s2">Step 1 — Data acquisition and visualization (first 10 s)</h2>
<p><b>Method.</b> The record is read with <code>wfdb.rdrecord</code> (first 60 s only), the second
signal column (<code>p_signal[:, 1]</code>, in mV) is selected, and a time axis
<i>t = n / f<sub>s</sub></i> with f<sub>s</sub> = 250 Hz is built. The first 10 s (2 500 samples) are
plotted as amplitude versus time.</p>
{code("1.1")}{code("1.2")}{code("1.3")}
{shots_for("1.", caption="Step 1 in JupyterLab — notebook cells {cells} with their outputs.")}
{fig("fig01_raw_ecg_10s.png", "Raw ECG, AHA DB record 0001, channel 2, first 10 s (fs = 250 Hz).")}
<p><b>Observation.</b> Thirteen heartbeats are visible. In this lead the normal QRS complex is an
<i>rS</i> pattern: a small positive R wave (≈ 0.4–0.55 mV) followed by a deep S wave (≈ −1 mV). At
≈ 3.2 s there is one much larger (≈ 3 mV), wider complex that occurs early and is followed by a long
pause — an ectopic (PVC-like) beat, analysed in Steps 3–6. Small high-frequency ripples and a slow
baseline drift are also visible.</p>
</section>
""")

H.append(f"""
<section class="page">
<h2 id="s3">Step 2 — ECG filtering with an FIR band-stop filter (first 10 s)</h2>
<p><b>Design.</b> Power-line interference is a narrow-band sinusoid, so a band-stop FIR filter centred
on 50 Hz is used. It is designed with the window method (Kaiser window; order chosen by
<code>scipy.signal.kaiserord</code> for ≥ 40 dB attenuation with a 2-Hz transition band) and applied
forward–backward with <code>filtfilt</code>. A symmetric FIR filter has linear phase; forward–backward
filtering cancels its 0.56-s group delay, so the P-QRS-T waves are not shifted.</p>
{table(pd.DataFrame([
    ["Filter type", "FIR band-stop (window method, Kaiser window)"],
    ["Cut-off frequencies", "48 Hz and 52 Hz (centre 50 Hz)"],
    ["Number of taps (order)", f"{flt['numtaps']} taps (order {flt['numtaps'] - 1}), Kaiser β = {flt['beta']}"],
    ["Attenuation at 50 Hz", f"{flt['atten_50hz_db_single_pass']} dB per pass, "
                             f"{2 * flt['atten_50hz_db_single_pass']:.1f} dB with filtfilt"],
    ["Pass-band ripple (≤ 46 Hz, ≥ 54 Hz)", "< 0.07 dB"],
    ["Phase", "zero phase (forward–backward filtering)"]], columns=["Parameter", "Value"]),
    "FIR band-stop filter specification.", cls="data kv")}
{code("2.1")}{code("2.2")}{code("2.3")}
{shots_for("2.", caption="Step 2 in JupyterLab — notebook cells {cells} with their outputs.")}
{fig("fig02_fir_bandstop_response.png", "Magnitude response of the 281-tap FIR band-stop filter "
     "(single pass); grey band = stop band 48–52 Hz.")}
{fig("fig03_raw_vs_filtered_10s.png", "Comparison of (a) the raw ECG and (b) the filtered ECG over the "
     "first 10 s; (c) the component removed by the filter (raw − filtered), in µV.")}
{fig("fig03b_raw_vs_filtered_zoom.png", "One heartbeat (0.9–1.6 s): raw and filtered ECG overlaid — "
     "the two traces are practically identical.")}
{fig("fig04_psd_raw_vs_filtered.png", "Power spectral density (Welch) of the raw and filtered ECG: "
     "the filter cuts a deep notch at 48–52 Hz and leaves the rest of the spectrum unchanged.")}
{kv_table(fstats, "Quantitative effect of the filter (first 10 s).",
          fmt=lambda k, v: f"{v:.3g}" if "filtered (µV²)" in k else f"{v:.4f}" if "Correlation" in k else f"{v:.2f}")}
<div class="answer"><b>Does the filtering visibly improve the signal quality?</b> Only marginally,
because this recording contains very little 50-Hz interference: the 50-Hz peak in the raw spectrum is
only ≈ {fstats.iloc[0]:.1f} dB above the neighbouring noise floor. The filter works as designed — the
power in 49–51 Hz falls by ≈ {fstats.iloc[3]:.0f} dB — but the removed component is tiny (RMS ≈
{fstats.iloc[5]:.1f} µV, {fstats.iloc[6]:.1f} % of the ECG RMS), so the raw and filtered traces are
almost identical (correlation {fstats.iloc[7]:.4f}) and the P-QRS-T morphology is preserved without any
time shift. The disturbances that are <i>visible</i> — slow baseline wander and broadband muscle noise —
lie outside the 48–52 Hz stop band and would need a high-pass (≈ 0.5 Hz) and low-pass (≈ 40 Hz) filter.
(The AHA recordings come from the USA, where mains is 60 Hz; setting <code>F0 = 60</code> re-targets
the same design, and the spectrum shows no dominant 60-Hz line either.)</div>
</section>
""")

H.append(f"""
<section class="page">
<h2 id="s4">Step 3 — R-peak detection (first 10 s)</h2>
<p><b>Method — Pan–Tompkins algorithm</b> on the band-stop-filtered ECG: (1) band-pass 5–15 Hz
(2nd-order Butterworth, zero phase) to keep the QRS energy; (2) five-point derivative to emphasise
the steep QRS slopes; (3) squaring; (4) 150-ms moving-window integration (MWI); (5) adaptive
thresholds on the MWI peaks (signal and noise levels updated after every peak, 200-ms refractory
period, T-wave rejection within 360 ms, search-back when a beat seems missed). Each R-peak is then
located as the maximum of the filtered ECG near the detected QRS.</p>
<p>Two simple plausibility checks make the detector robust to a muscle-noise burst later in the
recording (Step 4): (a) a detection that splits one normal RR interval into two is removed (the one
that least resembles the median beat), and (b) a beat that matches the median beat template poorly is
moved to the best match within ±150 ms. Each beat is labelled <i>Normal</i>, <i>Ectopic (PVC-like)</i>
— premature (RR &lt; 80 % of the median), followed by a compensatory pause (&gt; 120 %), abnormal
shape (template correlation &lt; 0.5) and &gt; 2× larger — or <i>Low-confidence (noise)</i>.</p>
{code("3.1")}{code("3.2")}{code("3.3")}{code("3.4")}
{shots_for("3.", "## Step 3", caption="Step 3 in JupyterLab — notebook cells {cells}.")}
{fig("fig05_pan_tompkins_stages_10s.png", "Pan–Tompkins intermediate signals for the first 10 s: "
     "band-pass, derivative, squared signal and moving-window integration with the adaptive threshold.")}
{fig("fig06_rpeaks_10s.png", f"Filtered ECG with the {S['rpeaks_first10s']} detected R-peaks "
     "(numbered); the diamond marks the ectopic (PVC-like) beat.")}
{table(r10, f"R-peaks detected in the first 10 s ({S['rpeaks_first10s']} beats). Template correlation = "
       "similarity with the median beat (NaN for the last beat, whose window extends past 10 s).",
       fmts={"t_R (s)": "{:.3f}", "R amplitude (mV)": "{:.3f}", "template corr": "{:.2f}"})}
<p><b>Result.</b> {S['rpeaks_first10s']} R-peaks were detected in the first 10 s; every visible QRS
complex is marked and there are no false detections. Beat 5 (t = 3.172 s) is the ectopic beat — its R
amplitude is ≈ 3 mV and its shape is unlike the other beats (correlation −0.17).</p>
</section>
""")

H.append(f"""
<section class="page">
<h2 id="s5">Step 4 — Heart-rhythm analysis (full 1-minute recording)</h2>
<p><b>Method.</b> The same filter and detector are applied to the complete 60-s signal. The heart
rate is computed from the R-peak count (HR = number of R-peaks / recording time in minutes) and,
as a check, from the mean RR interval (HR = 60 / mean RR). Classification: &lt; 60 bpm
Bradycardia, 60–100 bpm Normal, &gt; 100 bpm Tachycardia.</p>
{code("4.1")}{code("4.2")}{code("4.3")}{code("4.4")}
{shots_for("4.", caption="Step 4 in JupyterLab — notebook cells {cells}.")}
{fig("fig07_rpeaks_full_minute.png", f"The full minute of filtered ECG (one row per 10 s) with all "
     f"{hr['n_peaks']} detected R-peaks: filled circles = normal beats, diamonds = ectopic beats, "
     "ring = low-confidence beat inside the muscle-noise burst (30–36 s).")}
{fig("fig08_instantaneous_hr.png", "Instantaneous heart rate (60 / RR) over the minute; shaded band = "
     "normal range 60–100 bpm. The spikes and dips are the three ectopic beats and their compensatory "
     "pauses; the blip at ≈ 35 s comes from the uncertain beat position inside the noise burst.")}
{kv_table(hrs, "Heart-rate analysis of the 1-minute recording.",
          fmt=lambda k, v: v if not str(v).replace('.', '', 1).isdigit() else (f"{float(v):.1f}" if "bpm" in k else f"{float(v):.0f}"))}
<div class="answer"><b>Result.</b> Total number of detected R-peaks = <b>{hr['n_peaks']}</b> in 60 s →
<b>Heart Rate = {hr['hr_count_bpm']:.0f} bpm → {hr['classification']}</b> (60–100 bpm). The mean-RR
method gives {hr['hr_mean_rr_bpm']} bpm, the same class. The rhythm is regular except for three
isolated ectopic (PVC-like) beats at 3.2, 45.3 and 57.1 s, which do not change the classification.</div>
<p><b>Independent check.</b> The first minute of record 0001 has no reference beat annotations (the
annotations start at 2.5 h), so the channel-2 detections were compared with an independent detector
(wfdb XQRS) run on channel 1, which is cleaner in this record. Channel 1 gives the same total of 76 beats,
and 75 of the 76 channel-2 beats lie within 100 ms of a channel-1 beat. The only unmatched beat is the
low-confidence beat at 35.0 s inside the muscle-noise burst. Its presence is confirmed, so the beat
count is correct, but its exact position on channel 2 is uncertain. That beat is therefore excluded
from the feature extraction.</p>
</section>
""")

H.append(f"""
<section class="page">
<h2 id="s6">Step 5 — ECG feature extraction (every 10 s)</h2>
<p><b>Definitions used.</b></p>
<ul>
<li><b>RR interval</b> of beat <i>i</i>: RR<sub>i</sub> = t(R<sub>i</sub>) − t(R<sub>i−1</sub>), i.e. the
interval that ends at that beat (the same set of intervals as t(R<sub>i+1</sub>) − t(R<sub>i</sub>)).
A beat belongs to the 10-s segment that contains its R-peak.</li>
<li><b>Q point</b> = minimum of the filtered ECG within 40 ms before R (a Q wave lasts &lt; 40 ms);
<b>S point</b> = minimum within 80 ms after R; <b>QRS duration = t<sub>S</sub> − t<sub>Q</sub></b>.</li>
<li><b>Reported value per segment = the MEAN</b> of the individual beats in the segment (SD and
median are also given). The RR mean uses only normal-to-normal (NN) intervals and the QRS mean uses
only normal beats. The ectopic beats, the compensatory-pause intervals that follow them and the
low-confidence beat are excluded because they do not describe the normal rhythm (see the table of
excluded beats below). All individual beat values are listed in Appendix A.</li>
</ul>
{code("5.1")}{code("5.2")}{code("5.3")}{code("5.4")}
{shots_for("5.", "## Step 5", caption="Step 5 in JupyterLab — notebook cells {cells}.")}
{table(seg, "Extracted ECG features per 10-s segment (assignment format). Values are the MEAN ± SD "
       "of the normal beats in each segment.", cls="data main")}
{table(segd, "Segment features in detail: number of beats, mean, SD and median of the RR interval "
       "(NN beats) and of the QRS duration (normal beats).",
       fmts={c: "{:.4f}" for c in segd.columns if "(s)" in c} | {c: "{:.0f}" for c in segd.columns
             if c in ("beats detected", "excluded beats", "RR n", "QRS n")})}
{table(excluded[["beat", "segment", "t_R (s)", "RR (s)", "QRS (s)", "beat type", "reason"]],
       f"The {len(excluded)} beats whose RR interval is NOT used for the segment means or for the Normal "
       "class in Step 6, and why (their QRS is still used if the beat itself is normal).",
       fmts={"t_R (s)": "{:.3f}", "RR (s)": "{:.3f}", "QRS (s)": "{:.3f}"})}
{fig("fig09_qrs_fiducials.png", "Q, R and S points on two normal beats and on the ectopic beat; the "
     "shaded span is the QRS duration t<sub>S</sub> − t<sub>Q</sub>.")}
{fig("fig10_segment_features.png", "Individual RR intervals (NN beats) and QRS durations (normal beats) "
     "in each segment (dots) with the segment mean ± SD.")}
<p><b>Observation.</b> The features are stable across the six segments: mean RR
{segd['RR mean (s)'].min():.3f}–{segd['RR mean (s)'].max():.3f} s (≈ 75–77 bpm) and mean QRS
{segd['QRS mean (s)'].min():.3f}–{segd['QRS mean (s)'].max():.3f} s, far below the 0.12-s limit of a
wide QRS. For the ectopic beats the QRS is 0.108–0.112 s and the RR only 0.44–0.46 s.</p>
</section>
""")

dsum = dataset.groupby("label")[["RR (s)", "QRS (s)"]].agg(["count", "mean", "std", "min", "max"])
dsum.columns = [f"{a} {b}" for a, b in dsum.columns]
dsum = dsum.reset_index().rename(columns={"label": "class"})
split = dataset.groupby(["split", "label"]).size().unstack(fill_value=0).reset_index()
H.append(f"""
<section class="page">
<h2 id="s7">Step 6 — Machine-learning classification</h2>
<p><b>A. Normal-class data.</b> The RR interval and QRS duration of the {ml['n_normal']} individual
NN beats from Step 5 are labelled <i>Normal</i>. Individual beat values are used rather than only the
six segment means because the arrhythmia samples are also individual beats: both classes are then
described in the same unit (one sample = one beat), and the classes are reasonably balanced
({ml['n_normal']} vs {ml['n_arrhythmia']} instead of 6 vs 50). The six-segment-mean version is
tested as a robustness check (Step 6.6).<br>
<b>B. Arrhythmia-class data.</b> The {ml['n_arrhythmia']} beats from the assignment table
(<code>data/arrhythmia_training_data.csv</code>), labelled <i>Arrhythmia</i>.<br>
<b>C. Model.</b> X = [RR interval, QRS duration], y = [Normal, Arrhythmia]. A Support Vector Machine
with an RBF kernel (C = 1, γ = 'scale') is trained after feature standardisation, using a stratified
70 % / 30 % train/test split (random_state = 42). An RBF kernel is chosen because arrhythmic beats lie on
<i>both</i> sides of the normal RR range (too short and too long). A single straight boundary in RR
cannot separate them; a curved boundary can.</p>
{code("6.1")}{code("6.2")}{code("6.3")}{code("6.4")}{code("6.5")}{code("6.6")}{code("7")}
{shots_for("6.", "7", "## Step 6", caption="Step 6 in JupyterLab — notebook cells {cells}.")}
{table(dsum, "Combined data set — summary per class.",
       fmts={c: "{:.3f}" for c in dsum.columns if c != "class" and "count" not in c})}
{fig("fig11_dataset_scatter.png", "The combined data set in feature space: X = [RR interval, QRS duration].", "86%")}
{table(split, "Stratified train/test split (70 % / 30 %).")}
{table(pred, f"Predicted class of the {len(pred)} test samples (decision score &lt; 0 → Normal, "
       "&gt; 0 → Arrhythmia).", fmts={"RR (s)": "{:.4f}", "QRS (s)": "{:.4f}", "decision score": "{:.2f}"},
       cls="data compact")}
{fig("fig12_confusion_matrix.png", "Confusion matrix of the test set (positive class = Arrhythmia).", "46%")}
{kv_table(metrics, "Model performance on the test set (positive class = Arrhythmia) and 5-fold "
          "cross-validation on the whole data set.",
          fmt=lambda k, v: f"{v:.3f}" if "SD" in k else pct(v))}
<p class="formula">Accuracy = (TP + TN) / N = ({tp} + {tn}) / {tp + tn + fp + fn} · Sensitivity = TP / (TP + FN) =
{tp} / {tp + fn} · Specificity = TN / (TN + FP) = {tn} / {tn + fp} · Precision = TP / (TP + FP) = {tp} / {tp + fp}</p>
{fig("fig13_decision_boundary.png", "SVM decision regions (blue = Normal, orange = Arrhythmia) with the "
     "training (filled) and test (open) samples; green diamonds = the candidate's three ectopic beats, "
     "which were not used for training or testing.", "90%")}
{table(excl, "Model applied to beats that were excluded from the data set: the three ectopic beats and "
       "the sinus beats that follow their compensatory pauses.",
       fmts={"t_R (s)": "{:.3f}", "RR (s)": "{:.3f}", "QRS (s)": "{:.3f}", "decision score": "{:.2f}"})}
<p><b>Robustness check.</b> Using only the six segment means as the Normal class (6 Normal + 50
Arrhythmia samples) also gives a test accuracy of {100 * ml['segment_mean_check_accuracy']:.0f} %.</p>
<div class="answer"><b>Interpretation of the model performance.</b> The SVM classified all
{len(pred)} test beats correctly: TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}. This gives accuracy,
sensitivity (recall), specificity and precision of 100 %, and 5-fold cross-validation agrees (100 %).
The result is expected because the two classes do not overlap in feature space. Every arrhythmia
sample has a wide QRS (≥ 0.120 s, normal ≤ 0.080 s), and its RR interval is either short (&lt; 0.59 s)
or long (&gt; 1.01 s), while the normal RR range is 0.744–0.844 s. The SVM learned a closed region
around the normal beats. It also labelled the candidate's three real ectopic beats as
<i>Arrhythmia</i>, which is physiologically consistent, although it never saw them.
<br>The perfect score must be read with care:
(1) the data set is small and perfectly separable;
(2) all normal beats come from one person in one minute, and neighbouring beats are correlated, so a
random beat-level split is optimistic;
(3) the model has never seen beats with a normal QRS but an unusual RR, such as the sinus beats after
the compensatory pauses, which lie on the decision boundary (scores between −0.1 and +0.25) and are
classified inconsistently;
(4) the arrhythmia values may have been measured with a different QRS method.
A real screening model would need many patients, beat types and inter-patient validation.</div>
</section>
""")

H.append(f"""
<section class="page">
<h2 id="s8">Interpretation, conclusion and limitations</h2>
<ul class="concl">
<li><b>Filtering.</b> The {flt['numtaps']}-tap FIR band-stop filter (48–52 Hz, zero phase) removes the
50-Hz band (≈ −{fstats.iloc[3]:.0f} dB) without distorting the ECG. The visible improvement is negligible
because the recording contains very little power-line interference.</li>
<li><b>R-peaks.</b> Pan–Tompkins with two plausibility checks detected {S['rpeaks_first10s']} R-peaks in the
first 10 s and {hr['n_peaks']} in the full minute. 75 of the 76 beats agree with an independent
detector on channel 1, and the remaining beat lies in a muscle-noise burst.</li>
<li><b>Heart rhythm.</b> Heart Rate = {hr['hr_count_bpm']:.0f} bpm ({hr['hr_mean_rr_bpm']} bpm from the mean
RR) → <b>{hr['classification']}</b>. The rhythm is regular apart from three isolated premature, wide,
high-amplitude beats with compensatory pauses (premature ventricular complexes). This is a
signal-processing exercise, not a medical diagnosis.</li>
<li><b>Features.</b> Segment means: RR {segd['RR mean (s)'].min():.3f}–{segd['RR mean (s)'].max():.3f} s,
QRS {segd['QRS mean (s)'].min():.3f}–{segd['QRS mean (s)'].max():.3f} s. The features are stable and
typical of a normal rhythm with narrow QRS complexes.</li>
<li><b>Machine learning.</b> The SVM (RBF) reaches 100 % accuracy, sensitivity, specificity and precision
on the test set, because the two classes are clearly separated by QRS duration and RR interval. The
estimate is optimistic: the data set is small and comes from one subject, and the model is uncertain for
normal-QRS beats with unusual RR intervals.</li>
</ul>
{shots_for("## Interpretation", caption="The conclusion cell in JupyterLab.")}
</section>
""")

nbt = beats.copy()
H.append(f"""
<section class="page appendix">
<h2 id="a1">Appendix A — Beat-level feature table (all {len(beats)} beats, 1 minute)</h2>
{table(nbt, "All detected beats with Q, R, S times, RR interval (ending at the beat), QRS duration, "
       "beat type and whether the beat is a normal-to-normal (NN) beat used in Steps 5–6.",
       fmts={c: "{:.3f}" for c in nbt.columns if "(s)" in c}, cls="data compact")}
<h2 id="a2">Appendix B — Machine-learning data set ({len(dataset)} samples)</h2>
<div class="cap"><b>{nxt("Table")}.</b> Combined data set: N01–N{ml['n_normal']:02d} = Normal beats
from Step 5; A01–A50 = Arrhythmia beats from the assignment table.</div>
<div class="twocol">{halves(dataset[["sample", "RR (s)", "QRS (s)", "label", "split"]])}</div>
<h2 id="a3">Appendix C — Reproducibility</h2>
<p>Environment printed by the notebook (Step 0.1):</p>
<pre class="env">{esc(ENV)}</pre>
<table class="data kv"><tbody>
<tr><td>Notebook</td><td><code>{NB_NAME}</code> (executed top-to-bottom in JupyterLab, execution counts 1–27)</td></tr>
<tr><td>Data</td><td><code>data/ahadb/0001.*</code> (PhysioNet ahadb v1.0.0, SHA-256 verified);
<code>data/arrhythmia_training_data.csv</code></td></tr>
<tr><td>Outputs</td><td><code>figures/</code> (plots, 200 dpi), <code>results/</code> (CSV/JSON tables),
<code>screenshots/</code> (JupyterLab screenshots)</td></tr>
<tr><td>Re-run</td><td><code>pip install -r requirements.txt</code>, then <code>jupyter lab</code> → Run ▸ Run All Cells</td></tr>
</tbody></table>
<h2 id="a4">References</h2>
<ol class="refs">
<li>Goldberger A. L. et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource
for complex physiologic signals. <i>Circulation</i> 101(23):e215–e220, 2000.</li>
<li>PhysioNet AHA Database (ahadb), version 1.0.0. https://physionet.org/content/ahadb/1.0.0/</li>
<li>Pan J., Tompkins W. J. A real-time QRS detection algorithm. <i>IEEE Trans. Biomed. Eng.</i>
32(3):230–236, 1985.</li>
<li>Virtanen P. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python.
<i>Nature Methods</i> 17:261–272, 2020.</li>
<li>Pedregosa F. et al. Scikit-learn: machine learning in Python. <i>J. Mach. Learn. Res.</i>
12:2825–2830, 2011.</li>
</ol>
</section>
""")

CSS = """
@page { size: A4; margin: 17mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body { font-family: Inter, Arial, sans-serif; font-size: 9.4pt; line-height: 1.45; color: #0b0b0b;
       margin: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
h1, h2, h3 { font-weight: 650; color: #0b0b0b; break-after: avoid; }
h2 { font-size: 14pt; margin: 0 0 3mm; padding-bottom: 1.5mm; border-bottom: 1.5px solid #2a78d6; }
h3 { font-size: 11pt; margin: 6mm 0 2mm; }
p { margin: 0 0 2.6mm; } ul { margin: 0 0 3mm; padding-left: 5mm; } li { margin-bottom: 1.2mm; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 0.86em; background: #f3f3f0;
       padding: 0 1mm; border-radius: 2px; }
section.page { break-before: page; }
.muted { color: #52514e; }
.cover { padding-top: 14mm; }
.kicker { font-size: 10pt; font-weight: 600; color: #2a78d6; letter-spacing: .02em; }
h1.title { font-size: 24pt; line-height: 1.15; margin: 3mm 0 3mm; }
.subtitle { font-size: 11.5pt; color: #52514e; margin-bottom: 8mm; }
table.meta { border-collapse: collapse; margin-bottom: 8mm; font-size: 9.6pt; }
table.meta th { text-align: left; color: #52514e; font-weight: 500; padding: 1.4mm 6mm 1.4mm 0; }
table.meta td { padding: 1.4mm 0; }
.tiles { display: flex; gap: 3.5mm; margin-bottom: 8mm; }
.tile { flex: 1; border: 1px solid #e1e0d9; border-radius: 6px; padding: 3.5mm 4mm; background: #fcfcfb; }
.tile .lbl { font-size: 8.4pt; color: #52514e; }
.tile .val { font-size: 21pt; font-weight: 650; margin: 1mm 0 .5mm; }
.tile .sub { font-size: 8pt; color: #52514e; }
.cap { font-size: 8.4pt; color: #52514e; margin: 0 0 1.2mm; }
.listing { margin: 2.5mm 0 3.5mm; }
.listing .cap { font-weight: 600; color: #0b0b0b; }
.hl { background: #f7f7f5; border: 1px solid #e1e0d9; border-radius: 4px; padding: 2.2mm 3mm; }
.hl pre { margin: 0; font-family: 'DejaVu Sans Mono', monospace; font-size: 7.15pt; line-height: 1.36;
          white-space: pre; }
.hl .linenos { color: #a8a7a0; padding-right: 2mm; }
figure { margin: 3mm 0 5mm; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 236mm; }
figure.shot img { border: 1px solid #c3c2b7; box-shadow: 0 0.5mm 1.5mm rgba(0,0,0,.08); }
figcaption { font-size: 8.3pt; color: #52514e; margin-top: 1.6mm; text-align: left; }
figcaption .src { color: #898781; font-size: 7.5pt; }
.tbl { margin: 3mm 0 5mm; break-inside: avoid; }
.tbl .note { font-size: 8pt; color: #52514e; margin-top: 1mm; }
table.data { border-collapse: collapse; width: 100%; font-size: 8.4pt; font-variant-numeric: tabular-nums; }
table.data th { text-align: left; font-weight: 600; background: #f0efec; padding: 1.4mm 2mm;
                border-bottom: 1px solid #c3c2b7; }
table.data td { padding: 1.1mm 2mm; border-bottom: 1px solid #e9e8e3; vertical-align: top; }
table.data.compact { font-size: 7.6pt; } table.data.compact td { padding: .6mm 1.6mm; }
table.data.compact th { padding: 1mm 1.6mm; }
table.data.kv td:first-child { color: #52514e; width: 46%; }
table.data.main { font-size: 10pt; } table.data.main td, table.data.main th { padding: 1.8mm 3mm; }
.tbl:has(table.compact) { break-inside: auto; }
table.glance td:first-child { width: 8mm; } table.glance td:nth-child(2) { width: 52mm; color: #52514e; }
.answer { border-left: 3px solid #2a78d6; background: #f4f8fd; padding: 3mm 4mm; margin: 3mm 0 4mm;
          break-inside: avoid; }
.formula { font-size: 8.6pt; color: #52514e; }
ol.toc li { margin-bottom: 1.6mm; font-size: 10.5pt; }
.twocol { display: flex; gap: 6mm; align-items: flex-start; }\ntable.toc { width: 100%; border-collapse: collapse; margin: 2mm 0 6mm; font-size: 10.5pt; }\ntable.toc td { padding: 1.8mm 0; border-bottom: 1px dotted #c3c2b7; }\ntable.toc td.pg { text-align: right; width: 14mm; color: #52514e; }\ntable.toc a { color: #0b0b0b; text-decoration: none; }
pre.env { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.6pt; background: #f7f7f5;
          border: 1px solid #e1e0d9; padding: 2.5mm 3mm; border-radius: 4px; }
ol.refs li { font-size: 8.8pt; }
"""
pyg_css = HtmlFormatter(style="default").get_style_defs(".hl")
doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>ECG Mini Project — {STUDENT}</title>"
       f"<style>{CSS}\n{pyg_css}</style></head><body>{''.join(H)}</body></html>")
REPORT.mkdir(exist_ok=True)

hdr = ("<div style='font-family:Inter,Arial,sans-serif;font-size:7px;color:#898781;width:100%;"
       "padding:0 15mm;display:flex;justify-content:space-between'>"
       f"<span>FACE IT 2026 · Hands-on Mini Project · ECG</span><span>{STUDENT}</span></div>")
ftr = ("<div style='font-family:Inter,Arial,sans-serif;font-size:7px;color:#898781;width:100%;"
       "padding:0 15mm;display:flex;justify-content:space-between'>"
       "<span>AHA DB record 0001 · channel 2 · fs = 250 Hz</span>"
       "<span>Page <span class='pageNumber'></span> / <span class='totalPages'></span></span></div>")
def render(html_doc):
    (REPORT / "report.html").write_text(html_doc, encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROMIUM)
        page = browser.new_page()
        page.goto((REPORT / "report.html").as_uri())
        page.wait_for_load_state("networkidle")
        page.pdf(path=str(PDF), format="A4", print_background=True, display_header_footer=True,
                 header_template=hdr, footer_template=ftr,
                 margin={"top": "17mm", "bottom": "16mm", "left": "15mm", "right": "15mm"})
        browser.close()


render(doc)                                    # pass 1: find on which page each section starts
from pypdf import PdfReader
import re
norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
texts = [norm(pg.extract_text() or "") for pg in PdfReader(str(PDF)).pages]
for anchor, title in TOC:
    key = norm(title.split(" (")[0])[:32]
    pageno = next((i + 1 for i, t in enumerate(texts) if i >= 2 and key in t), None)
    doc = doc.replace(f"@@{anchor}@@", str(pageno) if pageno else "")
render(doc)                                    # pass 2: contents with page numbers
print(f"Report written: {PDF}  ({PDF.stat().st_size / 1e6:.1f} MB)")
