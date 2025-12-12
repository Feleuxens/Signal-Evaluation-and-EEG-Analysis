from mne_bids import BIDSPath, read_raw_bids
from matplotlib import pyplot as plt
import mne
import numpy as np
import annotations
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from ica import run_ica_and_interpolate
import asrpy
import plots

# --- Constants ---
tmin, tmax = -0.5, 1.0  # epoch time range in seconds
baseline = (-0.25, 0.0)  # baseline correction period
l_freq, h_freq = 0.1, 40.0  # highpass, lowpass filter
notch_freqs = [50, 100, 150, 200]  # notch filter frequencies
bad_channel_z_thresh = 3.0  # z-score threshold for bad channel detection
# --- Artifact correction settings ---
ASR_CUTOFF = 10  # ASR cutoff (lower = more aggressive, 5-20 typical)
ICA_N_COMPONENTS = 0.99  # Variance explained or number of components


def analyze_subject(subject_id, bids_root="../data/", use_ica=True, use_asr=False):

    bids_path = BIDSPath(
        subject=subject_id,
        root=bids_root,
        datatype="eeg",
        suffix="eeg",
        task="jacobsen",
    )

    # --- Load the data ---
    raw = read_raw_bids(bids_path)
    raw.load_data()
    raw_unprocessed = raw.copy()  # keep a copy of unprocessed data

    data = raw.get_data(picks="eeg")
    channel_names = raw.ch_names

    # --- Automatic detection of bad channels ---
    # 1) Variance z-score across channels
    variances = np.var(data, axis=1)
    z_var = (variances - variances.mean()) / variances.std()
    bad_channels_var = [
        channel
        for channel, z in zip(channel_names, z_var)
        if np.abs(z) > bad_channel_z_thresh
    ]

    # 2) Low correlation channels (flat or noisy channels)
    corr_matrix = np.corrcoef(data)
    mean_corr = np.median(corr_matrix, axis=0)
    z_corr = (mean_corr - mean_corr.mean()) / mean_corr.std()
    bad_channels_corr = [
        channel
        for channel, z in zip(channel_names, z_corr)
        if z < -bad_channel_z_thresh
    ]

    automatic_bad_channels = sorted(set(bad_channels_var + bad_channels_corr))

    raw.info["bads"].extend(automatic_bad_channels)

    print(
        f"Automatically detected bad channels for subject {subject_id}: {raw.info['bads']}"
    )

    # --- Filtering ---
    # 1) Notch filter to remove line noise
    raw.notch_filter(notch_freqs, picks="eeg", method="spectrum_fit")

    # 2) Bandpass filter
    raw.filter(l_freq, h_freq, picks="eeg", fir_design="firwin")

    # --- Artifact correction ---
    # ASR first (handles transient artifacts)
    if use_asr:
        try:
            print(f"Running ASR with cutoff={ASR_CUTOFF}...")
            asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=ASR_CUTOFF)
            asr.fit(raw)
            raw = asr.transform(raw)
            print("ASR complete.")
        except Exception as e:
            print(f"ASR failed: {e}. Continuing without ASR.")

    # ICA second (handles stereotyped artifacts like blinks)
    ica = None
    if use_ica:
        try:
            raw, ica = run_ica_and_interpolate(
                raw,
                n_components=0.99,
                method="fastica",
                random_state=42,
                reject_ecg=True,
            )
            print(
                "ICA cleaning applied. Excluded components:",
                getattr(ica, "exclude", []),
            )
        except Exception as e:
            print(f"ICA failed: {e}. Continuing without ICA.")

    # # Fallback: robust ICA-based artefact rejection
    # ica = ICA(n_components=0.99, method="fastica", random_state=42, max_iter="auto")
    # ica.fit(raw)
    # # Find EOG artifacts and exclude
    # eog_epochs = create_eog_epochs(raw, reject_by_annotation=True)
    # eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
    # ica.exclude.extend(eog_inds)
    # # Optionally find ECG components
    # try:
    #     ecg_epochs = create_ecg_epochs(raw)
    #     ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs)
    #     ica.exclude.extend(ecg_inds)
    # except Exception:
    #     pass
    # ica.apply(raw)

    # TODO: implement

    # --- Interpolate bad channels ---
    if raw.info["bads"]:
        try:
            # Ensure montage is set for interpolation
            if not raw.info.get("dig"):
                raw.set_montage("standard_1020", on_missing="ignore")
            raw.interpolate_bads(reset_bads=False, mode="accurate")
            print(f"Interpolated bad channels: {raw.info['bads']}")
        except Exception as e:
            print(f"Interpolation failed: {e}")

    # --- Interpolate previously marked bad channels ---
    # raw.interpolate_bads(reset_bads=False)
    # TODO: doesnt work
    # File "~/.local/share/virtualenvs/eeg-Bt72Ye1s/lib/python3.13/site-packages/mne/bem.py", line 1048, in get_fitting_dig
    # raise ValueError(f"No digitization points found for dig_kinds={dig_kinds}")
    # raw.interpolate_bads(reset_bads=False)

    # --- Events and Epoching ---
    # 1) Load and attach annotations
    annotations.load_and_attach_annotations(bids_path, raw)

    print(f"Found {len(raw.annotations)} annotations.")

    events, event_dict = mne.events_from_annotations(raw)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=True,
    )

    # --- Baseline correction: mean subtraction from -0.25 to 0.0 s ---
    epochs.apply_baseline(baseline)

    # --- Display results ---
    print(f"Subject {subject_id} analysis complete. Number of epochs: {len(epochs)}")

    data = epochs.get_data()

    # TODO: Just for testing

    ch = next((c for c in ["Fp1", "Fp2", "Fpz", "AFz"] if c in raw.ch_names), "Fp1")
    sf = raw.info["sfreq"]
    t_min, duration = 100.0, 10.0
    start = int(t_min * sf)
    stop = int((t_min + duration) * sf)
    un = raw_unprocessed.copy().pick(ch).get_data(start=start, stop=stop).ravel() * 1e6
    pr = raw.copy().pick(ch).get_data(start=start, stop=stop).ravel() * 1e6
    times = np.linspace(t_min, t_min + duration, un.size)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(times, un, color="k", label="Unprocessed")
    plt.plot(times, pr, color="r", label="Processed", alpha=0.8)
    plt.title(f"{ch} — unprocessed (black) vs processed (red)")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(times, un - pr, color="m")
    plt.title("Difference (unprocessed - processed)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    # plots.power_spectral_density_plot(raw, epochs, 0, 100)
    #
    # plots.ica_topography_plot(ica, raw)
    #
    # plots.one_channel_erp_plot(raw, epochs, baseline)
    #
    # plots.all_channel_erp_plot(epochs, baseline)

    plots.unprocessed_vs_processed_plot(raw_unprocessed, raw)

    # plots.butterfly_plot(epochs)
