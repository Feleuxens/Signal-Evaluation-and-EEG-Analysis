from mne_bids import BIDSPath, read_raw_bids
from matplotlib import pyplot as plt
import mne
import numpy as np
import annotations

# --- Constants ---
tmin, tmax = -0.5, 1.0  # epoch time range in seconds
baseline = (-0.25, 0.0)  # baseline correction period
l_freq, h_freq = 0.1, 40.0  # highpass, lowpass filter
notch_freqs = [50, 100, 150, 200]  # notch filter frequencies
bad_channel_z_thresh = 3.0  # z-score threshold for bad channel detection


def analyze_subject(subject_id, bids_root="../data/"):

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

    # --- Artefact removal with ASR ---
    # TODO: implement

    # --- Interpolate bad channels ---
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
        baseline=None,
        preload=True,
        reject_by_annotation=True,
    )

    # --- Baseline correction: mean subtraction from -0.25 to 0.0 s ---
    epochs.apply_baseline(baseline)

    # --- Display results ---
    print(f"Subject {subject_id} analysis complete. Number of epochs: {len(epochs)}")

    data = epochs.get_data()

    # TODO: Just for testing
    desc_random = "random"
    desc_regular = "regular"

    id_random = event_dict.get(desc_random)
    id_regular = event_dict.get(desc_regular)
    if id_random is None or id_regular is None:
        raise ValueError(
            f"Descriptions not found. Available: {list(event_dict.keys())}"
        )

    # select and baseline-correct
    epochs_random = epochs[id_random].copy().apply_baseline(baseline)
    epochs_regular = epochs[id_regular].copy().apply_baseline(baseline)

    # average (mean across all occurrences)
    evoked_random = epochs_random.average()
    evoked_regular = epochs_regular.average()

    # For type checking reasons
    if type(evoked_random) is not mne.Evoked:
        raise ValueError("evoked_random is not an instance of mne.Evoked")

    if type(evoked_regular) is not mne.Evoked:
        raise ValueError("evoked_regular is not an instance of mne.Evoked")

    # simple channel-mean time series plot
    times = evoked_random.times
    mean_random = evoked_random.data.mean(axis=0)
    mean_regular = evoked_regular.data.mean(axis=0)

    plt.plot(times, mean_random * 1e6, label="Random")  # convert to µV if data in V
    plt.plot(times, mean_regular * 1e6, label="Regular")
    plt.axvspan(baseline[0], baseline[1], color="gray", alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.title("Mean across channels — Random vs Regular")
    plt.show()

    return
