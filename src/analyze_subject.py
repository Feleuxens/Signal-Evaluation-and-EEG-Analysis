from mne_bids import BIDSPath
from matplotlib import pyplot as plt
import numpy as np
import plots

from step01_loading import load_data
from step02_badchannels import detect_bad_channels
from step03_filtering import filter_data
from step04_downsampling import downsample_data
from step05_referencing import rereference_data
from step06_asr import run_asr
from step07_ica import run_ica
from step08_interpolation import interpolate_bad_channels
from step09_epoching import epoch_data

BASELINE = (-0.25, 0.0)  # baseline correction period


def analyze_subject(subject_id, bids_root="../data/"):

    bids_path = BIDSPath(
        subject=subject_id,
        root=bids_root,
        datatype="eeg",
        suffix="eeg",
        task="jacobsen",
    )

    # 01) Load data
    print(f"\n\nStep 01: Loading data for subject {subject_id}...")
    raw = load_data(bids_path)
    raw_unprocessed = raw.copy()  # keep a copy of unprocessed data

    # 02) Automatic detection of bad channels
    print(f"\n\nStep 02: Detecting bad channels")
    raw = detect_bad_channels(raw)

    # 03) Filtering
    print(f"\n\nStep 03: Filtering")
    raw = filter_data(raw)

    # 04) Downsampling
    print(f"\n\nStep 04: Downsampling")
    raw = downsample_data(raw)

    # 05) Rereference
    print(f"\n\nStep 05: Rereferencing")
    raw = rereference_data(raw)

    # 06) Artifact correction
    print(f"\n\nStep 06: Artifact correction")
    raw, asr = run_asr(raw)

    # 07) ICA cleaning
    print(f"\n\nStep 07: ICA cleaning")
    raw, ica = run_ica(raw)

    # 08) Interpolate bad channels
    print(f"\n\nStep 08: Interpolating bad channels")
    raw = interpolate_bad_channels(raw)

    # 09) Epoching
    print(f"\n\nStep 09: Epoching")
    epochs, events, event_dict = epoch_data(raw, bids_path, baseline=BASELINE)

    # TODO: Update further analysis steps...

    # --- Display results ---
    print(f"Subject {subject_id} analysis complete. Number of epochs: {len(epochs)}")

    data = epochs.get_data()

    # TODO: Just for testing

    # ch = next((c for c in ["Fp1", "Fp2", "Fpz", "AFz"] if c in raw.ch_names), "Fp1")
    # sf = raw.info["sfreq"]
    # t_min, duration = 100.0, 10.0
    # start = int(t_min * sf)
    # stop = int((t_min + duration) * sf)
    # un = raw_unprocessed.copy().pick(ch).get_data(start=start, stop=stop).ravel() * 1e6
    # pr = raw.copy().pick(ch).get_data(start=start, stop=stop).ravel() * 1e6
    # times = np.linspace(t_min, t_min + duration, un.size)
    #
    # plt.figure(figsize=(12, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(times, un, color="k", label="Unprocessed")
    # plt.plot(times, pr, color="r", label="Processed", alpha=0.8)
    # plt.title(f"{ch} — unprocessed (black) vs processed (red)")
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.plot(times, un - pr, color="m")
    # plt.title("Difference (unprocessed - processed)")
    # plt.xlabel("Time (s)")
    # plt.tight_layout()
    # plt.show()
    #
    plots.power_spectral_density_plot(raw, epochs, 0, 64)

    plots.ica_topography_plot(ica, raw)

    plots.one_channel_erp_plot(raw, epochs, BASELINE)

    plots.all_channel_erp_plot(epochs, BASELINE)

    plots.unprocessed_vs_processed_plot(raw_unprocessed, raw)

    plots.butterfly_plot(epochs)
