from time import time
from os import mkdir
from os.path import isdir
from mne_bids import BIDSPath
from utils import get_subjectlist
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import re
from scipy import signal

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

def main(bids_root="../data/"):
    # Plot EOG channels
    subject_id = "005"
    bids_path = BIDSPath(
        subject=subject_id,
        root=bids_root,
        datatype="eeg",
        suffix="eeg",
        task="jacobsen",
    )

    # 01) Load data
    raw = load_data(bids_path)

    # Rename EXG -> EOG if not already
    rename_map = {f"EXG{i}": f"EOG{i}" for i in range(1, 9)}
    raw.rename_channels(rename_map)

    # Optionally pick only EOG channels
    eog_chs = [f"EOG{i}" for i in range(1, 9)]
    raw_eog = raw.copy().pick_channels(eog_chs)

    # Simple plot: scrollable multi-channel trace
    raw_eog.plot(n_channels=8, scalings='auto', duration=10, start=0.0,
                title='EOG channels (EXG -> EOG)', show=True, block=True)




def main2(bids_root="../data/"):
    # Plot blink guesses by thershold
    subject_id = "005"
    bids_path = BIDSPath(
        subject=subject_id,
        root=bids_root,
        datatype="eeg",
        suffix="eeg",
        task="jacobsen",
    )

    # 01) Load data
    raw = load_data(bids_path)
    # === Parameters ===
    eog_ch_name = "EXG1"
    l_freq, h_freq = 1.0, 15.0        # bandpass for blink detection
    threshold_uv = 100.0              # µV threshold for peak detection
    min_distance_s = 0.25             # minimum distance between peaks (s)
    plot_seconds = 100                 # seconds of data to plot (None = whole recording)
    plot_start_s = 300.0               # start time in seconds for the plotted window
    sfreq = raw.info['sfreq']         # assumes `raw` is already loaded in workspace

    # === Prepare EOG data ===
    if eog_ch_name not in raw.ch_names:
        raise ValueError(f"Channel {eog_ch_name} not found in raw.ch_names")

    raw_eog = raw.copy().pick_channels([eog_ch_name])
    eog_data = raw_eog.get_data()[0]             # shape (n_samples,)
    n_samples = eog_data.shape[0]
    total_duration_s = n_samples / sfreq

    # clamp plot_start_s
    if plot_start_s < 0:
        plot_start_s = 0.0
    if plot_start_s >= total_duration_s:
        raise ValueError(f"plot_start_s ({plot_start_s}s) is beyond recording duration ({total_duration_s:.1f}s)")

    # === Filter EOG to emphasize blinks ===
    eog_filt = mne.filter.filter_data(
        eog_data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method='iir'
    )

    # === Convert threshold to Volts (raw usually in Volts) ===
    threshold_v = threshold_uv * 1e-6

    # === Find blink peaks on absolute filtered signal ===
    abs_eog = np.abs(eog_filt)
    min_distance_samples = int(min_distance_s * sfreq)
    peaks, props = signal.find_peaks(abs_eog, height=threshold_v, distance=min_distance_samples)

    # === Create boolean mask per sample indicating within-blink window (small window around peak) ===
    blink_half_width_s = 0.1
    blink_half_width_samples = int(blink_half_width_s * sfreq)

    blink_mask = np.zeros(n_samples, dtype=bool)
    for p in peaks:
        start = max(0, p - blink_half_width_samples)
        end = min(n_samples, p + blink_half_width_samples + 1)
        blink_mask[start:end] = True

    # === Prepare times for plotting ===
    times = np.arange(n_samples) / sfreq  # in seconds

    # Determine plot slice indices based on plot_start_s and plot_seconds
    start_sample = int(round(plot_start_s * sfreq))
    if plot_seconds is None:
        end_sample = n_samples
    else:
        end_sample = min(n_samples, start_sample + int(round(plot_seconds * sfreq)))

    times_plot = times[start_sample:end_sample]
    eog_plot = eog_filt[start_sample:end_sample]
    mask_plot = blink_mask[start_sample:end_sample]

    # === Plot: EOG with blink markers ===
    plt.figure(figsize=(12, 4))
    plt.plot(times_plot, eog_plot * 1e6, color='C0', label='EXG1 (filtered)')  # convert to µV for display

    # Mark detected peak positions within the plotted window
    peak_mask_in_plot = (peaks >= start_sample) & (peaks < end_sample)
    peak_times_plot = peaks[peak_mask_in_plot] / sfreq
    peak_vals_plot = eog_filt[peaks[peak_mask_in_plot]] * 1e6
    plt.plot(peak_times_plot, peak_vals_plot, 'rx', label='Detected blink peaks')

    # Shade blink regions (contiguous True segments in mask_plot)
    in_blink = False
    start_t = None
    for i, val in enumerate(mask_plot):
        if val and not in_blink:
            in_blink = True
            start_t = times_plot[i]
        elif not val and in_blink:
            end_t = times_plot[i]
            plt.axvspan(start_t, end_t, color='red', alpha=0.15)
            in_blink = False
    if in_blink:
        plt.axvspan(start_t, times_plot[-1], color='red', alpha=0.15)

    plt.xlabel('Time (s)')
    plt.ylabel('EXG1 (µV)')
    plt.title(f'EOG blink detection on subject {subject_id} and channel {eog_ch_name} — window {times_plot[0]:.1f}s to {times_plot[-1]:.1f}s')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # === Optional: print summary for the plotted window ===
    n_peaks_plot = peak_mask_in_plot.sum()
    print(f"Plotted window: {times_plot[0]:.2f}s–{times_plot[-1]:.2f}s")
    print(f"Detected {n_peaks_plot} blinks (peaks above {threshold_uv} µV) in plotted window.")

# def main(bids_root="../data/"):
#     subject_id = "005"
#     bids_path = BIDSPath(
#         subject=subject_id,
#         root=bids_root,
#         datatype="eeg",
#         suffix="eeg",
#         task="jacobsen",
#     )

#     # 01) Load data
#     raw = load_data(bids_path)
        
#     # Parameters
#     # raw must be loaded
#     rename_exg_to_eog = True    # set False to keep original names
#     filter_l, filter_h = 0.5, 15.0
#     plot_start_s = 300.0          # start time in seconds for plotting
#     plot_duration_s = 60.0      # duration in seconds to plot
#     scalings_uV = 200.0         # vertical scale in µV for plotting visual spacing

#     # Find EXG channels: either by channel type 'exg' or by name pattern
#     exg_chs = [ch for ch in raw.ch_names if raw.get_channel_types(picks=[ch])[0] == 'exg']
#     # Fallback: include channels named like 'EOG' or starting with 'EXG' if none found
#     if len(exg_chs) == 0:
#         exg_chs = [ch for ch in raw.ch_names if ch.upper().startswith('EOG') or ch.upper().startswith('EXG')]

#     if len(exg_chs) == 0:
#         raise RuntimeError("No EXG/EOG channels found. Adjust selection logic.")

#     # Optionally rename to EOG1..En (works in-place on a copy)
#     if rename_exg_to_eog:
#         mapping = {orig: f"EOG{ix+1}" for ix, orig in enumerate(exg_chs)}
#         raw = raw.copy().rename_channels(mapping)
#         exg_chs = [mapping[ch] for ch in exg_chs]

#     # Pick and filter data for plotting
#     sfreq = raw.info['sfreq']
#     start_s = max(0.0, plot_start_s)
#     stop_s = min(raw.times[-1], plot_start_s + plot_duration_s)
#     start_sample = int(start_s * sfreq)
#     stop_sample = int(stop_s * sfreq)

#     data = raw.copy().pick_channels(exg_chs).get_data()[:, start_sample:stop_sample]  # channels x samples
#     times = np.arange(start_sample, stop_sample) / sfreq

#     # Bandpass filter for display (avoid filtering in-place on original)
#     data_filt = mne.filter.filter_data(data, sfreq=sfreq, l_freq=filter_l, h_freq=filter_h, verbose=False)

#     # Convert to µV for plotting
#     data_uV = data_filt * 1e6

#     # Plot stacked traces
#     n_ch = data_uV.shape[0]
#     plt.figure(figsize=(10, 1.2 * n_ch))
#     offsets = np.arange(n_ch) * scalings_uV
#     for i, ch in enumerate(exg_chs):
#         plt.plot(times, data_uV[i] + offsets[i], color='C0')
#         plt.text(times[0] - 0.01*(stop_s-start_s), offsets[i], ch, verticalalignment='center', fontsize=9)

#     plt.xlim(start_s, stop_s)
#     plt.xlabel('Time (s)')
#     plt.yticks([])
#     plt.title(f'EXG channels (renamed) — {start_s:.1f}s to {stop_s:.1f}s')
#     plt.tight_layout()
#     plt.show()


# subject_ids = ["003", "004"]

# def main(bids_root="../data/"):
#     # === User parameters ===
#     eog_ch_name = "EXG1"
#     l_freq, h_freq = 1.0, 15.0
#     threshold_uv = 100.0
#     min_distance_s = 0.25
#     blink_half_width_s = 0.1
#     poi_channels = ["PO7", "PO8"]
#     t_baseline = (None, 0)
#     plot_time_window = (-0.2, 0.8)

#     # === Helpers ===
#     def detect_blinks_in_raw(raw):
#         sfreq = raw.info["sfreq"]
#         eog_data = raw.copy().pick_channels([eog_ch_name]).get_data()[0]
#         eog_filt = mne.filter.filter_data(eog_data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method="iir")
#         threshold_v = threshold_uv * 1e-6
#         abs_eog = np.abs(eog_filt)
#         min_distance_samples = int(min_distance_s * sfreq)
#         peaks, _ = signal.find_peaks(abs_eog, height=threshold_v, distance=min_distance_samples)
#         n_samples = eog_data.shape[0]
#         blink_half_width_samples = int(blink_half_width_s * sfreq)
#         blink_mask = np.zeros(n_samples, dtype=bool)
#         for p in peaks:
#             start = max(0, p - blink_half_width_samples)
#             end = min(n_samples, p + blink_half_width_samples + 1)
#             blink_mask[start:end] = True
#         blink_samples = np.where(blink_mask)[0]
#         return blink_samples

#     # === Collect evoked PO7+PO8 means per subject per condition/blink ===
#     groups = ["reg_blink", "reg_noblink", "rnd_blink", "rnd_noblink"]
#     grand_means = {g: [] for g in groups}

#     for subj in subject_ids:


#         bids_path = BIDSPath(
#             subject=subj,
#             root=bids_root,
#             datatype="eeg",
#             suffix="eeg",
#             task="jacobsen",
#         )

#         # 01) Load data
#         raw = load_data(bids_path)
#         raw_copy = raw.copy()  # keep a copy of unprocessed data

#         # 02) Automatic detection of bad channels
#         print(f"\n\nStep 02: Detecting bad channels")
#         raw = detect_bad_channels(raw)

#         # 03) Filtering
#         print(f"\n\nStep 03: Filtering")
#         raw = filter_data(raw)

#         # 04) Downsampling
#         print(f"\n\nStep 04: Downsampling")
#         raw = downsample_data(raw)

#         # 05) Rereference
#         print(f"\n\nStep 05: Rereferencing")
#         raw = rereference_data(raw)

#         # 06) Artifact correction
#         print(f"\n\nStep 06: Artifact correction")
#         raw, asr = run_asr(raw)

#         # 07) ICA cleaning
#         print(f"\n\nStep 07: ICA cleaning")
#         raw, ica, number_excluded_components = run_ica(raw)

#         # 08) Interpolate bad channels
#         print(f"\n\nStep 08: Interpolating bad channels")
#         raw = interpolate_bad_channels(raw)

#         # 09) Epoching
#         print(f"\n\nStep 09: Epoching")
#         epochs, events, event_dict = epoch_data(raw, bids_path, baseline=BASELINE)

#         blink_samples = detect_blinks_in_raw(raw)

#         sfreq = raw.info["sfreq"]
#         epoch_sample_starts = epochs.events[:, 0] + int(round(epochs.tmin * sfreq))
#         epoch_sample_ends = epochs.events[:, 0] + int(round(epochs.tmax * sfreq))
#         epoch_has_blink = np.array([
#             np.any((blink_samples >= start) & (blink_samples <= end))
#             for start, end in zip(epoch_sample_starts, epoch_sample_ends)
#         ])

#         # conditions
#         if epochs.metadata is not None and "condition" in epochs.metadata.columns:
#             conditions = epochs.metadata["condition"].astype(str).str.lower().values
#         else:
#             if isinstance(epochs.event_id, dict):
#                 inv_map = {v: k for k, v in epochs.event_id.items()}
#                 event_codes = epochs.events[:, 2]
#                 conditions = np.array([inv_map.get(code, str(code)).lower() for code in event_codes])
#             else:
#                 conditions = np.array(["unknown"] * len(epochs))

#         is_regular = np.array(["regular" in c for c in conditions])
#         is_random = np.array(["random" in c for c in conditions])
#         if not is_regular.any() and not is_random.any():
#             n = len(epochs)
#             is_regular = np.zeros(n, dtype=bool)
#             is_regular[: n // 2] = True
#             is_random = ~is_regular

#         regular_idx = np.where(is_regular)[0]
#         random_idx = np.where(is_random)[0]

#         reg_blink_idx = regular_idx[epoch_has_blink[regular_idx]]
#         reg_noblink_idx = regular_idx[~epoch_has_blink[regular_idx]]

#         rnd_blink_idx = random_idx[epoch_has_blink[random_idx]]
#         rnd_noblink_idx = random_idx[~epoch_has_blink[random_idx]]

#         # create evokeds and store PO7+PO8 mean time series
#         def evoked_poi_mean_from_idx(idxs):
#             if len(idxs) == 0:
#                 return None
#             sub = epochs[idxs].copy().pick_channels(poi_channels)
#             sub.apply_baseline(t_baseline)
#             ev = sub.average()
#             return ev.data.mean(axis=0) * 1e6  # µV

#         m = evoked_poi_mean_from_idx(reg_blink_idx)
#         if m is not None:
#             grand_means["reg_blink"].append(m)
#         m = evoked_poi_mean_from_idx(reg_noblink_idx)
#         if m is not None:
#             grand_means["reg_noblink"].append(m)
#         m = evoked_poi_mean_from_idx(rnd_blink_idx)
#         if m is not None:
#             grand_means["rnd_blink"].append(m)
#         m = evoked_poi_mean_from_idx(rnd_noblink_idx)
#         if m is not None:
#             grand_means["rnd_noblink"].append(m)

#     # === Compute grand average (across subjects) for each group ===
#     # determine common time vector from last loaded epochs
#     times = epochs.times
#     grand_avgs = {}
#     for g in groups:
#         if len(grand_means[g]) == 0:
#             grand_avgs[g] = None
#         else:
#             arr = np.vstack(grand_means[g])  # subjects x time
#             grand_avgs[g] = arr.mean(axis=0)

#     # === Plot grand averages ===
#     plt.figure(figsize=(8, 5))
#     def plot_if_present(ts, label, color, linestyle='-'):
#         if ts is None:
#             return
#         mask = (times >= plot_time_window[0]) & (times <= plot_time_window[1])
#         plt.plot(times[mask], ts[mask], label=label, color=color, linestyle=linestyle)

#     plot_if_present(grand_avgs["reg_blink"], "regular + blink", "C0")
#     plot_if_present(grand_avgs["reg_noblink"], "regular + no blink", "C0", linestyle='--')
#     plot_if_present(grand_avgs["rnd_blink"], "random + blink", "C1")
#     plot_if_present(grand_avgs["rnd_noblink"], "random + no blink", "C1", linestyle='--')

#     plt.axvline(0, color="k", linewidth=0.7)
#     plt.xlabel("Time (s)")
#     plt.ylabel("PO7+PO8 (µV)")
#     plt.title("Grand average PO7+PO8 by condition and blink presence")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # === Print subject counts per group ===
#     for g in groups:
#         print(f"{g}: subjects contributing = {len(grand_means[g])}")



if __name__ == "__main__":
    main()
