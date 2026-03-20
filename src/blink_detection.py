from time import time
from os import getenv, mkdir, remove
from os.path import isdir, isfile
from mne_bids import BIDSPath
import mne
from mne.io.edf.edf import RawEDF
from mne import Epochs, read_epochs
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.lines import Line2D

from pipeline.step01_loading import load_data
from pipeline.step02_badchannels import detect_bad_channels
from pipeline.step03_filtering import filter_data
from pipeline.step04_downsampling import downsample_data
from pipeline.step05_referencing import rereference_data
from pipeline.step06_asr import run_asr
from pipeline.step07_ica import run_ica
from pipeline.step08_interpolation import interpolate_bad_channels
from pipeline.step09_epoching import epoch_data
from pipeline.step10_trialrejection import reject_trials

from utils.config import load_config, PipelineConfig
from utils.utils import get_config_path, get_subject_list, average_channel, pairwise_average


def main():
    bids_root = getenv("BIDS_ROOT", "../data/")
    bids_root = bids_root.rstrip("/")
    config_root = getenv("CONFIG_ROOT", "../config/")
    config_root = config_root.rstrip("/")

    # load default config
    config_path = get_config_path(config_root, 1) 
    config = load_config(config_path)

    output_folder = bids_root + "/processed_blinkdetection"

    print("Which action do you want to perform?")
    print("1 - Plot blink detection for one subjects")
    print("2 - Process data for for blink comparison (all subjects)")
    print("3 - Plot data for blink comparison (all subjects)")

    i = input(": ")
    if i.lower() == "1":
        s = f"{int(input("Subject ID: ")):03d}"
        eeg_plus_eog_one_subject(bids_root, s, config)

    elif i.lower() == "2":
        start_time = time()
        precompute_all_epochs(bids_root, config, output_folder)
        total_time = time() - start_time
        print(f"\nElapsed time: {total_time} seconds\n")

    elif i.lower() == "3":
        start_time = time()
        all_subjects_plotting(bids_root, config, output_folder)
        total_time = time() - start_time
        print(f"\nElapsed time: {total_time} seconds\n")



def all_subjects_plotting(bids_root: str, config: PipelineConfig, output_folder: str):
    epochs_with_blinks, epochs_without_blinks = load_all_epochs(bids_root, output_folder)
    plot_average_data(bids_root, epochs_with_blinks, epochs_without_blinks)


def save_blink_epochs(
    output_folder: str,
    subject_id: str,
    epochs_with_blink: Epochs,
    epochs_without_blink: Epochs,
):
    epochs_with_blink.save(f"{output_folder}/sub-{subject_id}_with_blinks_epo.fif", overwrite=True)
    epochs_without_blink.save(f"{output_folder}/sub-{subject_id}_without_blinks_epo.fif", overwrite=True)


def read_blink_epochs(
    data_folder: str, subject_id: str
) -> tuple[Epochs, Epochs]:
    if not isdir(data_folder):
        raise FileNotFoundError(f"{data_folder} is not a directory")

    epochs_with_blinks = None
    epochs_without_blinks = None

    file = f"{data_folder}/sub-{subject_id}_with_blinks_epo.fif"
    if isfile(file):
        epochs_with_blinks = read_epochs(file, preload=True)
    else:
        raise FileNotFoundError(f"{data_folder} doesn't have file: {file}")
    
    file = f"{data_folder}/sub-{subject_id}_without_blinks_epo.fif"
    if isfile(file):
        epochs_without_blinks = read_epochs(file, preload=True)
    else:
        raise FileNotFoundError(f"{data_folder} doesn't have file: {file}")
    
    return epochs_with_blinks, epochs_without_blinks # pyright: ignore[reportReturnType]



def precompute_all_epochs(bids_root: str, config: PipelineConfig, output_folder: str):

    subject_ids = get_subject_list(bids_root)

    for i, subject_id in enumerate(subject_ids):

        epochs_after, _, raw_after = process_subject_with_blinkdetection(bids_root, subject_id, config)

        # eeg_chs = ['PO7','PO8']
        eog_chs = ['EOG5','EOG6']

        blink_intervals, _ = detect_blinks_on_raw(
            raw_after,
            eog_chs=eog_chs,
            l_freq=1.0, h_freq=15.0,
            envelope_smooth_ms=20.0,
            mad_mult=6.0
        )

        has_blink = epochs_have_blinks(epochs_after, blink_intervals)

        epochs_with_blinks, epochs_without_blinks = filter_blinks(epochs_after, has_blink)

        save_blink_epochs(output_folder, subject_id, epochs_with_blinks, epochs_without_blinks)


def load_all_epochs(bids_root: str, output_folder: str) -> tuple[
    dict[int, Epochs],
    dict[int, Epochs]
]:
    with_blinks = {}
    without_blinks = {}

    subject_ids = get_subject_list(bids_root)

    for i, subject_id in enumerate(subject_ids):

        epochs_with_blinks, epochs_without_blinks = read_blink_epochs(output_folder, subject_id)

        with_blinks[i] = epochs_with_blinks
        without_blinks[i] = epochs_without_blinks
    
    return (with_blinks, without_blinks)


def filter_blinks(epochs: Epochs, has_blink: np.typing.NDArray[np.bool]) -> tuple[Epochs, Epochs]:
    if has_blink.ndim != 1 or has_blink.shape[0] != len(epochs):
        raise ValueError("filter must be a 1D boolean array with length == len(epochs)")

    print(f"hasblinks: {len(has_blink)}, epochs: {len(epochs)}")
    epochs_with_blink = epochs.copy().drop(~has_blink, reason="blink")
    epochs_without_blink = epochs.drop(has_blink, reason="no blink")

    return (epochs_with_blink, epochs_without_blink)


def process_subject_with_blinkdetection(bids_root: str, subject_id: str, config: PipelineConfig) -> tuple[Epochs, Epochs, RawEDF]:
    """
    computes all epochs once with ASR and once without.
    First Epochs are with ASR,
    Second Epochs are without ASR
    """
    bids_path = BIDSPath(
        subject=subject_id,
        root=bids_root,
        datatype="eeg",
        suffix="eeg",
        task="jacobsen",
    )

    print("\nStep 01: Loading data")
    raw = load_data(bids_path)

    if config.bad_channels.enabled:
        print("\nStep 02: Detecting bad channels")
        raw = detect_bad_channels(raw, config.bad_channels)

    if config.filtering.enabled:
        print(f"\nStep 03: Filtering")
        raw = filter_data(raw, config.filtering)

    if config.downsampling.enabled:
        print(f"\nStep 04: Downsampling")
        raw = downsample_data(raw, config.downsampling)

    if config.rereferencing.enabled:
        print(f"\nStep 05: Rereferencing")
        raw = rereference_data(raw, config.rereferencing)

    raw_before = raw.copy()
    raw_after = raw

    if config.asr.enabled:
        print(f"\nStep 06: Artifact correction")
        raw_after, _ = run_asr(raw_after, config.asr)

    if config.ica.enabled:
        print(f"\nStep 07: ICA cleaning")
        raw_after, _, _ = run_ica(raw_after, config.ica)
        raw_before, _, _ = run_ica(raw_before, config.ica)

    if config.interpolation.enabled:
        print(f"\nStep 08: Interpolating bad channels")
        raw_after = interpolate_bad_channels(raw_after, config.interpolation)
        raw_before = interpolate_bad_channels(raw_before, config.interpolation)

    print(f"\nStep 09: Epoching")
    epochs_after, _, _ = epoch_data(raw_after, bids_path, config.epoching)
    epochs_before, _, _ = epoch_data(raw_before, bids_path, config.epoching)

    return (epochs_after, epochs_before, raw_after)

def plot_average_data(bids_root: str, epochs_with_blinks: dict[int, Epochs], epochs_without_blinks: dict[int, Epochs]):

    output_folder = bids_root + "/processed_blinkdetection"

    data_random_po7_with_blink, data_regular_po7_with_blink, times_po7_with_blink, n_subjects_with_blink, _ =  average_channel("PO7", epochs_with_blinks)
    data_random_po8_with_blink, data_regular_po8_with_blink, _, _, _ =  average_channel("PO8", epochs_with_blinks)

    data_random_po7_without_blink, data_regular_po7_without_blink, times_po7_without_blink, n_subjects_without_blink, _ =  average_channel("PO7", epochs_without_blinks)
    data_random_po8_without_blink, data_regular_po8_without_blink, _, _, _ =  average_channel("PO8", epochs_without_blinks)

    data_random_with_blink = pairwise_average(data_random_po7_with_blink, data_random_po8_with_blink)
    data_random_without_blink = pairwise_average(data_random_po7_without_blink, data_random_po8_without_blink)
    data_regular_with_blink = pairwise_average(data_regular_po7_with_blink, data_regular_po8_with_blink)
    data_regular_without_blink = pairwise_average(data_regular_po7_without_blink, data_regular_po8_without_blink)

    n_epochs_with_blinks = 0
    for epochs in epochs_with_blinks.values():
        n_epochs_with_blinks += len(epochs)

    n_epochs_without_blinks = 0
    for epochs in epochs_without_blinks.values():
        n_epochs_without_blinks += len(epochs)


    if not isdir(output_folder):
        mkdir(output_folder)

    file = f"{output_folder}/fig-average_po7po8_with_blinks.png"
    # if isfile(file):
    #     remove(file)
        
    plot_channel(
        file,
        "with blinks",
        "PO7+PO8",
        data_random_with_blink,
        data_regular_with_blink,
        times_po7_with_blink,
        n_subjects_with_blink,
        n_epochs_with_blinks
    )

    file = f"{output_folder}/fig-average_po7po8_without_blinks.png"
    # if isfile(file):
    #     remove(file)

    plot_channel(
        file,
        "without blinks",
        "PO7+PO8",
        data_random_without_blink,
        data_regular_without_blink,
        times_po7_without_blink,
        n_subjects_without_blink,
        n_epochs_without_blinks
    )


def plot_channel(
    output_file,
    title,
    channel,
    data_random,
    data_regular,
    times,
    n_subjects,
    n_epochs
):

    plt.figure(figsize=(10, 5))
    plt.plot(times * 1000, data_random, "r-", linewidth=2, label="Random")
    plt.plot(times * 1000, data_regular, "b-", linewidth=2, label="Regular")
    plt.axhline(0, color="k", linestyle="--", linewidth=0.5)
    plt.axvline(0, color="k", linestyle="--", linewidth=0.5)
    plt.yticks([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.title(f"Grand Average ERP for condition: {title} at {channel} (n={n_subjects} subjects, k={n_epochs} epochs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")

    return data_random, data_regular, times


def eeg_plus_eog_one_subject(bids_root: str, subject_id: str, config: PipelineConfig):
    """
    Plot EOG channels as well as before and after ASR eeg data for each epoch for one subject.
    """
    epochs_after, epochs_before, raw_after = process_subject_with_blinkdetection(bids_root, subject_id, config)

    eeg_chs = ['PO7','PO8']
    eog_chs = ['EOG5','EOG6']

    blink_intervals, durations = detect_blinks_on_raw(
        raw_after,
        eog_chs=eog_chs,
        l_freq=1.0, h_freq=15.0,
        envelope_smooth_ms=20.0,
        mad_mult=6.0
    )

    print("Detected blinks:", len(blink_intervals))
    print("mean duration of blinks:", np.mean(durations))

    fig, ax = plot_epochs_before_after(
        subject_id,
        epochs_after,
        epochs_before,
        blink_intervals,
        picks=eeg_chs,
        eog_picks=eog_chs,
        scale=1e6
    )

def detect_blinks_on_raw(
    raw: mne.io.edf.edf.RawEDF,
    eog_chs: list[str],
    l_freq=1.0,
    h_freq=15.0,
    envelope_smooth_ms=20.0,
    mad_mult=6.0,
    min_distance_s=0.05,
    merge_gap_s=0.02,
) :
    """
    Detect blink intervals on raw — one interval per blink:
      - bandpass EOG channels
      - build smoothed envelope = max(abs(channels))
      - threshold via median + mad_mult*MAD to find candidate peaks
      - for each candidate peak expand left/right until envelope <= baseline_level
        (baseline_level = median + 0.5 * MAD), which avoids splitting rise/fall into multiple blinks
      - merge overlapping/nearby intervals
    Returns:
      intervals: list of (start_s, end_s)
      durations: np.array of durations (s)
      (optionally) env_smooth, threshold
    """

    sfreq = raw.info['sfreq']
    eog_idx = mne.pick_channels(raw.ch_names, include=eog_chs)
    eog_data = raw.get_data(picks=eog_idx)

    # Bandpass filter for blink detection
    eog_filtered = mne.filter.filter_data(
        eog_data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method='iir', verbose=False
    )

    # envelope: max absolute across eog channels
    env = np.max(np.abs(eog_filtered), axis=0)

    # smooth envelope
    win = int(round(envelope_smooth_ms * 1e-3 * sfreq))
    win = max(1, win)
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode='same')

    # robust baseline & threshold
    med = np.median(env_smooth)
    mad = np.median(np.abs(env_smooth - med))
    thresh = med + mad_mult * mad
    baseline_level = med + 0.5 * mad

    min_dist = int(round(min_distance_s * sfreq))
    peaks, _ = signal.find_peaks(env_smooth, height=thresh, distance=min_dist)

    intervals = []
    n = env_smooth.size
    for p in peaks:
        s = p
        while s > 0 and env_smooth[s] > baseline_level:
            s -= 1
        e = p
        while e < n - 1 and env_smooth[e] > baseline_level:
            e += 1
        intervals.append((max(0, s), min(n - 1, e)))

    # convert to seconds 
    intervals_s = [(s / sfreq, e / sfreq) for s, e in intervals]
    if not intervals_s:
        return [], np.array([])

    # merge nearby/overlapping intervals
    intervals_s.sort()
    merged = []
    cur_s, cur_e = intervals_s[0]
    for s, e in intervals_s[1:]:
        if s <= cur_e + merge_gap_s:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    durations = np.array([e - s for s, e in merged])

    return merged, durations

def epochs_have_blinks(epochs: mne.Epochs, blink_intervals) -> np.typing.NDArray[np.bool] :
    """
    Return a boolean numpy array of length n_epochs where True means the epoch
    overlaps at least one blink interval.
    epochs: mne.Epochs
    blink_intervals: list of (start_s, end_s) in absolute seconds (same reference as epochs.events)
    """
    sfreq = epochs.info['sfreq']
    event_samples = epochs.events[:, 0]
    epoch_start_times = event_samples / sfreq + epochs.tmin
    epoch_duration = epochs.times[-1] - epochs.times[0]
    epoch_end_times = epoch_start_times + epoch_duration

    has_blink = np.zeros(len(epoch_start_times), dtype=bool)
    if not blink_intervals:
        return has_blink

    bi = np.array(blink_intervals)  
    for i, (s0, e0) in enumerate(zip(epoch_start_times, epoch_end_times)):
        overlaps = np.logical_not((bi[:,1] <= s0) | (bi[:,0] >= e0))
        if overlaps.any():
            has_blink[i] = True

    return has_blink

def plot_epochs_before_after(
    subject_id: str,
    epochs_after: mne.Epochs,
    epochs_before: mne.Epochs,
    blink_intervals,
    picks: list[str], 
    eog_picks: list[str],
    scale=1e6,
    figsize=(15, 15),
):
    plot_chs = picks + eog_picks
    picks_idx = [epochs_after.ch_names.index(ch) for ch in plot_chs]

    times = epochs_after.times
    n_epochs = len(epochs_after)

    data_after = epochs_after.get_data()[:, picks_idx, :] * scale
    data_before = epochs_before.get_data()[:, picks_idx, :] * scale

    fixed_spacing = 100
    offsets = [0, fixed_spacing, 2*fixed_spacing, 3*fixed_spacing]

    event_samples = epochs_after.events[:, 0]
    sfreq = epochs_after.info['sfreq']
    epoch_start_times_raw = event_samples / sfreq + epochs_after.tmin

    has_blink = epochs_have_blinks(epochs_before, blink_intervals)

    # Plot
    current = 0
    fig, ax = plt.subplots(figsize=figsize)

    legend_handles = [
        Line2D([0], [0], color='tab:blue', linewidth=1, label='After (ASR)'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=1, label='Before'),
    ]

    def redraw(idx):
        ax.cla()
        for i, ch in enumerate(plot_chs):
            ax.plot(times, data_after[idx, i, :] + offsets[i], color='tab:blue', linewidth=0.9)
            if ch.startswith("PO"):
                ax.plot(times, data_before[idx, i, :] + offsets[i], color='orange', linestyle='--', linewidth=0.9)
        
        ax.set_yticks(offsets)
        ax.set_yticklabels(plot_chs)
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Epoch {idx+1} / {n_epochs}, with {"blink" if has_blink[idx] else "no blink"} for subject {subject_id}")
        ax.grid(True, linewidth=0.3, alpha=0.6)
        ax.legend(handles=legend_handles, loc='upper right')

        # add new blink shading
        t0 = epoch_start_times_raw[idx]
        t_end = t0 + (times[-1] - times[0])
        for s, e in blink_intervals:
            if e <= t0 or s >= t_end:
                continue
            ov_s = max(s, t0); ov_e = min(e, t_end)
            rel_s = ov_s - t0 + times[0]; rel_e = ov_e - t0 + times[0]
            ax.axvspan(rel_s, rel_e, color='red', alpha=0.15)

        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal current
        if event.key in ('right', 'pagedown') and current < n_epochs - 1:
            current += 1; redraw(current)
        elif event.key in ('left', 'pageup') and current > 0:
            current -= 1; redraw(current)
        elif event.key == 'home':
            current = 0; redraw(current)
        elif event.key == 'end':
            current = n_epochs - 1; redraw(current)

    fig.canvas.mpl_connect('key_press_event', on_key)
    redraw(0)
    plt.tight_layout()
    plt.show()

    return fig, ax


if __name__ == "__main__":
    main()
