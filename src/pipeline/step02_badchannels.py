import numpy as np
from mne.io.edf.edf import RawEDF

from utils.config import StepBadChannels


def detect_bad_channels(raw: RawEDF, config: StepBadChannels) -> RawEDF:
    """Automatic detection and marking of bad channels"""

    channel_names = raw.ch_names

    # pick EEG channel indices excluding EXG* and Status channels
    eeg_picks = [
        i
        for i, ch in enumerate(channel_names)
        if not ch.upper().startswith(config.exg_prefix)
        and not ch.startswith(config.status_prefix)
        and raw.get_channel_types()[i] == "eeg"
    ]

    channel_names = [raw.ch_names[i] for i in eeg_picks]

    if not eeg_picks:
        print("No EEG channels found for bad channel detection.")
        return raw

    data = raw.get_data(picks=eeg_picks)

    bad_channels_var = _zscore_bad_channel_detection(
        data, channel_names, config.z_thresh
    )
    bad_channels_corr = _correlation_bad_channel_detection(
        data, channel_names, config.z_thresh
    )

    bad_channels = sorted(set(bad_channels_var + bad_channels_corr))

    # Mark detected bad channels in raw.info['bads']
    raw.info["bads"].extend(bad_channels)

    print(f"Automatically detected bad channels: {raw.info['bads']}")

    return raw


def _zscore_bad_channel_detection(data, channel_names, bad_channel_z_thresh):
    """1. Variance z-score across channels"""
    variances = np.var(data, axis=1)
    z_var = (variances - variances.mean()) / variances.std()
    bad_channels_var = [
        channel
        for channel, z in zip(channel_names, z_var)
        if np.abs(z) > bad_channel_z_thresh
    ]

    return bad_channels_var


def _correlation_bad_channel_detection(data, channel_names, bad_channel_z_thresh):
    """2. Low correlation channels (flat or noisy channels)"""
    corr_matrix = np.corrcoef(data)
    mean_corr = np.median(corr_matrix, axis=0)
    z_corr = (mean_corr - mean_corr.mean()) / mean_corr.std()
    bad_channels_corr = [
        channel
        for channel, z in zip(channel_names, z_corr)
        if z < -bad_channel_z_thresh
    ]

    return bad_channels_corr
