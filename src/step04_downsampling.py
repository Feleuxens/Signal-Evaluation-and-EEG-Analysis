from mne.io.edf.edf import RawEDF


TARGET_SFREQ = 128  # Target sampling frequency in Hz


def downsample_data(raw: RawEDF, target_sfreq=TARGET_SFREQ) -> RawEDF:
    """Downsample the raw data to the target sampling frequency."""

    raw.resample(target_sfreq, npad="auto")

    return raw
