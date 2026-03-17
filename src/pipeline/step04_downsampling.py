from mne.io.edf.edf import RawEDF

from utils.config import StepDownsampling


def downsample_data(raw: RawEDF, config: StepDownsampling) -> RawEDF:
    """Downsample the raw data to the target sampling frequency."""

    raw.resample(config.target_sfreq, npad="auto")

    return raw
