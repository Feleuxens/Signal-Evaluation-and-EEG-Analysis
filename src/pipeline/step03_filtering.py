from array import array
from mne.io.edf.edf import RawEDF

from utils.config import StepFiltering


def filter_data(raw: RawEDF, config: StepFiltering) -> RawEDF:
    """Apply notch and bandpass filtering to the raw data."""

    notch_freqs = array("f", config.notch_frequencies)
    if config.notch_filter_enabled:
        raw.notch_filter(
            notch_freqs,
            picks=config.notch_filter_pick,
            method=config.notch_filter_method,
        )
    if config.pass_filter_enabled:
        raw.filter(
            config.high_pass,
            config.low_pass,
            picks=config.notch_filter_pick,
            method=config.pass_filter_method,
        )

    return raw
