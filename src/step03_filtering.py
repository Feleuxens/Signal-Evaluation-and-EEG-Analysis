from mne.io.edf.edf import RawEDF


LOW_PASS = 40.0  # Hz
HIGH_PASS = 0.1  # Hz
NOTCH_FREQUENCIES = [50, 100, 150, 200]  # Hz


def filter_data(raw: RawEDF) -> RawEDF:
    """Apply notch and bandpass filtering to the raw data."""

    raw = _notch_and_bandpass_filter(raw)

    return raw


def _notch_and_bandpass_filter(
    raw, l_freq=HIGH_PASS, h_freq=LOW_PASS, notch_freqs=NOTCH_FREQUENCIES
) -> RawEDF:
    """Apply notch and bandpass filtering to raw data."""

    # 1) Notch filter to remove line noise
    raw.notch_filter(notch_freqs, picks="eeg", method="spectrum_fit")

    # 2) Bandpass filter
    raw.filter(l_freq, h_freq, picks="eeg", fir_design="firwin")

    return raw
