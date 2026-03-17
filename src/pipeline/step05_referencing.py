from mne.io.edf.edf import RawEDF

from utils.config import StepRereferencing


def rereference_data(raw: RawEDF, config: StepRereferencing) -> RawEDF:
    """Rereference the raw data to the average reference."""

    raw.set_eeg_reference(config.ref_channels, projection=config.projection)

    return raw
