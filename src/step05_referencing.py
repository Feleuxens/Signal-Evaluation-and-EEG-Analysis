from mne.io.edf.edf import RawEDF


def rereference_data(raw: RawEDF) -> RawEDF:
    """Rereference the raw data to the average reference."""

    raw.set_eeg_reference("average", projection=False)

    return raw
