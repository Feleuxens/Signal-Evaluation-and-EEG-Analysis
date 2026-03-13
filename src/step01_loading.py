from mne_bids import BIDSPath, read_raw_bids
from mne.io.edf.edf import RawEDF


def load_data(bids_path: BIDSPath) -> RawEDF:
    """Load raw EEG data from BIDS path."""

    raw = read_raw_bids(bids_path)
    raw.load_data()

    # TODO: Test
    rename_map = {f"EXG{i}": f"EOG{i}" for i in range(1, 9)}
    raw.rename_channels(rename_map)

    return raw
