from mne_bids import BIDSPath, read_raw_bids
from mne.io.edf.edf import RawEDF


def load_data(bids_path: BIDSPath) -> RawEDF:
    """Load raw EEG data from BIDS path."""

    raw: RawEDF = read_raw_bids(bids_path)
    raw.load_data()

    # TODO: confirm?
    # Apperantly only EXOG4 and EXG5 are EOG channels
    # rename_map = {"EXG4": "EOG4", "EXG5": "EOG5"}
    # raw.rename_channels(rename_map)

    for ch in ["EXG4", "EXG5"]:
        raw.set_channel_types({ch: "eog"})
        raw.rename_channels({ch: ch.replace('X', 'O')})

    for ch in ["EXG1", "EXG2", "EXG3", "EXG6", "EXG7", "EXG8"]:
        raw.set_channel_types({ch: "misc"})

    return raw
