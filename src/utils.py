import mne_bids


def get_subjectlist(bids_root="../data/"):
    """Get list of subject IDs from BIDS dataset."""

    subject_list = mne_bids.get_entity_vals(bids_root, entity_key="subject")
    subject_list = [f"{int(s):03d}" for s in subject_list]  # zero-pad to 3 digits

    return subject_list
