import pandas as pd
from mne import Annotations

def load_events_from_bids(bids_path):
    """Load events from BIDS *events.tsv file."""
    events_tsv = bids_path.copy().update(suffix='events', extension='.tsv').fpath
    df = pd.read_csv(events_tsv, sep='\t')

    # BIDS events.tsv has onset (s), duration (s), and trial_type or value
    onsets = df['onset'].astype(float).to_list()
    durations = [0.0] * len(onsets)

    value_mapping = {1: 'regular', 3: 'random'}
    values = df['value'].astype(int).to_list()
    descriptions = []

    for v in values:
        descriptions.append(value_mapping.get(v, 'event'))

    return Annotations(onset=onsets, duration=durations, description=descriptions)


def load_and_attach_annotations(bids_path, raw):
    """Load events from BIDS *events.tsv file and attach as annotations to raw."""
    annotations = load_events_from_bids(bids_path)
    raw.set_annotations(annotations)
