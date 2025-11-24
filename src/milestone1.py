from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
import mne
from annotations import load_and_attach_annotations


# --- user params: adjust these ---
bids_root = '../data/'  # path to BIDS root directory
subject = '001'
task = 'jacobsen'   # use the actual task label in your dataset
session = None      # set if your dataset has sessions, e.g. '01'
start_sec = 140.0    # where to start the plot
duration_sec = start_sec + 15
# duration_sec = 150.0 # how many seconds to show
out_png = '../outputs/continuous_subject_001.png'
# -------------------------------

# build BIDSPath
bids_path = BIDSPath(root=bids_root, subject=subject, task=task, session=session, suffix='eeg', datatype='eeg')

raw = read_raw_bids(bids_path=bids_path)

load_and_attach_annotations(bids_path, raw)

# print summary metadata for the slide / notebook cell
print(f"Subject: {subject}, task: {task}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Number of channels: {len(raw.info['ch_names'])}")
print("Channel types summary:")
print(mne.pick_types(raw.info, eeg=True, eog=True, meg=False, stim=False, misc=True))
# (Optional) list first 10 channel names
print("First channels:", raw.info['ch_names'][:10])

print(f"Found {len(raw.annotations)} annotations; plotting as vertical bands.")
# convert annotations to events for plotting vertical lines
events, event_id = mne.events_from_annotations(raw)


for start in range(30, 401, 15):
# Plot interactive overview (recommended)
# raw.plot is interactive in notebooks; here we also save a static snapshot.
    out_png = f'../outputs/continuous_subject_{subject}_{start}s-{start+15}s.png'
    fig = raw.plot(start=start, duration=start+15, n_channels=len(raw.info['ch_names']),
                   scalings='auto', show=False, title=f'Sub-{subject} continuous ({start}s–{start+15}s)')

    fig.set_size_inches(18, 10)

    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"Saved snapshot to {out_png}")





