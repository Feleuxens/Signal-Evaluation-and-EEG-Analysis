from utils import get_subjectlist
from analyze_subject import analyze_subject


# path where to save the datasets.
bids_root = "../data/"
subject_id = '001'


get_subjectlist(bids_root)

analyze_subject(subject_id, bids_root)

exit()

# INFO: Bellow are old runs
bids_path = BIDSPath(subject=subject_id,datatype='eeg', suffix='eeg',task='jacobsen',root=bids_root)

# read the file
raw = read_raw_bids(bids_path)

print(raw.info['ch_names'])


# # Plot overview of first 30 seconds of data
# # get data for first 30 s
# start, duration = 0.0, 30.0
# data, times = raw.get_data(return_times=True, start=int(start*raw.info['sfreq']),
#                            stop=int((start+duration)*raw.info['sfreq']))
# ch_names = raw.info['ch_names']
#
# n_ch = data.shape[0]
# cols = 4
# rows = (n_ch + cols - 1) // cols
# fig, axes = plt.subplots(rows, cols, figsize=(12, rows*1.5), sharex=True)
# axes = axes.flatten()
# for i in range(n_ch):
#     axes[i].plot(times, data[i] * 1e6)  # convert to µV if data in V
#     axes[i].set_title(ch_names[i], fontsize=8)
#     axes[i].tick_params(labelsize=6)
# for ax in axes[n_ch:]:
#     ax.axis('off')
# fig.tight_layout()
# fig.savefig('outputs/eeg_overview_grid_30s.png', dpi=200, bbox_inches='tight')

from annotations import load_and_attach_annotations

load_and_attach_annotations(bids_path, raw)

# def load_events_from_bids(bids_path):
#     """Load events from BIDS *events.tsv file."""
#     events_tsv = bids_path.copy().update(suffix='events', extension='.tsv').fpath
#     df = pd.read_csv(events_tsv, sep='\t')
#
#     # BIDS events.tsv has onset (s), duration (s), and trial_type or value
#     onset = df['onset'].to_list()
#     duration = df['duration'].fillna(0).to_list() if 'duration' in df.columns else [0.0]*len(onset)
#     if 'trial_type' in df.columns:
#         description = df['trial_type'].astype(str).to_list()
#     elif 'value' in df.columns:
#         description = df['value'].astype(str).to_list()
#     else:
#         description = ['event'] * len(onset)
#
#     annot = Annotations(onset=onset, duration=duration, description=description)
#
#     return annot
#
#
# annotations = load_events_from_bids(bids_path)
# raw.set_annotations(annotations)
#
# Convert annotations to events if needed:
events, event_id = mne.events_from_annotations(raw)

print(f"Found {len(raw.annotations)} annotations.")

# fix the annotations readin
# ccs_eeg_utils.read_annotations_core(bids_path,raw)
#
print(f"Sampling Frequency: {raw.info['sfreq']} Hz")
#
#
# # Plot the channel 1
# print("Plotting channel 1")
# plt.plot(raw[1,:][0].T)
# plt.show()


# Epoching

# raw_subselect = raw.copy().pick(['Cz'])
#
# # print(raw_subselect.annotations)
#
# evts,evts_dict = mne.events_from_annotations(raw)
#
# print(evts)
# print(evts_dict)
#
# # get all keys which contain "stimulus"
# wanted_keys = [e for e in evts_dict.keys() if "stimulus" in e]
# # subset the large event-dictionairy
# evts_dict_stim=dict((k, evts_dict[k]) for k in wanted_keys if k in evts_dict)
#
# epochs = mne.Epochs(raw, evts, evts_dict_stim, tmin=-0.1, tmax=1)
#
# data = epochs.get_data()
#
# # plt.plot(data[1,0,:].T)
# # plt.title(f"Subject {subject_id} - Channel Cz - Epochs for all stimuli")
# # plt.show()
# #
#
# # My first ERP
#
# target = ["stimulus:{}{}".format(k,k) for k in [1,2,3,4,5]]
# distractor = ["stimulus:{}{}".format(k,j) for k in [1,2,3,4,5] for j in [1,2,3,4,5] if k!=j]
#
# # evoked = epochs[1].average()
# # evoked.plot()
# target_evoked = []
# for t in target:
#     target_evoked.append(epochs[t].average())
#
# distractor_evoked = []
# for d in distractor:
#     distractor_evoked.append(epochs[d].average())
#
# # mne.viz.plot_compare_evokeds(target_evoked)
#
# # There are to many distractor conditions to plot them all at once
# mne.viz.plot_compare_evokeds(distractor_evoked[:10])
