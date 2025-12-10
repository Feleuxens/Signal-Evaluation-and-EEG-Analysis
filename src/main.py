from time import sleep
from utils import get_subjectlist
from analyze_subject import analyze_subject


# path where to save the datasets.
bids_root = "../data/"
# subject_id = "001"

subject_id = "003"


subjects = get_subjectlist(bids_root)

print(f"Subjects: {subjects}")

sleep(2)

analyze_subject(subject_id, bids_root)
