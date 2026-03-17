import asrpy
from mne.io.edf.edf import RawEDF

from utils.config import StepASR


def run_asr(raw: RawEDF, config: StepASR) -> tuple[RawEDF, asrpy.ASR | None]:
    """Apply ASR artifact correction to the raw data."""

    asr = None

    try:
        asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=config.cutoff)
        asr.fit(raw)
        raw = asr.transform(raw)

    except Exception as e:
        print(f"ASR failed: {e}. Continuing without ASR.")

    # finally:
    return raw, asr
