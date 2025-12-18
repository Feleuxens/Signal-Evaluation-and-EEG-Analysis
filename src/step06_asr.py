import asrpy
from mne.io.edf.edf import RawEDF


ASR_CUTOFF = 10  # ASR cutoff (lower = more aggressive, 5-20 typical)


def run_asr(raw: RawEDF) -> tuple[RawEDF, asrpy.ASR | None]:
    """Apply ASR artifact correction to the raw data."""

    asr = None

    try:
        asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=ASR_CUTOFF)
        asr.fit(raw)
        raw = asr.transform(raw)

    except Exception as e:
        print(f"ASR failed: {e}. Continuing without ASR.")

    finally:
        return raw, asr
