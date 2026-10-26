"""Training-free baselines (Sec. 4.3 / 5.4): Bag of Hallucinations and an external VAD gate."""
from collections import Counter
from dataclasses import replace

import torch
import whisper

from .metrics import normalize


def build_bag(hallucinated_texts, top_k=11, min_count=2):
    """BoH (Baranski et al., 2025) rebuilt as the paper did: the most frequent normalised
    hallucinations on the ESC-50 calibration folds."""
    counts = Counter(t for t in map(normalize, hallucinated_texts) if t)
    return [t for t, c in counts.most_common(top_k) if c >= min_count]


def apply_bag(results, bag):
    """Remove outputs that exactly match a bag phrase after normalisation."""
    bag = set(bag)
    return [replace(r, text="") if normalize(r.text) in bag else r for r in results]


def apply_vad(results, paths, threshold=0.5):
    """External-VAD baseline with Silero VAD: clips with no detected speech become empty.
    Simplification: WhisperX also re-segments speech before decoding; this only gates."""
    from silero_vad import get_speech_timestamps, load_silero_vad
    vad = load_silero_vad()
    out = []
    for r, p in zip(results, paths):
        wav = torch.from_numpy(whisper.load_audio(p))
        speech = get_speech_timestamps(wav, vad, threshold=threshold, sampling_rate=16000)
        out.append(r if speech else replace(r, text="", rejected=True))
    return out
