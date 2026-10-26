"""Dataset manifests (Sec. 4.1). Every loader returns [(audio_path, reference_text), ...].
Non-speech clips have reference "", so any emitted text is a hallucination."""
import csv
from pathlib import Path

import torch
import whisper


def esc50(root, folds):
    """ESC-50 (github.com/karolpiczak/ESC-50). Calibration: folds 1-3. Test: folds 4-5."""
    rows = csv.DictReader(open(Path(root, "meta", "esc50.csv")))
    return [(str(Path(root, "audio", r["filename"])), "") for r in rows if int(r["fold"]) in folds]


def urbansound8k(root):
    """UrbanSound8K, all 10 folds (8,732 clips)."""
    rows = csv.DictReader(open(Path(root, "metadata", "UrbanSound8K.csv")))
    return [(str(Path(root, "audio", f"fold{r['fold']}", r["slice_file_name"])), "") for r in rows]


# FSD50K labels are propagated up the AudioSet ontology, so these parents cover every
# speech / vocal / music child label (Male_speech..., Singing, Guitar, ...).
FSD_EXCLUDE = {"Speech", "Human_voice", "Singing", "Music", "Musical_instrument"}


def fsd50k_nonspeech(root, split="eval"):
    """FSD50K with speech, vocal and music clips removed (paper reports 8,621 clips)."""
    rows = csv.DictReader(open(Path(root, "FSD50K.ground_truth", f"{split}.csv")))
    keep = [r for r in rows if not FSD_EXCLUDE & set(r["labels"].split(","))]
    return [(str(Path(root, f"FSD50K.{split}_audio", f"{r['fname']}.wav")), "") for r in keep]


def librispeech(root, subset, limit=None):
    """LibriSpeech in the OpenSLR layout: root/<subset>/<spk>/<chapter>/*.trans.txt."""
    items = []
    for trans in sorted(Path(root, subset).rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            items.append((str(trans.parent / f"{uid}.flac"), text))
    return items[:limit]


def load_mels(paths, n_mels):
    """Decode with ffmpeg, pad/trim to 30 s, log-Mel (80 bins; 128 for large-v3)."""
    audio = [whisper.pad_or_trim(whisper.load_audio(p)) for p in paths]
    return torch.stack([whisper.log_mel_spectrogram(a, n_mels) for a in audio])
