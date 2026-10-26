"""HR, WER and FRR (Sec. 4.2), with per-utterance records so trade-offs can be recomputed offline."""
import json

import jiwer
import torch
from whisper.normalizers import EnglishTextNormalizer

normalize = EnglishTextNormalizer()


def _errors(ref, hyp):
    m = jiwer.process_words(ref, hyp)
    return m.substitutions + m.deletions + m.insertions


def records(results, items):
    """One JSON-able dict per input. Speech inputs also get word-error counts for the final
    text and for the raw (unfiltered) text, so WER at any other tau can be recomputed."""
    out = []
    for r, (path, ref) in zip(results, items):
        rec = dict(path=path, ref=ref, **vars(r))
        if ref:
            ref_n = normalize(ref)
            rec.update(errors=_errors(ref_n, normalize(r.text)),
                       raw_errors=_errors(ref_n, normalize(r.raw_text)),
                       ref_words=len(ref_n.split()))
        out.append(rec)
    return out


def summarize(recs):
    """Non-speech set -> HR (%). Speech set -> WER (%) and FRR (%), plus WER on accepted only."""
    n = len(recs)
    if not recs[0]["ref"]:
        hall = [bool(r["text"]) for r in recs]
        return {"n": n, "HR": 100 * sum(hall) / n, "HR_ci95": bootstrap_ci(hall)}
    acc = [r for r in recs if not r["rejected"]]
    wer = lambda rs: 100 * sum(r["errors"] for r in rs) / max(1, sum(r["ref_words"] for r in rs))
    rej = [r["rejected"] for r in recs]
    return {"n": n, "WER": wer(recs), "FRR": 100 * sum(rej) / n, "FRR_ci95": bootstrap_ci(rej),
            "WER_accepted": wer(acc)}  # analysis A2: rejection vs. mis-transcription


def bootstrap_ci(values, n_boot=1000, seed=0):
    """95% bootstrap interval for the mean of a 0/1 (or real) vector, in %."""
    v = torch.as_tensor(values, dtype=torch.float)
    g = torch.Generator().manual_seed(seed)
    means = v[torch.randint(len(v), (n_boot, len(v)), generator=g)].mean(1)
    return [100 * means.quantile(q).item() for q in (0.025, 0.975)]


def save_jsonl(recs, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]
