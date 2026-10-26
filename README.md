# Reducing Hallucinations in Whisper

An implementation of **hallucination space projection** for OpenAI's Whisper speech recognition model, built as a course project for Speech and Audio Processing.

> Paper: [*Reducing Hallucinated Transcripts in Whisper via Hallucination Space Projection*](https://arxiv.org/abs/2609.04561) (Abbasihafshejani and Jadliwala, arXiv:2609.04561, 2026)

## Introduction

Whisper is a widely used foundation model for automatic speech recognition (ASR). Because its decoder is generative, it can produce fluent but entirely made-up text when the input contains little or no speech, such as silence, environmental noise or background sounds. Typical outputs are short phrases like "thank you", "you" or "thanks for watching". Downstream systems may treat these hallucinated transcripts as real speech.

For audio with no speech, the correct transcript is empty, so any text Whisper emits is a hallucination.

## What this project is about

This project re-implements the method from the paper above and studies how well it reduces non-speech hallucinations in Whisper, and what it costs on real speech. The goal is to understand the method, reproduce its evaluation, and examine the trade-off it introduces between three quantities:

- **HR** (hallucination rate): the share of non-speech clips for which Whisper still outputs text.
- **WER** (word error rate): transcription quality on genuine speech.
- **FRR** (false rejection rate): the share of genuine speech clips wrongly rejected as non-speech.

The evaluation follows the paper's setup. Non-speech audio comes from ESC-50, UrbanSound8K and a filtered subset of FSD50K. Speech comes from LibriSpeech. The project also includes extra analyses beyond the paper, such as a gate-only ablation, confidence intervals, and sensitivity of the learned subspace to calibration choices.

## Method

The method is training-free and works at inference time. It does not change Whisper's weights, and it needs no external voice activity detector (VAD) or phrase list.

**1. Calibration.** Whisper is run on a non-speech calibration set (ESC-50 folds 1 to 3). Each clip falls into one of two groups:
- clips for which Whisper produces a hallucinated, non-empty transcript;
- clips that Whisper correctly leaves empty.

Decoder hidden states are collected at each decoder layer for both groups.

**2. Low-rank hallucination subspace.** For a chosen decoder layer, the hidden states of hallucinating clips are paired with those of correctly empty clips and subtracted, giving a difference matrix. A singular value decomposition of this matrix gives the directions along which the two groups differ most. The top *r* right singular vectors form an orthonormal basis **B** of the hallucination-associated subspace. This basis is estimated once and then kept fixed.

**3. Inference-time projection.** During decoding, the hidden state **h** at the chosen decoder layer is moved away from the subspace:

```
h' = h - alpha * (h B^T) B
```

where *alpha* sets the projection strength. This raises Whisper's own no-speech probability on non-speech input, so hallucinated outputs are rejected by the standard no-speech threshold.

**4. Two variants.**
- **Always-on:** the projection is applied to every input. It gives the strongest hallucination reduction but also hurts real speech the most.
- **Gated:** the projection is applied only when Whisper's unprojected no-speech probability is at least a gate threshold *gamma*. It keeps most of the hallucination reduction with far fewer false rejections of real speech.

The final decision is the usual Whisper rule: an output is dropped as non-speech when its no-speech probability exceeds the threshold *tau* (0.6 by default).

The method has four tunable parameters: the decoder layer, the projection rank *r*, the strength *alpha* and the gate threshold *gamma*. The paper selects them separately for each Whisper size (small, medium, large-v3).

## Reference

Maryam Abbasihafshejani and Murtuza Jadliwala. *Reducing Hallucinated Transcripts in Whisper via Hallucination Space Projection.* arXiv:2609.04561, 2026. https://arxiv.org/abs/2609.04561
