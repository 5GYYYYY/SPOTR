# SPOTR
[IJCAI 2026] SPOTR: Spatio-temporal Pooling One-Token Reconstruction for Universal Physiological Signal Self-supervised Learning

## Overview

Physiological signals such as **EEG**, **iEEG**, **ECG**, and **PPG** are widely used in clinical monitoring and health assessment. However, existing self-supervised learning methods for physiological signals often suffer from three practical limitations:

1. **Limited cross-dataset generalization** under heterogeneous acquisition settings.
2. **Shortcut learning** caused by temporal continuity and cross-channel redundancy in masked reconstruction.
3. **High computational cost** when Transformer models flatten spatiotemporal tokens into long sequences.

We propose **SPOTR** (**S**patio-temporal **P**ooling **O**ne-**T**oken **R**econstruction), a universal self-supervised learning framework for physiological signals. SPOTR compresses each input waveform into a **single global token** and reconstructs the signal only from this bottleneck, encouraging compact, global, and transferable representations.



## Method

SPOTR contains three main components:

1. **ST Compactor**
   Compresses the input waveform into compact temporal tokens and spatial tokens through attention-based spatiotemporal pooling.
2. **Latent Aggregator**
   Aggregates the compact spatiotemporal tokens into a single global class token.
3. **Latent Renderer**
   Reconstructs the original physiological signal from mask tokens conditioned only on the single global token.

This design forces reconstruction to rely on globally organized information instead of local visible context, reducing shortcut learning and improving representation transfer.
