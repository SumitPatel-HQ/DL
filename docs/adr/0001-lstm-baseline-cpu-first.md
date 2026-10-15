# PyTorch LSTM baseline, CPU-first, no cloud inference

P0 uses a compact MFCC to 1-2 layer LSTM classifier trained with cross-entropy and Adam on CPU, with local `.pt` weights. P0 vocabulary is 8-class (`left/right/up/down/go/stop/unknown/silence`) but kept configurable for later words like Next/Play. No external transcription API or GPU dependency.
