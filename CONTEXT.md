# Keyword Spotting

Local keyword-spotting system that finds a small trained vocabulary in recorded audio and live microphone input using an LSTM classifier.

## Language

**Keyword**:
A trained word in the P0 vocabulary that the model can detect.
_Avoid_: command, instruction, transcription

**Unknown**:
Speech that is not a Keyword or Silence, used to reject unrelated words.
_Avoid_: other, background speech

**Silence**:
Non-speech audio including quiet, noise, and music without keywords.
_Avoid_: background, empty

**Detection**:
A predicted Keyword occurrence with approximate start/end times and a score.
_Avoid_: prediction, match, recognition

**Confidence**:
Per-Detection model score, not a measure of correctness.
_Avoid_: accuracy, precision, probability

**Evaluation Run**:
Labeled measurement of model quality with fixed checkpoint, dataset split, and config.
_Avoid_: test, benchmark, score report
