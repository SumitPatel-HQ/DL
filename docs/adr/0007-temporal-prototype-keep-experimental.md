# Temporal prototype gate outcome: keep experimental

ADR-0006 asked for one bounded temporal prototype scored by the same frozen
protocol. The T01 frame BiLSTM was trained on speaker-disjoint AMI meetings, its
post-processing was tuned on validation only, and it ran once on the held-out AMI
IN1002 test split with the same ±0.5 s center tolerance, one-to-one match rule and
per-Keyword duplicate suppression. Test event-F1 is 0.057 (2 hits, 52 false
Detections, 14 misses) with 316 false alarms/hour in annotated Unknown/Silence. The
sliding-window baseline scored 0.010 and 2,216 false alarms/hour. The preliminary
target is event-F1 ≥ 0.70. It is unmet and is not redefined.

Decision: keep the prototype experimental. It improves on the baseline, but it hears
the same 2 of 16 Keywords. Every missed Keyword is read as background. Validation
event-F1 (0.278) is five times the test value, and the held-out dev meetings already
showed a Keyword frame F1 of 0.016. A bounded retune is not taken: no threshold,
smoothing or minimum-length setting reaches the target, and the gap needs verified
labels and far more real Keyword tokens, not parameter changes. Isolated Keyword
classification stays the validated core, and the sliding-window detector stays
experimental under ADR-0006.

The prototype ships as a separate model module, training and evaluation CLI, and
stored Evaluation Runs. It adds no serving contract: Audio Analysis and Voice Control
still use the E02 classifier. The dashboard shows the stored runs labeled
experimental, with Confidence described as a smoothed frame score and the run linked.
Any later UI must keep that wording and must not claim reliable search of natural
speech. Evidence: `docs/evaluations/temporal-v1/`.
