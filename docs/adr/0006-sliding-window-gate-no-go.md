# Sliding-window gate outcome: no-go, temporal model as a separate prototype

The ADR-0002 gate ran on held-out natural AMI meeting speech (IN1002, 600 s,
speakers disjoint from validation and Speech Commands). The test used the E02
checkpoint and detector parameters that were frozen on validation. The result is
event-F1 0.007 (1 hit, 259 false Detections, 15 misses) and 1,559 false
alarms/hour in annotated Unknown/Silence. The preliminary target is event-F1
≥ 0.70. The best validation event-F1 over the whole tuning grid is 0.047. Most
false Detections score ≥ 0.85 inside fluent speech, so no threshold, stride or
consistency setting is a credible small improvement path. The one-bounded-retune
option is therefore not taken.

Decision: the sliding-window detector is not presented as a working continuous
spotting feature. Isolated Keyword classification stays the validated core. The
detector stays available as an explicitly experimental mode. Any UI that shows its
Detections must show Confidence only and link this measured Evaluation Run. Any
UI must not claim reliable search of natural speech. An LSTM+CTC or other temporal
model is a separately scoped prototype. It is outside P0 delivery and must be
evaluated with the same frozen protocol on the same test split. The 0.70 target
is reported as unmet, not redefined. Evidence: `docs/evaluations/continuous-v1/`.
