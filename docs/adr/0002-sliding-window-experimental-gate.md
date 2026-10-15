# Sliding-window continuous spotting is experimental with a go/no-go gate

Continuous Detections reuse the isolated-word model over overlapping ~1.0s windows with threshold, smoothing, and merging. Window size, stride, thresholds, and ±0.5s match tolerance are validation-tuned parameters, not guarantees. Isolated-word accuracy does not imply continuous-event performance; Phase 2 decides whether to continue, tune once, or prototype LSTM+CTC separately.
