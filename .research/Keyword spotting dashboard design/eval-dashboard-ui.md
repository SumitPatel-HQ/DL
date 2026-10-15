# Keyword Spotting Dashboard Design

## How do best ML experiment UIs display training/validation loss/accuracy curves, confusion matrices, per-class metrics so students can explain overfitting and errors beyond averages?

### Takeaway
Best experiment UIs pair synchronized train/val curves (loss + accuracy) with a normalized confusion matrix and a per-class precision/recall/F1 table so overfitting (diverging loss) and weak classes (Unknown/Silence) are visible without relying on single accuracy.

### Cited Findings
- TensorBoard tracks experiment metrics like loss and accuracy via `tf.keras.callbacks.TensorBoard` and renders them as Scalars/Time-Series that change every epoch, plus histogram/distribution plugins for weights/gradients — [Source](https://www.tensorflow.org/tensorboard/get_started)
- MLflow autologging logs training and validation metrics including loss and any `model.compile()` metrics per epoch via `mlflow.tensorflow.autolog()` or `MlflowCallback`, visualizable as time-series run-metric plots — [Source](https://mlflow.org/docs/latest/deep-learning/tensorflow/guide/)
- MLflow `mlflow.log_metrics({...}, step=epoch)` creates a consistent x-axis for all curves so loss curves, accuracy trends and divergence patterns persist as run history after ephemeral logs disappear — [Source](https://theneuralbase.com/mlflow/learn/intermediate/logging-training-curves/)
- Overfitting pattern is defined as training loss keeps falling while validation loss rises, watched in training and validation curves in MLflow or CloudWatch — [Source](https://www.mindmeshacademy.com/certifications/aws/aws-certified-machine-learning-engineer-associate/study-guide/3-3-3-debugging-model-convergence-mlflow-tensorboard-and-cloudwatch)
- W&B `wandb.plot.confusion_matrix(y_true, preds, class_names)` logs a multi-class confusion matrix in one line to Custom Charts, with hover for exact per-cell counts — [Source](https://docs.wandb.ai/models/ref/python/custom-charts/confusion_matrix)
- W&B confusion-matrix chart supports toggling runs via eye icon, filtering to a subset of classes via gear icon, and normalizing counts to probabilities per cell for comparing across dataset sizes — [Source](https://wandb.ai/wandb/plots/reports/Confusion-Matrix-Usage-and-Examples--VmlldzozMDg1NTM)
- W&B provides `confusion_matrix()`, `roc_curve()`, `pr_curve()`, `line()`, `line_series()`, `bar()`, `histogram()` helpers covering classification (confusion/PR/ROC), regression and Vega-Lite custom charts — [Source](https://docs.wandb.ai/models/ref/python/custom-charts)
- W&B model-evaluation presets create PR curves via `wandb.plot.pr_curve(ground_truth, predictions)` and ROC via `wandb.plot.roc_curve(ground_truth, predictions)` in one line — [Source](https://docs.wandb.ai/guides/track/log/plots/)
- HF Evaluate defines Metric (compare predictions to ground truth), Comparison (compare models) and Measurement (dataset/prediction properties) as complementary evaluation types with per-metric Spaces and limitation cards — [Source](https://huggingface.co/docs/evaluate/en/index)
- HF `evaluate` added a `confusion_matrix` metric returning the raw matrix array, with multilabel support via `multilabel_confusion_matrix` (per-class 2x2 arrays) — [Source](https://github.com/huggingface/evaluate/pull/528); [Source](https://github.com/huggingface/evaluate/pull/533)
- Practitioner guidance: confusion-matrix rows = actual class, columns = predicted class, cell ink = row-normalized share (“of actual X, where did predictions go?”), diagonal accented by inset stroke not color, always show one-line key next to every example — [Source](https://microcharts.dev/docs/charts/confusion-grid)
- Confusion-matrix heatmap guidance: label axes “True Label” (y) vs “Predicted Label” (x), annotate cells with counts/percentages, support normalization modes none / by-row (recall) / by-column (precision) / total, use sequential colormap with colorbar — [Source](https://anyplot.ai/confusion-matrix)
- Recharts 3.x adds `HeatMap` + `HeatMapChart` cartesian primitives with tooltip, legend, labels, clipping, animation and z-index support, closing long-standing heatmap gap (issue #237) — [Source](https://github.com/recharts/recharts/pull/7225)
- Recharts `LineChart` supports multi-series lines, `CartesianGrid`, `Tooltip`, `Legend`, `ReferenceLine/ReferenceArea`, `Brush` zoom and `syncId` synchronized tooltips across charts — [Source](https://recharts.github.io/en-US/api/LineChart/)
- Nivo `HeatMap` (`@nivo/heatmap`) offers SVG/canvas/HTTP-API renderers, selectable color scales, rect/circle/custom cell shapes and custom tooltips — [Source](https://nivo.rocks/heatmap/)

### Inferences
- For the student dashboard: one synchronized dual-line chart (train vs val loss) + second chart (train vs val accuracy/F1) with shared epoch x-axis and best-epoch marker directly teaches the E02 overfit signature (train acc 0.97 at epoch 20 vs val acc 0.67 at best epoch).
- Confusion matrix must default to row-normalized (recall view) with toggle to raw counts, because Unknown-as-weak-class (E02 test F1 0.46) only shows when “predicted as Unknown vs predicted as keyword” distribution is visible; always co-display per-class precision/recall/F1 + support table beside the matrix.
- For imbalanced keyword spotting with dominant Unknown, PR curves are more honest than ROC; keep ROC out of default view to avoid optimistic reading.

### Gaps
- No reliable source found on exact per-class table UX ordering (worst-F1-first vs alphabetical) in W&B/MLflow/TensorBoard; needs design judgment.
- No reliable source found on calibration-curve patterns specific to small KWS models; only general PR/ROC/confidence-histogram guidance was located.

## What patterns keep isolated classification metrics visually/logically separate from continuous-event detection metrics (precision/recall/F1, FA/hour) to stay honest about sliding-window limits?

### Takeaway
Keyword-spotting literature evaluates isolated accuracy separately from detection operating points (DET curves, FRR@FAR, FA/hour); dashboards should enforce this with separate tabs/sections, distinct units, and frozen detector parameters so a 0.74 isolated F1 never implies a working 0.007 event-F1 detector.

### Cited Findings
- State-of-the-art KWS models report 99.34–99.72% accuracy on Google Speech Commands yet suffer 103.6–902.1 false alarms/hour on real-world negative audio, proving accuracy alone cannot benchmark robustness — [Source](https://doi.org/10.3233/faia230695)
- KWS detection-task metrics of interest are False Alarm Rate and False Rejection Rate at given operating points shown on a DET curve by sweeping threshold, plus EER where FAR=FRR and FRR@fixed-FAR (e.g., detection rate at 5 false alarms) — [Source](https://arxiv.org/html/2410.16647)
- Production KWS comparison fixes FA/h (e.g., FA/h=0.1) then compares FRR, because FRR and FA/h are not independent and threshold trades one for the other — [Source](https://www.isca-archive.org/interspeech_2025/saladukha25_interspeech.pdf)
- Word-spotting usability requires confidence-score quality measures beyond ROC: threshold-setting utility (KS statistic on miss distributions) vs result-confidence utility (Bayesian posterior of hit given score) — [Source](https://doi.org/10.3115/1626307.1626311)
- Standard KWS evaluation plots ROC of false positives vs true positives/false negatives per keyword, with Figure-of-Merit defined as average detection rate from 0–10 FA/keyword/hour (FOMq 0–2.5 FA/kw/h for low-FA region) — [Source](https://www.wseas.us/e-library/conferences/2009/budapest/MIV-SSIP/MIV-SSIP07.pdf)
- Successive-refinement KWS work frames the task as continuous audio-stream processing where false alarm means triggering when keyword not uttered, reducible by staged speech→keyword-like→keyword classifiers — [Source](https://doi.org/10.1109/icassp49357.2023.10095428)

### Inferences
- Dashboard must use two top-level sections/tabs — “Isolated classification (1-s clips)” vs “Continuous detection (experimental)” — with different metric vocabularies (macro-F1/accuracy vs event-P/R/F1 + FA/hour) and a persistent banner on the continuous tab linking the frozen detector JSON and test run.
- Continuous tab should show DET-style threshold sweep from validation tuning, mark the frozen operating point (stride 0.2 s, threshold 0.50, min_consecutive 1), and state that most false detections score ≥0.85 so threshold cannot fix them — preventing “just raise threshold” misreading.
- Unlabeled audio must show Detections with Confidence only (score histogram + list with timestamps), never event precision/recall, matching the confidence-score framing in word-spotting literature.

### Gaps
- No reliable source found on a standard React component pattern for side-by-side isolated-vs-streaming views; separation is inferred from metric literature, not from an existing dashboard teardown.
- I found no reliable sources on Comet/Neptune continuous-event dashboard patterns within the tool-call budget; W&B/MLflow/TensorBoard/HF coverage above is the basis.

## How to design run-based provenance: run selector, checkpoint hash + dataset/split + config display, model switching / run deletion updating views, reproducibility links?

### Takeaway
MLflow/W&B/HF model-card conventions converge on: run as primary key (params + metrics time-series + artifacts), immutable hashes/IDs for checkpoint/dataset/config, compare-runs view for model switching, and explicit tombstoning on deletion so views never silently show stale numbers.

### Cited Findings
- MLflow Tracking is organized around runs (single `python train.py` execution); each run records metrics, parameters, start/end times plus artifacts (weights, images, data files), with `artifact_uri`/`artifact_location` for retrieval — [Source](https://mlflow.org/docs/latest/ml/tracking/)
- MLflow 3 supports multiple model checkpoints per run with unique identifiers, direct `models:/<model_id>` URIs, and links between models, metrics and datasets for traceability — [Source](https://mlflow.org/docs/latest/ml/tracking/)
- MLflow compare-runs view: select 2+ runs via checkboxes → Compare button → side-by-side parameter table (identical params hidden), metric tables, parallel-coordinates plot and shared artifacts section — [Source](https://apxml.com/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow/comparing-mlflow-runs); [Source](https://github.com/mlflow/mlflow/pull/13145)
- MLflow `search_runs()` with filter string (e.g., `metrics.accuracy > 0.9`) plus `get_metric_history()` for full curves vs last-value columns enables programmatic run selection and best-run lookup — [Source](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments-mlflow?view=azureml-api-2)
- Deleted MLflow runs are not auto-purged from backend/artifact stores; `mlflow gc` CLI permanently removes metadata + artifacts, implying UI must handle deleted-run references explicitly — [Source](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/)
- Provenance research finds MLflow native provenance rudimentary (incomplete traces, coarse granularity); MLflow2PROV extracts W3C-PROV graphs from Git + MLflow (commit ID, branch, repo URL) to answer “who created best-accuracy model, when, from what code/dataset” queries — [Source](https://doi.org/10.1016/j.is.2024.102495); [Source](https://dl.acm.org/doi/10.1007/978-3-031-42914-9_17)
- HF model cards require model description, intended uses/limitations, training params, datasets used and evaluation results, with YAML front-matter (`datasets`, `base_model`, `library_name`, `license`, `model-index` eval results) powering discovery and Papers-with-Code leaderboard indexing — [Source](https://huggingface.co/docs/hub/model-cards)

### Inferences
- Dashboard run selector should be a dropdown/table keyed by run ID (E01/E02 + split), showing checkpoint SHA-256 (truncated + copy), dataset/split version, config hash, timestamp and dirty-flag warning; switching runs reloads every chart/table from that run’s JSON, never mixing curves from one run with metrics from another.
- Reproducibility panel should expose copy-paste `uv run` commands, manifest/normalization hashes and detector-file hash (continuous), mirroring MLflow artifact + HF model-card practice at student scale with static JSON.
- Deletion (or missing run file) must yield an explicit “Run deleted/missing — views cleared” empty state, not fallback to another run’s numbers, consistent with MLflow gc semantics.

### Gaps
- No reliable source found on best-practice copy for run-deletion empty states in ML experiment UIs; guidance is inferred from artifact-store semantics.
- No reliable source found on Papers-with-Code leaderboard critique within budget; HF model-card eval-results indexing is the closest located evidence.

## What empty-state and unmet-target designs work (Not evaluated, target ≥0.85 macro-F1 / ≥0.70 event-F1 reported as unmet, not redefined) for course evaluator trust?

### Takeaway
Honest-ML practice (model cards, eval-results metadata, per-split reporting) favors explicit “Not evaluated / target unmet” badges with raw numbers and denominators over hiding failures; continuous detection at 0.007 event-F1 / 1559 FA/hour must be labeled experimental/failing, not averaged away.

### Cited Findings
- HF model cards must document intended uses, potential limitations/biases (Mitchell 2018), training params, datasets and evaluation results — limitations are a required section, not optional — [Source](https://huggingface.co/docs/hub/model-cards)
- HF `model-index` eval metadata records task + dataset + metric name/type/value + source (e.g., Open LLM Leaderboard URL) so scores render with provenance instead of bare numbers — [Source](https://huggingface.co/docs/hub/model-cards)
- HF Evaluate metric cards include limitations/usage documentation per metric Space, framing metrics as complementary (Metric vs Comparison vs Measurement) rather than single-score verdicts — [Source](https://huggingface.co/docs/evaluate/en/index)
- KWS FAR benchmark explicitly reports both accuracy (99%+) and FAR (100–900 FA/h) side by side to prevent “great accuracy but unusable” misreading — [Source](https://doi.org/10.3233/faia230695)
- MLflow compare view hides identical parameters by default to focus on what changed, and parallel-coordinates plots surface trade-offs — a pattern for “target vs actual” focus without redefining the target — [Source](https://apxml.com/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow/comparing-mlflow-runs)
- Deleted-run artifacts require explicit `gc`; until purged, stale files persist — implying empty states must distinguish “never evaluated” from “deleted” — [Source](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/)

### Inferences
- Use three distinct states: (a) “Not evaluated” (grey, no run file for that split), (b) “Target unmet” (amber/red badge showing e.g., macro-F1 0.740 vs target ≥0.85, event-F1 0.007 vs ≥0.70, 1559 FA/h), (c) “Experimental only” banner on continuous tab stating detector is not a working search feature.
- Targets must render as fixed reference lines/bands on charts (e.g., 0.85 line on F1 bar, 0.70 line on event-F1) with numeric delta (“−0.11”, “−0.693”), never rescaling axes to make gaps look small.
- Course-evaluator trust comes from showing denominators and exposure (1017 test clips; 16 keyword events, 561 s Unknown + 8.8 s Silence, 600 s test audio) next to every headline metric so small-sample anecdote risk is explicit.

### Gaps
- No reliable source found with controlled studies on which empty-state copy (“Not evaluated” vs “No data”) maximizes evaluator trust; recommendation is inferred from model-card limitation-reporting norms.
- No reliable source found on Tremor/Plotly-React-specific target-badge components within budget; Recharts ReferenceLine + custom badge (above) is the implementable substitute.
