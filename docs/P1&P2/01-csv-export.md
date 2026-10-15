# P1-01: CSV export (PR-14)

Parent: P1+P2 spec ([P1&P2 Spec](P1&P2%20Spec.md)), #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** CSV download for a Recording's stored Detections:
new `GET /api/recordings/{id}/detections.csv` returning header
`recording_id,keyword,start_s,end_s,confidence,model_id` with one row per
stored Detection of the latest completed analysis, plus an Audio Analysis UI
export button wired to it.

**Blocked by:** P0-06 (Analysis UI), P0-07 (acceptance report).

**Status:** ready-for-agent

- [ ] Completed analysis returns 200 CSV whose rows exactly match `GET .../detections` (same order, same values, Confidence as score)
- [ ] Empty-result analysis returns headers-only CSV with 200 (never invents a label)
- [ ] Pre-completion / failed / missing Recording return the existing JSON error shapes (no CSV body)
- [ ] Filenames with paths or command characters are rejected per existing upload rules; response omits local paths
- [ ] API test covers populated, empty, gated, and injection-filename cases
