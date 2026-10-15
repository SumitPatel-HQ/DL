# P2-02: Opt-in OS volume + external media actions (PR-16 + PR-20, gated)

Parent: P1+P2 spec ([P1&P2 Spec](P1&P2%20Spec.md)), #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** Optional, off-by-default Windows volume control and external
PowerPoint/media actions on the same Windows machine: explicit user opt-in,
explicit target + action permission, one isolated action-execution seam
separate from inference, full accept/reject/action logging. Extends the
existing upload/event injection rejection to the new seam (no executable paths
or commands cross the API boundary).

**Blocked by:** P1-04 gate passed (event-F1 + ≤1 false action per 30 min
controlled negative testing) AND a new ADR superseding ADR-0004 for opt-in OS
control. Default recommendation: `wontfix` for the course deliverable if gates
are unmet, with written rationale.

**Status:** ready-for-agent (gated; do not start until blockers close)

- [ ] Microphone and OS control are both explicit opt-ins; launch state does nothing
- [ ] Only allowlisted in-app actions plus the explicitly permitted OS/media targets execute; everything else is rejected and logged
- [ ] Scripted Keyword + negative-session test records false-action counts meeting the PRD target on target hardware
- [ ] New ADR documents consent, scope, and rollback; README states limits
