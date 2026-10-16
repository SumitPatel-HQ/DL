# In-app allowlisted actions only for P0

P2 scope superseded by [ADR-0009](0009-experimental-opt-in-os-actions.md).
The original P0 decision below remains the default behavior.

Accepted Detections map strictly to predefined in-app demo actions with cooldown and rejection logging. No shell execution, arbitrary app control, or OS volume/PowerPoint integration in P0; those require explicit opt-in and a separate decision.
