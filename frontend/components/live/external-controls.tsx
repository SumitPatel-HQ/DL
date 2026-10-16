"use client";

import { useMemo, useState } from "react";
import { API_BASE } from "../../lib/api.ts";
import type { ExternalState, NegativeReport } from "../../lib/live/session.ts";
import type { Action } from "../../lib/live/stage.ts";

const STORAGE_KEY = "kws_external_action_settings";

type SavedSettings = {
  targetKind: string;
  targetId?: string;
  actions: Action[];
  acknowledged: boolean;
};

function loadSavedSettings(): SavedSettings | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as SavedSettings) : null;
  } catch {
    return null;
  }
}

function saveSettings(settings: SavedSettings | null) {
  if (typeof window === "undefined") return;
  try {
    if (settings) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    // Ignore quota or security errors
  }
}

export function ExternalControls({ state, report, send, pending }: {
  state: ExternalState;
  report: NegativeReport | null;
  send: (control: Record<string, unknown>) => void;
  pending: boolean;
}) {
  const saved = useMemo(() => loadSavedSettings(), []);

  const [targetIdOverride, setTargetIdOverride] = useState<string>("");
  const [actionsOverride, setActionsOverride] = useState<Action[] | null>(null);
  const [acknowledgedOverride, setAcknowledgedOverride] = useState<boolean | null>(null);
  const [noKeywords, setNoKeywords] = useState(false);

  // Derive effective target:
  const target = useMemo(() => {
    if (!state.targets.length) return undefined;
    if (targetIdOverride) {
      const match = state.targets.find((item) => item.id === targetIdOverride);
      if (match) return match;
    }
    if (state.target_id) {
      const match = state.targets.find((item) => item.id === state.target_id);
      if (match) return match;
    }
    if (saved) {
      const match = state.targets.find((item) => item.id === saved.targetId || item.kind === saved.targetKind);
      if (match) return match;
    }
    return state.targets[0];
  }, [state.targets, targetIdOverride, state.target_id, saved]);

  const effectiveTargetId = target?.id ?? "";

  // Derive effective actions:
  const effectiveActions = useMemo(() => {
    if (actionsOverride !== null) return actionsOverride;
    if (state.allowed_actions.length > 0) return state.allowed_actions;
    if (saved?.actions.length && (!target || saved.targetKind === target.kind)) return saved.actions;
    return target?.actions ? [...target.actions] : [];
  }, [actionsOverride, state.allowed_actions, saved, target]);

  // Derive effective acknowledged:
  const effectiveAcknowledged = useMemo(() => {
    if (acknowledgedOverride !== null) return acknowledgedOverride;
    if (state.enabled) return true;
    return saved?.acknowledged ?? false;
  }, [acknowledgedOverride, state.enabled, saved]);

  function submit(control: Record<string, unknown>) {
    send(control);
  }

  function handleSelectTarget(newId: string) {
    setTargetIdOverride(newId);
    const selected = state.targets.find((item) => item.id === newId);
    const newActions = selected ? [...selected.actions] : [];
    setActionsOverride(newActions);
    if (selected) {
      saveSettings({
        targetKind: selected.kind,
        targetId: selected.id,
        actions: newActions,
        acknowledged: effectiveAcknowledged,
      });
    }
  }

  function handleToggleAction(action: Action, checked: boolean) {
    const updated = checked
      ? [...effectiveActions, action]
      : effectiveActions.filter((a) => a !== action);
    setActionsOverride(updated);
    if (target) {
      saveSettings({
        targetKind: target.kind,
        targetId: target.id,
        actions: updated,
        acknowledged: effectiveAcknowledged,
      });
    }
  }

  function handleToggleAcknowledge(checked: boolean) {
    setAcknowledgedOverride(checked);
    if (target) {
      saveSettings({
        targetKind: target.kind,
        targetId: target.id,
        actions: effectiveActions,
        acknowledged: checked,
      });
    }
  }

  function handleGrant() {
    if (!target || !effectiveActions.length || !effectiveAcknowledged) return;
    saveSettings({
      targetKind: target.kind,
      targetId: target.id,
      actions: effectiveActions,
      acknowledged: true,
    });
    submit({
      type: "external_grant",
      target_id: effectiveTargetId,
      actions: effectiveActions,
      acknowledge_experimental: true,
    });
  }

  function handleRevoke() {
    submit({ type: "external_revoke" });
  }

  function downloadReport() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `negative-session-${report?.id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="card" aria-labelledby="external-title" style={{ padding: 20, display: "grid", gap: 14 }}>
      <p>
        OS control is off by default. The event-F1 target is unmet and the 30-minute live false-action gate
        has not passed. Opting in can cause accidental actions on this computer. Keyboard and click controls stay in the demo.
      </p>
      {!state.supported ? (
        <p>External actions require a Windows backend.</p>
      ) : (
        <>
          <p role="status">
            {state.enabled
              ? `Enabled for ${state.targets.find((item) => item.id === state.target_id)?.label ?? "selected target"}: ${state.allowed_actions.join(", ")}. Other voice actions are rejected.`
              : "External actions off. Voice actions control the in-app demo."}
          </p>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <button
              type="button"
              className="btn"
              disabled={pending}
              onClick={() => submit({ type: "external_targets" })}
            >
              Refresh targets
            </button>
            {state.enabled ? (
              <button
                type="button"
                className="btn solid-danger"
                disabled={pending}
                onClick={handleRevoke}
              >
                Revoke OS control
              </button>
            ) : null}
          </div>
          {state.targets.length ? (
            <>
              <label>
                Target on this computer{" "}
                <select value={target?.id ?? ""} onChange={(event) => handleSelectTarget(event.target.value)}>
                  <option value="">Select a target</option>
                  {state.targets.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
              {target ? (
                <fieldset style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
                  <legend>Permit each action explicitly</legend>
                  {target.actions.map((action) => (
                    <label key={action}>
                      <input
                        type="checkbox"
                        checked={effectiveActions.includes(action)}
                        onChange={(event) => handleToggleAction(action, event.target.checked)}
                      />{" "}
                      {action.replaceAll("_", " ")}
                    </label>
                  ))}
                </fieldset>
              ) : null}
              <label>
                <input
                  type="checkbox"
                  checked={effectiveAcknowledged}
                  onChange={(event) => handleToggleAcknowledge(event.target.checked)}
                />{" "}
                I understand the unmet quality gates and consent to experimental control of this target.
              </label>
              <button
                type="button"
                className="btn primary"
                disabled={pending || !target || !effectiveActions.length || !effectiveAcknowledged}
                onClick={handleGrant}
              >
                {state.enabled ? "Update active OS actions" : "Enable selected OS actions"}
              </button>
            </>
          ) : (
            <p>Refresh after opening a PowerPoint slideshow or starting a Windows media session. No application is launched by this feature.</p>
          )}
          {state.error ? (
            <p role="alert">
              {state.error === "executor_unavailable"
                ? "Windows target discovery or execution failed. Permissions were revoked. Refresh targets to try again."
                : "Permission request rejected. Check the target, selected actions, and experimental consent."}
            </p>
          ) : null}
          {state.enabled ? (
            <div style={{ display: "grid", gap: 10 }}>
              <h4>Controlled negative test</h4>
              <p>
                Keep this tab visible for at least 30 minutes of scored microphone audio. Use unrelated speech, noise and Silence;
                no trained Keywords. Every accepted Detection counts conservatively as a false action. Stop or Revoke ends the test as not passed.
              </p>
              <label>
                <input
                  type="checkbox"
                  checked={noKeywords}
                  onChange={(event) => setNoKeywords(event.target.checked)}
                />{" "}
                I confirm the test audio contains no trained Keywords.
              </label>
              {state.negative_running ? (
                <button
                  type="button"
                  className="btn"
                  onClick={() => submit({ type: "negative_finish" })}
                >
                  Finish and save negative report
                </button>
              ) : (
                <button
                  type="button"
                  className="btn"
                  disabled={pending || !noKeywords}
                  onClick={() => submit({ type: "negative_start", no_keywords_confirmed: true })}
                >
                  Start controlled negative test
                </button>
              )}
            </div>
          ) : null}
        </>
      )}
      {report ? (
        <div role="status">
          <p>Negative exposure: {report.negative_exposure_s.toFixed(1)} s. False actions: {report.false_actions}. Verdict: {report.verdict}.</p>
          <button type="button" className="btn" onClick={downloadReport}>Download report JSON</button>
        </div>
      ) : null}
      <p>
        Stop, disconnect, hiding this tab or Revoke clears live permissions. An already-dispatched action may still complete and cannot be undone by revocation.
        A timed-out native action has an unknown outcome and is never retried automatically.
      </p>
      <p>
        <a href={`${API_BASE}/api/external-actions/logs`} target="_blank" rel="noreferrer">Stored action audit</a>{" · "}
        <a href={`${API_BASE}/api/external-actions/negative-reports`} target="_blank" rel="noreferrer">Stored negative reports</a>
      </p>
    </section>
  );
}
