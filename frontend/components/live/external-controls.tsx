"use client";

import { useState } from "react";
import { API_BASE } from "../../lib/api.ts";
import type { ExternalState, NegativeReport } from "../../lib/live/session.ts";
import type { Action } from "../../lib/live/stage.ts";

export function ExternalControls({ state, report, send, pending }: {
  state: ExternalState;
  report: NegativeReport | null;
  send: (control: Record<string, unknown>) => void;
  pending: boolean;
}) {
  const [targetId, setTargetId] = useState("");
  const [actions, setActions] = useState<Action[]>([]);
  const [acknowledged, setAcknowledged] = useState(false);
  const [noKeywords, setNoKeywords] = useState(false);
  const target = state.targets.find((item) => item.id === targetId);

  function submit(control: Record<string, unknown>) {
    send(control);
  }

  function downloadReport() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `negative-session-${report?.id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return <section className="card" aria-labelledby="external-title" style={{ padding: 20, display: "grid", gap: 14, marginTop: 16 }}>
    <h3 id="external-title">Optional Windows actions · experimental</h3>
    <p>OS control is off by default. The event-F1 target is unmet and the 30-minute live false-action gate
      has not passed. Opting in can cause accidental actions on this computer. Keyboard and click controls stay in the demo.</p>
    {!state.supported ? <p>External actions require a Windows backend.</p> : <>
      <p role="status">{state.enabled
        ? `Enabled for ${state.targets.find((item) => item.id === state.target_id)?.label ?? "selected target"}: ${state.allowed_actions.join(", ")}. Other voice actions are rejected.`
        : "External actions off. Voice actions control the in-app demo."}</p>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <button type="button" className="btn" disabled={pending} onClick={() => submit({ type: "external_targets" })}>
          Refresh targets and revoke permissions
        </button>
        {state.enabled ? <button type="button" className="btn solid-danger" onClick={() => send({ type: "external_revoke" })}>Revoke OS control</button> : null}
      </div>
      {state.targets.length ? <>
        <label>Target on this computer{" "}
          <select value={target?.id ?? ""} onChange={(event) => {
            setTargetId(event.target.value); setActions([]); setAcknowledged(false);
          }}>
            <option value="">Select a target</option>
            {state.targets.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
          </select>
        </label>
        {target ? <fieldset style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
          <legend>Permit each action explicitly</legend>
          {target.actions.map((action) => <label key={action}>
            <input type="checkbox" checked={actions.includes(action)} onChange={(event) =>
              setActions((current) => event.target.checked ? [...current, action] : current.filter((item) => item !== action))} />{" "}
            {action.replaceAll("_", " ")}
          </label>)}
        </fieldset> : null}
        <label><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} />{" "}
          I understand the unmet quality gates and consent to experimental control of this target for this mic session.
        </label>
        <button type="button" className="btn primary" disabled={pending || !target || !actions.length || !acknowledged}
          onClick={() => submit({ type: "external_grant", target_id: targetId, actions, acknowledge_experimental: true })}>
          Enable selected OS actions
        </button>
      </> : <p>Refresh after opening a PowerPoint slideshow or starting a Windows media session. No application is launched by this feature.</p>}
      {state.error ? <p role="alert">{state.error === "executor_unavailable"
        ? "Windows target discovery or execution failed. Permissions were revoked. Refresh targets to try again."
        : "Permission request rejected. Check the target, selected actions, and experimental consent."}</p> : null}
      {state.enabled ? <div style={{ display: "grid", gap: 10 }}>
        <h4>Controlled negative test</h4>
        <p>Keep this tab visible for at least 30 minutes of scored microphone audio. Use unrelated speech, noise and Silence;
          no trained Keywords. Every accepted Detection counts conservatively as a false action. Stop or Revoke ends the test as not passed.</p>
        <label><input type="checkbox" checked={noKeywords} onChange={(event) => setNoKeywords(event.target.checked)} />{" "}
          I confirm the test audio contains no trained Keywords.
        </label>
        {state.negative_running
          ? <button type="button" className="btn" onClick={() => submit({ type: "negative_finish" })}>Finish and save negative report</button>
          : <button type="button" className="btn" disabled={pending || !noKeywords} onClick={() => submit({ type: "negative_start", no_keywords_confirmed: true })}>Start controlled negative test</button>}
      </div> : null}
    </>}
    {report ? <div role="status">
      <p>Negative exposure: {report.negative_exposure_s.toFixed(1)} s. False actions: {report.false_actions}. Verdict: {report.verdict}.</p>
      <button type="button" className="btn" onClick={downloadReport}>Download report JSON</button>
    </div> : null}
    <p>Stop, disconnect, hiding this tab or Revoke clears permissions. An already-dispatched action may still complete and cannot be undone by revocation.
      A timed-out native action has an unknown outcome and is never retried automatically.</p>
    <p><a href={`${API_BASE}/api/external-actions/logs`} target="_blank" rel="noreferrer">Stored action audit</a>{" · "}
      <a href={`${API_BASE}/api/external-actions/negative-reports`} target="_blank" rel="noreferrer">Stored negative reports</a></p>
  </section>;
}
