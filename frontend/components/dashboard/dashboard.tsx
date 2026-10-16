"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { api, errorCode, type RunDetail, type RunKind, type RunSummary } from "../../lib/api.ts";
import { modelOptions, runsFor, type ModelOption } from "../../lib/evaluations.ts";
import { formatTimestamp, shortHash } from "../../lib/format.ts";
import { ChartIcon, DatabaseIcon, TrashIcon, XIcon } from "../icons";
import { useBackend } from "../shell/backend";
import { Banner, EmptyState, ErrorBanner } from "../ui";
import { ComparisonSection } from "./comparison";
import { NoiseComparisonSection } from "./noise";
import { ContinuousView, IsolatedView, ProvenancePanel } from "./views";

type Detail =
  | { state: "ready"; run: RunDetail }
  | { state: "error"; id: string; code: string };

type ViewMode = "single" | "compare" | "noise";

export function EvaluationDashboard({ initialTab, initialRun }: { initialTab: RunKind; initialRun: string | null }) {
  const { backend, refresh } = useBackend();
  const activeId = backend.status === "ready" ? backend.model.id : null;
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [tab, setTab] = useState<RunKind>(initialTab);
  const [mode, setMode] = useState<ViewMode>("single");
  const [checkpoint, setCheckpoint] = useState<string | null>(null);
  // undefined: follow the first run; null: nothing selected (after a deletion, never a fallback).
  const [runId, setRunId] = useState<string | null | undefined>(initialRun ?? undefined);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [deleted, setDeleted] = useState<RunSummary | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const online = backend.status !== "offline";

  useEffect(() => {
    if (!online) return;
    let cancelled = false;
    api
      .evaluations()
      .then((result) => {
        if (cancelled) return;
        setRuns(result.runs);
        setListError(null);
      })
      .catch((error: unknown) => {
        if (!cancelled) setListError(errorCode(error));
      });
    return () => {
      cancelled = true;
    };
  }, [online, reload]);

  const options = useMemo<ModelOption[]>(() => {
    const list = modelOptions(runs ?? [], activeId);
    if (activeId && backend.status === "ready" && !list.some((o) => o.checkpoint === activeId)) {
      list.unshift({ checkpoint: activeId, experiment: backend.model.version ?? "active", active: true, runs: 0 });
    }
    return list;
  }, [runs, activeId, backend]);

  const initialRunSummary = runs?.find((r) => r.id === initialRun);
  const model =
    checkpoint ?? initialRunSummary?.provenance.checkpoint_sha256 ?? options[0]?.checkpoint ?? null;
  const tabRuns = useMemo(() => runsFor(runs ?? [], tab, model), [runs, tab, model]);
  const selectedId = runId === undefined ? tabRuns[0]?.id ?? null : runId;
  const selected = tabRuns.find((r) => r.id === selectedId) ?? null;
  const shown =
    detail?.state === "ready" && detail.run.id === selectedId && detail.run.kind === tab ? detail.run : null;
  const detailError = detail?.state === "error" && detail.id === selectedId ? detail.code : null;

  useEffect(() => {
    if (!selectedId || !online) return;
    let cancelled = false;
    api
      .evaluation(selectedId)
      .then((run) => {
        if (!cancelled) setDetail({ state: "ready", run });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const code = errorCode(error);
        setDetail({ state: "error", id: selectedId, code });
        if (code === "evaluation_not_found") setRuns((list) => list?.filter((r) => r.id !== selectedId) ?? list);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId, online, reload]);

  const syncUrl = (nextTab: RunKind, nextRun: string | null) => {
    const params = new URLSearchParams();
    params.set("tab", nextTab);
    if (nextRun) params.set("run", nextRun);
    window.history.replaceState(null, "", `?${params.toString()}`);
  };

  const chooseTab = (next: RunKind) => {
    setTab(next);
    setRunId(undefined);
    setDeleted(null);
    syncUrl(next, null);
  };
  const chooseModel = (next: string) => {
    setCheckpoint(next);
    setRunId(undefined);
    setDeleted(null);
    syncUrl(tab, null);
  };
  const chooseRun = (next: string) => {
    setRunId(next);
    setDeleted(null);
    syncUrl(tab, next);
  };

  const importRuns = async () => {
    setBusy(true);
    setNotice(null);
    try {
      const result = await api.importEvaluations();
      setNotice(
        `Imported ${result.imported.length} run${result.imported.length === 1 ? "" : "s"}` +
          (result.rejected.length
            ? `; rejected ${result.rejected.map((r) => `${r.source} (${r.code})`).join(", ")}`
            : "") +
          ".",
      );
      setDeleted(null);
      setReload((n) => n + 1);
    } catch (error) {
      setListError(errorCode(error));
    } finally {
      setBusy(false);
    }
  };

  const deleteRun = async () => {
    if (!selected) return;
    setBusy(true);
    try {
      await api.deleteEvaluation(selected.id);
    } catch (error) {
      if (errorCode(error) !== "evaluation_not_found") {
        setListError(errorCode(error));
        setBusy(false);
        dialogRef.current?.close();
        return;
      }
    }
    dialogRef.current?.close();
    setRuns((list) => list?.filter((r) => r.id !== selected.id) ?? list);
    setDeleted(selected);
    setDetail(null);
    setRunId(null);
    syncUrl(tab, null);
    setBusy(false);
  };

  const activeOption = options.find((o) => o.checkpoint === model);

  return (
    <div className="page page-dashboard">
      <div className="page-head">
        <div>
          <p className="eyebrow">Evaluation</p>
          <h1>Evidence from saved Evaluation Runs</h1>
          <p>
            Every number below is read from one stored run with its checkpoint hash, dataset and split version,
            config, and timestamp. Isolated clip classification and continuous-event detection never share a view.
          </p>
        </div>
        <button type="button" className="btn" onClick={importRuns} disabled={busy || !online}>
          <DatabaseIcon size={15} /> Import committed runs
        </button>
      </div>

      <div style={{ display: "grid", gap: 12, marginBottom: 16 }}>
        {backend.status === "offline" ? (
          <ErrorBanner code="offline" handlers={{ retry: () => { refresh(); setReload((n) => n + 1); } }} />
        ) : listError ? (
          <ErrorBanner code={listError} handlers={{ retry: () => setReload((n) => n + 1) }} />
        ) : null}
        {notice ? (
          <Banner tone="info" title="Import finished" live actions={
            <button type="button" className="btn small icon ghost" aria-label="Dismiss" onClick={() => setNotice(null)}><XIcon size={14} /></button>
          }>
            {notice}
          </Banner>
        ) : null}
      </div>

      <div className="tabs" role="tablist" aria-label="Evaluation kind">
        <button type="button" role="tab" aria-selected={tab === "isolated"} onClick={() => chooseTab("isolated")}>
          Isolated classification <span className="tag">1 s clips</span>
        </button>
        <button type="button" role="tab" aria-selected={tab === "continuous"} onClick={() => chooseTab("continuous")}>
          Continuous detection <span className="tag warn">experimental</span>
        </button>
      </div>

      <div className="controls">
        <div className="control">
          <span className="label">View</span>
          <div className="seg" role="group" aria-label="Single or comparison view">
            <button type="button" aria-pressed={mode === "single"} onClick={() => setMode("single")}>
              Single run
            </button>
            <button type="button" aria-pressed={mode === "compare"} onClick={() => setMode("compare")}>
              Compare E01 vs E02
            </button>
            <button type="button" aria-pressed={mode === "noise"} onClick={() => setMode("noise")}>
              Clean vs noisy
            </button>
          </div>
        </div>
      </div>

      {mode === "compare" ? (
        <ComparisonSection
          runs={runs ?? []}
          kind={tab}
          online={online}
          onImport={importRuns}
          busy={busy}
        />
      ) : null}

      {mode === "noise" ? (
        <NoiseComparisonSection
          runs={runs ?? []}
          kind={tab}
          online={online}
          onImport={importRuns}
          busy={busy}
        />
      ) : null}

      {mode === "single" ? (
      <>
      <div className="controls">
        <div className="control">
          <span className="label">Model checkpoint</span>
          <div className="seg" role="group" aria-label="Model checkpoint">
            {options.length === 0 ? (
              <button type="button" disabled>No checkpoints with runs</button>
            ) : (
              options.map((o) => (
                <button key={o.checkpoint} type="button" aria-pressed={o.checkpoint === model} onClick={() => chooseModel(o.checkpoint)}>
                  <b>{o.experiment}</b> <span className="mono">{shortHash(o.checkpoint, 8)}</span>
                  {o.active ? <span className="tag ok">active</span> : null}
                </button>
              ))
            )}
          </div>
        </div>
        <div className="control">
          <span className="label">Evaluation Run</span>
          <div className="run-row" role="group" aria-label="Evaluation Run">
            {tabRuns.length === 0 ? (
              <span className="table-note" style={{ padding: "8px 0" }}>No {tab} runs for this checkpoint</span>
            ) : (
              tabRuns.map((r) => (
                <button key={r.id} type="button" className="run-chip" aria-pressed={r.id === selectedId} onClick={() => chooseRun(r.id)}>
                  <b>{r.provenance.split} split</b>
                  <span>{formatTimestamp(r.created_at)}</span>
                </button>
              ))
            )}
          </div>
        </div>
        <span className="shell-spacer" />
        {selected ? (
          <button type="button" className="btn danger small" onClick={() => dialogRef.current?.showModal()} disabled={busy}>
            <TrashIcon size={14} /> Delete run
          </button>
        ) : null}
      </div>

      {runs === null && !listError && online ? (
        <div style={{ display: "grid", gap: 16 }}>
          <div className="skeleton" style={{ height: 120 }} />
          <div className="skeleton" style={{ height: 320 }} />
        </div>
      ) : deleted && !selected ? (
        <section className="card state-panel">
          <EmptyState icon={<TrashIcon size={22} />} title="Run deleted · views cleared">
            <code>{deleted.id}</code> was deleted, so every chart that depended on it was cleared. Nothing falls
            back to another run. Pick a remaining run above, or re-import the committed runs.
          </EmptyState>
        </section>
      ) : detailError === "evaluation_not_found" ? (
        <section className="card state-panel">
          <EmptyState icon={<TrashIcon size={22} />} title="Run no longer exists · views cleared">
            This Evaluation Run was deleted elsewhere. No numbers are shown in its place.
          </EmptyState>
        </section>
      ) : detailError ? (
        <ErrorBanner code={detailError} handlers={{ retry: () => setReload((n) => n + 1) }} />
      ) : runs !== null && tabRuns.length === 0 ? (
        <section className="card state-panel">
          <EmptyState
            icon={<ChartIcon size={22} />}
            title="Not evaluated"
            actions={
              online ? (
                <button type="button" className="btn primary" onClick={importRuns} disabled={busy}>
                  <DatabaseIcon size={15} /> Import committed runs
                </button>
              ) : null
            }
          >
            No stored {tab === "isolated" ? "isolated classification" : "continuous-event"} Evaluation Run exists for{" "}
            {activeOption ? `${activeOption.experiment} (${shortHash(activeOption.checkpoint, 8)})` : "this checkpoint"}.
            Without a labeled run there is no number to show, so none is guessed.
          </EmptyState>
        </section>
      ) : shown ? (
        <>
          <ProvenancePanel run={shown} />
          {shown.kind === "isolated" ? <IsolatedView run={shown} /> : <ContinuousView run={shown} />}
        </>
      ) : selectedId ? (
        <div style={{ display: "grid", gap: 16 }} aria-busy="true">
          <div className="skeleton" style={{ height: 120 }} />
          <div className="skeleton" style={{ height: 320 }} />
        </div>
      ) : null}
      </>
      ) : null}

      <dialog ref={dialogRef} className="modal" aria-labelledby="delete-run-title">
        <div className="modal-body">
          <h2 id="delete-run-title">Delete this Evaluation Run?</h2>
          <p><code>{selected?.id}</code> will be removed from the evaluation store.</p>
          <ul>
            <li>Every chart built from it is cleared, with no fallback to another run.</li>
            <li>The committed run file is untouched; Import committed runs restores it.</li>
          </ul>
        </div>
        <div className="modal-actions">
          <button type="button" className="btn ghost" onClick={() => dialogRef.current?.close()} disabled={busy}>Cancel</button>
          <button type="button" className="btn solid-danger" onClick={deleteRun} disabled={busy}>
            <TrashIcon size={15} /> Delete run
          </button>
        </div>
      </dialog>
    </div>
  );
}
