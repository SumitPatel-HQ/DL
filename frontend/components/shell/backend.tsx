"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, errorCode, type ActiveModel } from "../../lib/api.ts";

export type Backend =
  | { status: "checking" }
  | { status: "offline" }
  | { status: "unavailable"; code: string; setupAction: string }
  | { status: "ready"; model: ActiveModel };

type BackendContext = { backend: Backend; refresh: () => void };

const Context = createContext<BackendContext>({ backend: { status: "checking" }, refresh: () => { } });

const POLL_MS = 5000;

// One readiness source for every slice: API reachability plus the active model.
export function BackendProvider({ children }: { children: ReactNode }) {
  const [backend, setBackend] = useState<Backend>({ status: "checking" });
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      let next: Backend;
      try {
        const health = await api.health();
        if (health.model.status === "ready") {
          const model = await api.activeModel();
          next = { status: "ready", model };
        } else {
          next = {
            status: "unavailable",
            code: health.model.code,
            setupAction: health.model.setup_action,
          };
        }
      } catch (error) {
        const code = errorCode(error);
        next = code === "offline" ? { status: "offline" } : { status: "unavailable", code, setupAction: "" };
      }
      if (cancelled) return;
      setBackend((previous) => (same(previous, next) ? previous : next));
    };
    void check();
    const timer = window.setInterval(check, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [tick]);

  const refresh = useCallback(() => setTick((value) => value + 1), []);
  const value = useMemo(() => ({ backend, refresh }), [backend, refresh]);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

function same(a: Backend, b: Backend): boolean {
  if (a.status !== b.status) return false;
  if (a.status === "ready" && b.status === "ready") return a.model.id === b.model.id;
  if (a.status === "unavailable" && b.status === "unavailable") return a.code === b.code;
  return true;
}

export function useBackend(): BackendContext {
  return useContext(Context);
}

export function BackendPill() {
  const { backend } = useBackend();
  switch (backend.status) {
    case "checking":
      return (
        <span className="pill" role="status">
          <span className="dot busy" /> Checking API
        </span>
      );
    case "offline":
      return (
        <span className="pill" role="status">
          <span className="dot critical" /> <strong>API offline</strong>
        </span>
      );
    case "unavailable":
      return (
        <span className="pill" role="status" title={backend.setupAction}>
          <span className="dot warn" /> API up · <strong>model unavailable</strong>
        </span>
      );
    case "ready":
      return (
        <span className="pill" role="status" title={`Ready · checkpoint ${backend.model.id}`}>
          <span className="dot ok" /> <strong>Model</strong> {backend.model.version ?? "ready"}
        </span>
      );
  }
}
