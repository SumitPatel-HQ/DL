import { DEPLOYED } from "./deploy.ts";

export type Health = {
  status: "ok";
  model: {
    status: "unavailable";
    code: "checkpoint_not_installed" | "model_incompatible" | "worker_unavailable" | "vocabulary_not_reviewed";
    setup_action: string;
  } | {
    status: "ready";
    code: "model_ready";
  };
};

export async function getHealth(): Promise<Health> {
  // Deployed, the Vercel service binding (vercel.json) supplies the API's internal URL.
  const base = new URL(
    (DEPLOYED ? process.env.BACKEND_URL : process.env.API_BASE_URL) ?? "http://localhost:8000",
  );
  if (!DEPLOYED && base.href !== "http://localhost:8000/" && base.href !== "http://127.0.0.1:8000/") {
    throw new Error("API_BASE_URL must be http://localhost:8000 or http://127.0.0.1:8000.");
  }
  const response = await fetch(new URL("/api/health", base), {
    cache: "no-store",
    signal: AbortSignal.timeout(3000),
  });
  if (!response.ok) throw new Error("Backend health request failed.");
  const health: unknown = await response.json();
  if (
    typeof health !== "object" ||
    health === null ||
    !("status" in health) ||
    health.status !== "ok" ||
    !("model" in health) ||
    typeof health.model !== "object" ||
    health.model === null
  ) {
    throw new Error("Unexpected health response.");
  }
  const model = health.model;
  if ("status" in model && "code" in model) {
    if (model.status === "ready" && model.code === "model_ready") {
      return { status: "ok", model: { status: "ready", code: "model_ready" } };
    }
    if (
      model.status === "unavailable" &&
      (model.code === "checkpoint_not_installed" || model.code === "model_incompatible" ||
        model.code === "worker_unavailable" || model.code === "vocabulary_not_reviewed") &&
      "setup_action" in model &&
      typeof model.setup_action === "string"
    ) {
      return {
        status: "ok",
        model: {
          status: "unavailable",
          code: model.code,
          setup_action: model.setup_action,
        },
      };
    }
  }
  throw new Error("Unexpected health response.");
}
