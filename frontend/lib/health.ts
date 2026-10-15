export type Health = {
  status: "ok";
  model: {
    status: "unavailable";
    code: "checkpoint_not_installed";
    setup_action: string;
  };
};

export async function getHealth(): Promise<Health> {
  const base = new URL(process.env.API_BASE_URL ?? "http://127.0.0.1:8000");
  if (
    base.protocol !== "http:" ||
    !["127.0.0.1", "localhost", "[::1]"].includes(base.hostname)
  ) {
    throw new Error("API_BASE_URL must use HTTP on loopback.");
  }
  const response = await fetch(new URL("/api/health", base), {
    cache: "no-store",
    signal: AbortSignal.timeout(3000),
  });
  if (!response.ok) throw new Error("Backend health request failed.");
  const health: Health = await response.json();
  if (
    health.status !== "ok" ||
    health.model?.status !== "unavailable" ||
    health.model.code !== "checkpoint_not_installed" ||
    typeof health.model.setup_action !== "string"
  ) {
    throw new Error("Unexpected health response.");
  }
  return health;
}
