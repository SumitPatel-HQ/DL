import { DEPLOYED } from "../lib/deploy";
import { getHealth } from "../lib/health";

export async function ModelStatus() {
  let health;
  try {
    health = await getHealth();
  } catch {
    return (
      <section className="banner error" aria-labelledby="status-heading" role="alert">
        <div className="banner-body">
          <strong id="status-heading">Backend disconnected</strong>
          {DEPLOYED ? (
            <span>The API did not answer. It may be starting up; refresh this page in a few seconds.</span>
          ) : (
            <span>Start the API with <code>npm run api</code>, check API_BASE_URL, then refresh this page.</span>
          )}
        </div>
      </section>
    );
  }

  if (health.model.status !== "ready") {
    return (
      <section className="banner experimental" aria-labelledby="status-heading" role="status">
        <div className="banner-body">
          <strong id="status-heading">API connected · model unavailable</strong>
          <span>{health.model.setup_action} No trained keywords are available.</span>
        </div>
      </section>
    );
  }

  // Model is ready: navbar BackendPill already displays model checkpoint & readiness.
  // Avoid repeating redundant banner per guidelines.
  return null;
}
