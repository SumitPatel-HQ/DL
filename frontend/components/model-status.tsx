import { getHealth } from "../lib/health";

export async function ModelStatus() {
  let health;
  try {
    health = await getHealth();
  } catch {
    return (
      <section className="banner error" aria-labelledby="status-heading" role="status">
        <div className="banner-body">
          <strong id="status-heading">Backend disconnected</strong>
          <span>Start the API with <code>npm run api</code>, check API_BASE_URL, then refresh this page.</span>
        </div>
      </section>
    );
  }

  return health.model.status === "ready" ? (
    <section className="banner info" aria-labelledby="status-heading" role="status">
      <div className="banner-body">
        <strong id="status-heading">API connected · model ready</strong>
        <span>All three slices can use the active checkpoint.</span>
      </div>
    </section>
  ) : (
    <section className="banner experimental" aria-labelledby="status-heading" role="status">
      <div className="banner-body">
        <strong id="status-heading">API connected · model unavailable</strong>
        <span>{health.model.setup_action} No trained keywords are available.</span>
      </div>
    </section>
  );
}
