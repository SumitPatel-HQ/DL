import { getHealth } from "../lib/health";

export async function ModelStatus() {
  let health;
  try {
    health = await getHealth();
  } catch {
    return (
      <section aria-labelledby="status-heading" role="status">
        <h2 id="status-heading">Backend disconnected</h2>
        <p>Start the API with <code>npm run api</code>, check API_BASE_URL, then refresh this page.</p>
      </section>
    );
  }

  return (
    <section aria-labelledby="status-heading" role="status">
      <p className="eyebrow">API connected</p>
      <h2 id="status-heading">Model unavailable</h2>
      <p>{health.model.setup_action}</p>
      <p>No trained keywords are available.</p>
    </section>
  );
}
