import { Suspense } from "react";
import { ModelStatus } from "../components/model-status";

export const dynamic = "force-dynamic";

export default function Home() {
  return (
    <main>
      <p className="eyebrow">Local audio lab / Phase 0</p>
      <h1>Keyword Spotting</h1>
      <p className="intro">Find trained keywords in recordings and explore bounded voice control with a local LSTM model.</p>
      <Suspense fallback={<p role="status">Checking backend readiness...</p>}>
        <ModelStatus />
      </Suspense>
      <section aria-labelledby="scope-heading">
        <h2 id="scope-heading">Project setup</h2>
        <p>This scaffold checks connectivity. Recording analysis and microphone sessions arrive in later tickets.</p>
        <p>Evaluation: Not evaluated. No Evaluation Run has been recorded.</p>
      </section>
    </main>
  );
}
