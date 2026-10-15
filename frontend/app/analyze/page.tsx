import type { Metadata } from "next";
import { AnalysisWorkbench } from "../../components/analysis/workbench";

export const metadata: Metadata = { title: "Audio Analysis · KWS Lab" };

export default function AnalyzePage() {
  return <AnalysisWorkbench />;
}
