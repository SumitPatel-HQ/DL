import type { Metadata } from "next";
import { EvaluationDashboard } from "../../components/dashboard/dashboard";

export const metadata: Metadata = { title: "Evaluation · KWS Lab" };

export default async function DashboardPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const tab = params.tab === "continuous" ? "continuous" : "isolated";
  const run = typeof params.run === "string" ? params.run : null;
  return <EvaluationDashboard key={`${tab}:${run ?? ""}`} initialTab={tab} initialRun={run} />;
}
