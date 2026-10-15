import type { Metadata } from "next";
import { LiveSession } from "../../components/live/live-session";

export const metadata: Metadata = { title: "Voice Control · KWS Lab" };

export default function LivePage() {
  return <LiveSession />;
}
