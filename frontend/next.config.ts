import type { NextConfig } from "next";

// Vercel sets VERCEL=1 during builds; KWS_DEPLOYED=1 reproduces a deployed build locally.
const vercel = process.env.VERCEL === "1";
const deployed = vercel || process.env.KWS_DEPLOYED === "1";

const config: NextConfig = {
  env: { KWS_DEPLOYED: deployed ? "1" : "" },
  // On Vercel, vercel.json routes /api to the Python service. A local deployed-mode
  // build has no router, so it forwards /api to the local API instead.
  ...(deployed && !vercel
    ? { rewrites: async () => [{ source: "/api/:path*", destination: `${process.env.API_BASE_URL ?? "http://localhost:8000"}/api/:path*` }] }
    : {}),
};

export default config;
