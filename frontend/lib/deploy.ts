// True in builds for the hosted site (see next.config.ts). The hosted API sits on
// the same origin under /api, cannot hold WebSockets, and runs on Linux, so live
// sessions score in the browser and Windows Actions are unavailable.
export const DEPLOYED = process.env.KWS_DEPLOYED === "1";
