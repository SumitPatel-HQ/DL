// Copy the onnxruntime-web WASM runtime into public/ort/ so live sessions load it
// from the site itself (lib/live/browser-socket.ts sets wasmPaths to /ort/).
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = dirname(createRequire(import.meta.url).resolve("onnxruntime-web/ort-wasm-simd-threaded.wasm"));
const target = join(root, "public", "ort");
mkdirSync(target, { recursive: true });
for (const name of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) {
  copyFileSync(join(dist, name), join(target, name));
}
