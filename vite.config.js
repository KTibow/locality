import { defineConfig } from "vite";

export default defineConfig({
  root: "./src",
  publicDir: "../public",
  appType: "mpa",
  server: {
    headers: {
      "cross-origin-opener-policy": "same-origin",
      "cross-origin-embedder-policy": "require-corp",
    },
  },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
});
