import react from "@vitejs/plugin-react";

/** @type {import("vite").UserConfig} */
export default {
  plugins: [react()],
  base: "/static/dist/",
  build: {
    outDir: "../humangpt/web/static/dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/v1": "http://127.0.0.1:8000",
      "/api": "http://127.0.0.1:8000",
      "/events": "http://127.0.0.1:8000",
    },
  },
};