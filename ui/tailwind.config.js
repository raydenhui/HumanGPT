/** @type {import("tailwindcss").Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        base: "#0d1116",
        panel: "#161b22",
        panel2: "#1f2530",
        ink: "#d8dee6",
        dim: "#8b94a1",
        accent: "#4f9cf0",
        ok: "#2ea869",
        warn: "#d9a514",
        err: "#d45555",
        line: "#2a323c",
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ].join(", "),
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"].join(", "),
      },
    },
  },
  plugins: [],
};