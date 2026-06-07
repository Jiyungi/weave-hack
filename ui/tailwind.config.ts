import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#080b12",
        panel: "#121926",
        panel2: "#0d1320",
        line: "#222e44",
        "line-strong": "#30405c",
        text: "#eaf0f9",
        muted: "#93a2bb",
        accent: "#5b9dff",
        accent2: "#8b7bff",
        good: "#3fb950",
        bad: "#f85149",
        warn: "#d29922",
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        card: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.6)",
        glow: "0 0 0 1px rgba(91,157,255,0.35), 0 8px 30px -8px rgba(91,157,255,0.35)",
      },
      backgroundImage: {
        "accent-grad": "linear-gradient(135deg, #6aa8ff 0%, #8b7bff 100%)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.25s ease-out",
      },
    },
  },
  plugins: [],
};

export default config;
