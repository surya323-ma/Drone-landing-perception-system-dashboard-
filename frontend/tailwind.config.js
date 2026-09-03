/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Pulled straight from the perception system's own on-screen HUD
        // palette (main.py / main.cpp Palette class) so the dashboard reads
        // as an extension of the same instrument, not a bolted-on admin UI.
        hud: {
          bg: "#0A0D12",
          panel: "#12181F",
          panel2: "#161D26",
          grid: "#32465A",
          cyan: "#3CF0FF",
          cyanSoft: "#5AC8E6",
          amber: "#EBAA3C",
          green: "#78EB8C",
          red: "#EB5050",
          white: "#F5F5F5",
          dim: "#8FA0AC",
        },
      },
      fontFamily: {
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
        display: ["'Space Grotesk'", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 24px rgba(60, 240, 255, 0.15)",
      },
    },
  },
  plugins: [],
};
