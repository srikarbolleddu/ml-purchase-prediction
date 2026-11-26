import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Served from https://srikarbolleddu.github.io/ml-purchase-prediction/
export default defineConfig({
  base: "./",
  plugins: [react()],
});
