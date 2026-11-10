import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// base './' makes the build work from any sub-path (GitHub Pages project sites).
export default defineConfig({
  base: './',
  plugins: [react()],
  optimizeDeps: { exclude: ['onnxruntime-web'] },
})
