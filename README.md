# Live Vision

Real-time multi-camera object and face tracking, frame enhancement, pixel-difference movement alerts, and Gemini visual scene analysis in the browser and on Node.js.

**Try it yourself:** https://ai.studio/apps/30c1a814-3fca-4e94-8877-8ace09020392

## Features

- **Multi-camera switching**: Uses local PC webcams (`enumerateDevices`) and lets phones stream 720p JPEG frames over WebSockets (`ws`) with Screen Wake Lock and automatic frame dropping above 400 KB buffer.
- **Live WebGL enhancement**: Auto levels (1st–99th percentile), dark-scene-only gamma correction, and 3x3 sharpening with a 2D canvas fallback.
- **Object & face tracking**: Runs TensorFlow.js COCO-SSD and MediaPipe Face Detection off the render loop with class-matched greedy IoU tracking and persistent IDs.
- **Movement watch zone**: Draw a rectangular watch zone directly on the video canvas to monitor frame-to-frame pixel differences, trigger visual/sound alerts, and optionally inspect the zone with Gemini.
- **Detected panel & Gemini Vision**: Captures 160 px thumbnails for live and recently seen items, supports **Ask Gemini** per item, and offers **Scan all things** (`gemini-4-hydrogen` / `gemini-3.1-flash-lite`) to locate up to 30 objects across the full view.

## Run locally

Requires **Node.js 20+** (Node.js 22 recommended).

1. Install dependencies:
   ```bash
   npm install
   ```
2. Copy `.env.example` to `.env` and add your Gemini API key:
   ```env
   GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open `http://localhost:3000` in your browser (use HTTPS when connecting phones over a remote network so `getUserMedia` is permitted).

### Production build

```bash
npm run build
npm start
```
