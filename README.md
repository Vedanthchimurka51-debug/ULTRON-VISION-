# Live Vision

**Real-Time Multi-Camera Computer Vision, WebGL Frame Enhancement, Spatial Watch-Zone Motion Detection, and Server-Side Gemini Scene Understanding**

[**Try it yourself**](https://ai.studio/apps/30c1a814-3fca-4e94-8877-8ace09020392)

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Repository Structure](#repository-structure)
4. [Core Subsystems & Algorithms](#core-subsystems--algorithms)
   - [1. Zero-Delay Render & Async Inference Pipeline](#1-zero-delay-render--async-inference-pipeline)
   - [2. WebGL Frame Enhancement Shader](#2-webgl-frame-enhancement-shader)
   - [3. Dual-Model Object & Face Detection + Greedy IoU Tracker](#3-dual-model-object--face-detection--greedy-iou-tracker)
   - [4. Pixel-Difference Watch Zone & Alert Engine](#4-pixel-difference-watch-zone--alert-engine)
   - [5. Low-Latency Binary WebSocket Phone Relay](#5-low-latency-binary-websocket-phone-relay)
   - [6. Server-Side Gemini Multimodal Gateway](#6-server-side-gemini-multimodal-gateway)
5. [Protocol & API Reference](#protocol--api-reference)
6. [Technology Stack](#technology-stack)
7. [Local Setup & Usage](#local-setup--usage)
8. [Accessibility & UI Engineering](#accessibility--ui-engineering)

---

## Overview

**Live Vision** is a full-stack browser + Node.js computer vision workbench built without native C++, Python, or Rust extensions. It unifies local PC webcams and remote mobile phone cameras into a single low-latency 30 fps viewport, enhances frames on the GPU via WebGL before inference, tracks objects and faces continuously with stable non-reusing IDs, monitors user-defined rectangular regions for pixel-level movement, and delegates semantic inspection and exhaustive scene localization to Gemini models on the server.

### Key Capabilities

- **Multi-Source Camera Switching**: Automatic enumeration of local webcams (`navigator.mediaDevices.enumerateDevices`) with live `devicechange` listeners, manual **Rescan**, and seamless hot-switching to any connected mobile device streaming over WebSockets.
- **Non-Blocking 30 fps Pipeline**: Video rendering runs on `requestAnimationFrame` while TensorFlow.js inference runs asynchronously on snapshot buffers—display frame rate never blocks on model execution.
- **GPU Frame Enhancement**: Single-pass WebGL fragment shader performing percentile auto-levels, non-darkening dark-scene gamma correction, and 3x3 unsharp sharpening, with an automatic 2D canvas LUT fallback.
- **Dual-Tracker Persistence**: Separate class-gated greedy IoU trackers for objects (COCO-SSD) and faces (MediaPipe Face Detector) with 15-frame occlusion hysteresis and zero ID reuse.
- **Watch Zone Pixel-Diff Alerts**: Interactive click/touch drag-to-draw rectangular ROI directly on the canvas, real-time luminance delta evaluation, visual/audio alerts, and optional cooldown-gated Gemini auto-inspection.
- **Detected Panel & Exhaustive Scene Scan**: Live 160 px object/face thumbnail crops refreshed every 2 seconds without DOM flicker, retaining up to 80 historical detections (30 displayed in **Seen earlier**), paired with per-item **Ask Gemini** and full-frame **Scan all things** localization.

---

## System Architecture

```text
+-----------------------------------------------------------------------------------+
|                               BROWSER CLIENT (React + Vite)                       |
|                                                                                   |
|  [PC Webcam Stream] ----\                                                         |
|                          +--> [FrameEnhancer] --> [displayCanvas] (Clean Frame)   |
|  [Remote Phone Frame] --/      (WebGL / 2D LUT)        |                          |
|   (WS ArrayBuffer ->                                   +--> [detectCanvas]        |
|    createImageBitmap)                                  |       |                  |
|                                                        |       +--> TF.js Models  |
|  [Phone Camera Sender]                                 |       |    (COCO-SSD +   |
|   getUserMedia (720p) -> JPEG (q=0.72)                 |       |     MediaPipe)   |
|   -> drop if bufferedAmount > 400 KB                   |       |       |          |
|   -> Screen Wake Lock                                  |       |       v          |
|         |                                              |       |  [IoUTracker x2] |
|         |                                              |       v       |          |
|         |                                              |  [ZoneMotionDetector]    |
|         |                                              |       |       |          |
|         |                                              |       v       v          |
|         |                                              +--> [overlayCanvas]       |
|         |                                                   (Boxes + Watch Zone)  |
+---------|------------------------------------------------------|------------------+
          | WSS /ws (Binary JPEG + JSON Control)                 | HTTPS POST /api/*
          v                                                      v
+-----------------------------------------------------------------------------------+
|                           NODE.JS SERVER (Express + ws)                           |
|                                                                                   |
|  [WebSocket Relay (/ws)]                           [Gemini Gateway]               |
|   - Persistent clientId -> "Phone N" mapping        - POST /api/describe          |
|   - Binary frame framing: [1B idLen][id][JPEG]      - POST /api/scan-all          |
|   - Viewer backpressure drop (> 400 KB)             - Models: gemini-4-hydrogen,  |
|   - 25s heartbeat ping/pong                           gemini-3.1-flash-lite       |
+-----------------------------------------------------------------------------------+
```

---

## Repository Structure

```text
live-vision/
├── .env.example                       # Environment template (GEMINI_API_KEY, PORT)
├── .gitignore                         # Ignores node_modules, dist, logs, and .env*
├── README.md                          # Architecture, math, protocol, and setup guide
├── index.html                         # Document entry point and metadata
├── metadata.json                      # Application capability & permission manifest
├── package.json                       # Scripts (dev, build, start, clean, lint) & deps
├── server.ts                          # Express HTTP server, /ws binary relay & Gemini API
├── tsconfig.json                      # TypeScript compiler configuration
├── vite.config.ts                     # Vite 8 + Tailwind v4 + MediaPipe shim config
└── src/
    ├── main.tsx                       # React 19 root mount
    ├── index.css                      # Neutral dark grey tokens, sliders, reduced-motion
    ├── App.tsx                        # Main orchestrator: rAF, WS, zone drawing, UI
    ├── components/
    │   ├── TopBar.tsx                 # Camera switcher, Detected count, alert & WS status
    │   ├── DetectedPanel.tsx          # Left drawer: Gemini scan, In view now, Seen earlier
    │   └── ControlSidebar.tsx         # Tabbed right dock: Zone & AI / Camera & Enhance
    ├── shims/
    │   └── mediapipe-face-detection.ts # ESM shim for TF.js runtime bundling under Vite
    ├── types/
    │   └── vision.ts                  # Shared interfaces for cameras, tracks, zones & AI
    └── vision/
        ├── enhancer.ts                # WebGL shader + 96x54 histogram + 2D LUT fallback
        ├── detector.ts                # Singleton TF.js COCO-SSD & FaceDetector runner
        ├── tracker.ts                 # Class-gated greedy IoU multi-object tracker
        ├── motion.ts                  # 240px ROI pixel-difference motion evaluator
        ├── crop.ts                    # Reusable offscreen canvas 160px JPEG cropper
        └── alertSound.ts              # Rate-limited Web Audio API alert synthesizer
```

---

## Core Subsystems & Algorithms

### 1. Zero-Delay Render & Async Inference Pipeline

Decoupling presentation from neural inference prevents frame drops regardless of client GPU/CPU speed:

1. On each `requestAnimationFrame` tick, the pipeline inspects either the active `<video>` element (`currentTime` delta check) or the latest decoded `ImageBitmap` from a remote phone.
2. Old undecoded or unconsumed remote bitmaps are immediately `.close()`d so only the newest frame ever reaches the GPU.
3. The enhanced frame is written to `displayCanvas` (clean, un-annotated pixels) so thumbnail crops and Gemini snapshots never contain bounding boxes or labels.
4. If no TensorFlow.js pass is currently in flight (`!isDetecting`), `displayCanvas` is blitted to `detectCanvas`, and `detectObjectsAndFaces` executes asynchronously.
5. `overlayCanvas` sits pixel-aligned atop `displayCanvas` and is repainted immediately on detection completion, pointer zone dragging, or motion state transitions.

---

### 2. WebGL Frame Enhancement Shader

Executed in `src/vision/enhancer.ts` with a target budget under 8 ms per 720p frame:

1. **Downscaled Statistics (`96 x 54` = 5,184 px)**:
   Computes ITU-R BT.709 luminance per sampled pixel:
   $$Y = 0.2126R + 0.7152G + 0.0722B$$
   Builds a 256-bin histogram to extract the 1st percentile ($p_1$) and 99th percentile ($p_{99}$), smoothed across frames via exponential moving average ($\alpha = 0.25$) to eliminate exposure flicker.
2. **Dark-Scene-Only Gamma ($\gamma \in [0.5, 1.0]$)**:
   Bright scenes remain untouched ($\gamma = 1.0$, never darkening). Low-light scenes interpolate smoothly based on mean frame luminance $\mu_Y$:
   $$\gamma_{\text{auto}} = \begin{cases} 1.0 & \text{if } \mu_Y \ge 0.45 \\ \max\left(0.5,\, 0.5 + \frac{\mu_Y}{0.45}\cdot 0.5\right) & \text{otherwise} \end{cases}$$
   Users can also override gamma manually via the `[0.50, 1.00]` slider or revert to auto mode.
3. **3x3 Unsharp Filter + Levels Stretch (Single Fragment Pass)**:
   $$\mathbf{c}_{\text{sharp}} = \text{clamp}\big(\mathbf{c} + 0.35(\mathbf{c} - \tfrac{1}{4}(\mathbf{n}+\mathbf{s}+\mathbf{e}+\mathbf{w})),\; 0,\; 1\big)$$
   $$\mathbf{c}_{\text{out}} = \left(\text{clamp}\left(\frac{\mathbf{c}_{\text{sharp}} - b}{\max(0.08,\, w - b)},\; 0,\; 1\right)\right)^{\gamma}$$
4. **Automatic Fallback**: If WebGL initialization fails or context is lost, the enhancer builds a 256-entry `Uint8Array` lookup table from the same downscaled statistics and applies it on a 2D canvas.

---

### 3. Dual-Model Object & Face Detection + Greedy IoU Tracker

- **Object Detection (`@tensorflow-models/coco-ssd`)**:
  - Profiles: **Fast (lite)** (`lite_mobilenet_v2`) and **High detail** (`mobilenet_v2`), up to **50** detections per frame, adjustable confidence threshold (`0.25`–`0.75`, default `0.40`).
- **Face Detection (`@tensorflow-models/face-detection`)**:
  - `MediaPipeFaceDetector` on the TF.js runtime (`short` or `full` range model, up to **25** simultaneous faces).
- **Greedy IoU Tracker (`src/vision/tracker.ts`)**:
  - Maintains two independent tracker instances (`object` and `face`) with strictly monotonic IDs (`nextId++`, never reused).
  - Computes pairwise Intersection-over-Union across same-class tracks and detections:
    $$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} \ge 0.30$$
  - Sorts candidate pairs descending by IoU and assigns 1-to-1 greedy matches.
  - Tracks missing for `> 15` consecutive detection passes are pruned; both trackers and detection history reset cleanly on camera source switch.

---

### 4. Pixel-Difference Watch Zone & Alert Engine

Implemented in `src/vision/motion.ts`:

- Users define a normalized rectangular region $Z = (x, y, w, h) \in [0,1]^4$ by dragging directly over `overlayCanvas` with mouse or touch.
- Each incoming frame is downsampled to a 240 px-wide analysis buffer and cropped to $Z$ via `getImageData`.
- Integer luma is computed per pixel:
  $$Y = \frac{77R + 150G + 29B}{256}$$
- A pixel is flagged changed when $|Y_t - Y_{t-1}| > \tau_{\text{pix}}$, where $\tau_{\text{pix}}$ and area trigger percentage $\tau_{\text{area}}$ scale continuously from the user's **Motion sensitivity** slider ($S \in [10, 95]$):
  $$\tau_{\text{pix}} = \text{round}\left(46 - 0.32 S\right), \qquad \tau_{\text{area}} = \max\left(0.8,\; 7.5 - 0.065 S\right)\%$$
- When changed pixel area exceeds $\tau_{\text{area}}$, the zone enters alert state with a **1,800 ms hold window**, highlights the ROI and top status bar in alert red, optionally plays a rate-limited Web Audio two-tone chime, and optionally triggers a cooldown-gated (**8,000 ms**) **Gemini zone inspection**.

---

### 5. Low-Latency Binary WebSocket Phone Relay

- **Device Identity**: Each browser persists a random `clientId` in `localStorage` (`live_vision_client_id`) and connects to `/ws?clientId=<id>`.
- **Ordered Naming & Reconnect Stability**: The server maps `clientId` to `"Phone 1"`, `"Phone 2"`, ... in order of first stream start and preserves that name across disconnects and reconnects.
- **Stream Switching Rules**:
  - Newly connected phones appear in the top bar of all viewers without interrupting the viewer's active camera.
  - The sending phone automatically selects its own feed and shares its local stream directly with its canvas to eliminate network echo and hardware contention.
  - Disconnected phones are removed from the list unless a viewer is actively watching that phone, in which case it remains selected in `offline` state until switched away or reconnected.
- **Backpressure Control**:
  - Phone capture requests `1280x720` @ `30 fps`, encodes JPEG (`quality = 0.72`), and drops frames whenever `ws.bufferedAmount > 400 KB`.
  - Server relay packets prepend a compact header (`[1-byte idLen][UTF-8 phoneId][JPEG bytes]`) and skip slow viewers whose `bufferedAmount > 400 KB`.
- **Wake Lock**: Phones acquire `navigator.wakeLock.request('screen')` while streaming and re-acquire automatically on `visibilitychange`.

---

### 6. Server-Side Gemini Multimodal Gateway

All Gemini calls execute exclusively on Node.js via `@google/genai` (`GEMINI_API_KEY` never leaves the server):

- **Model Selection**: Strictly allowlisted free-tier-compatible models:
  - `gemini-4-hydrogen` (default, configured with `ThinkingLevel.LOW` for fast multimodal reasoning)
  - `gemini-3.1-flash-lite` (ultra-low-latency identification)
- **Item & Zone Inspection (`POST /api/describe`)**:
  - `item` mode: concise single-sentence description of a 160 px object/face crop.
  - `zone` mode: focused 1–2 sentence analysis of movement inside the user's watch zone.
  - `scene` mode: 2–3 sentence full-frame overview.
- **Exhaustive Scene Localization (`POST /api/scan-all`)**:
  - Uses structured JSON output (`responseMimeType: "application/json"` + `responseSchema`) to locate up to **30 distinct items** across the entire view—catching objects, tools, screens, accessories, and background details beyond COCO-SSD's 80 classes—returning normalized `[ymin, xmin, ymax, xmax]` coordinates (`0–1000`) and cropping thumbnails for the **Detected** panel.

---

## Protocol & API Reference

### HTTP Endpoints

#### `POST /api/describe`
Describes a cropped item, watch zone, or scene frame.

**Request (`application/json`)**
```json
{
  "imageBase64": "data:image/jpeg;base64,...",
  "label": "person #1",
  "model": "gemini-4-hydrogen",
  "mode": "item"
}
```
- `model`: `"gemini-4-hydrogen"` | `"gemini-3.1-flash-lite"`
- `mode`: `"item"` | `"zone"` | `"scene"`

**Response (`200 OK`)**
```json
{
  "description": "A person wearing dark headphones seated in front of a laptop.",
  "model": "gemini-4-hydrogen"
}
```

#### `POST /api/scan-all`
Performs exhaustive multi-object localization on the current frame.

**Request (`application/json`)**
```json
{
  "imageBase64": "data:image/jpeg;base64,...",
  "model": "gemini-4-hydrogen"
}
```

**Response (`200 OK`)**
```json
{
  "summary": "A desk workspace with a laptop, external monitor, mug, and keyboard.",
  "model": "gemini-4-hydrogen",
  "items": [
    { "label": "laptop", "ymin": 410, "xmin": 220, "ymax": 790, "xmax": 610 },
    { "label": "ceramic mug", "ymin": 620, "xmin": 680, "ymax": 770, "xmax": 765 }
  ]
}
```

### WebSocket Protocol (`/ws?clientId=<id>`)

| Direction | Frame Type | Payload | Purpose |
| :--- | :--- | :--- | :--- |
| Client -> Server | Text (JSON) | `{"type":"phone:start"}` | Register device as an active phone camera |
| Client -> Server | Text (JSON) | `{"type":"phone:stop"}` | Unregister phone camera |
| Client -> Server | Binary | Raw JPEG bytes (`q=0.72`) | Upload phone camera frame |
| Server -> Client | Text (JSON) | `{"type":"phone:registered","id":"...","name":"Phone 1"}` | Confirm assigned persistent phone name |
| Server -> Client | Text (JSON) | `{"type":"phones","phones":[{"id":"...","name":"Phone 1"}]}` | Broadcast active phone roster |
| Server -> Client | Binary | `[1B idLen][UTF-8 id][JPEG bytes]` | Relay phone frame to viewers |

---

## Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | React 19, TypeScript, Tailwind CSS v4, Vite 8 |
| **Client ML & Vision** | TensorFlow.js (`@tensorflow/tfjs`), COCO-SSD (`@tensorflow-models/coco-ssd`), MediaPipe Face Detection (`@tensorflow-models/face-detection`), Custom WebGL1 Fragment Shader, 2D Canvas ROI Pixel-Diff Engine |
| **Backend** | Node.js 22, Express 4, `ws` WebSocket Server, `tsx` |
| **Multimodal AI** | `@google/genai` SDK (`gemini-4-hydrogen`, `gemini-3.1-flash-lite`) |

---

## Local Setup & Usage

### Prerequisites

- **Node.js 20+** (Node.js 22 LTS recommended)
- A **Gemini API key**
- A webcam (for PC viewing) and/or a smartphone on HTTPS (or `localhost` / tunneled HTTPS) for mobile camera streaming

### Installation

1. Clone the repository and install dependencies:
   ```bash
   npm install
   ```

2. Configure environment variables by copying `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   *(Windows Command Prompt: `copy .env.example .env`)*

   Edit `.env`:
   ```env
   GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   PORT=3000
   ```

3. Start the full-stack development server (Express + WebSockets + Vite middleware):
   ```bash
   npm run dev
   ```

4. Open `http://localhost:3000` in your browser.

### Available Scripts

| Command | Description |
| :--- | :--- |
| `npm run dev` | Starts `server.ts` via `tsx` with live Vite middleware on port 3000 |
| `npm run build` | Compiles production frontend bundle into `dist/` |
| `npm start` | Runs `node server.ts` serving `dist/` and `/api/*` + `/ws` |
| `npm run lint` | Runs strict TypeScript typechecking (`tsc --noEmit`) |
| `npm run clean` | Cross-platform removal of `dist/` and temporary build files |

---

## Accessibility & UI Engineering

- **Neutral Dark Viewfinder Palette**: Non-black charcoal workspace (`#1e2024` / `#27292f`) paired with off-white typography (`#f3f2ee`) and single-accent amber (`#f2b632`) corner brackets and primary actions.
- **Non-Color-Only Status**: Every camera source explicitly renders text states (`live`, `waiting`, `offline`), and alerts combine color changes with text percentage readouts.
- **Touch & Keyboard Ergonomics**: All interactive controls enforce `min-h-[44px]`, clear `focus-visible` outlines, `Escape` drawer dismissal, `tabular-nums` alignment for telemetry, and `@media (prefers-reduced-motion: reduce)` compliance.
