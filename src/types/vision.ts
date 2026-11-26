export type CameraState = 'live' | 'waiting' | 'offline';

export interface PcCameraSource {
  id: string;
  kind: 'pc';
  deviceId: string;
  name: string;
  deviceLabel: string;
  state: CameraState;
}

export interface PhoneCameraSource {
  id: string;
  kind: 'phone';
  phoneId: string;
  name: string;
  state: CameraState;
  isLocal: boolean;
}

export type CameraSource = PcCameraSource | PhoneCameraSource;

export type DetectionKind = 'object' | 'face';

export interface RawDetection {
  kind: DetectionKind;
  className: string;
  /** [x, y, width, height] in canvas pixel coordinates */
  box: [number, number, number, number];
  score: number | null;
}

export interface TrackedItem {
  id: number;
  key: string;
  kind: DetectionKind;
  className: string;
  box: [number, number, number, number];
  score: number | null;
  missedFrames: number;
  lastSeenAt: number;
}

export type GeminiStatus = 'idle' | 'loading' | 'done' | 'error';

export type GeminiModelId = 'gemini-4-hydrogen' | 'gemini-3.1-flash-lite';

export type DetectionProfile = 'fast' | 'accurate';

export interface DetectedHistoryItem {
  key: string;
  id: number;
  kind: DetectionKind;
  className: string;
  label: string;
  score: number | null;
  inView: boolean;
  firstSeenAt: number;
  lastSeenAt: number;
  lastCropAt: number;
  cropDataUrl: string;
  geminiStatus: GeminiStatus;
  geminiText?: string;
  geminiError?: string;
}

export interface EnhancementSettings {
  enabled: boolean;
  autoGamma: boolean;
  gamma: number;
}

export interface NormalizedZone {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface MotionZoneSettings {
  enabled: boolean;
  zone: NormalizedZone | null;
  sensitivity: number;
  soundAlert: boolean;
  autoInspectGemini: boolean;
}

export interface MotionAlertState {
  active: boolean;
  changedPercent: number;
  triggerThreshold: number;
  lastTriggeredAt: number | null;
  alertCount: number;
}

export interface GeminiDetectedObject {
  id: string;
  label: string;
  /** Pixel coordinates [x, y, width, height] relative to scanned frame */
  box: [number, number, number, number];
  /** Normalized 0..1 coordinates for responsive overlay rendering */
  normBox: { x: number; y: number; w: number; h: number };
  cropDataUrl: string;
}

export interface SceneScanResult {
  summary: string;
  model: GeminiModelId;
  scannedAt: number;
  items: GeminiDetectedObject[];
}
