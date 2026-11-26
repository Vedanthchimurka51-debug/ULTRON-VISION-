import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { ControlSidebar } from './components/ControlSidebar';
import { DetectedPanel } from './components/DetectedPanel';
import { TopBar } from './components/TopBar';
import type {
  CameraSource,
  CameraState,
  DetectedHistoryItem,
  DetectionProfile,
  EnhancementSettings,
  GeminiDetectedObject,
  GeminiModelId,
  MotionAlertState,
  MotionZoneSettings,
  NormalizedZone,
  PcCameraSource,
  PhoneCameraSource,
  SceneScanResult,
  TrackedItem,
} from './types/vision';
import { playMotionAlertTone } from './vision/alertSound';
import { captureCroppedThumbnail } from './vision/crop';
import {
  detectObjectsAndFaces,
  loadDetectionModels,
  type LoadedModels,
} from './vision/detector';
import { FrameEnhancer } from './vision/enhancer';
import { ZoneMotionDetector } from './vision/motion';
import { IoUTracker } from './vision/tracker';

const CLIENT_ID_STORAGE_KEY = 'live_vision_client_id';
const MAX_HISTORY_ITEMS = 80;
const MAX_SEEN_EARLIER = 30;
const CROP_REFRESH_MS = 2000;
const MAX_WS_BUFFER_BYTES = 400 * 1024;
const MOTION_HOLD_MS = 1800;
const AUTO_INSPECT_COOLDOWN_MS = 8000;

function getOrCreateClientId(): string {
  try {
    const existing = window.localStorage.getItem(CLIENT_ID_STORAGE_KEY);
    if (existing && /^[a-zA-Z0-9_-]{6,64}$/.test(existing)) {
      return existing;
    }
    const generated =
      typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : `lv_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
    window.localStorage.setItem(CLIENT_ID_STORAGE_KEY, generated);
    return generated;
  } catch {
    return `lv_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
  }
}

function formatCameraPermissionError(err: unknown): string {
  const name = err instanceof DOMException ? err.name : '';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
    return 'Camera permission was denied. Open your browser site settings (click the lock or camera icon in the address bar), set Camera to Allow, and click Rescan or retry.';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'No camera hardware was found on this device.';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'Camera is currently in use by another application or tab.';
  }
  if (err instanceof Error && err.message) {
    return `Could not open camera: ${err.message}`;
  }
  return 'Could not open camera.';
}

function captureZoneSnapshotDataUrl(
  canvas: HTMLCanvasElement,
  zone: NormalizedZone
): string {
  const srcW = canvas.width;
  const srcH = canvas.height;
  if (srcW === 0 || srcH === 0) return '';

  const sx = Math.max(0, Math.floor(zone.x * srcW));
  const sy = Math.max(0, Math.floor(zone.y * srcH));
  const sw = Math.max(8, Math.min(srcW - sx, Math.ceil(zone.w * srcW)));
  const sh = Math.max(8, Math.min(srcH - sy, Math.ceil(zone.h * srcH)));

  const maxSide = 640;
  const scale = Math.min(1, maxSide / Math.max(sw, sh));
  const outW = Math.max(8, Math.round(sw * scale));
  const outH = Math.max(8, Math.round(sh * scale));

  const temp = document.createElement('canvas');
  temp.width = outW;
  temp.height = outH;
  const ctx = temp.getContext('2d');
  if (!ctx) return '';
  ctx.drawImage(canvas, sx, sy, sw, sh, 0, 0, outW, outH);
  return temp.toDataURL('image/jpeg', 0.86);
}

function captureFullFrameDataUrl(canvas: HTMLCanvasElement): string {
  const srcW = canvas.width;
  const srcH = canvas.height;
  if (srcW === 0 || srcH === 0) return '';
  const maxSide = 960;
  const scale = Math.min(1, maxSide / Math.max(srcW, srcH));
  const outW = Math.max(1, Math.round(srcW * scale));
  const outH = Math.max(1, Math.round(srcH * scale));
  const temp = document.createElement('canvas');
  temp.width = outW;
  temp.height = outH;
  const ctx = temp.getContext('2d');
  if (!ctx) return '';
  ctx.drawImage(canvas, 0, 0, srcW, srcH, 0, 0, outW, outH);
  return temp.toDataURL('image/jpeg', 0.85);
}

interface OverlayDrawParams {
  overlayCanvas: HTMLCanvasElement;
  width: number;
  height: number;
  tracks: TrackedItem[];
  geminiBoxes: GeminiDetectedObject[];
  showGeminiBoxes: boolean;
  zone: NormalizedZone | null;
  draftZone: NormalizedZone | null;
  zoneEnabled: boolean;
  motionActive: boolean;
  changedPercent: number;
}

function drawCanvasOverlay({
  overlayCanvas,
  width,
  height,
  tracks,
  geminiBoxes,
  showGeminiBoxes,
  zone,
  draftZone,
  zoneEnabled,
  motionActive,
  changedPercent,
}: OverlayDrawParams): void {
  if (overlayCanvas.width !== width || overlayCanvas.height !== height) {
    overlayCanvas.width = width;
    overlayCanvas.height = height;
  }
  const ctx = overlayCanvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, width, height);

  ctx.font =
    '600 13px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  ctx.textBaseline = 'top';

  // 1. Gemini exhaustive scan boxes (dashed cool white/lavender)
  if (showGeminiBoxes && geminiBoxes.length > 0) {
    for (const item of geminiBoxes) {
      const x = Math.max(0, Math.round(item.normBox.x * width));
      const y = Math.max(0, Math.round(item.normBox.y * height));
      const w = Math.max(4, Math.round(item.normBox.w * width));
      const h = Math.max(4, Math.round(item.normBox.h * height));

      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#c9d4ff';
      ctx.strokeRect(x + 1, y + 1, Math.max(1, w - 2), Math.max(1, h - 2));
      ctx.restore();

      const paddingX = 5;
      const tagHeight = 18;
      const textWidth = ctx.measureText(item.label).width;
      const tagWidth = Math.ceil(textWidth + paddingX * 2);
      const tagX = Math.min(Math.max(0, x), Math.max(0, width - tagWidth));
      const tagY = Math.min(y + 2, Math.max(0, height - tagHeight));

      ctx.fillStyle = '#c9d4ff';
      ctx.fillRect(tagX, tagY, tagWidth, tagHeight);
      ctx.fillStyle = '#1a1b1e';
      ctx.fillText(item.label, tagX + paddingX, tagY + 2);
    }
  }

  // 2. Live TensorFlow.js object & face tracks
  for (const track of tracks) {
    const [rawX, rawY, rawW, rawH] = track.box;
    const x = Math.max(0, Math.round(rawX));
    const y = Math.max(0, Math.round(rawY));
    const w = Math.max(2, Math.round(rawW));
    const h = Math.max(2, Math.round(rawH));

    const isFace = track.kind === 'face';
    const strokeColor = isFace ? '#5ec2b7' : '#f2b632';
    const textColor = '#1a1b1e';

    ctx.lineWidth = 2;
    ctx.strokeStyle = strokeColor;
    ctx.strokeRect(x + 1, y + 1, Math.max(1, w - 2), Math.max(1, h - 2));

    const labelText =
      track.score !== null
        ? `${track.className} #${track.id} ${track.score.toFixed(2)}`
        : `${track.className} #${track.id}`;

    const paddingX = 6;
    const tagHeight = 20;
    const textWidth = ctx.measureText(labelText).width;
    const tagWidth = Math.ceil(textWidth + paddingX * 2);

    const tagX = Math.min(Math.max(0, x), Math.max(0, width - tagWidth));
    const tagY = y >= tagHeight ? y - tagHeight : Math.min(y + 2, height - tagHeight);

    ctx.fillStyle = strokeColor;
    ctx.fillRect(tagX, tagY, tagWidth, tagHeight);

    ctx.fillStyle = textColor;
    ctx.fillText(labelText, tagX + paddingX, tagY + 3);
  }

  // 3. Watch zone rectangle (draft while dragging, or active saved zone)
  const activeZone = draftZone ?? zone;
  if (activeZone) {
    const zx = Math.round(activeZone.x * width);
    const zy = Math.round(activeZone.y * height);
    const zw = Math.max(4, Math.round(activeZone.w * width));
    const zh = Math.max(4, Math.round(activeZone.h * height));

    const isDraft = draftZone !== null;
    const borderColor = isDraft
      ? '#f2b632'
      : !zoneEnabled
        ? '#a9acb5'
        : motionActive
          ? '#e5534b'
          : '#f2b632';

    const fillColor = isDraft
      ? 'rgba(242, 182, 50, 0.14)'
      : !zoneEnabled
        ? 'rgba(169, 172, 181, 0.08)'
        : motionActive
          ? 'rgba(229, 83, 75, 0.18)'
          : 'rgba(242, 182, 50, 0.08)';

    ctx.fillStyle = fillColor;
    ctx.fillRect(zx, zy, zw, zh);

    ctx.save();
    if (!motionActive || isDraft) {
      ctx.setLineDash([7, 5]);
    }
    ctx.lineWidth = motionActive && !isDraft ? 3 : 2;
    ctx.strokeStyle = borderColor;
    ctx.strokeRect(zx + 1, zy + 1, Math.max(1, zw - 2), Math.max(1, zh - 2));
    ctx.restore();

    const zoneTag = isDraft
      ? 'Set watch zone'
      : !zoneEnabled
        ? 'Zone paused'
        : motionActive
          ? `Movement · ${changedPercent.toFixed(1)}%`
          : `Watch zone · ${changedPercent.toFixed(1)}%`;

    const paddingX = 7;
    const tagHeight = 20;
    const textWidth = ctx.measureText(zoneTag).width;
    const tagWidth = Math.ceil(textWidth + paddingX * 2);
    const tagX = Math.min(Math.max(0, zx), Math.max(0, width - tagWidth));
    const tagY = zy >= tagHeight ? zy - tagHeight : Math.min(zy + 2, height - tagHeight);

    ctx.fillStyle = borderColor;
    ctx.fillRect(tagX, tagY, tagWidth, tagHeight);
    ctx.fillStyle = motionActive && !isDraft ? '#f3f2ee' : '#1a1b1e';
    ctx.fillText(zoneTag, tagX + paddingX, tagY + 3);
  }
}

export default function App() {
  const clientId = useMemo(() => getOrCreateClientId(), []);

  const [pcDevices, setPcDevices] = useState<
    Array<{ id: string; deviceId: string; name: string; deviceLabel: string }>
  >([]);
  const [serverPhones, setServerPhones] = useState<
    Array<{ id: string; name: string }>
  >([]);
  const [watchedOfflinePhone, setWatchedOfflinePhone] = useState<{
    id: string;
    name: string;
  } | null>(null);

  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [activeCameraState, setActiveCameraState] = useState<CameraState>('waiting');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [pcRetryNonce, setPcRetryNonce] = useState<number>(0);

  const [wsConnected, setWsConnected] = useState<boolean>(false);

  const [detectionProfile, setDetectionProfile] =
    useState<DetectionProfile>('fast');
  const [minConfidence, setMinConfidence] = useState<number>(0.4);
  const [modelsLoading, setModelsLoading] = useState<boolean>(true);
  const [modelsStatus, setModelsStatus] = useState<string>(
    'Loading detection models...'
  );
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [isSendingCamera, setIsSendingCamera] = useState<boolean>(false);
  const [sendingPhoneName, setSendingPhoneName] = useState<string | null>(null);
  const [phoneFacingMode, setPhoneFacingMode] = useState<'environment' | 'user'>(
    'environment'
  );
  const [wakeLockActive, setWakeLockActive] = useState<boolean>(false);
  const [sendError, setSendError] = useState<string | null>(null);

  const [enhancement, setEnhancement] = useState<EnhancementSettings>({
    enabled: true,
    autoGamma: true,
    gamma: 0.85,
  });
  const [effectiveGamma, setEffectiveGamma] = useState<number>(1.0);
  const [enhanceMode, setEnhanceMode] = useState<'webgl' | 'canvas' | 'off'>(
    'webgl'
  );

  // Motion watch zone state
  const [motionSettings, setMotionSettings] = useState<MotionZoneSettings>({
    enabled: true,
    zone: null,
    sensitivity: 65,
    soundAlert: false,
    autoInspectGemini: false,
  });
  const [motionAlert, setMotionAlert] = useState<MotionAlertState>({
    active: false,
    changedPercent: 0,
    triggerThreshold: 2.8,
    lastTriggeredAt: null,
    alertCount: 0,
  });
  const [isDrawingZone, setIsDrawingZone] = useState<boolean>(false);
  const [draftZone, setDraftZone] = useState<NormalizedZone | null>(null);

  // Gemini model & scene/zone analysis state
  const [geminiModel, setGeminiModel] =
    useState<GeminiModelId>('gemini-4-hydrogen');
  const [isScanningAll, setIsScanningAll] = useState<boolean>(false);
  const [sceneScan, setSceneScan] = useState<SceneScanResult | null>(null);
  const [sceneScanError, setSceneScanError] = useState<string | null>(null);
  const [showGeminiBoxes, setShowGeminiBoxes] = useState<boolean>(true);

  const [isInspectingZone, setIsInspectingZone] = useState<boolean>(false);
  const [zoneInspectionText, setZoneInspectionText] = useState<string | null>(null);
  const [zoneInspectionError, setZoneInspectionError] = useState<string | null>(
    null
  );
  const [zoneInspectionModel, setZoneInspectionModel] =
    useState<GeminiModelId | null>(null);

  const [fps, setFps] = useState<number>(0);
  const [detectionMs, setDetectionMs] = useState<number>(0);
  const [hasPicture, setHasPicture] = useState<boolean>(false);

  const [isDetectedOpen, setIsDetectedOpen] = useState<boolean>(false);
  const [historyItems, setHistoryItems] = useState<DetectedHistoryItem[]>([]);

  // Refs for real-time loops
  const wsRef = useRef<WebSocket | null>(null);
  const selectedCameraIdRef = useRef<string | null>(null);
  const isSendingCameraRef = useRef<boolean>(false);
  const enhancementRef = useRef<EnhancementSettings>(enhancement);
  const modelsRef = useRef<LoadedModels | null>(null);
  const minConfidenceRef = useRef<number>(minConfidence);
  const geminiModelRef = useRef<GeminiModelId>(geminiModel);

  const motionSettingsRef = useRef<MotionZoneSettings>(motionSettings);
  const draftZoneRef = useRef<NormalizedZone | null>(null);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const motionDetectorRef = useRef<ZoneMotionDetector | null>(null);
  const motionHoldUntilRef = useRef<number>(0);
  const motionActiveRef = useRef<boolean>(false);
  const latestChangedPercentRef = useRef<number>(0);
  const latestTriggerThresholdRef = useRef<number>(2.8);
  const lastMotionUiSyncRef = useRef<number>(0);
  const lastAutoInspectAtRef = useRef<number>(0);
  const isInspectingZoneRef = useRef<boolean>(false);

  const latestTracksRef = useRef<TrackedItem[]>([]);
  const sceneScanRef = useRef<SceneScanResult | null>(null);
  const showGeminiBoxesRef = useRef<boolean>(true);

  const displayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const viewerVideoRef = useRef<HTMLVideoElement | null>(null);
  const senderVideoRef = useRef<HTMLVideoElement | null>(null);
  const viewerStreamRef = useRef<MediaStream | null>(null);
  const senderStreamRef = useRef<MediaStream | null>(null);
  const wakeLockRef = useRef<WakeLockSentinel | null>(null);

  const enhancerRef = useRef<FrameEnhancer | null>(null);
  const objectTrackerRef = useRef<IoUTracker>(new IoUTracker('object'));
  const faceTrackerRef = useRef<IoUTracker>(new IoUTracker('face'));
  const historyMapRef = useRef<Map<string, DetectedHistoryItem>>(new Map());

  const latestRemoteBitmapRef = useRef<ImageBitmap | null>(null);
  const remoteBitmapDirtyRef = useRef<boolean>(false);
  const remoteDecodeSeqRef = useRef<number>(0);
  const cameraEpochRef = useRef<number>(0);
  const lastVideoTimeRef = useRef<number>(-1);
  const isDetectingRef = useRef<boolean>(false);
  const pendingDetectionFrameRef = useRef<boolean>(false);
  const frameTimestampsRef = useRef<number[]>([]);
  const lastUiStatsSyncRef = useRef<number>(0);
  const enhanceDirtyRef = useRef<boolean>(true);

  const redrawOverlayNow = useCallback(() => {
    const display = displayCanvasRef.current;
    const overlay = overlayCanvasRef.current;
    if (!display || !overlay || display.width === 0 || display.height === 0) {
      return;
    }
    drawCanvasOverlay({
      overlayCanvas: overlay,
      width: display.width,
      height: display.height,
      tracks: latestTracksRef.current,
      geminiBoxes: sceneScanRef.current?.items ?? [],
      showGeminiBoxes: showGeminiBoxesRef.current,
      zone: motionSettingsRef.current.zone,
      draftZone: draftZoneRef.current,
      zoneEnabled: motionSettingsRef.current.enabled,
      motionActive: motionActiveRef.current,
      changedPercent: latestChangedPercentRef.current,
    });
  }, []);

  useEffect(() => {
    selectedCameraIdRef.current = selectedCameraId;
  }, [selectedCameraId]);

  useEffect(() => {
    isSendingCameraRef.current = isSendingCamera;
  }, [isSendingCamera]);

  useEffect(() => {
    enhancementRef.current = enhancement;
    enhanceDirtyRef.current = true;
  }, [enhancement]);

  useEffect(() => {
    minConfidenceRef.current = minConfidence;
  }, [minConfidence]);

  useEffect(() => {
    geminiModelRef.current = geminiModel;
  }, [geminiModel]);

  useEffect(() => {
    motionSettingsRef.current = motionSettings;
    redrawOverlayNow();
  }, [motionSettings, redrawOverlayNow]);

  useEffect(() => {
    draftZoneRef.current = draftZone;
    redrawOverlayNow();
  }, [draftZone, redrawOverlayNow]);

  useEffect(() => {
    sceneScanRef.current = sceneScan;
    redrawOverlayNow();
  }, [sceneScan, redrawOverlayNow]);

  useEffect(() => {
    showGeminiBoxesRef.current = showGeminiBoxes;
    redrawOverlayNow();
  }, [showGeminiBoxes, redrawOverlayNow]);

  useEffect(() => {
    isInspectingZoneRef.current = isInspectingZone;
  }, [isInspectingZone]);

  // Unified camera list
  const cameras: CameraSource[] = useMemo(() => {
    const pcList: PcCameraSource[] = pcDevices.map((dev) => {
      const isSelected = dev.id === selectedCameraId;
      return {
        id: dev.id,
        kind: 'pc',
        deviceId: dev.deviceId,
        name: dev.name,
        deviceLabel: dev.deviceLabel,
        state: isSelected ? activeCameraState : 'waiting',
      };
    });

    const phoneList: PhoneCameraSource[] = serverPhones.map((phone) => {
      const id = `phone-${phone.id}`;
      const isSelected = id === selectedCameraId;
      const isLocal = phone.id === clientId && isSendingCamera;
      return {
        id,
        kind: 'phone',
        phoneId: phone.id,
        name: phone.name,
        state: isSelected ? activeCameraState : 'live',
        isLocal,
      };
    });

    if (
      watchedOfflinePhone &&
      !serverPhones.some((p) => p.id === watchedOfflinePhone.id)
    ) {
      phoneList.push({
        id: `phone-${watchedOfflinePhone.id}`,
        kind: 'phone',
        phoneId: watchedOfflinePhone.id,
        name: watchedOfflinePhone.name,
        state: 'offline',
        isLocal: false,
      });
    }

    return [...pcList, ...phoneList];
  }, [
    pcDevices,
    serverPhones,
    watchedOfflinePhone,
    selectedCameraId,
    activeCameraState,
    clientId,
    isSendingCamera,
  ]);

  const selectedCamera = useMemo(
    () => cameras.find((c) => c.id === selectedCameraId) ?? null,
    [cameras, selectedCameraId]
  );

  const syncHistoryState = useCallback(() => {
    const items = Array.from(historyMapRef.current.values());
    setHistoryItems(items);
  }, []);

  const resetTrackersAndHistory = useCallback(() => {
    cameraEpochRef.current += 1;
    objectTrackerRef.current.reset();
    faceTrackerRef.current.reset();
    historyMapRef.current.clear();
    latestTracksRef.current = [];
    setHistoryItems([]);
    setSceneScan(null);
    setSceneScanError(null);
    setZoneInspectionText(null);
    setZoneInspectionError(null);
    enhancerRef.current?.resetStats();
    motionDetectorRef.current?.reset();
    motionHoldUntilRef.current = 0;
    motionActiveRef.current = false;
    latestChangedPercentRef.current = 0;
    setMotionAlert((prev) => ({
      ...prev,
      active: false,
      changedPercent: 0,
    }));
    setFps(0);
    setDetectionMs(0);
    setHasPicture(false);
    frameTimestampsRef.current = [];
    lastVideoTimeRef.current = -1;
    pendingDetectionFrameRef.current = false;

    if (latestRemoteBitmapRef.current) {
      latestRemoteBitmapRef.current.close();
      latestRemoteBitmapRef.current = null;
    }
    remoteBitmapDirtyRef.current = false;

    const overlay = overlayCanvasRef.current;
    if (overlay) {
      const ctx = overlay.getContext('2d');
      ctx?.clearRect(0, 0, overlay.width, overlay.height);
    }
    const display = displayCanvasRef.current;
    if (display) {
      const ctx = display.getContext('2d');
      ctx?.clearRect(0, 0, display.width, display.height);
    }
  }, []);

  // Load TensorFlow.js models when profile changes
  useEffect(() => {
    let cancelled = false;
    setModelsLoading(true);
    setModelsError(null);

    loadDetectionModels(detectionProfile, (status) => {
      if (!cancelled) setModelsStatus(status);
    })
      .then((loaded) => {
        if (cancelled) return;
        modelsRef.current = loaded;
        setModelsLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setModelsError(
          err instanceof Error
            ? `Failed to load detection models: ${err.message}`
            : 'Failed to load detection models.'
        );
        setModelsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [detectionProfile]);

  // Enumerate PC cameras
  const scanPcCameras = useCallback(async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      setPcDevices([]);
      return;
    }
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((d) => d.kind === 'videoinput');
      const mapped = videoInputs.map((d, idx) => ({
        id: d.deviceId ? `pc-${d.deviceId}` : `pc-slot-${idx}`,
        deviceId: d.deviceId,
        name: `PC camera ${idx + 1}`,
        deviceLabel: d.label || `Camera ${idx + 1}`,
      }));

      setPcDevices(mapped);

      setSelectedCameraId((prev) => {
        if (!prev) {
          return mapped.length > 0 ? mapped[0].id : null;
        }
        if (prev.startsWith('pc-slot-')) {
          const slotIndex = Number(prev.replace('pc-slot-', ''));
          if (mapped[slotIndex]) {
            return mapped[slotIndex].id;
          }
        }
        return prev;
      });
    } catch {
      setPcDevices([]);
    }
  }, []);

  useEffect(() => {
    void scanPcCameras();
    const mediaDevices = navigator.mediaDevices;
    if (!mediaDevices || !mediaDevices.addEventListener) return;
    const handleDeviceChange = () => {
      void scanPcCameras();
    };
    mediaDevices.addEventListener('devicechange', handleDeviceChange);
    return () => {
      mediaDevices.removeEventListener('devicechange', handleDeviceChange);
    };
  }, [scanPcCameras]);

  // WebSocket client with auto-reconnect
  useEffect(() => {
    let disposed = false;
    let reconnectTimer: number | null = null;
    let attempt = 0;

    const connect = () => {
      if (disposed) return;
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws?clientId=${encodeURIComponent(
        clientId
      )}`;
      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        if (disposed) return;
        attempt = 0;
        setWsConnected(true);
        if (isSendingCameraRef.current) {
          ws.send(JSON.stringify({ type: 'phone:start' }));
        }
      };

      ws.onmessage = (event) => {
        if (disposed) return;
        if (typeof event.data === 'string') {
          try {
            const msg = JSON.parse(event.data) as {
              type?: string;
              phones?: Array<{ id: string; name: string }>;
              id?: string;
              name?: string;
            };
            if (msg.type === 'phones' && Array.isArray(msg.phones)) {
              const incoming = msg.phones;
              setServerPhones((prevPhones) => {
                const currentSelected = selectedCameraIdRef.current;
                if (currentSelected && currentSelected.startsWith('phone-')) {
                  const watchedId = currentSelected.slice('phone-'.length);
                  const stillOnline = incoming.some((p) => p.id === watchedId);
                  if (!stillOnline) {
                    const prevInfo = prevPhones.find((p) => p.id === watchedId);
                    if (prevInfo) {
                      setWatchedOfflinePhone(prevInfo);
                      setActiveCameraState('offline');
                    }
                  } else {
                    setWatchedOfflinePhone(null);
                  }
                }
                return incoming;
              });
            } else if (msg.type === 'phone:registered' && msg.id && msg.name) {
              setSendingPhoneName(msg.name);
              setSelectedCameraId((prev) => {
                const ownId = `phone-${msg.id}`;
                return prev === ownId ? prev : ownId;
              });
            }
          } catch {
            // Ignore malformed JSON
          }
          return;
        }

        if (event.data instanceof ArrayBuffer) {
          const buffer = event.data;
          if (buffer.byteLength < 3) return;
          const view = new Uint8Array(buffer);
          const idLen = view[0];
          if (idLen === 0 || 1 + idLen >= buffer.byteLength) return;

          const phoneId = new TextDecoder().decode(view.subarray(1, 1 + idLen));
          if (selectedCameraIdRef.current !== `phone-${phoneId}`) {
            return;
          }

          const jpegSlice = view.subarray(1 + idLen);
          const blob = new Blob([jpegSlice], { type: 'image/jpeg' });
          const decodeSeq = ++remoteDecodeSeqRef.current;
          const epochAtReceive = cameraEpochRef.current;

          createImageBitmap(blob)
            .then((bitmap) => {
              if (
                disposed ||
                cameraEpochRef.current !== epochAtReceive ||
                decodeSeq < remoteDecodeSeqRef.current
              ) {
                bitmap.close();
                return;
              }
              if (latestRemoteBitmapRef.current) {
                latestRemoteBitmapRef.current.close();
              }
              latestRemoteBitmapRef.current = bitmap;
              remoteBitmapDirtyRef.current = true;
              setActiveCameraState('live');
              setHasPicture(true);
            })
            .catch(() => {
              // Drop corrupt JPEG frame
            });
        }
      };

      ws.onclose = () => {
        if (disposed) return;
        setWsConnected(false);
        if (wsRef.current === ws) {
          wsRef.current = null;
        }
        const delayMs = Math.min(5000, 1000 * Math.pow(2, attempt));
        attempt += 1;
        reconnectTimer = window.setTimeout(connect, delayMs);
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connect();

    return () => {
      disposed = true;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [clientId]);

  const handleSelectCamera = useCallback(
    (cameraId: string) => {
      if (cameraId === selectedCameraIdRef.current) return;
      if (
        watchedOfflinePhone &&
        cameraId !== `phone-${watchedOfflinePhone.id}`
      ) {
        setWatchedOfflinePhone(null);
      }
      setCameraError(null);
      setSelectedCameraId(cameraId);
    },
    [watchedOfflinePhone]
  );

  const stopViewerStream = useCallback(() => {
    if (viewerStreamRef.current) {
      for (const track of viewerStreamRef.current.getTracks()) {
        track.stop();
      }
      viewerStreamRef.current = null;
    }
    if (viewerVideoRef.current) {
      viewerVideoRef.current.srcObject = null;
    }
  }, []);

  const prevSourceKeyRef = useRef<string | null>(null);
  const selectedPcDeviceId =
    selectedCamera?.kind === 'pc' ? selectedCamera.deviceId : null;
  const selectedKind = selectedCamera?.kind ?? null;
  const selectedIsLocalPhone =
    selectedCamera?.kind === 'phone' && selectedCamera.isLocal;

  useEffect(() => {
    const sourceKey = selectedCameraId ?? 'none';
    if (prevSourceKeyRef.current !== sourceKey) {
      const wasPlaceholderUpgrade =
        prevSourceKeyRef.current?.startsWith('pc-slot-') &&
        selectedCameraId?.startsWith('pc-') &&
        !selectedCameraId?.startsWith('pc-slot-');
      prevSourceKeyRef.current = sourceKey;

      if (!wasPlaceholderUpgrade) {
        resetTrackersAndHistory();
      }
    }

    if (!selectedCameraId) {
      stopViewerStream();
      setActiveCameraState('offline');
      return;
    }

    if (selectedKind === 'phone') {
      stopViewerStream();
      if (selectedIsLocalPhone) {
        setCameraError(null);
        setActiveCameraState(senderStreamRef.current ? 'live' : 'waiting');
        return;
      }

      const isOffline =
        watchedOfflinePhone !== null &&
        selectedCameraId === `phone-${watchedOfflinePhone.id}` &&
        !serverPhones.some((p) => p.id === watchedOfflinePhone.id);

      setCameraError(null);
      setActiveCameraState(isOffline ? 'offline' : 'waiting');
      return;
    }

    let cancelled = false;

    const startPcCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraError('Camera API is unavailable in this browser.');
        setActiveCameraState('offline');
        return;
      }

      setCameraError(null);
      setActiveCameraState('waiting');
      stopViewerStream();

      try {
        const videoConstraints: MediaTrackConstraints = {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 },
        };
        if (selectedPcDeviceId) {
          videoConstraints.deviceId = { exact: selectedPcDeviceId };
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: videoConstraints,
          audio: false,
        });

        if (cancelled) {
          for (const t of stream.getTracks()) t.stop();
          return;
        }

        viewerStreamRef.current = stream;
        const videoEl = viewerVideoRef.current;
        if (videoEl) {
          videoEl.srcObject = stream;
          await videoEl.play().catch(() => {});
        }

        const track = stream.getVideoTracks()[0];
        if (track) {
          track.addEventListener('ended', () => {
            if (!cancelled) {
              setActiveCameraState('offline');
            }
          });
        }

        setActiveCameraState('live');
        void scanPcCameras();
      } catch (err) {
        if (cancelled) return;
        setCameraError(formatCameraPermissionError(err));
        setActiveCameraState('offline');
      }
    };

    void startPcCamera();

    return () => {
      cancelled = true;
      stopViewerStream();
    };
  }, [
    selectedCameraId,
    selectedKind,
    selectedPcDeviceId,
    selectedIsLocalPhone,
    watchedOfflinePhone,
    serverPhones,
    stopViewerStream,
    resetTrackersAndHistory,
    scanPcCameras,
    pcRetryNonce,
  ]);

  // Wake Lock management
  const acquireWakeLock = useCallback(async () => {
    if (!('wakeLock' in navigator) || !navigator.wakeLock?.request) {
      return;
    }
    try {
      if (wakeLockRef.current) {
        await wakeLockRef.current.release().catch(() => {});
      }
      const sentinel = await navigator.wakeLock.request('screen');
      wakeLockRef.current = sentinel;
      setWakeLockActive(true);
      sentinel.addEventListener('release', () => {
        if (wakeLockRef.current === sentinel) {
          wakeLockRef.current = null;
          setWakeLockActive(false);
        }
      });
    } catch {
      setWakeLockActive(false);
    }
  }, []);

  const releaseWakeLock = useCallback(async () => {
    if (wakeLockRef.current) {
      const s = wakeLockRef.current;
      wakeLockRef.current = null;
      await s.release().catch(() => {});
    }
    setWakeLockActive(false);
  }, []);

  useEffect(() => {
    if (!isSendingCamera) return;
    const handleVisibility = () => {
      if (document.visibilityState === 'visible' && isSendingCameraRef.current) {
        void acquireWakeLock();
      }
    };
    document.addEventListener('visibilitychange', handleVisibility);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibility);
    };
  }, [isSendingCamera, acquireWakeLock]);

  const stopSendingCamera = useCallback(() => {
    setIsSendingCamera(false);
    setSendingPhoneName(null);
    if (senderStreamRef.current) {
      for (const t of senderStreamRef.current.getTracks()) {
        t.stop();
      }
      senderStreamRef.current = null;
    }
    if (senderVideoRef.current) {
      senderVideoRef.current.srcObject = null;
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'phone:stop' }));
    }
    void releaseWakeLock();

    if (selectedCameraIdRef.current === `phone-${clientId}`) {
      setSelectedCameraId(pcDevices.length > 0 ? pcDevices[0].id : null);
    }
  }, [clientId, pcDevices, releaseWakeLock]);

  const startSendingCamera = useCallback(
    async (facingMode: 'environment' | 'user') => {
      setSendError(null);
      if (!navigator.mediaDevices?.getUserMedia) {
        setSendError('Camera API is unavailable in this browser.');
        return;
      }

      stopViewerStream();

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: facingMode },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 },
          },
          audio: false,
        });

        if (senderStreamRef.current) {
          for (const t of senderStreamRef.current.getTracks()) t.stop();
        }
        senderStreamRef.current = stream;

        const senderVideo = senderVideoRef.current;
        if (senderVideo) {
          senderVideo.srcObject = stream;
          await senderVideo.play().catch(() => {});
        }

        setIsSendingCamera(true);
        setSelectedCameraId(`phone-${clientId}`);
        setActiveCameraState('live');
        void acquireWakeLock();

        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'phone:start' }));
        }
      } catch (err) {
        setSendError(formatCameraPermissionError(err));
        setIsSendingCamera(false);
      }
    },
    [acquireWakeLock, clientId, stopViewerStream]
  );

  const handleToggleSendCamera = useCallback(() => {
    if (isSendingCamera) {
      stopSendingCamera();
    } else {
      void startSendingCamera(phoneFacingMode);
    }
  }, [isSendingCamera, phoneFacingMode, startSendingCamera, stopSendingCamera]);

  const handleChangeFacingMode = useCallback(
    (mode: 'environment' | 'user') => {
      setPhoneFacingMode(mode);
      if (isSendingCamera) {
        void startSendingCamera(mode);
      }
    },
    [isSendingCamera, startSendingCamera]
  );

  // Phone JPEG WebSocket sender loop
  useEffect(() => {
    if (!isSendingCamera) return;

    let cancelled = false;
    let encoding = false;
    const sendCanvas = document.createElement('canvas');
    const sendCtx = sendCanvas.getContext('2d');
    let lastSentTime = -1;
    let lastSrcObject: MediaProvider | null = null;

    const timer = window.setInterval(() => {
      if (cancelled || encoding) return;
      const ws = wsRef.current;
      const video = senderVideoRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      if (ws.bufferedAmount > MAX_WS_BUFFER_BYTES) return;
      if (!video || video.readyState < 2 || video.videoWidth === 0) return;
      if (video.srcObject !== lastSrcObject) {
        lastSrcObject = video.srcObject;
        lastSentTime = -1;
      }
      if (video.currentTime === lastSentTime) return;
      lastSentTime = video.currentTime;

      const targetW = Math.min(1280, video.videoWidth);
      const targetH = Math.min(
        720,
        Math.round((video.videoHeight / video.videoWidth) * targetW)
      );

      if (sendCanvas.width !== targetW || sendCanvas.height !== targetH) {
        sendCanvas.width = targetW;
        sendCanvas.height = targetH;
      }

      if (!sendCtx) return;
      sendCtx.drawImage(video, 0, 0, targetW, targetH);

      encoding = true;
      sendCanvas.toBlob(
        (blob) => {
          encoding = false;
          if (cancelled || !blob) return;
          const activeWs = wsRef.current;
          if (
            activeWs &&
            activeWs.readyState === WebSocket.OPEN &&
            activeWs.bufferedAmount <= MAX_WS_BUFFER_BYTES
          ) {
            activeWs.send(blob);
          }
        },
        'image/jpeg',
        0.72
      );
    }, 33);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [isSendingCamera]);

  useEffect(() => {
    return () => {
      if (senderStreamRef.current) {
        for (const t of senderStreamRef.current.getTracks()) t.stop();
      }
      void releaseWakeLock();
    };
  }, [releaseWakeLock]);

  // Gemini zone inspection helper
  const inspectWatchZoneWithGemini = useCallback(async () => {
    const display = displayCanvasRef.current;
    const zone = motionSettingsRef.current.zone;
    if (!display || display.width === 0 || !zone || isInspectingZoneRef.current) {
      return;
    }

    const imageBase64 = captureZoneSnapshotDataUrl(display, zone);
    if (!imageBase64) return;

    const modelToUse = geminiModelRef.current;
    setIsInspectingZone(true);
    setZoneInspectionError(null);

    try {
      const res = await fetch('/api/describe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageBase64,
          model: modelToUse,
          mode: 'zone',
        }),
      });
      const data = (await res.json()) as {
        description?: string;
        model?: GeminiModelId;
        error?: string;
      };
      if (!res.ok || !data.description) {
        setZoneInspectionError(
          data.error || `Zone inspection failed (${res.status}).`
        );
      } else {
        setZoneInspectionText(data.description);
        setZoneInspectionModel(data.model ?? modelToUse);
      }
    } catch (err) {
      setZoneInspectionError(
        err instanceof Error
          ? `Network error: ${err.message}`
          : 'Failed to reach server.'
      );
    } finally {
      setIsInspectingZone(false);
    }
  }, []);

  // Gemini exhaustive full-scene object scan ("Scan all things")
  const handleScanAllObjects = useCallback(async () => {
    const display = displayCanvasRef.current;
    if (!display || display.width === 0 || display.height === 0) {
      setSceneScanError('Wait for a live camera frame before scanning.');
      return;
    }

    const imageBase64 = captureFullFrameDataUrl(display);
    if (!imageBase64) return;

    const modelToUse = geminiModelRef.current;
    const epochAtStart = cameraEpochRef.current;
    setIsScanningAll(true);
    setSceneScanError(null);

    try {
      const res = await fetch('/api/scan-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageBase64,
          model: modelToUse,
        }),
      });
      const data = (await res.json()) as {
        summary?: string;
        model?: GeminiModelId;
        items?: Array<{
          label: string;
          ymin: number;
          xmin: number;
          ymax: number;
          xmax: number;
        }>;
        error?: string;
      };

      if (cameraEpochRef.current !== epochAtStart) return;

      if (!res.ok) {
        setSceneScanError(data.error || `Scene scan failed (${res.status}).`);
        return;
      }

      const rawItems = Array.isArray(data.items) ? data.items : [];
      const scannedItems: GeminiDetectedObject[] = rawItems.map((entry, idx) => {
        const nx = entry.xmin / 1000;
        const ny = entry.ymin / 1000;
        const nw = (entry.xmax - entry.xmin) / 1000;
        const nh = (entry.ymax - entry.ymin) / 1000;
        const pxBox: [number, number, number, number] = [
          nx * display.width,
          ny * display.height,
          nw * display.width,
          nh * display.height,
        ];
        const cropDataUrl = captureCroppedThumbnail(display, pxBox);
        return {
          id: `g-${Date.now()}-${idx}`,
          label: entry.label,
          box: pxBox,
          normBox: { x: nx, y: ny, w: nw, h: nh },
          cropDataUrl,
        };
      });

      setSceneScan({
        summary: data.summary || `Found ${scannedItems.length} items in view.`,
        model: data.model ?? modelToUse,
        scannedAt: Date.now(),
        items: scannedItems,
      });
      setShowGeminiBoxes(true);
    } catch (err) {
      if (cameraEpochRef.current !== epochAtStart) return;
      setSceneScanError(
        err instanceof Error
          ? `Network error: ${err.message}`
          : 'Failed to reach server.'
      );
    } finally {
      setIsScanningAll(false);
    }
  }, []);

  // Main requestAnimationFrame loop: render -> enhance -> motion zone -> non-blocking detection
  useEffect(() => {
    const enhancer = new FrameEnhancer();
    enhancerRef.current = enhancer;
    const motionDetector = new ZoneMotionDetector();
    motionDetectorRef.current = motionDetector;
    const detectCanvas = document.createElement('canvas');
    detectCanvasRef.current = detectCanvas;

    let rafId = 0;

    const runDetection = (epoch: number) => {
      const models = modelsRef.current;
      const displayCanvas = displayCanvasRef.current;
      const overlayCanvas = overlayCanvasRef.current;
      if (!models || !displayCanvas || !overlayCanvas) return;
      if (displayCanvas.width === 0 || displayCanvas.height === 0) return;

      if (
        detectCanvas.width !== displayCanvas.width ||
        detectCanvas.height !== displayCanvas.height
      ) {
        detectCanvas.width = displayCanvas.width;
        detectCanvas.height = displayCanvas.height;
      }
      const dCtx = detectCanvas.getContext('2d', { alpha: false });
      if (!dCtx) return;
      dCtx.drawImage(displayCanvas, 0, 0);

      isDetectingRef.current = true;
      pendingDetectionFrameRef.current = false;

      detectObjectsAndFaces(models, detectCanvas, minConfidenceRef.current)
        .then(({ objects, faces, durationMs }) => {
          isDetectingRef.current = false;
          if (cameraEpochRef.current !== epoch) return;

          const now = Date.now();
          const trackedObjects = objectTrackerRef.current.update(objects, now);
          const trackedFaces = faceTrackerRef.current.update(faces, now);
          const visibleTracks = [...trackedObjects, ...trackedFaces];
          latestTracksRef.current = visibleTracks;

          drawCanvasOverlay({
            overlayCanvas,
            width: detectCanvas.width,
            height: detectCanvas.height,
            tracks: visibleTracks,
            geminiBoxes: sceneScanRef.current?.items ?? [],
            showGeminiBoxes: showGeminiBoxesRef.current,
            zone: motionSettingsRef.current.zone,
            draftZone: draftZoneRef.current,
            zoneEnabled: motionSettingsRef.current.enabled,
            motionActive: motionActiveRef.current,
            changedPercent: latestChangedPercentRef.current,
          });

          setDetectionMs(Math.round(durationMs));

          const map = historyMapRef.current;
          for (const item of map.values()) {
            item.inView = false;
          }

          for (const track of visibleTracks) {
            const label = `${track.className} #${track.id}`;
            const existing = map.get(track.key);
            if (existing) {
              existing.inView = true;
              existing.score = track.score;
              existing.lastSeenAt = now;
              if (now - existing.lastCropAt >= CROP_REFRESH_MS) {
                const updatedCrop = captureCroppedThumbnail(
                  detectCanvas,
                  track.box
                );
                if (updatedCrop) {
                  existing.cropDataUrl = updatedCrop;
                  existing.lastCropAt = now;
                }
              }
            } else {
              const cropDataUrl = captureCroppedThumbnail(
                detectCanvas,
                track.box
              );
              if (cropDataUrl) {
                map.set(track.key, {
                  key: track.key,
                  id: track.id,
                  kind: track.kind,
                  className: track.className,
                  label,
                  score: track.score,
                  inView: true,
                  firstSeenAt: now,
                  lastSeenAt: now,
                  lastCropAt: now,
                  cropDataUrl,
                  geminiStatus: 'idle',
                });
              }
            }
          }

          if (map.size > MAX_HISTORY_ITEMS) {
            const sorted = Array.from(map.values()).sort((a, b) => {
              if (a.inView !== b.inView) return a.inView ? 1 : -1;
              return a.lastSeenAt - b.lastSeenAt;
            });
            const toRemove = map.size - MAX_HISTORY_ITEMS;
            for (let i = 0; i < toRemove; i++) {
              map.delete(sorted[i].key);
            }
          }

          syncHistoryState();
        })
        .catch(() => {
          isDetectingRef.current = false;
        });
    };

    const tick = (nowPerf: number) => {
      rafId = window.requestAnimationFrame(tick);

      const displayCanvas = displayCanvasRef.current;
      if (!displayCanvas) return;

      const selectedId = selectedCameraIdRef.current;
      if (!selectedId) return;

      const isLocalPhone =
        isSendingCameraRef.current && selectedId === `phone-${clientId}`;
      const isRemotePhone = selectedId.startsWith('phone-') && !isLocalPhone;

      let source: HTMLVideoElement | ImageBitmap | null = null;
      let width = 0;
      let height = 0;
      let isNewFrame = false;

      if (isRemotePhone) {
        const bmp = latestRemoteBitmapRef.current;
        if (bmp && (remoteBitmapDirtyRef.current || enhanceDirtyRef.current)) {
          source = bmp;
          width = bmp.width;
          height = bmp.height;
          isNewFrame = remoteBitmapDirtyRef.current;
          remoteBitmapDirtyRef.current = false;
        }
      } else {
        const videoEl = isLocalPhone
          ? senderVideoRef.current
          : viewerVideoRef.current;
        if (videoEl && videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
          const videoChanged = videoEl.currentTime !== lastVideoTimeRef.current;
          if (videoChanged || enhanceDirtyRef.current) {
            source = videoEl;
            width = videoEl.videoWidth;
            height = videoEl.videoHeight;
            isNewFrame = videoChanged;
            lastVideoTimeRef.current = videoEl.currentTime;
          }
        }
      }

      if (source && width > 0 && height > 0) {
        enhanceDirtyRef.current = false;
        const stats = enhancer.render(
          source,
          width,
          height,
          displayCanvas,
          enhancementRef.current
        );

        if (isNewFrame) {
          setHasPicture(true);
          pendingDetectionFrameRef.current = true;

          const ts = frameTimestampsRef.current;
          ts.push(nowPerf);
          while (ts.length > 0 && nowPerf - ts[0] > 1000) {
            ts.shift();
          }

          // Pixel-difference motion evaluation inside watch zone
          const mCfg = motionSettingsRef.current;
          const nowMs = Date.now();
          if (mCfg.zone && mCfg.enabled) {
            const evalRes = motionDetector.evaluate(
              displayCanvas,
              mCfg.zone,
              mCfg.sensitivity
            );
            latestChangedPercentRef.current = evalRes.changedPercent;
            latestTriggerThresholdRef.current = evalRes.triggerThreshold;

            const wasActive = motionActiveRef.current;
            if (evalRes.triggered) {
              motionHoldUntilRef.current = nowMs + MOTION_HOLD_MS;
              motionActiveRef.current = true;

              if (!wasActive) {
                if (mCfg.soundAlert) {
                  playMotionAlertTone();
                }
                setMotionAlert((prev) => ({
                  active: true,
                  changedPercent: evalRes.changedPercent,
                  triggerThreshold: evalRes.triggerThreshold,
                  lastTriggeredAt: nowMs,
                  alertCount: prev.alertCount + 1,
                }));
              }

              if (
                mCfg.autoInspectGemini &&
                nowMs - lastAutoInspectAtRef.current >= AUTO_INSPECT_COOLDOWN_MS &&
                !isInspectingZoneRef.current
              ) {
                lastAutoInspectAtRef.current = nowMs;
                void inspectWatchZoneWithGemini();
              }
            } else if (nowMs > motionHoldUntilRef.current) {
              motionActiveRef.current = false;
            }

            redrawOverlayNow();

            if (nowPerf - lastMotionUiSyncRef.current >= 140) {
              lastMotionUiSyncRef.current = nowPerf;
              setMotionAlert((prev) => ({
                ...prev,
                active: motionActiveRef.current,
                changedPercent: evalRes.changedPercent,
                triggerThreshold: evalRes.triggerThreshold,
              }));
            }
          } else if (motionActiveRef.current) {
            motionActiveRef.current = false;
            redrawOverlayNow();
            setMotionAlert((prev) => ({ ...prev, active: false }));
          }
        }

        if (nowPerf - lastUiStatsSyncRef.current >= 250) {
          lastUiStatsSyncRef.current = nowPerf;
          setFps(frameTimestampsRef.current.length);
          setEffectiveGamma(stats.effectiveGamma);
          setEnhanceMode(stats.mode);
        }
      } else if (nowPerf - lastUiStatsSyncRef.current >= 500) {
        lastUiStatsSyncRef.current = nowPerf;
        const ts = frameTimestampsRef.current;
        while (ts.length > 0 && nowPerf - ts[0] > 1000) {
          ts.shift();
        }
        setFps(ts.length);

        if (motionActiveRef.current && Date.now() > motionHoldUntilRef.current) {
          motionActiveRef.current = false;
          redrawOverlayNow();
          setMotionAlert((prev) => ({ ...prev, active: false }));
        }
      }

      if (
        pendingDetectionFrameRef.current &&
        !isDetectingRef.current &&
        modelsRef.current
      ) {
        runDetection(cameraEpochRef.current);
      }
    };

    rafId = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(rafId);
      enhancer.dispose();
      enhancerRef.current = null;
      motionDetectorRef.current = null;
    };
  }, [clientId, inspectWatchZoneWithGemini, redrawOverlayNow, syncHistoryState]);

  // Interactive zone drawing handlers
  const getNormalizedCanvasPoint = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>): { x: number; y: number } | null => {
      const target = overlayCanvasRef.current;
      if (!target) return null;
      const rect = target.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return null;
      const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      return { x, y };
    },
    []
  );

  const handleCanvasPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>) => {
      if (!isDrawingZone) return;
      const pt = getNormalizedCanvasPoint(e);
      if (!pt) return;
      e.currentTarget.setPointerCapture(e.pointerId);
      dragStartRef.current = pt;
      setDraftZone({ x: pt.x, y: pt.y, w: 0.01, h: 0.01 });
    },
    [getNormalizedCanvasPoint, isDrawingZone]
  );

  const handleCanvasPointerMove = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>) => {
      if (!isDrawingZone || !dragStartRef.current) return;
      const pt = getNormalizedCanvasPoint(e);
      if (!pt) return;
      const start = dragStartRef.current;
      const x = Math.min(start.x, pt.x);
      const y = Math.min(start.y, pt.y);
      const w = Math.abs(pt.x - start.x);
      const h = Math.abs(pt.y - start.y);
      setDraftZone({ x, y, w, h });
    },
    [getNormalizedCanvasPoint, isDrawingZone]
  );

  const handleCanvasPointerUp = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>) => {
      if (!isDrawingZone || !dragStartRef.current) return;
      const pt = getNormalizedCanvasPoint(e);
      const start = dragStartRef.current;
      dragStartRef.current = null;
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }

      if (pt) {
        const x = Math.min(start.x, pt.x);
        const y = Math.min(start.y, pt.y);
        const w = Math.abs(pt.x - start.x);
        const h = Math.abs(pt.y - start.y);

        if (w >= 0.03 && h >= 0.03) {
          motionDetectorRef.current?.reset();
          motionHoldUntilRef.current = 0;
          motionActiveRef.current = false;
          setMotionSettings((prev) => ({
            ...prev,
            enabled: true,
            zone: { x, y, w, h },
          }));
          setMotionAlert((prev) => ({
            ...prev,
            active: false,
            changedPercent: 0,
          }));
          setZoneInspectionText(null);
          setZoneInspectionError(null);
          setIsDrawingZone(false);
        }
      }

      setDraftZone(null);
    },
    [getNormalizedCanvasPoint, isDrawingZone]
  );

  const handleClearZone = useCallback(() => {
    setIsDrawingZone(false);
    setDraftZone(null);
    motionDetectorRef.current?.reset();
    motionHoldUntilRef.current = 0;
    motionActiveRef.current = false;
    latestChangedPercentRef.current = 0;
    setMotionSettings((prev) => ({ ...prev, zone: null }));
    setMotionAlert({
      active: false,
      changedPercent: 0,
      triggerThreshold: 2.8,
      lastTriggeredAt: null,
      alertCount: 0,
    });
    setZoneInspectionText(null);
    setZoneInspectionError(null);
  }, []);

  // Ask Gemini handler for individual item in DetectedPanel
  const handleAskGemini = useCallback(
    async (itemKey: string) => {
      const item = historyMapRef.current.get(itemKey);
      if (!item || !item.cropDataUrl) return;

      item.geminiStatus = 'loading';
      item.geminiError = undefined;
      syncHistoryState();

      const epochAtCall = cameraEpochRef.current;

      try {
        const res = await fetch('/api/describe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            imageBase64: item.cropDataUrl,
            label: item.label,
            model: geminiModelRef.current,
            mode: 'item',
          }),
        });

        const data = (await res.json()) as {
          description?: string;
          error?: string;
        };

        if (cameraEpochRef.current !== epochAtCall) return;
        const current = historyMapRef.current.get(itemKey);
        if (!current) return;

        if (!res.ok || !data.description) {
          current.geminiStatus = 'error';
          current.geminiError =
            data.error || `Request failed (${res.status}).`;
        } else {
          current.geminiStatus = 'done';
          current.geminiText = data.description;
          current.geminiError = undefined;
        }
      } catch (err) {
        if (cameraEpochRef.current !== epochAtCall) return;
        const current = historyMapRef.current.get(itemKey);
        if (!current) return;
        current.geminiStatus = 'error';
        current.geminiError =
          err instanceof Error
            ? `Network error: ${err.message}`
            : 'Failed to reach server.';
      }

      syncHistoryState();
    },
    [syncHistoryState]
  );

  const inViewItems = useMemo(
    () =>
      historyItems
        .filter((i) => i.inView)
        .sort((a, b) => b.firstSeenAt - a.firstSeenAt),
    [historyItems]
  );

  const seenEarlierItems = useMemo(
    () =>
      historyItems
        .filter((i) => !i.inView)
        .sort((a, b) => b.lastSeenAt - a.lastSeenAt)
        .slice(0, MAX_SEEN_EARLIER),
    [historyItems]
  );

  const waitingMessage = useMemo(() => {
    if (cameraError) return null;
    if (!selectedCamera) {
      return 'No camera selected. Connect a webcam, click Rescan, or tap Send this camera on a phone.';
    }
    if (selectedCamera.kind === 'phone') {
      if (selectedCamera.state === 'offline') {
        return `${selectedCamera.name} is offline. Waiting for it to reconnect.`;
      }
      if (!hasPicture) {
        return `Waiting for ${selectedCamera.name}`;
      }
    } else if (selectedCamera.kind === 'pc' && !hasPicture) {
      return `Waiting for ${selectedCamera.name}`;
    }
    return null;
  }, [cameraError, hasPicture, selectedCamera]);

  return (
    <div className="flex min-h-screen flex-col bg-[#1e2024] text-[#f3f2ee]">
      <TopBar
        cameras={cameras}
        selectedCameraId={selectedCameraId}
        onSelectCamera={handleSelectCamera}
        onRescan={() => {
          void scanPcCameras();
          if (selectedKind === 'pc' && (cameraError || activeCameraState === 'offline')) {
            setPcRetryNonce((n) => n + 1);
          }
        }}
        detectedInViewCount={inViewItems.length}
        isDetectedOpen={isDetectedOpen}
        onToggleDetected={() => setIsDetectedOpen((v) => !v)}
        wsConnected={wsConnected}
        motionAlert={motionAlert}
        hasWatchZone={motionSettings.zone !== null && motionSettings.enabled}
      />

      <div className="flex flex-1 min-h-0 flex-col lg:flex-row">
        <DetectedPanel
          isOpen={isDetectedOpen}
          onClose={() => setIsDetectedOpen(false)}
          inViewItems={inViewItems}
          seenEarlierItems={seenEarlierItems}
          onAskGemini={(key) => void handleAskGemini(key)}
          sceneScan={sceneScan}
          onClearSceneScan={() => setSceneScan(null)}
          activeGeminiModel={geminiModel}
        />

        <main className="flex flex-1 min-w-0 flex-col p-3 sm:p-4 gap-3">
          {/* Viewfinder stage */}
          <div className="relative flex flex-1 min-h-[280px] items-center justify-center rounded-md border border-[#3f434e] bg-[#16171a] p-2.5">
            {/* Four amber viewfinder corner brackets */}
            <span
              aria-hidden="true"
              className="pointer-events-none absolute top-0 left-0 z-20 h-6 w-6 border-t-2 border-l-2 border-[#f2b632]"
            />
            <span
              aria-hidden="true"
              className="pointer-events-none absolute top-0 right-0 z-20 h-6 w-6 border-t-2 border-r-2 border-[#f2b632]"
            />
            <span
              aria-hidden="true"
              className="pointer-events-none absolute bottom-0 left-0 z-20 h-6 w-6 border-b-2 border-l-2 border-[#f2b632]"
            />
            <span
              aria-hidden="true"
              className="pointer-events-none absolute bottom-0 right-0 z-20 h-6 w-6 border-b-2 border-r-2 border-[#f2b632]"
            />

            <div className="relative inline-flex max-h-[68vh] max-w-full items-center justify-center overflow-hidden">
              <canvas
                ref={displayCanvasRef}
                className="block h-auto w-auto max-h-[68vh] max-w-full"
              />
              <canvas
                ref={overlayCanvasRef}
                onPointerDown={handleCanvasPointerDown}
                onPointerMove={handleCanvasPointerMove}
                onPointerUp={handleCanvasPointerUp}
                onPointerCancel={handleCanvasPointerUp}
                className={`absolute inset-0 h-full w-full ${
                  isDrawingZone
                    ? 'pointer-events-auto cursor-crosshair touch-none'
                    : 'pointer-events-none'
                }`}
              />
            </div>

            {isDrawingZone && (
              <div className="pointer-events-none absolute top-3 left-1/2 z-20 -translate-x-1/2 rounded border border-[#f2b632] bg-[#27292f] px-3.5 py-1.5 text-xs font-medium text-[#f3f2ee]">
                Drag a rectangle over the video to set the watch zone
              </div>
            )}

            {cameraError && (
              <div
                role="alert"
                className="absolute inset-4 z-10 flex items-center justify-center"
              >
                <div className="max-w-md rounded-md border border-[#e5534b] bg-[#27292f] p-4 text-sm leading-relaxed text-[#f3f2ee]">
                  <p className="font-semibold">Camera unavailable</p>
                  <p className="mt-1.5 text-[#a9acb5]">{cameraError}</p>
                </div>
              </div>
            )}

            {!cameraError && waitingMessage && (
              <div className="pointer-events-none absolute inset-4 z-10 flex items-center justify-center">
                <p className="rounded-md border border-[#3f434e] bg-[#27292f] px-4 py-2.5 text-sm text-[#f3f2ee]">
                  {waitingMessage}
                </p>
              </div>
            )}

            {modelsLoading && (
              <div
                role="status"
                aria-live="polite"
                className="absolute bottom-3 left-3 z-20 rounded border border-[#3f434e] bg-[#27292f] px-3 py-1.5 text-xs text-[#f3f2ee]"
              >
                {modelsStatus}
              </div>
            )}

            {modelsError && (
              <div
                role="alert"
                className="absolute bottom-3 left-3 z-20 rounded border border-[#e5534b] bg-[#3b2223] px-3 py-1.5 text-xs text-[#f3f2ee]"
              >
                {modelsError}
              </div>
            )}
          </div>

          {/* Telemetry & quick action strip */}
          <div className="flex flex-wrap items-center justify-between gap-2.5 rounded-md border border-[#3f434e] bg-[#27292f] px-3.5 py-2.5">
            <div className="flex flex-wrap items-center gap-3 text-xs text-[#a9acb5] tabular-nums">
              <span className="font-medium text-[#f3f2ee]">{fps} fps</span>
              <span aria-hidden="true">/</span>
              <span>Detection {detectionMs} ms</span>
              <span aria-hidden="true">/</span>
              <span>In view {inViewItems.length}</span>
              <span aria-hidden="true">/</span>
              <span>
                {selectedCamera
                  ? `${selectedCamera.name} · ${selectedCamera.state}`
                  : 'No camera active'}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  setIsDrawingZone((v) => !v);
                  setDraftZone(null);
                }}
                className={`min-h-[44px] rounded-md border px-3.5 py-2 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
                  isDrawingZone
                    ? 'border-[#f2b632] bg-[#f2b632] font-semibold text-[#1a1b1e]'
                    : 'border-[#3f434e] bg-[#1e2024] text-[#f3f2ee] hover:bg-[#31343c]'
                }`}
              >
                {isDrawingZone
                  ? 'Cancel draw'
                  : motionSettings.zone
                    ? 'Redraw watch zone'
                    : 'Draw watch zone'}
              </button>

              {motionSettings.zone && (
                <button
                  type="button"
                  disabled={isInspectingZone}
                  onClick={() => void inspectWatchZoneWithGemini()}
                  className="min-h-[44px] rounded-md border border-[#3f434e] bg-[#1e2024] px-3.5 py-2 text-xs font-medium text-[#f3f2ee] transition-colors hover:border-[#f2b632] hover:bg-[#31343c] disabled:opacity-60 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632]"
                >
                  {isInspectingZone ? 'Inspecting zone...' : 'Inspect zone'}
                </button>
              )}

              <button
                type="button"
                disabled={isScanningAll}
                onClick={() => void handleScanAllObjects()}
                className="min-h-[44px] rounded-md border border-[#f2b632] bg-[#f2b632] px-3.5 py-2 text-xs font-semibold text-[#1a1b1e] transition-colors hover:bg-[#e0a624] disabled:opacity-60 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f3f2ee]"
              >
                {isScanningAll ? 'Scanning view...' : 'Scan all things'}
              </button>
            </div>
          </div>

          {/* Movement alert banner */}
          {motionSettings.zone &&
            motionSettings.enabled &&
            (motionAlert.active || zoneInspectionText || zoneInspectionError) && (
              <div
                role="status"
                aria-live="polite"
                className={`rounded-md border px-3.5 py-2.5 text-xs ${
                  motionAlert.active
                    ? 'border-[#e5534b] bg-[#362022] text-[#f3f2ee]'
                    : 'border-[#3f434e] bg-[#27292f] text-[#f3f2ee]'
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-2.5">
                    <span className="font-semibold">
                      {motionAlert.active
                        ? 'Movement detected in watch zone'
                        : 'Watch zone report'}
                    </span>
                    <span className="tabular-nums text-[#a9acb5]">
                      Delta {motionAlert.changedPercent.toFixed(1)}% (threshold{' '}
                      {motionAlert.triggerThreshold.toFixed(1)}%)
                    </span>
                    {motionAlert.lastTriggeredAt && (
                      <span className="tabular-nums text-[#a9acb5]">
                        Last trigger{' '}
                        {new Date(motionAlert.lastTriggeredAt).toLocaleTimeString()}
                      </span>
                    )}
                  </div>

                  {zoneInspectionText && (
                    <button
                      type="button"
                      onClick={() => {
                        setZoneInspectionText(null);
                        setZoneInspectionError(null);
                      }}
                      className="min-h-[44px] rounded px-2.5 text-xs text-[#a9acb5] hover:text-[#f3f2ee] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632]"
                    >
                      Dismiss
                    </button>
                  )}
                </div>

                {zoneInspectionText && (
                  <p className="mt-2 border-t border-[#3f434e] pt-2 leading-relaxed text-[#f3f2ee]">
                    <span className="font-medium text-[#f2b632]">
                      {zoneInspectionModel ?? geminiModel}:{' '}
                    </span>
                    {zoneInspectionText}
                  </p>
                )}

                {zoneInspectionError && (
                  <p className="mt-2 border-t border-[#e5534b]/40 pt-2 text-[#f1948a]">
                    {zoneInspectionError}
                  </p>
                )}
              </div>
            )}

          {/* Gemini exhaustive scene scan result card */}
          {(sceneScan || sceneScanError) && (
            <div className="rounded-md border border-[#3f434e] bg-[#27292f] px-3.5 py-3 text-xs">
              {sceneScanError ? (
                <div className="flex items-center justify-between gap-2 text-[#f1948a]">
                  <span>{sceneScanError}</span>
                  <button
                    type="button"
                    onClick={() => setSceneScanError(null)}
                    className="min-h-[44px] rounded px-2.5 text-[#a9acb5] hover:text-[#f3f2ee]"
                  >
                    Dismiss
                  </button>
                </div>
              ) : (
                sceneScan && (
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-semibold text-[#f2b632]">
                          Full scene scan ({sceneScan.model})
                        </span>
                        <span className="tabular-nums text-[#a9acb5]">
                          {sceneScan.items.length} items located
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setShowGeminiBoxes((v) => !v)}
                          className="min-h-[44px] rounded border border-[#3f434e] bg-[#1e2024] px-3 py-1.5 text-xs text-[#f3f2ee] hover:bg-[#31343c] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632]"
                        >
                          {showGeminiBoxes ? 'Hide scan boxes' : 'Show scan boxes'}
                        </button>
                        <button
                          type="button"
                          onClick={() => setSceneScan(null)}
                          className="min-h-[44px] rounded border border-[#3f434e] bg-[#1e2024] px-3 py-1.5 text-xs text-[#a9acb5] hover:text-[#f3f2ee] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632]"
                        >
                          Clear
                        </button>
                      </div>
                    </div>

                    <p className="leading-relaxed text-[#f3f2ee]">
                      {sceneScan.summary}
                    </p>

                    {sceneScan.items.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 pt-1 text-xs text-[#a9acb5]">
                        {sceneScan.items.map((item, idx) => (
                          <span key={item.id}>
                            {item.label}
                            {idx < sceneScan.items.length - 1 ? ' · ' : ''}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )
              )}
            </div>
          )}
        </main>

        <ControlSidebar
          isSendingCamera={isSendingCamera}
          sendingPhoneName={sendingPhoneName}
          phoneFacingMode={phoneFacingMode}
          onChangeFacingMode={handleChangeFacingMode}
          onToggleSendCamera={handleToggleSendCamera}
          wakeLockActive={wakeLockActive}
          sendError={sendError}
          enhancement={enhancement}
          effectiveGamma={effectiveGamma}
          enhanceMode={enhanceMode}
          onToggleEnhancement={() =>
            setEnhancement((prev) => ({ ...prev, enabled: !prev.enabled }))
          }
          onChangeGamma={(val) =>
            setEnhancement((prev) => ({
              ...prev,
              autoGamma: false,
              gamma: val,
            }))
          }
          onResetAutoGamma={() =>
            setEnhancement((prev) => ({ ...prev, autoGamma: true }))
          }
          motionSettings={motionSettings}
          motionAlert={motionAlert}
          isDrawingZone={isDrawingZone}
          onToggleDrawingZone={() => {
            setIsDrawingZone((v) => !v);
            setDraftZone(null);
          }}
          onClearZone={handleClearZone}
          onToggleZoneEnabled={() =>
            setMotionSettings((prev) => ({ ...prev, enabled: !prev.enabled }))
          }
          onChangeSensitivity={(sensitivity) =>
            setMotionSettings((prev) => ({ ...prev, sensitivity }))
          }
          onToggleSoundAlert={() =>
            setMotionSettings((prev) => ({
              ...prev,
              soundAlert: !prev.soundAlert,
            }))
          }
          onToggleAutoInspectGemini={() =>
            setMotionSettings((prev) => ({
              ...prev,
              autoInspectGemini: !prev.autoInspectGemini,
            }))
          }
          geminiModel={geminiModel}
          onChangeGeminiModel={setGeminiModel}
          detectionProfile={detectionProfile}
          onChangeDetectionProfile={setDetectionProfile}
          minConfidence={minConfidence}
          onChangeMinConfidence={setMinConfidence}
          isScanningAll={isScanningAll}
          onScanAllObjects={() => void handleScanAllObjects()}
        />
      </div>

      {/* Offscreen video elements kept rendered so browsers decode frames reliably */}
      <video
        ref={viewerVideoRef}
        playsInline
        muted
        autoPlay
        className="pointer-events-none fixed -left-[9999px] top-0 h-px w-px opacity-0"
      />
      <video
        ref={senderVideoRef}
        playsInline
        muted
        autoPlay
        className="pointer-events-none fixed -left-[9999px] top-0 h-px w-px opacity-0"
      />
    </div>
  );
}
