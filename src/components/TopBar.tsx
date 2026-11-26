import type { CameraSource, MotionAlertState } from '../types/vision';

interface TopBarProps {
  cameras: CameraSource[];
  selectedCameraId: string | null;
  onSelectCamera: (cameraId: string) => void;
  onRescan: () => void;
  detectedInViewCount: number;
  isDetectedOpen: boolean;
  onToggleDetected: () => void;
  wsConnected: boolean;
  motionAlert: MotionAlertState;
  hasWatchZone: boolean;
}

export function TopBar({
  cameras,
  selectedCameraId,
  onSelectCamera,
  onRescan,
  detectedInViewCount,
  isDetectedOpen,
  onToggleDetected,
  wsConnected,
  motionAlert,
  hasWatchZone,
}: TopBarProps) {
  return (
    <header className="w-full border-b border-[#3f434e] bg-[#27292f] px-3 py-2.5 sm:px-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onToggleDetected}
            aria-expanded={isDetectedOpen}
            className={`min-h-[44px] shrink-0 whitespace-nowrap rounded-md border px-3.5 py-2 text-sm font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
              isDetectedOpen
                ? 'border-[#f2b632] bg-[#31343c] text-[#f3f2ee]'
                : 'border-[#3f434e] bg-[#1e2024] text-[#f3f2ee] hover:bg-[#31343c]'
            }`}
          >
            Detected (<span className="tabular-nums">{detectedInViewCount}</span>)
          </button>

          <span className="hidden text-sm font-semibold tracking-tight text-[#f3f2ee] sm:inline">
            Live Vision
          </span>
        </div>

        <div
          className="flex flex-1 flex-wrap items-center gap-2"
          role="group"
          aria-label="Camera sources"
        >
          {cameras.map((cam) => {
            const isActive = cam.id === selectedCameraId;
            return (
              <button
                key={cam.id}
                type="button"
                onClick={() => onSelectCamera(cam.id)}
                aria-pressed={isActive}
                className={`min-h-[44px] shrink-0 whitespace-nowrap rounded-md border px-3.5 py-2 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
                  isActive
                    ? 'border-[#f2b632] bg-[#f2b632] font-semibold text-[#1a1b1e]'
                    : 'border-[#3f434e] bg-[#1e2024] text-[#f3f2ee] hover:bg-[#31343c]'
                }`}
              >
                <span>{cam.name}</span>
                <span
                  className={`ml-2 text-xs font-normal ${
                    isActive ? 'text-[#1a1b1e]/80' : 'text-[#a9acb5]'
                  }`}
                >
                  {cam.state}
                </span>
              </button>
            );
          })}

          <button
            type="button"
            onClick={onRescan}
            className="min-h-[44px] shrink-0 whitespace-nowrap rounded-md border border-[#3f434e] bg-[#1e2024] px-3.5 py-2 text-sm text-[#f3f2ee] transition-colors hover:bg-[#31343c] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632]"
          >
            Rescan
          </button>
        </div>

        <div className="flex items-center gap-2">
          {hasWatchZone && motionAlert.active && (
            <div
              role="status"
              aria-live="assertive"
              className="min-h-[44px] flex items-center rounded-md border border-[#e5534b] bg-[#3b2223] px-3.5 py-2 text-sm font-semibold text-[#f3f2ee] whitespace-nowrap tabular-nums"
            >
              Zone alert · {motionAlert.changedPercent.toFixed(1)}%
            </div>
          )}

          {!wsConnected && (
            <div
              role="status"
              aria-live="polite"
              className="min-h-[44px] flex items-center rounded-md border border-[#e5534b] bg-[#3b2223] px-3.5 py-2 text-sm font-medium text-[#f3f2ee] whitespace-nowrap"
            >
              Connection lost
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
