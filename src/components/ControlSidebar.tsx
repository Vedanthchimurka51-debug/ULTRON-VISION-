import { useState } from 'react';
import type {
  DetectionProfile,
  EnhancementSettings,
  GeminiModelId,
  MotionAlertState,
  MotionZoneSettings,
} from '../types/vision';

interface ControlSidebarProps {
  // Phone camera relay
  isSendingCamera: boolean;
  sendingPhoneName: string | null;
  phoneFacingMode: 'environment' | 'user';
  onChangeFacingMode: (mode: 'environment' | 'user') => void;
  onToggleSendCamera: () => void;
  wakeLockActive: boolean;
  sendError: string | null;

  // Enhancement
  enhancement: EnhancementSettings;
  effectiveGamma: number;
  enhanceMode: 'webgl' | 'canvas' | 'off';
  onToggleEnhancement: () => void;
  onChangeGamma: (gamma: number) => void;
  onResetAutoGamma: () => void;

  // Watch zone & motion detection
  motionSettings: MotionZoneSettings;
  motionAlert: MotionAlertState;
  isDrawingZone: boolean;
  onToggleDrawingZone: () => void;
  onClearZone: () => void;
  onToggleZoneEnabled: () => void;
  onChangeSensitivity: (sensitivity: number) => void;
  onToggleSoundAlert: () => void;
  onToggleAutoInspectGemini: () => void;

  // Gemini & local detection models
  geminiModel: GeminiModelId;
  onChangeGeminiModel: (model: GeminiModelId) => void;
  detectionProfile: DetectionProfile;
  onChangeDetectionProfile: (profile: DetectionProfile) => void;
  minConfidence: number;
  onChangeMinConfidence: (val: number) => void;
  isScanningAll: boolean;
  onScanAllObjects: () => void;
}

export function ControlSidebar({
  isSendingCamera,
  sendingPhoneName,
  phoneFacingMode,
  onChangeFacingMode,
  onToggleSendCamera,
  wakeLockActive,
  sendError,
  enhancement,
  effectiveGamma,
  enhanceMode,
  onToggleEnhancement,
  onChangeGamma,
  onResetAutoGamma,
  motionSettings,
  motionAlert,
  isDrawingZone,
  onToggleDrawingZone,
  onClearZone,
  onToggleZoneEnabled,
  onChangeSensitivity,
  onToggleSoundAlert,
  onToggleAutoInspectGemini,
  geminiModel,
  onChangeGeminiModel,
  detectionProfile,
  onChangeDetectionProfile,
  minConfidence,
  onChangeMinConfidence,
  isScanningAll,
  onScanAllObjects,
}: ControlSidebarProps) {
  const [activeTab, setActiveTab] = useState<'zone-ai' | 'camera-enhance'>(
    'zone-ai'
  );

  const hasZone = motionSettings.zone !== null;

  return (
    <aside
      aria-label="Controls"
      className="flex w-full shrink-0 flex-col border-t border-[#3f434e] bg-[#27292f] lg:w-80 lg:border-t-0 lg:border-l"
    >
      <div
        role="tablist"
        aria-label="Control sections"
        className="grid grid-cols-2 gap-1.5 border-b border-[#3f434e] p-2.5"
      >
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'zone-ai'}
          onClick={() => setActiveTab('zone-ai')}
          className={`min-h-[44px] rounded-md border px-3 py-2 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
            activeTab === 'zone-ai'
              ? 'border-[#f2b632] bg-[#31343c] text-[#f3f2ee]'
              : 'border-transparent bg-[#1e2024] text-[#a9acb5] hover:text-[#f3f2ee]'
          }`}
        >
          Zone and AI
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'camera-enhance'}
          onClick={() => setActiveTab('camera-enhance')}
          className={`min-h-[44px] rounded-md border px-3 py-2 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
            activeTab === 'camera-enhance'
              ? 'border-[#f2b632] bg-[#31343c] text-[#f3f2ee]'
              : 'border-transparent bg-[#1e2024] text-[#a9acb5] hover:text-[#f3f2ee]'
          }`}
        >
          Camera and enhance
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {activeTab === 'zone-ai' ? (
          <>
            <section aria-labelledby="heading-watch-zone" className="space-y-3">
              <div className="flex items-center justify-between gap-2">
                <h2
                  id="heading-watch-zone"
                  className="text-sm font-semibold text-[#f3f2ee]"
                >
                  Movement watch zone
                </h2>
                <span className="text-xs text-[#a9acb5]">
                  {!hasZone
                    ? 'no zone'
                    : motionSettings.enabled
                      ? motionAlert.active
                        ? 'alert'
                        : 'armed'
                      : 'paused'}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={onToggleDrawingZone}
                  className={`min-h-[44px] rounded-md border px-3 py-2 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
                    isDrawingZone
                      ? 'border-[#f2b632] bg-[#f2b632] text-[#1a1b1e]'
                      : 'border-[#3f434e] bg-[#1e2024] text-[#f3f2ee] hover:bg-[#31343c]'
                  }`}
                >
                  {isDrawingZone
                    ? 'Cancel draw'
                    : hasZone
                      ? 'Redraw zone'
                      : 'Draw zone'}
                </button>

                <button
                  type="button"
                  disabled={!hasZone}
                  onClick={onClearZone}
                  className="min-h-[44px] rounded-md border border-[#3f434e] bg-[#1e2024] px-3 py-2 text-xs font-medium text-[#f3f2ee] transition-colors hover:bg-[#31343c] disabled:opacity-45 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632]"
                >
                  Clear zone
                </button>
              </div>

              {hasZone && (
                <div className="space-y-2.5 rounded-md border border-[#3f434e] bg-[#1e2024] p-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[#a9acb5]">Pixel difference</span>
                    <span
                      className={`tabular-nums font-semibold ${
                        motionAlert.active ? 'text-[#e5534b]' : 'text-[#f3f2ee]'
                      }`}
                    >
                      {motionAlert.changedPercent.toFixed(1)}% /{' '}
                      {motionAlert.triggerThreshold.toFixed(1)}%
                    </span>
                  </div>

                  <div className="h-2 w-full overflow-hidden rounded bg-[#27292f]">
                    <div
                      className={`h-full transition-all ${
                        motionAlert.active ? 'bg-[#e5534b]' : 'bg-[#f2b632]'
                      }`}
                      style={{
                        width: `${Math.min(
                          100,
                          (motionAlert.changedPercent /
                            Math.max(1, motionAlert.triggerThreshold * 2)) *
                            100
                        )}%`,
                      }}
                    />
                  </div>

                  <button
                    type="button"
                    onClick={onToggleZoneEnabled}
                    className="min-h-[44px] w-full rounded border border-[#3f434e] bg-[#27292f] px-3 py-2 text-xs font-medium text-[#f3f2ee] hover:bg-[#31343c] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632]"
                  >
                    {motionSettings.enabled ? 'Pause monitoring' : 'Resume monitoring'}
                  </button>
                </div>
              )}

              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="sensitivity-slider" className="text-[#f3f2ee]">
                    Motion sensitivity
                  </label>
                  <span className="tabular-nums text-[#a9acb5]">
                    {motionSettings.sensitivity}
                  </span>
                </div>
                <input
                  id="sensitivity-slider"
                  type="range"
                  min={10}
                  max={95}
                  step={5}
                  value={motionSettings.sensitivity}
                  onChange={(e) => onChangeSensitivity(Number(e.target.value))}
                  className="h-11 w-full cursor-pointer"
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  aria-pressed={motionSettings.soundAlert}
                  onClick={onToggleSoundAlert}
                  className={`min-h-[44px] rounded-md border px-2.5 py-2 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
                    motionSettings.soundAlert
                      ? 'border-[#f2b632] bg-[#31343c] text-[#f3f2ee]'
                      : 'border-[#3f434e] bg-[#1e2024] text-[#a9acb5] hover:text-[#f3f2ee]'
                  }`}
                >
                  Sound {motionSettings.soundAlert ? 'on' : 'off'}
                </button>

                <button
                  type="button"
                  aria-pressed={motionSettings.autoInspectGemini}
                  onClick={onToggleAutoInspectGemini}
                  className={`min-h-[44px] rounded-md border px-2.5 py-2 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
                    motionSettings.autoInspectGemini
                      ? 'border-[#f2b632] bg-[#31343c] text-[#f3f2ee]'
                      : 'border-[#3f434e] bg-[#1e2024] text-[#a9acb5] hover:text-[#f3f2ee]'
                  }`}
                >
                  Auto Gemini {motionSettings.autoInspectGemini ? 'on' : 'off'}
                </button>
              </div>
            </section>

            <section
              aria-labelledby="heading-ai-models"
              className="space-y-3 border-t border-[#3f434e] pt-5"
            >
              <h2
                id="heading-ai-models"
                className="text-sm font-semibold text-[#f3f2ee]"
              >
                Vision and Gemini models
              </h2>

              <div className="space-y-1.5">
                <span className="block text-xs text-[#a9acb5]">Gemini model</span>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    aria-pressed={geminiModel === 'gemini-4-hydrogen'}
                    onClick={() => onChangeGeminiModel('gemini-4-hydrogen')}
                    className={`min-h-[44px] rounded-md border px-2.5 py-2 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
                      geminiModel === 'gemini-4-hydrogen'
                        ? 'border-[#f2b632] bg-[#31343c] text-[#f3f2ee]'
                        : 'border-[#3f434e] bg-[#1e2024] text-[#a9acb5] hover:text-[#f3f2ee]'
                    }`}
                  >
                    Hydrogen
                  </button>
                  <button
                    type="button"
                    aria-pressed={geminiModel === 'gemini-3.1-flash-lite'}
                    onClick={() => onChangeGeminiModel('gemini-3.1-flash-lite')}
                    className={`min-h-[44px] rounded-md border px-2.5 py-2 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
                      geminiModel === 'gemini-3.1-flash-lite'
                        ? 'border-[#f2b632] bg-[#31343c] text-[#f3f2ee]'
                        : 'border-[#3f434e] bg-[#1e2024] text-[#a9acb5] hover:text-[#f3f2ee]'
                    }`}
                  >
                    Flash Lite
                  </button>
                </div>
              </div>

              <div className="space-y-1.5">
                <span className="block text-xs text-[#a9acb5]">
                  Live object and face model
                </span>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    aria-pressed={detectionProfile === 'fast'}
                    onClick={() => onChangeDetectionProfile('fast')}
                    className={`min-h-[44px] rounded-md border px-2.5 py-2 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
                      detectionProfile === 'fast'
                        ? 'border-[#f2b632] bg-[#31343c] text-[#f3f2ee]'
                        : 'border-[#3f434e] bg-[#1e2024] text-[#a9acb5] hover:text-[#f3f2ee]'
                    }`}
                  >
                    Fast (lite)
                  </button>
                  <button
                    type="button"
                    aria-pressed={detectionProfile === 'accurate'}
                    onClick={() => onChangeDetectionProfile('accurate')}
                    className={`min-h-[44px] rounded-md border px-2.5 py-2 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
                      detectionProfile === 'accurate'
                        ? 'border-[#f2b632] bg-[#31343c] text-[#f3f2ee]'
                        : 'border-[#3f434e] bg-[#1e2024] text-[#a9acb5] hover:text-[#f3f2ee]'
                    }`}
                  >
                    High detail
                  </button>
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="confidence-slider" className="text-[#f3f2ee]">
                    Detection threshold
                  </label>
                  <span className="tabular-nums text-[#a9acb5]">
                    {minConfidence.toFixed(2)}
                  </span>
                </div>
                <input
                  id="confidence-slider"
                  type="range"
                  min={0.25}
                  max={0.75}
                  step={0.05}
                  value={minConfidence}
                  onChange={(e) => onChangeMinConfidence(parseFloat(e.target.value))}
                  className="h-11 w-full cursor-pointer"
                />
              </div>

              <button
                type="button"
                disabled={isScanningAll}
                onClick={onScanAllObjects}
                className="min-h-[44px] w-full rounded-md border border-[#f2b632] bg-[#f2b632] px-4 py-2 text-sm font-semibold text-[#1a1b1e] transition-colors hover:bg-[#e0a624] disabled:opacity-60 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f3f2ee]"
              >
                {isScanningAll ? 'Scanning all items...' : 'Scan all things (Gemini)'}
              </button>
            </section>
          </>
        ) : (
          <>
            <section aria-labelledby="heading-send-camera" className="space-y-3">
              <h2
                id="heading-send-camera"
                className="text-sm font-semibold text-[#f3f2ee]"
              >
                Phone camera relay
              </h2>

              <div
                className="grid grid-cols-2 gap-2"
                role="group"
                aria-label="Phone lens"
              >
                <button
                  type="button"
                  onClick={() => onChangeFacingMode('environment')}
                  aria-pressed={phoneFacingMode === 'environment'}
                  className={`min-h-[44px] rounded-md border px-3 py-2 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
                    phoneFacingMode === 'environment'
                      ? 'border-[#f2b632] bg-[#31343c] font-medium text-[#f3f2ee]'
                      : 'border-[#3f434e] bg-[#1e2024] text-[#a9acb5] hover:text-[#f3f2ee]'
                  }`}
                >
                  Back camera
                </button>
                <button
                  type="button"
                  onClick={() => onChangeFacingMode('user')}
                  aria-pressed={phoneFacingMode === 'user'}
                  className={`min-h-[44px] rounded-md border px-3 py-2 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
                    phoneFacingMode === 'user'
                      ? 'border-[#f2b632] bg-[#31343c] font-medium text-[#f3f2ee]'
                      : 'border-[#3f434e] bg-[#1e2024] text-[#a9acb5] hover:text-[#f3f2ee]'
                  }`}
                >
                  Front camera
                </button>
              </div>

              <button
                type="button"
                onClick={onToggleSendCamera}
                className="min-h-[44px] w-full rounded-md border border-[#f2b632] bg-[#f2b632] px-4 py-2 text-sm font-semibold text-[#1a1b1e] transition-colors hover:bg-[#e0a624] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f3f2ee]"
              >
                {isSendingCamera ? 'Stop sending camera' : 'Send this camera'}
              </button>

              {isSendingCamera && (
                <p className="text-xs leading-relaxed text-[#a9acb5]">
                  Sending as {sendingPhoneName ?? 'phone'}.{' '}
                  {wakeLockActive ? 'Screen wake lock active.' : ''}
                </p>
              )}

              {sendError && (
                <p
                  role="alert"
                  className="rounded-md border border-[#e5534b] bg-[#3b2223] p-2.5 text-xs leading-relaxed text-[#f3f2ee]"
                >
                  {sendError}
                </p>
              )}
            </section>

            <section
              aria-labelledby="heading-enhancement"
              className="space-y-3 border-t border-[#3f434e] pt-5"
            >
              <div className="flex items-center justify-between gap-2">
                <h2
                  id="heading-enhancement"
                  className="text-sm font-semibold text-[#f3f2ee]"
                >
                  Frame enhancement
                </h2>
                <span className="text-xs text-[#a9acb5]">
                  {enhanceMode === 'off'
                    ? 'off'
                    : enhanceMode === 'webgl'
                      ? 'WebGL'
                      : 'canvas'}
                </span>
              </div>

              <button
                type="button"
                onClick={onToggleEnhancement}
                aria-pressed={enhancement.enabled}
                className={`min-h-[44px] w-full rounded-md border px-3.5 py-2 text-sm font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632] ${
                  enhancement.enabled
                    ? 'border-[#f2b632] bg-[#31343c] text-[#f3f2ee]'
                    : 'border-[#3f434e] bg-[#1e2024] text-[#a9acb5] hover:text-[#f3f2ee]'
                }`}
              >
                {enhancement.enabled ? 'Enhancement on' : 'Enhancement off'}
              </button>

              <div className="space-y-2 pt-1">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="gamma-slider" className="text-[#f3f2ee]">
                    Gamma {enhancement.autoGamma ? '(auto)' : '(manual)'}
                  </label>
                  <span className="tabular-nums text-[#a9acb5]">
                    {effectiveGamma.toFixed(2)}
                  </span>
                </div>

                <input
                  id="gamma-slider"
                  type="range"
                  min={0.5}
                  max={1.0}
                  step={0.05}
                  disabled={!enhancement.enabled}
                  value={enhancement.autoGamma ? effectiveGamma : enhancement.gamma}
                  onChange={(e) => onChangeGamma(parseFloat(e.target.value))}
                  className="h-11 w-full cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
                />

                {!enhancement.autoGamma && (
                  <button
                    type="button"
                    disabled={!enhancement.enabled}
                    onClick={onResetAutoGamma}
                    className="min-h-[44px] w-full rounded-md border border-[#3f434e] bg-[#1e2024] px-3 py-2 text-xs text-[#f3f2ee] hover:bg-[#31343c] disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632]"
                  >
                    Use auto gamma
                  </button>
                )}
              </div>
            </section>
          </>
        )}
      </div>
    </aside>
  );
}
