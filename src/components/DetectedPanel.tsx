import { useEffect, useState } from 'react';
import type {
  DetectedHistoryItem,
  GeminiModelId,
  SceneScanResult,
} from '../types/vision';

interface DetectedPanelProps {
  isOpen: boolean;
  onClose: () => void;
  inViewItems: DetectedHistoryItem[];
  seenEarlierItems: DetectedHistoryItem[];
  onAskGemini: (itemKey: string) => void;
  sceneScan: SceneScanResult | null;
  onClearSceneScan: () => void;
  activeGeminiModel: GeminiModelId;
}

function formatAgo(lastSeenAt: number, now: number): string {
  const diffSec = Math.max(0, Math.floor((now - lastSeenAt) / 1000));
  if (diffSec < 60) {
    return `${diffSec}s ago`;
  }
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) {
    return `${diffMin}m ago`;
  }
  const diffHr = Math.floor(diffMin / 60);
  return `${diffHr}h ago`;
}

interface ItemCardProps {
  item: DetectedHistoryItem;
  isEarlier: boolean;
  now: number;
  onAskGemini: (itemKey: string) => void;
}

function DetectedItemCard({ item, isEarlier, now, onAskGemini }: ItemCardProps) {
  return (
    <li className="rounded-md border border-[#3f434e] bg-[#1e2024] p-2.5">
      <div className="flex items-start gap-2.5">
        <img
          src={item.cropDataUrl}
          alt={item.label}
          className="h-20 w-20 shrink-0 rounded border border-[#3f434e] bg-[#17181b] object-contain"
        />

        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-2">
            <span className="truncate text-sm font-semibold text-[#f3f2ee]">
              {item.label}
            </span>
            <span className="shrink-0 text-xs text-[#a9acb5] tabular-nums">
              {isEarlier
                ? formatAgo(item.lastSeenAt, now)
                : item.score !== null
                  ? item.score.toFixed(2)
                  : 'face'}
            </span>
          </div>

          <div className="mt-2">
            <button
              type="button"
              onClick={() => onAskGemini(item.key)}
              disabled={item.geminiStatus === 'loading'}
              className="min-h-[44px] w-full rounded border border-[#3f434e] bg-[#27292f] px-3 py-2 text-left text-xs font-medium text-[#f3f2ee] transition-colors hover:border-[#f2b632] hover:bg-[#31343c] disabled:opacity-60 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632]"
            >
              {item.geminiStatus === 'loading' ? 'Asking Gemini...' : 'Ask Gemini'}
            </button>
          </div>
        </div>
      </div>

      {item.geminiStatus === 'done' && item.geminiText && (
        <p className="mt-2.5 border-t border-[#3f434e] pt-2 text-xs leading-relaxed text-[#f3f2ee]">
          {item.geminiText}
        </p>
      )}

      {item.geminiStatus === 'error' && item.geminiError && (
        <p className="mt-2.5 border-t border-[#3f434e] pt-2 text-xs leading-relaxed text-[#f1948a]">
          {item.geminiError}
        </p>
      )}
    </li>
  );
}

export function DetectedPanel({
  isOpen,
  onClose,
  inViewItems,
  seenEarlierItems,
  onAskGemini,
  sceneScan,
  onClearSceneScan,
  activeGeminiModel,
}: DetectedPanelProps) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!isOpen) return;
    setNow(Date.now());
    const timer = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);
    return () => window.clearInterval(timer);
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  return (
    <>
      <div
        className="fixed inset-0 z-30 bg-black/55 lg:hidden"
        onClick={onClose}
        aria-hidden="true"
      />

      <aside
        aria-label="Detected items"
        className="fixed inset-y-0 left-0 z-40 flex w-[85vw] max-w-[380px] flex-col border-r border-[#3f434e] bg-[#27292f] lg:static lg:z-auto lg:w-[clamp(260px,25vw,380px)] lg:shrink-0"
      >
        <div className="flex items-center justify-between border-b border-[#3f434e] px-3.5 py-2.5">
          <div>
            <h2 className="text-sm font-semibold text-[#f3f2ee]">Detected</h2>
            <p className="text-xs text-[#a9acb5]">Model: {activeGeminiModel}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="min-h-[44px] rounded-md border border-[#3f434e] bg-[#1e2024] px-3.5 py-2 text-sm text-[#f3f2ee] transition-colors hover:bg-[#31343c] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632]"
          >
            Close
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3.5 space-y-5">
          {sceneScan && sceneScan.items.length > 0 && (
            <section aria-labelledby="heading-gemini-scan">
              <div className="mb-2 flex items-center justify-between gap-2">
                <h3
                  id="heading-gemini-scan"
                  className="text-xs font-semibold text-[#f2b632]"
                >
                  All items found (<span className="tabular-nums">{sceneScan.items.length}</span>)
                </h3>
                <button
                  type="button"
                  onClick={onClearSceneScan}
                  className="min-h-[44px] rounded px-2.5 text-xs text-[#a9acb5] hover:text-[#f3f2ee] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f2b632]"
                >
                  Clear
                </button>
              </div>
              <ul className="space-y-2">
                {sceneScan.items.map((gItem) => (
                  <li
                    key={gItem.id}
                    className="flex items-center gap-2.5 rounded-md border border-[#3f434e] bg-[#1e2024] p-2"
                  >
                    {gItem.cropDataUrl && (
                      <img
                        src={gItem.cropDataUrl}
                        alt={gItem.label}
                        className="h-14 w-14 shrink-0 rounded border border-[#3f434e] bg-[#17181b] object-contain"
                      />
                    )}
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-[#f3f2ee]">
                        {gItem.label}
                      </p>
                      <p className="text-xs text-[#a9acb5] tabular-nums">
                        {sceneScan.model}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}

          <section aria-labelledby="heading-in-view">
            <h3
              id="heading-in-view"
              className="mb-2 text-xs font-semibold text-[#a9acb5]"
            >
              In view now (<span className="tabular-nums">{inViewItems.length}</span>)
            </h3>
            {inViewItems.length === 0 ? (
              <p className="text-xs text-[#a9acb5]">Nothing detected in view.</p>
            ) : (
              <ul className="space-y-2.5">
                {inViewItems.map((item) => (
                  <DetectedItemCard
                    key={item.key}
                    item={item}
                    isEarlier={false}
                    now={now}
                    onAskGemini={onAskGemini}
                  />
                ))}
              </ul>
            )}
          </section>

          <section aria-labelledby="heading-seen-earlier">
            <h3
              id="heading-seen-earlier"
              className="mb-2 text-xs font-semibold text-[#a9acb5]"
            >
              Seen earlier (<span className="tabular-nums">{seenEarlierItems.length}</span>)
            </h3>
            {seenEarlierItems.length === 0 ? (
              <p className="text-xs text-[#a9acb5]">No earlier items yet.</p>
            ) : (
              <ul className="space-y-2.5">
                {seenEarlierItems.map((item) => (
                  <DetectedItemCard
                    key={item.key}
                    item={item}
                    isEarlier={true}
                    now={now}
                    onAskGemini={onAskGemini}
                  />
                ))}
              </ul>
            )}
          </section>
        </div>
      </aside>
    </>
  );
}
