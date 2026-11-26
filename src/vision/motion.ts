import type { NormalizedZone } from '../types/vision';

const ANALYSIS_W = 240;

export interface MotionEvaluation {
  changedPercent: number;
  triggerThreshold: number;
  triggered: boolean;
}

export class ZoneMotionDetector {
  private analysisCanvas: HTMLCanvasElement;
  private analysisCtx: CanvasRenderingContext2D;
  private prevLuma: Uint8Array | null = null;
  private prevSignature = '';

  constructor() {
    this.analysisCanvas = document.createElement('canvas');
    this.analysisCanvas.width = ANALYSIS_W;
    this.analysisCanvas.height = 135;
    const ctx = this.analysisCanvas.getContext('2d', {
      willReadFrequently: true,
    });
    if (!ctx) {
      throw new Error('2D canvas context unavailable for motion detection.');
    }
    this.analysisCtx = ctx;
  }

  public reset(): void {
    this.prevLuma = null;
    this.prevSignature = '';
  }

  public evaluate(
    sourceCanvas: HTMLCanvasElement,
    zone: NormalizedZone,
    sensitivity: number
  ): MotionEvaluation {
    const srcW = sourceCanvas.width;
    const srcH = sourceCanvas.height;
    if (srcW === 0 || srcH === 0 || zone.w <= 0.01 || zone.h <= 0.01) {
      return { changedPercent: 0, triggerThreshold: 2.5, triggered: false };
    }

    const w = ANALYSIS_W;
    const h = Math.max(1, Math.round((ANALYSIS_W * srcH) / srcW));

    if (this.analysisCanvas.width !== w || this.analysisCanvas.height !== h) {
      this.analysisCanvas.width = w;
      this.analysisCanvas.height = h;
      this.prevLuma = null;
    }

    this.analysisCtx.drawImage(sourceCanvas, 0, 0, w, h);

    const x0 = Math.max(0, Math.min(w - 1, Math.floor(zone.x * w)));
    const y0 = Math.max(0, Math.min(h - 1, Math.floor(zone.y * h)));
    const x1 = Math.max(x0 + 1, Math.min(w, Math.ceil((zone.x + zone.w) * w)));
    const y1 = Math.max(y0 + 1, Math.min(h, Math.ceil((zone.y + zone.h) * h)));

    const rw = x1 - x0;
    const rh = y1 - y0;
    const pixelCount = rw * rh;

    const imgData = this.analysisCtx.getImageData(x0, y0, rw, rh).data;
    const currLuma = new Uint8Array(pixelCount);

    for (let i = 0, p = 0; i < imgData.length; i += 4, p++) {
      currLuma[p] =
        (77 * imgData[i] + 150 * imgData[i + 1] + 29 * imgData[i + 2]) >> 8;
    }

    const clampedSens = Math.max(1, Math.min(100, sensitivity));
    // Higher sensitivity -> smaller pixel delta required & smaller area % required
    const pixelDiffThreshold = Math.round(46 - (clampedSens / 100) * 32);
    const triggerThreshold = Number(
      Math.max(0.8, 7.5 - (clampedSens / 100) * 6.5).toFixed(1)
    );

    const signature = `${w}x${h}:${x0},${y0},${rw},${rh}`;
    if (!this.prevLuma || this.prevSignature !== signature) {
      this.prevLuma = currLuma;
      this.prevSignature = signature;
      return { changedPercent: 0, triggerThreshold, triggered: false };
    }

    let changedPixels = 0;
    const prev = this.prevLuma;
    for (let p = 0; p < pixelCount; p++) {
      const diff = currLuma[p] - prev[p];
      if (diff > pixelDiffThreshold || diff < -pixelDiffThreshold) {
        changedPixels++;
      }
      // Subtle blend keeps background model adaptive without masking active motion
      prev[p] = currLuma[p];
    }

    const changedPercent = Number(((changedPixels / pixelCount) * 100).toFixed(1));
    return {
      changedPercent,
      triggerThreshold,
      triggered: changedPercent >= triggerThreshold,
    };
  }
}
