import type { DetectionKind, RawDetection, TrackedItem } from '../types/vision';

const IOU_THRESHOLD = 0.3;
const MAX_MISSED_FRAMES = 15;

export function computeIoU(
  a: [number, number, number, number],
  b: [number, number, number, number]
): number {
  const [ax, ay, aw, ah] = a;
  const [bx, by, bw, bh] = b;

  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);

  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const interArea = interW * interH;

  if (interArea <= 0) return 0;

  const areaA = Math.max(0, aw) * Math.max(0, ah);
  const areaB = Math.max(0, bw) * Math.max(0, bh);
  const unionArea = areaA + areaB - interArea;

  return unionArea > 0 ? interArea / unionArea : 0;
}

export class IoUTracker {
  private kind: DetectionKind;
  private nextId = 1;
  private tracks: TrackedItem[] = [];

  constructor(kind: DetectionKind) {
    this.kind = kind;
  }

  public reset(): void {
    this.nextId = 1;
    this.tracks = [];
  }

  public update(detections: RawDetection[], now: number): TrackedItem[] {
    const candidates: Array<{ trackIdx: number; detIdx: number; iou: number }> = [];

    for (let t = 0; t < this.tracks.length; t++) {
      const track = this.tracks[t];
      for (let d = 0; d < detections.length; d++) {
        const det = detections[d];
        if (track.className !== det.className) continue;
        const iou = computeIoU(track.box, det.box);
        if (iou >= IOU_THRESHOLD) {
          candidates.push({ trackIdx: t, detIdx: d, iou });
        }
      }
    }

    candidates.sort((a, b) => b.iou - a.iou);

    const matchedTracks = new Set<number>();
    const matchedDets = new Set<number>();

    for (const pair of candidates) {
      if (matchedTracks.has(pair.trackIdx) || matchedDets.has(pair.detIdx)) {
        continue;
      }
      matchedTracks.add(pair.trackIdx);
      matchedDets.add(pair.detIdx);

      const track = this.tracks[pair.trackIdx];
      const det = detections[pair.detIdx];
      track.box = det.box;
      track.score = det.score;
      track.missedFrames = 0;
      track.lastSeenAt = now;
    }

    const survivingTracks: TrackedItem[] = [];

    for (let t = 0; t < this.tracks.length; t++) {
      const track = this.tracks[t];
      if (!matchedTracks.has(t)) {
        track.missedFrames += 1;
      }
      if (track.missedFrames <= MAX_MISSED_FRAMES) {
        survivingTracks.push(track);
      }
    }

    for (let d = 0; d < detections.length; d++) {
      if (matchedDets.has(d)) continue;
      const det = detections[d];
      const id = this.nextId++;
      survivingTracks.push({
        id,
        key: `${this.kind}-${id}`,
        kind: this.kind,
        className: det.className,
        box: det.box,
        score: det.score,
        missedFrames: 0,
        lastSeenAt: now,
      });
    }

    this.tracks = survivingTracks;
    return this.tracks.filter((t) => t.missedFrames === 0);
  }
}
