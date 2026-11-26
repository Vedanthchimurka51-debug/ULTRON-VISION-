let reusableCropCanvas: HTMLCanvasElement | null = null;
let reusableCropCtx: CanvasRenderingContext2D | null = null;

const MAX_CROP_SIDE = 160;
const PADDING_RATIO = 0.08;

export function captureCroppedThumbnail(
  sourceCanvas: HTMLCanvasElement,
  box: [number, number, number, number]
): string {
  const srcW = sourceCanvas.width;
  const srcH = sourceCanvas.height;
  if (srcW === 0 || srcH === 0) return '';

  const [bx, by, bw, bh] = box;
  const padX = bw * PADDING_RATIO;
  const padY = bh * PADDING_RATIO;

  const sx = Math.max(0, Math.floor(bx - padX));
  const sy = Math.max(0, Math.floor(by - padY));
  const ex = Math.min(srcW, Math.ceil(bx + bw + padX));
  const ey = Math.min(srcH, Math.ceil(by + bh + padY));

  const sw = Math.max(1, ex - sx);
  const sh = Math.max(1, ey - sy);

  const longest = Math.max(sw, sh);
  const scale = MAX_CROP_SIDE / longest;
  const outW = Math.max(1, Math.round(sw * scale));
  const outH = Math.max(1, Math.round(sh * scale));

  if (!reusableCropCanvas) {
    reusableCropCanvas = document.createElement('canvas');
    reusableCropCtx = reusableCropCanvas.getContext('2d');
  }
  if (!reusableCropCanvas || !reusableCropCtx) return '';

  if (reusableCropCanvas.width !== outW || reusableCropCanvas.height !== outH) {
    reusableCropCanvas.width = outW;
    reusableCropCanvas.height = outH;
  }

  reusableCropCtx.drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, outW, outH);
  return reusableCropCanvas.toDataURL('image/jpeg', 0.85);
}
