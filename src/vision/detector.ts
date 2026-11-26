import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as faceDetection from '@tensorflow-models/face-detection';
import type { DetectionProfile, RawDetection } from '../types/vision';

export interface LoadedModels {
  profile: DetectionProfile;
  coco: cocoSsd.ObjectDetection;
  face: faceDetection.FaceDetector;
}

const modelCache = new Map<DetectionProfile, Promise<LoadedModels>>();

export async function loadDetectionModels(
  profile: DetectionProfile = 'fast',
  onStatus?: (message: string) => void
): Promise<LoadedModels> {
  const cached = modelCache.get(profile);
  if (cached) {
    return cached;
  }

  const promise = (async () => {
    onStatus?.('Initializing TensorFlow.js backend...');
    await tf.ready();

    const cocoBase = profile === 'accurate' ? 'mobilenet_v2' : 'lite_mobilenet_v2';
    const faceModelType = profile === 'accurate' ? 'full' : 'short';
    const maxFaces = profile === 'accurate' ? 25 : 15;

    onStatus?.(
      profile === 'accurate'
        ? 'Loading high-detail object and face models...'
        : 'Loading object and face detection models...'
    );

    const [coco, face] = await Promise.all([
      cocoSsd.load({ base: cocoBase }),
      faceDetection.createDetector(
        faceDetection.SupportedModels.MediaPipeFaceDetector,
        {
          runtime: 'tfjs',
          modelType: faceModelType,
          maxFaces,
        }
      ),
    ]);

    return { profile, coco, face };
  })().catch((err) => {
    modelCache.delete(profile);
    throw err;
  });

  modelCache.set(profile, promise);
  return promise;
}

export async function detectObjectsAndFaces(
  models: LoadedModels,
  canvas: HTMLCanvasElement,
  minScore = 0.4
): Promise<{ objects: RawDetection[]; faces: RawDetection[]; durationMs: number }> {
  const t0 = performance.now();
  const safeMinScore = Math.max(0.2, Math.min(0.9, minScore));

  const [cocoResults, faceResults] = await Promise.all([
    models.coco.detect(canvas, 50, safeMinScore),
    models.face.estimateFaces(canvas, { flipHorizontal: false }),
  ]);

  const objects: RawDetection[] = cocoResults
    .filter((pred) => pred.score >= safeMinScore)
    .map((pred) => ({
      kind: 'object',
      className: pred.class,
      box: pred.bbox,
      score: pred.score,
    }));

  const faces: RawDetection[] = faceResults.map((face) => ({
    kind: 'face',
    className: 'face',
    box: [
      Math.max(0, face.box.xMin),
      Math.max(0, face.box.yMin),
      Math.max(1, face.box.width),
      Math.max(1, face.box.height),
    ],
    score: null,
  }));

  return {
    objects,
    faces,
    durationMs: performance.now() - t0,
  };
}
