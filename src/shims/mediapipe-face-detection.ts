// Shim for @mediapipe/face_detection UMD package when using @tensorflow-models/face-detection with runtime: 'tfjs'.
export class FaceDetection {
  constructor() {
    throw new Error('MediaPipe runtime is not used; use runtime: "tfjs".');
  }
}

export default { FaceDetection };
