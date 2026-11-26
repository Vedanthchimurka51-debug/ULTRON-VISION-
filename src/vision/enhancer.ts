export interface EnhanceOptions {
  enabled: boolean;
  autoGamma: boolean;
  gamma: number;
}

export interface EnhanceStats {
  mode: 'webgl' | 'canvas' | 'off';
  effectiveGamma: number;
  durationMs: number;
}

const STATS_W = 96;
const STATS_H = 54;

const VERTEX_SHADER_SRC = `
attribute vec2 a_position;
attribute vec2 a_texCoord;
varying vec2 v_texCoord;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_texCoord = a_texCoord;
}
`;

const FRAGMENT_SHADER_SRC = `
precision mediump float;
uniform sampler2D u_image;
uniform vec2 u_texelSize;
uniform float u_black;
uniform float u_invRange;
uniform float u_gamma;
uniform float u_sharpen;
varying vec2 v_texCoord;

void main() {
  vec3 c = texture2D(u_image, v_texCoord).rgb;
  vec3 n = texture2D(u_image, v_texCoord + vec2(0.0, -u_texelSize.y)).rgb;
  vec3 s = texture2D(u_image, v_texCoord + vec2(0.0, u_texelSize.y)).rgb;
  vec3 e = texture2D(u_image, v_texCoord + vec2(u_texelSize.x, 0.0)).rgb;
  vec3 w = texture2D(u_image, v_texCoord + vec2(-u_texelSize.x, 0.0)).rgb;

  vec3 avg4 = (n + s + e + w) * 0.25;
  vec3 sharpened = clamp(c + u_sharpen * (c - avg4), 0.0, 1.0);

  vec3 leveled = clamp((sharpened - vec3(u_black)) * u_invRange, 0.0, 1.0);
  vec3 corrected = pow(leveled, vec3(u_gamma));

  gl_FragColor = vec4(corrected, 1.0);
}
`;

function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

export class FrameEnhancer {
  private statsCanvas: HTMLCanvasElement;
  private statsCtx: CanvasRenderingContext2D;
  private histogram = new Uint32Array(256);

  private glCanvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private texture: WebGLTexture | null = null;
  private uTexelSize: WebGLUniformLocation | null = null;
  private uBlack: WebGLUniformLocation | null = null;
  private uInvRange: WebGLUniformLocation | null = null;
  private uGamma: WebGLUniformLocation | null = null;
  private uSharpen: WebGLUniformLocation | null = null;

  private fallbackCanvas: HTMLCanvasElement;
  private fallbackCtx: CanvasRenderingContext2D | null = null;
  private lut = new Uint8Array(256);

  private smoothBlack = 0.0;
  private smoothWhite = 1.0;
  private smoothAvg = 0.5;
  private statsInitialized = false;

  constructor() {
    this.statsCanvas = document.createElement('canvas');
    this.statsCanvas.width = STATS_W;
    this.statsCanvas.height = STATS_H;
    const sCtx = this.statsCanvas.getContext('2d', { willReadFrequently: true });
    if (!sCtx) {
      throw new Error('2D canvas context unavailable.');
    }
    this.statsCtx = sCtx;

    this.glCanvas = document.createElement('canvas');
    this.fallbackCanvas = document.createElement('canvas');
    this.fallbackCtx = this.fallbackCanvas.getContext('2d', {
      willReadFrequently: true,
    });

    this.initWebGL();
  }

  private initWebGL(): void {
    const gl = this.glCanvas.getContext('webgl', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    if (!gl) {
      this.gl = null;
      return;
    }

    const vert = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);
    if (!vert || !frag) {
      this.gl = null;
      return;
    }

    const program = gl.createProgram();
    if (!program) {
      this.gl = null;
      return;
    }
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      gl.deleteProgram(program);
      this.gl = null;
      return;
    }

    gl.useProgram(program);

    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW
    );
    const posLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    const texBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texBuffer);
    // Flip Y for HTMLVideoElement / ImageBitmap -> WebGL texture coords
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([0, 1, 1, 1, 0, 0, 1, 0]),
      gl.STATIC_DRAW
    );
    const texLoc = gl.getAttribLocation(program, 'a_texCoord');
    gl.enableVertexAttribArray(texLoc);
    gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    this.gl = gl;
    this.program = program;
    this.texture = texture;
    this.uTexelSize = gl.getUniformLocation(program, 'u_texelSize');
    this.uBlack = gl.getUniformLocation(program, 'u_black');
    this.uInvRange = gl.getUniformLocation(program, 'u_invRange');
    this.uGamma = gl.getUniformLocation(program, 'u_gamma');
    this.uSharpen = gl.getUniformLocation(program, 'u_sharpen');
  }

  private computeStats(
    source: HTMLVideoElement | ImageBitmap | HTMLCanvasElement
  ): { black: number; white: number; avg: number } {
    this.statsCtx.drawImage(source, 0, 0, STATS_W, STATS_H);
    const data = this.statsCtx.getImageData(0, 0, STATS_W, STATS_H).data;
    const hist = this.histogram;
    hist.fill(0);

    let sumLum = 0;
    const pixelCount = STATS_W * STATS_H;

    for (let i = 0; i < data.length; i += 4) {
      const lum = Math.min(
        255,
        Math.round(0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2])
      );
      hist[lum]++;
      sumLum += lum;
    }

    const lowTarget = Math.floor(pixelCount * 0.01);
    const highTarget = Math.floor(pixelCount * 0.99);

    let acc = 0;
    let p1 = 0;
    let p99 = 255;

    for (let i = 0; i < 256; i++) {
      acc += hist[i];
      if (acc >= lowTarget) {
        p1 = i;
        break;
      }
    }

    acc = 0;
    for (let i = 0; i < 256; i++) {
      acc += hist[i];
      if (acc >= highTarget) {
        p99 = i;
        break;
      }
    }

    // Prevent excessive contrast stretching on flat/blank frames
    if (p99 - p1 < 24) {
      p1 = 0;
      p99 = 255;
    }

    const rawBlack = p1 / 255;
    const rawWhite = p99 / 255;
    const rawAvg = sumLum / (pixelCount * 255);

    if (!this.statsInitialized) {
      this.smoothBlack = rawBlack;
      this.smoothWhite = rawWhite;
      this.smoothAvg = rawAvg;
      this.statsInitialized = true;
    } else {
      const alpha = 0.25;
      this.smoothBlack = this.smoothBlack * (1 - alpha) + rawBlack * alpha;
      this.smoothWhite = this.smoothWhite * (1 - alpha) + rawWhite * alpha;
      this.smoothAvg = this.smoothAvg * (1 - alpha) + rawAvg * alpha;
    }

    return {
      black: this.smoothBlack,
      white: this.smoothWhite,
      avg: this.smoothAvg,
    };
  }

  public resetStats(): void {
    this.statsInitialized = false;
  }

  public render(
    source: HTMLVideoElement | ImageBitmap | HTMLCanvasElement,
    width: number,
    height: number,
    targetCanvas: HTMLCanvasElement,
    options: EnhanceOptions
  ): EnhanceStats {
    const t0 = performance.now();

    if (targetCanvas.width !== width || targetCanvas.height !== height) {
      targetCanvas.width = width;
      targetCanvas.height = height;
    }

    const targetCtx = targetCanvas.getContext('2d', { alpha: false });
    if (!targetCtx) {
      return {
        mode: 'off',
        effectiveGamma: 1.0,
        durationMs: performance.now() - t0,
      };
    }

    if (!options.enabled) {
      targetCtx.drawImage(source, 0, 0, width, height);
      return {
        mode: 'off',
        effectiveGamma: 1.0,
        durationMs: performance.now() - t0,
      };
    }

    const { black, white, avg } = this.computeStats(source);
    const range = Math.max(0.08, white - black);

    // Brighten dark scenes only (never darken: gamma stays in [0.5, 1.0])
    const autoGamma =
      avg >= 0.45 ? 1.0 : Math.max(0.5, Math.min(1.0, 0.5 + (avg / 0.45) * 0.5));
    const effectiveGamma = options.autoGamma
      ? autoGamma
      : Math.max(0.5, Math.min(1.0, options.gamma));

    if (this.gl && this.program && this.texture && !this.gl.isContextLost()) {
      const gl = this.gl;
      if (this.glCanvas.width !== width || this.glCanvas.height !== height) {
        this.glCanvas.width = width;
        this.glCanvas.height = height;
        gl.viewport(0, 0, width, height);
      }

      gl.useProgram(this.program);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);

      gl.uniform2f(this.uTexelSize, 1 / width, 1 / height);
      gl.uniform1f(this.uBlack, black);
      gl.uniform1f(this.uInvRange, 1 / range);
      gl.uniform1f(this.uGamma, effectiveGamma);
      gl.uniform1f(this.uSharpen, 0.35);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      targetCtx.drawImage(this.glCanvas, 0, 0, width, height);

      return {
        mode: 'webgl',
        effectiveGamma,
        durationMs: performance.now() - t0,
      };
    }

    // 2D Canvas fallback with LUT from downscaled stats
    if (this.fallbackCtx) {
      if (this.fallbackCanvas.width !== width || this.fallbackCanvas.height !== height) {
        this.fallbackCanvas.width = width;
        this.fallbackCanvas.height = height;
      }

      for (let i = 0; i < 256; i++) {
        const leveled = Math.max(0, Math.min(1, (i / 255 - black) / range));
        this.lut[i] = Math.round(Math.pow(leveled, effectiveGamma) * 255);
      }

      this.fallbackCtx.drawImage(source, 0, 0, width, height);
      const imgData = this.fallbackCtx.getImageData(0, 0, width, height);
      const px = imgData.data;
      const lut = this.lut;

      for (let i = 0; i < px.length; i += 4) {
        px[i] = lut[px[i]];
        px[i + 1] = lut[px[i + 1]];
        px[i + 2] = lut[px[i + 2]];
      }

      targetCtx.putImageData(imgData, 0, 0);

      return {
        mode: 'canvas',
        effectiveGamma,
        durationMs: performance.now() - t0,
      };
    }

    targetCtx.drawImage(source, 0, 0, width, height);
    return {
      mode: 'off',
      effectiveGamma: 1.0,
      durationMs: performance.now() - t0,
    };
  }

  public dispose(): void {
    if (this.gl) {
      if (this.texture) this.gl.deleteTexture(this.texture);
      if (this.program) this.gl.deleteProgram(this.program);
      this.gl = null;
    }
  }
}
