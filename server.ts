import 'dotenv/config';
import express from 'express';
import http from 'node:http';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebSocketServer, WebSocket } from 'ws';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const IS_PROD =
  process.env.NODE_ENV === 'production' ||
  process.env.npm_lifecycle_event === 'start';

const MAX_RELAY_BUFFER_BYTES = 400 * 1024;
const MAX_IMAGE_BYTES = 1.5 * 1024 * 1024;

function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey || apiKey === 'YOUR_GEMINI_API_KEY' || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function sanitizeClientId(raw: string | null): string {
  if (!raw) return crypto.randomUUID();
  const cleaned = raw.replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 64);
  return cleaned.length >= 6 ? cleaned : crypto.randomUUID();
}

interface ActivePhone {
  id: string;
  name: string;
  ws: WebSocket;
}

interface ClientSocket extends WebSocket {
  isAlive?: boolean;
  clientId?: string;
}

const ALLOWED_GEMINI_MODELS = new Set([
  'gemini-4-hydrogen',
  'gemini-3.1-flash-lite',
]);

async function startServer(): Promise<void> {
  const app = express();
  app.use(express.json({ limit: '4mb' }));

  app.post('/api/describe', async (req, res) => {
    const ai = getGeminiClient();
    if (!ai) {
      res.status(500).json({
        error:
          'Gemini API key is not configured on the server. Set GEMINI_API_KEY in your .env file or server environment.',
      });
      return;
    }

    const body = req.body as {
      imageBase64?: unknown;
      label?: unknown;
      model?: unknown;
      mode?: unknown;
    } | null;
    if (!body || typeof body.imageBase64 !== 'string') {
      res.status(400).json({ error: 'Missing image data.' });
      return;
    }

    const requestedModel =
      typeof body.model === 'string' && ALLOWED_GEMINI_MODELS.has(body.model)
        ? body.model
        : 'gemini-4-hydrogen';

    const mode =
      body.mode === 'scene' || body.mode === 'zone' ? body.mode : 'item';

    let mimeType = 'image/jpeg';
    let base64Data = body.imageBase64.trim();

    const dataUrlMatch = base64Data.match(
      /^data:(image\/(?:jpeg|jpg|png|webp));base64,([A-Za-z0-9+/=]+)$/
    );
    if (dataUrlMatch) {
      mimeType = dataUrlMatch[1] === 'image/jpg' ? 'image/jpeg' : dataUrlMatch[1];
      base64Data = dataUrlMatch[2];
    } else if (!/^[A-Za-z0-9+/=]+$/.test(base64Data)) {
      res.status(400).json({ error: 'Invalid base64 image format.' });
      return;
    }

    const approxBytes = Math.floor((base64Data.length * 3) / 4);
    if (approxBytes === 0 || approxBytes > MAX_IMAGE_BYTES) {
      res.status(400).json({ error: 'Image size is out of range.' });
      return;
    }

    const detectedLabel =
      typeof body.label === 'string' && body.label.trim().length > 0
        ? body.label.trim().slice(0, 64)
        : '';

    let promptText: string;
    let systemInstruction: string;

    if (mode === 'scene') {
      promptText =
        'Inspect this full camera frame carefully. Identify all visible people, objects, text, and ongoing activity concisely.';
      systemInstruction =
        'Reply in 2 to 3 clear, factual sentences listing everything important visible in the scene. Do not use markdown, bullet points, or emojis.';
    } else if (mode === 'zone') {
      promptText =
        'Movement was detected in this watched camera zone. Describe clearly what is inside this zone and what appears to be happening.';
      systemInstruction =
        'Reply in 1 to 2 clear, factual sentences describing the subjects and movement in the zone. Do not use markdown, bullet points, or emojis.';
    } else {
      promptText = detectedLabel
        ? `This crop was detected as "${detectedLabel}". Describe what is visible in this cropped camera image in one short, factual sentence.`
        : 'Describe what is visible in this cropped camera image in one short, factual sentence.';
      systemInstruction =
        'Reply with exactly one short, clear, factual sentence describing the subject in the image. Do not use markdown, bullet points, or emojis.';
    }

    try {
      const config: {
        systemInstruction: string;
        thinkingConfig?: { thinkingLevel: ThinkingLevel };
      } = { systemInstruction };

      if (requestedModel === 'gemini-4-hydrogen') {
        config.thinkingConfig = { thinkingLevel: ThinkingLevel.LOW };
      }

      const response = await ai.models.generateContent({
        model: requestedModel,
        contents: {
          parts: [
            {
              inlineData: {
                mimeType,
                data: base64Data,
              },
            },
            { text: promptText },
          ],
        },
        config,
      });

      const description = response.text?.trim();
      if (!description) {
        res.status(502).json({
          error: 'Gemini returned an empty response for this image.',
        });
        return;
      }

      res.json({ description, model: requestedModel });
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Failed to call Gemini API.';
      const lower = message.toLowerCase();

      if (
        lower.includes('403') ||
        lower.includes('permission_denied') ||
        lower.includes('api_key_invalid') ||
        lower.includes('api key')
      ) {
        res.status(403).json({
          error:
            'Gemini API key was rejected. Check GEMINI_API_KEY in your .env file or server environment.',
        });
        return;
      }

      if (lower.includes('429') || lower.includes('resource_exhausted')) {
        res.status(429).json({
          error: 'Gemini rate limit reached. Try again in a moment.',
        });
        return;
      }

      res.status(502).json({ error: `Gemini request failed: ${message}` });
    }
  });

  app.post('/api/scan-all', async (req, res) => {
    const ai = getGeminiClient();
    if (!ai) {
      res.status(500).json({
        error:
          'Gemini API key is not configured on the server. Set GEMINI_API_KEY in your .env file or server environment.',
      });
      return;
    }

    const body = req.body as {
      imageBase64?: unknown;
      model?: unknown;
      zoneLabel?: unknown;
    } | null;

    if (!body || typeof body.imageBase64 !== 'string') {
      res.status(400).json({ error: 'Missing image data.' });
      return;
    }

    const requestedModel =
      typeof body.model === 'string' && ALLOWED_GEMINI_MODELS.has(body.model)
        ? body.model
        : 'gemini-4-hydrogen';

    let mimeType = 'image/jpeg';
    let base64Data = body.imageBase64.trim();

    const dataUrlMatch = base64Data.match(
      /^data:(image\/(?:jpeg|jpg|png|webp));base64,([A-Za-z0-9+/=]+)$/
    );
    if (dataUrlMatch) {
      mimeType = dataUrlMatch[1] === 'image/jpg' ? 'image/jpeg' : dataUrlMatch[1];
      base64Data = dataUrlMatch[2];
    } else if (!/^[A-Za-z0-9+/=]+$/.test(base64Data)) {
      res.status(400).json({ error: 'Invalid base64 image format.' });
      return;
    }

    const approxBytes = Math.floor((base64Data.length * 3) / 4);
    if (approxBytes === 0 || approxBytes > MAX_IMAGE_BYTES) {
      res.status(400).json({ error: 'Image size is out of range.' });
      return;
    }

    const scopeNote =
      typeof body.zoneLabel === 'string' && body.zoneLabel.trim().length > 0
        ? `Focus on "${body.zoneLabel.trim().slice(0, 64)}". `
        : '';

    try {
      const config: {
        systemInstruction: string;
        responseMimeType: string;
        responseSchema: Record<string, unknown>;
        thinkingConfig?: { thinkingLevel: ThinkingLevel };
      } = {
        systemInstruction:
          'You are an exhaustive visual detector. Identify every distinct object, person, face, accessory, screen, furniture piece, and background item visible in the image—not only the main subject. Return clean lowercase labels and 0-1000 normalized bounding boxes.',
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            summary: {
              type: Type.STRING,
              description: 'One concise factual sentence describing the whole view.',
            },
            items: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  label: {
                    type: Type.STRING,
                    description: 'Concise object or subject name.',
                  },
                  ymin: { type: Type.NUMBER },
                  xmin: { type: Type.NUMBER },
                  ymax: { type: Type.NUMBER },
                  xmax: { type: Type.NUMBER },
                },
                required: ['label', 'ymin', 'xmin', 'ymax', 'xmax'],
              },
            },
          },
          required: ['summary', 'items'],
        },
      };

      if (requestedModel === 'gemini-4-hydrogen') {
        config.thinkingConfig = { thinkingLevel: ThinkingLevel.LOW };
      }

      const response = await ai.models.generateContent({
        model: requestedModel,
        contents: {
          parts: [
            {
              inlineData: {
                mimeType,
                data: base64Data,
              },
            },
            {
              text: `${scopeNote}Detect and locate ALL visible items in this camera frame (up to 30 distinct objects). Coordinates ymin, xmin, ymax, xmax must be integers from 0 to 1000.`,
            },
          ],
        },
        config,
      });

      const rawText = (response.text ?? '{}')
        .trim()
        .replace(/^```(?:json)?\s*/i, '')
        .replace(/\s*```$/i, '');
      const parsed = JSON.parse(rawText) as {
        summary?: unknown;
        items?: unknown;
      };

      const summary =
        typeof parsed.summary === 'string' ? parsed.summary.trim() : '';
      const rawItems = Array.isArray(parsed.items) ? parsed.items : [];

      const items = rawItems
        .map((entry) => {
          if (!entry || typeof entry !== 'object') return null;
          const obj = entry as Record<string, unknown>;
          const label =
            typeof obj.label === 'string' ? obj.label.trim().slice(0, 60) : '';
          const ymin = Number(obj.ymin);
          const xmin = Number(obj.xmin);
          const ymax = Number(obj.ymax);
          const xmax = Number(obj.xmax);
          if (
            !label ||
            !Number.isFinite(ymin) ||
            !Number.isFinite(xmin) ||
            !Number.isFinite(ymax) ||
            !Number.isFinite(xmax)
          ) {
            return null;
          }
          const cy0 = Math.max(0, Math.min(990, ymin));
          const cx0 = Math.max(0, Math.min(990, xmin));
          const cy1 = Math.max(cy0 + 5, Math.min(1000, ymax));
          const cx1 = Math.max(cx0 + 5, Math.min(1000, xmax));
          return {
            label,
            ymin: cy0,
            xmin: cx0,
            ymax: cy1,
            xmax: cx1,
          };
        })
        .filter((item): item is NonNullable<typeof item> => item !== null)
        .slice(0, 30);

      res.json({
        summary,
        items,
        model: requestedModel,
      });
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Failed to scan scene with Gemini.';
      const lower = message.toLowerCase();

      if (
        lower.includes('403') ||
        lower.includes('permission_denied') ||
        lower.includes('api_key_invalid') ||
        lower.includes('api key')
      ) {
        res.status(403).json({
          error:
            'Gemini API key was rejected. Check GEMINI_API_KEY in your .env file or server environment.',
        });
        return;
      }

      if (lower.includes('429') || lower.includes('resource_exhausted')) {
        res.status(429).json({
          error: 'Gemini rate limit reached. Try again in a moment.',
        });
        return;
      }

      res.status(502).json({ error: `Gemini scene scan failed: ${message}` });
    }
  });

  const server = http.createServer(app);
  const wss = new WebSocketServer({ noServer: true, maxPayload: 2 * 1024 * 1024 });

  const phoneNames = new Map<string, string>();
  const activePhones = new Map<string, ActivePhone>();
  let nextPhoneNumber = 1;

  function getPhoneList(): Array<{ id: string; name: string }> {
    return Array.from(activePhones.values()).map((phone) => ({
      id: phone.id,
      name: phone.name,
    }));
  }

  function broadcastJson(payload: unknown): void {
    const serialized = JSON.stringify(payload);
    for (const client of wss.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(serialized);
      }
    }
  }

  function sendJson(ws: WebSocket, payload: unknown): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  server.on('upgrade', (request, socket, head) => {
    const reqUrl = new URL(
      request.url || '/',
      `http://${request.headers.host || 'localhost'}`
    );
    if (reqUrl.pathname !== '/ws') {
      return;
    }

    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit('connection', ws, request, reqUrl);
    });
  });

  wss.on(
    'connection',
    (rawWs: WebSocket, _req: http.IncomingMessage, reqUrl: URL) => {
      const ws = rawWs as ClientSocket;
      const clientId = sanitizeClientId(reqUrl.searchParams.get('clientId'));
      ws.clientId = clientId;
      ws.isAlive = true;

      sendJson(ws, {
        type: 'phones',
        phones: getPhoneList(),
      });

      ws.on('pong', () => {
        ws.isAlive = true;
      });

      ws.on('message', (data, isBinary) => {
        if (isBinary) {
          const phone = activePhones.get(clientId);
          if (!phone || phone.ws !== ws) {
            return;
          }

          const jpegBuf = Buffer.isBuffer(data)
            ? data
            : Array.isArray(data)
              ? Buffer.concat(data)
              : Buffer.from(data as ArrayBuffer);

          if (jpegBuf.byteLength === 0) return;

          const idBytes = Buffer.from(clientId, 'utf8');
          if (idBytes.byteLength > 255) return;

          const header = Buffer.allocUnsafe(1 + idBytes.byteLength);
          header.writeUInt8(idBytes.byteLength, 0);
          idBytes.copy(header, 1);

          const packet = Buffer.concat([header, jpegBuf]);

          for (const rawClient of wss.clients) {
            const client = rawClient as ClientSocket;
            if (
              client !== ws &&
              client.readyState === WebSocket.OPEN &&
              client.bufferedAmount <= MAX_RELAY_BUFFER_BYTES
            ) {
              client.send(packet, { binary: true });
            }
          }
          return;
        }

        let parsed: { type?: unknown } | null = null;
        try {
          parsed = JSON.parse(data.toString('utf8')) as { type?: unknown };
        } catch {
          return;
        }

        if (!parsed || typeof parsed.type !== 'string') return;

        if (parsed.type === 'phone:start') {
          let assignedName = phoneNames.get(clientId);
          if (!assignedName) {
            assignedName = `Phone ${nextPhoneNumber++}`;
            phoneNames.set(clientId, assignedName);
          }

          activePhones.set(clientId, {
            id: clientId,
            name: assignedName,
            ws,
          });

          sendJson(ws, {
            type: 'phone:registered',
            id: clientId,
            name: assignedName,
          });
          broadcastJson({
            type: 'phones',
            phones: getPhoneList(),
          });
        } else if (parsed.type === 'phone:stop') {
          const existing = activePhones.get(clientId);
          if (existing && existing.ws === ws) {
            activePhones.delete(clientId);
            broadcastJson({
              type: 'phones',
              phones: getPhoneList(),
            });
          }
        }
      });

      ws.on('close', () => {
        const existing = activePhones.get(clientId);
        if (existing && existing.ws === ws) {
          activePhones.delete(clientId);
          broadcastJson({
            type: 'phones',
            phones: getPhoneList(),
          });
        }
      });
    }
  );

  const heartbeatInterval = setInterval(() => {
    for (const rawClient of wss.clients) {
      const client = rawClient as ClientSocket;
      if (client.isAlive === false) {
        client.terminate();
        continue;
      }
      client.isAlive = false;
      client.ping();
    }
  }, 25000);

  wss.on('close', () => {
    clearInterval(heartbeatInterval);
  });

  if (!IS_PROD) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      const indexPath = path.join(distDir, 'index.html');
      if (fs.existsSync(indexPath)) {
        res.sendFile(indexPath);
      } else {
        res.status(404).send('Build output not found. Run npm run build first.');
      }
    });
  }

  server.listen(PORT, '0.0.0.0', () => {
    console.log(`Live Vision server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start Live Vision server:', err);
  process.exit(1);
});
