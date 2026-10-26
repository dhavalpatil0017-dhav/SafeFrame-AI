import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  // Dev server must run on port 3000 in AI Studio
  const PORT = process.env.NODE_ENV === 'production' && process.env.PORT ? Number(process.env.PORT) : 3000;

  // Body parser for base64 image data (up to 30MB)
  app.use(express.json({ limit: '30mb' }));

  // Initialize server-side Gemini client
  const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });

  // Health check endpoint
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'ok',
      hasApiKey: Boolean(process.env.GEMINI_API_KEY),
      primaryModel: 'gemini-3.1-flash-lite',
      fallbackModel: 'gemini-3.8-flash',
      serverTime: new Date().toISOString(),
    });
  });

  // Resilient runner with automatic model fallback and timeout
  async function runModelWithFallback(
    paramsFactory: (model: string) => any,
    timeoutMs = 28000
  ): Promise<{ text: string; modelUsed: string }> {
    // gemini-3.1-flash-lite is high-speed multimodal, then gemini-3.8-flash, then gemini-flash-latest
    const candidateModels = ['gemini-3.1-flash-lite', 'gemini-3.8-flash', 'gemini-flash-latest'];
    let lastError: any = null;

    for (const model of candidateModels) {
      try {
        console.log(`[SafeFrame AI] Querying model ${model}...`);
        const callPromise = ai.models.generateContent(paramsFactory(model));
        const timerPromise = new Promise<never>((_, reject) =>
          setTimeout(
            () => reject(new Error(`Model ${model} request exceeded timeout (${timeoutMs / 1000}s)`)),
            timeoutMs
          )
        );

        const result = await Promise.race([callPromise, timerPromise]);
        if (result && result.text) {
          console.log(`[SafeFrame AI] Model ${model} succeeded!`);
          return { text: result.text, modelUsed: model };
        }
      } catch (err: any) {
        console.warn(`[SafeFrame AI] ${model} attempt failed: ${err.message || err}. Trying next fallback...`);
        lastError = err;
      }
    }

    throw lastError || new Error('All Gemini model candidates failed or timed out.');
  }

  // Analyze individual CCTV evidence frame / image
  app.post('/api/analyze-evidence', async (req, res) => {
    try {
      const { imageBase64, mimeType = 'image/jpeg', fileName, cameraInfo, context } = req.body;

      if (!imageBase64) {
        return res.status(400).json({ error: 'Missing imageBase64 payload.' });
      }

      // Clean base64 data and auto-detect MIME
      let detectedMime = mimeType;
      let rawBase64 = imageBase64;
      const dataUrlMatch = imageBase64.match(/^data:([^;]+);base64,(.+)$/);
      if (dataUrlMatch) {
        detectedMime = dataUrlMatch[1];
        rawBase64 = dataUrlMatch[2];
      } else {
        rawBase64 = imageBase64.replace(/^data:[^;]+;base64,/, '');
      }

      const prompt = `Analyze this CCTV surveillance evidence frame from SafeFrame AI.
File: "${fileName || 'CCTV_Capture'}"
Camera/Source: "${cameraInfo || 'Security Camera'}"
Investigator Context: "${context || 'Routine incident investigation triage.'}"

Strict Forensic Protocol:
1. Clearly and strictly distinguish between CONFIRMED VISUAL OBSERVATIONS (what is irrefutably visible in the pixels: clothing, colors, postures, open doors, lighting, visible objects, position) and UNCERTAIN CONCLUSIONS / INFERENCES (speculations, suspected tools, motives, assumptions due to lighting or resolution).
2. Assess threat severity (LOW, MEDIUM, HIGH, or CRITICAL) with clear reasoning.
3. List key entities or anomalies detected with their spatial location in the frame.
4. Recommend concrete next investigative verification steps.`;

      const { text: responseText, modelUsed } = await runModelWithFallback((model) => ({
        model,
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: detectedMime,
                data: rawBase64,
              },
            },
            {
              text: prompt,
            },
          ],
        },
        config: {
          systemInstruction:
            'You are an expert digital forensics examiner and physical security analyst. You inspect CCTV footage and frame captures. You never present assumptions as facts. You rigorously separate confirmed pixel-level observations from speculative conclusions, probabilistic estimates, and hypotheses.',
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              summary: {
                type: Type.STRING,
                description: 'Concise 2-sentence forensic overview of the frame.',
              },
              threatLevel: {
                type: Type.STRING,
                description: 'Threat level: LOW, MEDIUM, HIGH, or CRITICAL.',
              },
              threatJustification: {
                type: Type.STRING,
                description: 'Why this threat level was assigned.',
              },
              confirmedObservations: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: 'Verified visual facts only. What is directly visible in the frame.',
              },
              uncertainConclusions: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: 'Inferences, hypotheses, uncertain items, or probabilities.',
              },
              entitiesDetected: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    label: { type: Type.STRING },
                    location: { type: Type.STRING },
                    confidence: { type: Type.STRING },
                    details: { type: Type.STRING },
                  },
                },
                description: 'Detected subjects, vehicles, equipment, or objects.',
              },
              lightingAndQualityAssessment: {
                type: Type.STRING,
                description: 'Sensor clarity, illumination, perspective, lens distortion, or visibility conditions.',
              },
              anomalyHighlights: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: 'Security anomalies detected (e.g., unauthorized zone entry, unbadged presence, darkness manipulation).',
              },
              investigativeActions: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: 'Actionable next steps for security teams (e.g., cross-check badge reader, check exterior cameras).',
              },
            },
          },
        },
      }));

      const parsed = JSON.parse(responseText);
      return res.json({ success: true, data: parsed, modelUsed });
    } catch (error: any) {
      console.error('[SafeFrame AI] Error analyzing CCTV evidence:', error);
      return res.status(500).json({
        error: error.message || 'Failed to analyze CCTV frame via AI.',
        details: error.toString(),
      });
    }
  });

  // Synthesize multi-source incident case
  app.post('/api/synthesize-incident', async (req, res) => {
    try {
      const { caseId, title, evidenceList, timelineEvents } = req.body;

      const prompt = `Synthesize an integrated CCTV Security Incident Report for SafeFrame AI:
Case ID: ${caseId || 'SF-UNKNOWN'}
Case Title: ${title || 'Physical Security Anomaly'}

Evidence Items:
${JSON.stringify(evidenceList || [], null, 2)}

Timeline Events:
${JSON.stringify(timelineEvents || [], null, 2)}

Generate a consolidated incident assessment adhering to digital forensics standards. Rigorously separate verified timeline facts from investigative deductions.`;

      const { text, modelUsed } = await runModelWithFallback((model) => ({
        model,
        contents: prompt,
        config: {
          systemInstruction:
            'You are a Lead Forensic Incident Investigator. Provide a formal, impartial incident briefing synthesizing multiple CCTV frames and timeline logs into a coherent investigation briefing.',
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              executiveSummary: { type: Type.STRING },
              incidentClassification: { type: Type.STRING },
              calculatedSeverity: { type: Type.STRING },
              verifiedSequenceOfEvents: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
              investigativeHypotheses: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
              criticalVulnerabilitiesExposed: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
              recommendedRemediation: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
            },
          },
        },
      }));

      return res.json({ success: true, data: JSON.parse(text), modelUsed });
    } catch (error: any) {
      console.error('[SafeFrame AI] Error synthesizing incident:', error);
      return res.status(500).json({
        error: error.message || 'Failed to synthesize incident narrative.',
      });
    }
  });

  // Attach Vite middleware for dev or serve static files in prod
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[SafeFrame AI] Server active on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('[SafeFrame AI] Failed to start server:', err);
  process.exit(1);
});
