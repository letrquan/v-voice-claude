import { useRef, useState, useCallback } from "react";

export interface AudioData {
  samples: Float32Array;
  sampleRate: number;
}

/** Turn a getUserMedia rejection into something worth showing a user. */
function describeCaptureError(err: unknown): string {
  const name = err instanceof DOMException ? err.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Microphone access was denied.";
    case "NotFoundError":
      return "No microphone was found.";
    case "OverconstrainedError":
      return "The selected microphone is unavailable. Pick another in Settings.";
    case "NotReadableError":
      return "The microphone is in use by another app.";
    default:
      return `Could not start the microphone: ${
        err instanceof Error ? err.message : String(err)
      }`;
  }
}

export function useAudioCapture() {
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const sinkNodeRef = useRef<GainNode | null>(null);
  const samplesRef = useRef<Float32Array[]>([]);
  const sampleCountRef = useRef(0);

  /** Tear down every node, the media stream, and the context. Safe to call twice. */
  const releaseResources = useCallback(async () => {
    sourceRef.current?.disconnect();
    sourceRef.current = null;
    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;
    sinkNodeRef.current?.disconnect();
    sinkNodeRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (audioContextRef.current) {
      try {
        await audioContextRef.current.close();
      } catch {
        // Already closed
      }
      audioContextRef.current = null;
    }

    setAnalyserNode(null);
    setIsCapturing(false);
    samplesRef.current = [];
    sampleCountRef.current = 0;
  }, []);

  /** Start capture. Pass a deviceId to use a specific microphone, or "" for default.
   *  Rejects if the microphone cannot be opened — the caller must not assume
   *  that returning means audio is flowing.
   */
  const start = useCallback(async (deviceId?: string) => {
    try {
      const audioConstraints: MediaTrackConstraints = {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      };
      if (deviceId) {
        audioConstraints.deviceId = { exact: deviceId };
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: audioConstraints,
      });
      streamRef.current = stream;

      // Try 16kHz (ideal for Whisper), fallback to default
      let audioContext: AudioContext;
      try {
        audioContext = new AudioContext({ sampleRate: 16000 });
      } catch {
        audioContext = new AudioContext();
      }
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      // AnalyserNode for waveform visualization
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.75;
      source.connect(analyser);
      setAnalyserNode(analyser);

      // AudioWorklet for recording PCM data
      await audioContext.audioWorklet.addModule("/audio-processor.js");
      const workletNode = new AudioWorkletNode(
        audioContext,
        "audio-processor"
      );
      workletNodeRef.current = workletNode;

      const sinkNode = audioContext.createGain();
      sinkNode.gain.value = 0;
      sinkNodeRef.current = sinkNode;

      samplesRef.current = [];
      sampleCountRef.current = 0;
      workletNode.port.onmessage = (event) => {
        if (event.data.type === "audio-data") {
          const samples = event.data.samples instanceof Float32Array
            ? event.data.samples
            : new Float32Array(event.data.samples);
          samplesRef.current.push(samples);
          sampleCountRef.current += samples.length;
        }
      };

      source.connect(workletNode);
      workletNode.connect(sinkNode);
      sinkNode.connect(audioContext.destination);
      workletNode.port.postMessage({ command: "start" });
      setIsCapturing(true);
    } catch (err) {
      // Roll back anything that did open, then let the caller show the reason
      // instead of leaving a widget that looks live and records nothing.
      await releaseResources();
      throw new Error(describeCaptureError(err));
    }
  }, [releaseResources]);

  /** Get a snapshot of the accumulated audio buffer without stopping recording. */
  const getBuffer = useCallback((): AudioData | null => {
    if (!audioContextRef.current || samplesRef.current.length === 0) return null;

    const sampleRate = audioContextRef.current.sampleRate;
    const totalLength = sampleCountRef.current;
    if (totalLength === 0) return null;

    const allSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of samplesRef.current) {
      allSamples.set(chunk, offset);
      offset += chunk.length;
    }

    return { samples: allSamples, sampleRate };
  }, []);

  /** Consume the accumulated audio buffer, returning it and clearing it.
   *  New audio chunks arriving after this call start a fresh buffer.
   */
  const consumeBuffer = useCallback((): AudioData | null => {
    if (!audioContextRef.current || samplesRef.current.length === 0) return null;

    const sampleRate = audioContextRef.current.sampleRate;
    const chunks = samplesRef.current;
    const totalLength = sampleCountRef.current;
    if (totalLength === 0) return null;

    // Swap the queue before merging so audio arriving after this snapshot is
    // kept for the next transcription window.
    samplesRef.current = [];
    sampleCountRef.current = 0;

    const allSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      allSamples.set(chunk, offset);
      offset += chunk.length;
    }

    return { samples: allSamples, sampleRate };
  }, []);

  const stop = useCallback(async (): Promise<AudioData | null> => {
    if (!audioContextRef.current || !workletNodeRef.current) return null;

    // Signal worklet to stop and flush remaining buffer
    workletNodeRef.current.port.postMessage({ command: "stop" });

    // Wait for final data to arrive
    await new Promise((resolve) => setTimeout(resolve, 150));

    const sampleRate = audioContextRef.current.sampleRate;

    // Merge all sample chunks into one Float32Array
    const chunks = samplesRef.current;
    const totalLength = sampleCountRef.current;
    const allSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      allSamples.set(chunk, offset);
      offset += chunk.length;
    }

    await releaseResources();

    return { samples: allSamples, sampleRate };
  }, [releaseResources]);

  return { start, stop, getBuffer, consumeBuffer, analyserNode, isCapturing };
}
