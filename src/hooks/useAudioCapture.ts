import { useRef, useState, useCallback } from "react";

export interface AudioData {
  samples: Float32Array;
  sampleRate: number;
}

export function useAudioCapture() {
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const samplesRef = useRef<Float32Array[]>([]);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
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

      samplesRef.current = [];
      workletNode.port.onmessage = (event) => {
        if (event.data.type === "audio-data") {
          samplesRef.current.push(new Float32Array(event.data.samples));
        }
      };

      source.connect(workletNode);
      workletNode.port.postMessage({ command: "start" });
      setIsCapturing(true);
    } catch (err) {
      console.error("Failed to start audio capture:", err);
    }
  }, []);

  const stop = useCallback(async (): Promise<AudioData | null> => {
    if (!audioContextRef.current || !workletNodeRef.current) return null;

    // Signal worklet to stop and flush remaining buffer
    workletNodeRef.current.port.postMessage({ command: "stop" });

    // Wait for final data to arrive
    await new Promise((resolve) => setTimeout(resolve, 150));

    const sampleRate = audioContextRef.current.sampleRate;

    // Merge all sample chunks into one Float32Array
    const totalLength = samplesRef.current.reduce(
      (sum, chunk) => sum + chunk.length,
      0
    );
    const allSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of samplesRef.current) {
      allSamples.set(chunk, offset);
      offset += chunk.length;
    }

    // Cleanup
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    await audioContextRef.current.close();

    setAnalyserNode(null);
    setIsCapturing(false);
    audioContextRef.current = null;
    workletNodeRef.current = null;
    samplesRef.current = [];

    return { samples: allSamples, sampleRate };
  }, []);

  return { start, stop, analyserNode, isCapturing };
}
