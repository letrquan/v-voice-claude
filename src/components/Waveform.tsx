import { useRef, useEffect } from "react";

interface WaveformProps {
  analyserNode: AnalyserNode | null;
  onFrame?: () => void;
}

export function Waveform({ analyserNode, onFrame }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);

  useEffect(() => {
    if (!analyserNode || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;
    const freqArr = new Uint8Array(analyserNode.frequencyBinCount);

    // Hi-DPI support
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = rect.width;
    const H = rect.height;

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      analyserNode.getByteFrequencyData(freqArr);

      ctx.clearRect(0, 0, W, H);

      const N = 28;
      const barW = Math.max(1.5, (W - 10) / N - 1.5);
      const gap = (W - N * barW) / (N + 1);
      const sl = Math.floor(freqArr.length / N);

      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let j = 0; j < sl; j++) s += freqArr[i * sl + j];
        const norm = s / sl / 255;
        const bH = Math.max(2, norm * (H - 4));
        const x = gap + i * (barW + gap);
        const cy = H / 2;

        // Mint green → soft purple gradient per bar
        const g = ctx.createLinearGradient(0, cy - bH / 2, 0, cy + bH / 2);
        g.addColorStop(0, `rgba(125, 249, 196, ${0.4 + norm * 0.6})`);
        g.addColorStop(1, `rgba(167, 139, 250, ${0.4 + norm * 0.6})`);
        ctx.fillStyle = g;

        // Glow on loud bars
        if (norm > 0.5) {
          ctx.shadowColor = "rgba(125, 249, 196, 0.45)";
          ctx.shadowBlur = 6;
        }

        ctx.beginPath();
        ctx.roundRect(x, cy - bH / 2, barW, bH, barW / 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      // Run VAD callback each frame
      onFrame?.();
    };

    draw();

    return () => {
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [analyserNode, onFrame]);

  return (
    <canvas
      ref={canvasRef}
      data-tauri-drag-region
      style={{ width: "100%", height: "100%", display: "block" }}
    />
  );
}
