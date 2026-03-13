import { useRef, useEffect } from "react";

interface WaveformProps {
  analyserNode: AnalyserNode | null;
  onFrame?: (rms: number) => void;
}

export function Waveform({ analyserNode, onFrame }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const heightsRef = useRef<number[]>([]);

  useEffect(() => {
    if (!analyserNode || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;
    const freqArr = new Uint8Array(analyserNode.frequencyBinCount);
    // time-domain data for RMS volume calculation to pass to parent
    const timeArr = new Uint8Array(analyserNode.fftSize);

    // Hi-DPI support
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = rect.width;
    const H = rect.height;

    const N = 24; // fewer, thicker bars
    if (heightsRef.current.length !== N) {
      heightsRef.current = new Array(N).fill(2); // init heights
    }

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      analyserNode.getByteFrequencyData(freqArr);
      analyserNode.getByteTimeDomainData(timeArr);

      // calc RMS
      let sum = 0;
      for (let i = 0; i < timeArr.length; i++) {
        const v = (timeArr[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / timeArr.length);

      ctx.clearRect(0, 0, W, H);

      const barW = Math.max(2, (W - 10) / N - 2);
      const gap = (W - N * barW) / (N + 1);
      const sl = Math.floor(freqArr.length / N);

      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let j = 0; j < sl; j++) s += freqArr[i * sl + j];
        const norm = s / sl / 255;
        // give the bars a baseline height, max height is slightly less than canvas height
        const targetH = Math.max(4, norm * (H - 6));
        
        // spring physics / lerp
        const currentH = heightsRef.current[i];
        const nextH = currentH + (targetH - currentH) * 0.25;
        heightsRef.current[i] = nextH;

        const x = gap + i * (barW + gap);
        const cy = H / 2;

        const g = ctx.createLinearGradient(0, cy - nextH / 2, 0, cy + nextH / 2);
        g.addColorStop(0, `rgba(125, 249, 196, ${0.5 + norm * 0.5})`);
        g.addColorStop(1, `rgba(167, 139, 250, ${0.5 + norm * 0.5})`);
        ctx.fillStyle = g;

        if (norm > 0.4) {
          ctx.shadowColor = "rgba(125, 249, 196, 0.5)";
          ctx.shadowBlur = 8;
        } else {
          ctx.shadowBlur = 0;
        }

        ctx.beginPath();
        ctx.roundRect(x, cy - nextH / 2, barW, nextH, barW / 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      onFrame?.(rms);
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
