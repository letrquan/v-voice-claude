/**
 * AudioWorklet processor that captures PCM audio data and sends it
 * to the main thread in buffered chunks.
 */
class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = false;
    this.buffer = [];
    this.bufferSize = 4096; // Send every 4096 samples (~256ms at 16kHz)

    this.port.onmessage = (event) => {
      if (event.data.command === "start") {
        this.recording = true;
        this.buffer = [];
      } else if (event.data.command === "stop") {
        this.recording = false;
        // Flush remaining buffer
        if (this.buffer.length > 0) {
          this.port.postMessage({
            type: "audio-data",
            samples: this.buffer.slice(),
          });
          this.buffer = [];
        }
      }
    };
  }

  process(inputs) {
    if (this.recording && inputs[0] && inputs[0][0]) {
      const channelData = inputs[0][0];
      for (let i = 0; i < channelData.length; i++) {
        this.buffer.push(channelData[i]);
      }

      // Send when buffer is full
      if (this.buffer.length >= this.bufferSize) {
        this.port.postMessage({
          type: "audio-data",
          samples: this.buffer.slice(),
        });
        this.buffer = [];
      }
    }
    return true;
  }
}

registerProcessor("audio-processor", AudioProcessor);
