// Mixes mic input to mono PCM_s16le and posts 2048-sample frames (4 KiB).
// Frames go straight to the loopback socket; nothing is kept here.
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Int16Array(2048);
    this.length = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channels = input.length;
      const samples = input[0].length;
      for (let i = 0; i < samples; i++) {
        let sum = 0;
        for (let c = 0; c < channels; c++) sum += input[c][i];
        const value = Math.max(-1, Math.min(1, sum / channels));
        this.frame[this.length++] = value < 0 ? value * 32768 : value * 32767;
        if (this.length === this.frame.length) {
          this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
          this.frame = new Int16Array(2048);
          this.length = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
