// Browser mic pipeline. Called only from an explicit Start; stop() releases
// every track and closes the audio context so the OS mic indicator turns off.
export type MicPipeline = { sampleRate: number; analyser: AnalyserNode; stop: () => void };

export async function openMic(onFrame: (pcm: ArrayBuffer) => void, dsp = true): Promise<MicPipeline> {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new DOMException("Microphone needs a secure context.", "SecurityError");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    // Browser DSP lifts quiet speech, but noise suppression can clip short word
    // onsets; `dsp = false` (?dsp=off on /live) lets the two be compared.
    // Frames are never skipped: the server derives time from the sample count.
    audio: { channelCount: 1, echoCancellation: dsp, noiseSuppression: dsp, autoGainControl: true },
  });
  let context: AudioContext | null = null;
  try {
    context = new AudioContext();
    if (context.state === "suspended") await context.resume();
    await context.audioWorklet.addModule("/pcm-worklet.js");
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, "pcm-capture", { numberOfInputs: 1, numberOfOutputs: 0 });
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    node.port.onmessage = (event: MessageEvent<ArrayBuffer>) => onFrame(event.data);
    // A silent sink makes the browser pull the analyser; nothing reaches the speakers.
    const sink = context.createGain();
    sink.gain.value = 0;
    source.connect(node);
    source.connect(analyser);
    analyser.connect(sink);
    sink.connect(context.destination);
    const opened = context;
    let stopped = false;
    return {
      sampleRate: opened.sampleRate,
      analyser,
      stop: () => {
        if (stopped) return;
        stopped = true;
        node.port.onmessage = null;
        source.disconnect();
        node.disconnect();
        analyser.disconnect();
        sink.disconnect();
        for (const track of stream.getTracks()) track.stop();
        void opened.close();
      },
    };
  } catch (error) {
    for (const track of stream.getTracks()) track.stop();
    void context?.close();
    throw error;
  }
}

export async function micPermission(): Promise<PermissionState | null> {
  try {
    return (await navigator.permissions.query({ name: "microphone" as PermissionName })).state;
  } catch {
    return null;
  }
}
