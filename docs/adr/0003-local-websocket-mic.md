# Browser mic to local FastAPI over WebSocket, local-only

Live input uses browser `getUserMedia`/AudioWorklet streaming mono PCM over versioned WebSocket to FastAPI on loopback, which owns resampling to 16kHz and ring-buffer inference. Microphone runs only after explicit Start and stops on Stop/disconnect, with no raw mic retained by default.
