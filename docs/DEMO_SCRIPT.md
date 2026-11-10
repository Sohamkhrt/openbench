# 90-second demo video script

Record at 1440x900 in Chrome, live mode. Close other tabs so latency is stable.

| Time | On screen | Voice-over |
|---|---|---|
| 0:00-0:10 | Landing page, cursor on the "100% client-side · $0 compute" badge | "Moving ML models to edge hardware fails on latency spikes, size and memory you never saw in the notebook. OpenBench shows those trade-offs in the browser, for free." |
| 0:10-0:20 | Open DevTools Network tab briefly, then close | "No server, no GPU, no API keys. Three ONNX models run right here through onnxruntime-web and WebAssembly." |
| 0:20-0:35 | Click **Run benchmark**, progress bars fill per model | "It warms each model up, then times every inference with performance.now() on 20 road-vibration sensor windows: smooth road, rough road, potholes and speed bumps." |
| 0:35-0:50 | Model cards and EVS chart | "The Edge Viability Score combines accuracy, latency and size. INT8 quantization cuts the network to a quarter of its size with no accuracy loss, and a tiny random forest wins outright." |
| 0:50-1:00 | Latency timeline, hover a spike | "Here is per-inference jitter. Spikes like this are what break real-time loops on a microcontroller." |
| 1:00-1:10 | Drag latency and size budget sliders, badges flip | "Set your device's budget and see which models still fit." |
| 1:10-1:20 | Drag sensor noise to 0.2 g, rerun, open a pothole in the signal inspector | "Add sensor noise to stress-test robustness and see which model breaks first." |
| 1:20-1:30 | Switch to Playback, click Export JSON, end on GitHub link | "A recorded playback guarantees the demo never fails, and every run exports as JSON. OpenBench is open source and costs nothing to run." |
