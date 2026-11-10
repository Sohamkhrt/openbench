# 90-second demo video script

Record at 1440x900 in Chrome, live mode, animation speed "Normal". Close other tabs so latency is stable.

| Time | On screen | Voice-over |
|---|---|---|
| 0:00-0:10 | Hero text, cursor on the "100% client-side · $0 compute" badge | "Moving ML models to edge hardware fails on latency spikes, size and memory you never saw in the notebook. OpenBench shows those trade-offs in your browser, for free." |
| 0:10-0:30 | Click **Run benchmark on 9 models**; the sweep line crosses the signal, the pipeline lights up, probabilities fill, scoreboard turns green and red | "Nine ONNX models run right here through onnxruntime-web. You can watch each one read 1.28 seconds of accelerometer data and decide: smooth road, rough road, pothole or speed bump. Every square is a real inference." |
| 0:30-0:45 | Scroll to Prediction quality and Speed; open one "What this means for you" | "Fourteen metrics, each ranked and explained. Hazard recall is the safety number: how many potholes it actually catches. Latency is timed with performance.now(), p50 and p95." |
| 0:45-0:55 | Footprint section, point at hatched INT8 bars | "INT8 quantization makes the network four times smaller with no accuracy loss. kNN drags its whole training set along, which is why it's a bad edge choice." |
| 0:55-1:15 | What should I pick?: switch device from Raspberry Pi Pico to Arduino Nano, drag minimum accuracy | "Tell it your hardware. On a Pico it picks the random forest; on an Arduino with 256 KB it switches to gradient boosting, and explains what each choice means." |
| 1:15-1:30 | Switch to Playback, then Export JSON, end on GitHub link | "A recorded playback guarantees the demo never fails, and every run exports as JSON. OpenBench is open source and costs nothing to run." |
