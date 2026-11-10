"""
OpenBench model training + ONNX export.

Generates a physically-modelled road-vibration dataset (3-axis accelerometer,
100 Hz, 1.28 s windows), trains three edge models and exports them to ONNX:

  1. mlp_fp32.onnx   - baseline FP32 neural net (scikit-learn MLP)
  2. mlp_int8.onnx   - same network, dynamic INT8 quantization (onnxruntime)
  3. rf_sklearn.onnx - ultralight Random Forest (scikit-learn -> skl2onnx)

Also writes 20 static JSON fixtures (5 per class) for the browser playground,
and a models manifest with held-out accuracy and file sizes.

Usage:  python training/train.py
Deps:   pip install numpy scikit-learn skl2onnx onnx onnxruntime
"""
import json
import os
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_MODELS = os.path.join(ROOT, "public", "models")
OUT_FIX = os.path.join(ROOT, "public", "fixtures")
os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_FIX, exist_ok=True)

FS = 100            # Hz
N = 128             # samples per window (1.28 s)
CLASSES = ["smooth", "rough", "pothole", "speed_bump"]
rng = np.random.default_rng(42)
t = np.arange(N) / FS


def colored_noise(n, alpha, scale):
    """1/f^alpha noise (road texture is roughly pink/brown)."""
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = 1
    spec = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / f ** (alpha / 2)
    x = np.fft.irfft(spec, n)
    return scale * x / (x.std() + 1e-9)


def damped(t0, f, zeta, amp):
    """Damped oscillation starting at t0 (suspension / wheel-hop response)."""
    tt = np.clip(t - t0, 0, None)
    w = 2 * np.pi * f
    return amp * np.exp(-zeta * w * tt) * np.sin(w * tt) * (t >= t0)


def sample(cls):
    speed = rng.uniform(15, 60)                 # km/h
    engine_f = rng.uniform(22, 40)              # Hz
    base_rough = rng.uniform(0.02, 0.06)
    x = colored_noise(N, 1.2, base_rough * 0.6)
    y = colored_noise(N, 1.2, base_rough * 0.6)
    z = colored_noise(N, 1.0, base_rough)
    eng = rng.uniform(0.01, 0.03) * np.sin(2 * np.pi * engine_f * t + rng.uniform(0, 6.28))
    x, y, z = x + eng * 0.5, y + eng * 0.5, z + eng

    if cls == "rough":
        z += colored_noise(N, 0.6, rng.uniform(0.08, 0.2))
        x += colored_noise(N, 0.6, rng.uniform(0.03, 0.08))
        y += colored_noise(N, 0.6, rng.uniform(0.03, 0.08))
    elif cls == "pothole":
        t0 = rng.uniform(0.15, 1.0)
        depth = rng.uniform(0.25, 0.9)
        z += -depth * np.exp(-((t - t0) / 0.025) ** 2)          # drop
        z += damped(t0 + 0.04, rng.uniform(9, 14), 0.25, depth * 1.4)  # wheel hop
        z += damped(t0 + 0.04, rng.uniform(1.2, 2.0), 0.3, depth * 0.3)
        x += damped(t0 + 0.04, rng.uniform(6, 10), 0.3, depth * 0.35)
        # rear wheel follows (wheelbase ~2.6 m)
        t1 = t0 + 2.6 / (speed / 3.6)
        if t1 < t[-1]:
            z += damped(t1, rng.uniform(9, 14), 0.25, depth * 0.8)
    elif cls == "speed_bump":
        t0 = rng.uniform(0.1, 0.5)
        width = rng.uniform(0.12, 0.3) * (30 / speed)
        h = rng.uniform(0.2, 0.55)
        z += h * np.exp(-((t - t0) / width) ** 2)
        z += damped(t0 + width, rng.uniform(1.0, 1.8), 0.2, h * 0.5)
        t1 = t0 + 2.6 / (speed / 3.6)
        if t1 < t[-1]:
            z += 0.8 * h * np.exp(-((t - t1) / width) ** 2)
        x += 0.15 * h * np.gradient(np.exp(-((t - t0) / width) ** 2)) * 10

    # mild random mounting tilt + gravity on z
    z += 1.0
    w = np.stack([x, y, z]).astype(np.float32)   # (3, N)
    return w


def make(n_per_class):
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(sample(c).reshape(-1))
            y.append(ci)
    X, y = np.array(X, np.float32), np.array(y, np.int64)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def mb(p):
    return os.path.getsize(p) / (1024 * 1024)


def onnx_acc(path, X, y):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    outs = [o.name for o in s.get_outputs()]
    pi = next(i for i, o in enumerate(s.get_outputs()) if "prob" in o.name)
    t0 = time.perf_counter()
    probs = s.run(None, {name: X})[pi]
    dt = (time.perf_counter() - t0) * 1000 / len(X)
    return accuracy_score(y, probs.argmax(1)), dt, outs


print("Generating dataset ...")
Xtr, ytr = make(700)
Xte, yte = make(250)
print("train", Xtr.shape, "test", Xte.shape)

print("Training MLP ...")
mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200,
                    random_state=0, early_stopping=True, alpha=1e-2))
mlp.fit(Xtr, ytr)
print("MLP acc", mlp.score(Xte, yte))

print("Training Random Forest ...")
rf = RandomForestClassifier(n_estimators=40, max_depth=10, random_state=0, n_jobs=-1)
rf.fit(Xtr, ytr)
print("RF acc", rf.score(Xte, yte))

opts = {"zipmap": False}
p_fp32 = os.path.join(OUT_MODELS, "mlp_fp32.onnx")
p_int8 = os.path.join(OUT_MODELS, "mlp_int8.onnx")
p_rf = os.path.join(OUT_MODELS, "rf_sklearn.onnx")

for model, path in [(mlp, p_fp32), (rf, p_rf)]:
    clf = model.steps[-1][1] if hasattr(model, "steps") else model
    onx = to_onnx(model, Xtr[:1], options={id(clf): opts}, target_opset=17)
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

quantize_dynamic(p_fp32, p_int8, weight_type=QuantType.QInt8)

manifest = {"classes": CLASSES, "input": {"channels": 3, "length": N, "fs": FS,
                                           "layout": "[x(128), y(128), z(128)] flattened, units g"},
            "models": []}
meta = [
    ("mlp_fp32", "TinyMLP FP32", "StandardScaler + MLP 384-256-128-4, float32 weights", "fp32", p_fp32),
    ("mlp_int8", "TinyMLP INT8", "Same net, dynamic INT8 weight quantization (onnxruntime)", "int8", p_int8),
    ("rf_sklearn", "Random Forest", "40 trees, depth 10, scikit-learn -> skl2onnx", "trees", p_rf),
]
for mid, name, desc, prec, path in meta:
    acc, ms, outs = onnx_acc(path, Xte, yte)
    print(f"{mid}: acc={acc:.4f} cpu={ms:.3f}ms/sample size={mb(path):.3f}MB outputs={outs}")
    manifest["models"].append({
        "id": mid, "name": name, "description": desc, "precision": prec,
        "file": f"models/{os.path.basename(path)}", "sizeMB": round(mb(path), 4),
        "heldOutAccuracy": round(float(acc), 4), "heldOutSamples": int(len(yte)),
    })

with open(os.path.join(OUT_MODELS, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# 20 fixtures: 5 per class, from a fresh draw
fixtures = []
for ci, c in enumerate(CLASSES):
    for k in range(5):
        w = sample(c)
        fixtures.append({"id": f"{c}-{k + 1}", "label": c, "labelIndex": ci,
                         "data": [round(float(v), 4) for v in w.reshape(-1)]})
with open(os.path.join(OUT_FIX, "road_vibration.json"), "w") as f:
    json.dump({"name": "Road vibration (3-axis accelerometer)", "fs": FS, "length": N,
               "classes": CLASSES, "samples": fixtures}, f)
print("Wrote", len(fixtures), "fixtures")
