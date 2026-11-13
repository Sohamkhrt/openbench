"""
Tiny 1D-CNN for road-vibration windows, trained with JAX and exported to ONNX by hand.

Architecture (input is the flat [N, 384] window, reshaped inside the graph):
  Reshape -> (x - mean) / std
  Conv1d(3->16, k=7) ReLU MaxPool(2)
  Conv1d(16->32, k=5) ReLU MaxPool(2)
  Conv1d(32->48, k=3) ReLU
  GlobalMaxPool -> Dense(48->4) -> Softmax

Training adds random Gaussian sensor noise (sigma up to 0.12 g) to every batch.
"""
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import optax
from onnx import TensorProto, helper, numpy_helper

LAYERS = [(3, 16, 7), (16, 32, 5), (32, 48, 3)]
N_CLASSES = 4


def init(key):
    params = {}
    for i, (cin, cout, k) in enumerate(LAYERS):
        key, sub = jax.random.split(key)
        params[f"w{i}"] = jax.random.normal(sub, (cout, cin, k)) * np.sqrt(2.0 / (cin * k))
        params[f"b{i}"] = jnp.zeros(cout)
    key, sub = jax.random.split(key)
    params["wd"] = jax.random.normal(sub, (LAYERS[-1][1], N_CLASSES)) * np.sqrt(1.0 / LAYERS[-1][1])
    params["bd"] = jnp.zeros(N_CLASSES)
    return params


def conv(x, w, b):
    k = w.shape[-1]
    y = jax.lax.conv_general_dilated(x, w, (1,), [(k // 2, k // 2)], dimension_numbers=("NCH", "OIH", "NCH"))
    return y + b[None, :, None]


def pool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 1, 2), (1, 1, 2), "VALID")


def forward(params, x, mean, std):
    h = (x.reshape(-1, 3, 128) - mean[None, :, None]) / std[None, :, None]
    for i in range(len(LAYERS)):
        h = jax.nn.relu(conv(h, params[f"w{i}"], params[f"b{i}"]))
        if i < len(LAYERS) - 1:
            h = pool(h)
    h = h.max(axis=-1)
    return h @ params["wd"] + params["bd"]


def train(X, y, epochs=60, batch=64, seed=0, noise_aug=0.12):
    Xc = X.reshape(-1, 3, 128)
    mean = jnp.asarray(Xc.mean(axis=(0, 2)))
    std = jnp.asarray(Xc.std(axis=(0, 2)) + 1e-6)
    params = init(jax.random.PRNGKey(seed))
    steps = epochs * (len(X) // batch)
    opt = optax.adamw(optax.cosine_decay_schedule(3e-3, steps), weight_decay=1e-3)
    state = opt.init(params)

    def loss_fn(p, xb, yb):
        return optax.softmax_cross_entropy_with_integer_labels(forward(p, xb, mean, std), yb).mean()

    @jax.jit
    def step(p, s, xb, yb):
        loss, g = jax.value_and_grad(loss_fn)(p, xb, yb)
        up, s = opt.update(g, s, p)
        return optax.apply_updates(p, up), s, loss

    rng = np.random.default_rng(seed)
    yj = jnp.asarray(y)
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        for i in range(0, len(idx) - batch + 1, batch):
            b = idx[i:i + batch]
            # Augment with random sensor noise so the net does not rely on a clean signal.
            sigma = rng.uniform(0, noise_aug, size=(batch, 1)).astype(np.float32)
            xb = X[b] + sigma * rng.standard_normal((batch, X.shape[1]), dtype=np.float32)
            params, state, loss = step(params, state, jnp.asarray(xb), yj[b])
        if ep % 10 == 9:
            print(f"    epoch {ep + 1}: loss {float(loss):.4f}")
    return {k: np.asarray(v, np.float32) for k, v in params.items()}, np.asarray(mean, np.float32), np.asarray(std, np.float32)


def n_params(params):
    return int(sum(v.size for v in params.values()))


def export(params, mean, std, path):
    init_ = [
        numpy_helper.from_array(np.array([-1, 3, 128], np.int64), "shape"),
        numpy_helper.from_array(mean.reshape(1, 3, 1), "mean"),
        numpy_helper.from_array(std.reshape(1, 3, 1), "std"),
        numpy_helper.from_array(params["wd"], "wd"),
        numpy_helper.from_array(params["bd"], "bd"),
    ]
    nodes = [
        helper.make_node("Reshape", ["input", "shape"], ["x3"]),
        helper.make_node("Sub", ["x3", "mean"], ["xc"]),
        helper.make_node("Div", ["xc", "std"], ["h_in"]),
    ]
    prev = "h_in"
    for i, (_, _, k) in enumerate(LAYERS):
        init_ += [numpy_helper.from_array(params[f"w{i}"], f"w{i}"), numpy_helper.from_array(params[f"b{i}"], f"b{i}")]
        nodes.append(helper.make_node("Conv", [prev, f"w{i}", f"b{i}"], [f"c{i}"], kernel_shape=[k], pads=[k // 2, k // 2]))
        nodes.append(helper.make_node("Relu", [f"c{i}"], [f"r{i}"]))
        prev = f"r{i}"
        if i < len(LAYERS) - 1:
            nodes.append(helper.make_node("MaxPool", [prev], [f"p{i}"], kernel_shape=[2], strides=[2]))
            prev = f"p{i}"
    nodes += [
        helper.make_node("GlobalMaxPool", [prev], ["gmp"]),
        helper.make_node("Flatten", ["gmp"], ["flat"]),
        helper.make_node("Gemm", ["flat", "wd", "bd"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    graph = helper.make_graph(
        nodes, "tiny_cnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 384])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, N_CLASSES])],
        init_,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="openbench")
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
