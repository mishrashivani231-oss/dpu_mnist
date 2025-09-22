#!/usr/bin/env python3
import os, time, argparse
import numpy as np
from PIL import Image, ImageOps
import xir, vart

def dpu_subgraph(graph):
    root = graph.get_root_subgraph()
    subs = [s for s in root.toposort_child_subgraph()
            if s.has_attr("device") and s.get_attr("device").upper()=="DPU"]
    assert len(subs)==1, "Expected exactly 1 DPU subgraph"
    return subs[0]

def load28(img_path):
    """Return float32 image in [0,1] shaped (28,28). If no path, return zeros."""
    if img_path and os.path.isfile(img_path):
        img = Image.open(img_path).convert("L")
        # If the digit is dark on light background, invert so foreground is bright
        if np.asarray(img).mean() < 127: img = ImageOps.invert(img)
        img = img.resize((28,28))
        arr = np.asarray(img, dtype=np.float32)/255.0
    else:
        arr = np.zeros((28,28), dtype=np.float32)  # black image as fallback
    return arr

def to_dpu_dtype(x, it):
    """Quantize to int8 if required by input tensor dtype."""
    if "INT8" in str(it.dtype).upper():
        fix = it.get_attr("fix_point") if it.has_attr("fix_point") else 7
        return np.clip(np.round(x*(1<<fix)), -128, 127).astype(np.int8)
    return x.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deploy.xmodel")
    ap.add_argument("--image", default=None, help="28x28 grayscale PNG (optional)")
    ap.add_argument("--loops", type=int, default=100, help="number of inferences")
    args = ap.parse_args()

    assert os.path.isfile(args.model), f"Model not found: {args.model}"
    g = xir.Graph.deserialize(args.model)
    r = vart.Runner.create_runner(dpu_subgraph(g), "run")

    it = r.get_input_tensors()[0]
    ot = r.get_output_tensors()[0]
    ib_shape = tuple(it.dims)
    ob_shape = tuple(ot.dims)

    # Prepare one input sample
    x = load28(args.image)  # (28,28) float32 in [0,1]
    # NHWC vs NCHW
    if ib_shape[-1] == 1:
        x = x.reshape(1,28,28,1)
    else:
        x = x.reshape(1,1,28,28)
    x = to_dpu_dtype(x, it)

    # Allocate buffers once
    ib = [np.empty(ib_shape, dtype=x.dtype)]
    ob = [np.empty(ob_shape, dtype=np.int8 if "INT8" in str(ot.dtype).upper() else np.float32)]

    # Warmup
    np.copyto(ib[0], x)
    jid = r.execute_async(ib, ob); r.wait(jid)

    # Timed loop
    t0 = time.time()
    for _ in range(args.loops):
        np.copyto(ib[0], x)
        jid = r.execute_async(ib, ob)
        r.wait(jid)
    dt = time.time() - t0
    fps = args.loops / dt if dt > 0 else float("inf")

    # Decode a single result to show correctness
    y = ob[0].reshape(-1)
    if "INT8" in str(ot.dtype).upper():
        fix = ot.get_attr("fix_point") if ot.has_attr("fix_point") else 7
        y = y.astype(np.float32) / (1 << fix)
    pred = int(np.argmax(y))

    print("------------------------------------")
    print(f"Predicted digit: {pred}")
    print(f"Throughput: {fps:.2f} FPS  |  loops: {args.loops}  |  time: {dt:.4f} s")

if __name__ == "__main__":
    main()
