#!/usr/bin/env python3
import os, time, argparse, subprocess
import numpy as np
from PIL import Image, ImageOps
import xir, vart

def get_dpu_subgraphs(graph):
    """Return all DPU subgraphs (sorted largest-first)."""
    root = graph.get_root_subgraph()
    out = []
    def walk(sg):
        if sg.has_attr("device") and str(sg.get_attr("device")).upper() == "DPU":
            out.append(sg)
        for c in sg.children:
            walk(c)
    walk(root)
    try:
        out.sort(key=lambda s: len(s.get_ops()), reverse=True)
    except Exception:
        pass
    return out

def load28(img_path):
    """Return float32 image in [0,1] shaped (28,28). If no path, return zeros."""
    if img_path and os.path.isfile(img_path):
        img = Image.open(img_path).convert("L")
        # If the digit is dark on light background, invert so foreground is bright
        if np.asarray(img).mean() < 127: img = ImageOps.invert(img)
        img = img.resize((28,28))
        arr = np.asarray(img, dtype=np.float32)/255.0
    else:
        arr = np.zeros((28,28), dtype=np.float32)
    return arr

def to_dpu_dtype(x, it):
    """Quantize to int8 if required by input tensor dtype."""
    if "INT8" in str(it.dtype).upper():
        fix = it.get_attr("fix_point") if it.has_attr("fix_point") else 7
        return np.clip(np.round(x*(1<<fix)), -128, 127).astype(np.int8)
    return x.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mnist_int8_b4096.xmodel")
    ap.add_argument("--image", default=None, help="28x28 grayscale PNG (optional)")
    ap.add_argument("--loops", type=int, default=100, help="number of inferences")
    ap.add_argument("--subgraph-index", type=int, default=0,
                    help="which DPU subgraph to run if multiple are present (default: largest=0)")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    # Optional: print the model fingerprint for sanity
    try:
        out = subprocess.check_output(["xdputil", "xmodel", "-t", args.model], text=True)
        for line in out.splitlines():
            if "fingerprint" in line.lower():
                print(line.strip())
                break
    except Exception:
        pass

    g = xir.Graph.deserialize(args.model)
    dpu_subs = get_dpu_subgraphs(g)
    if not dpu_subs:
        raise RuntimeError("No DPU subgraph found in xmodel")
    if args.subgraph_index < 0 or args.subgraph_index >= len(dpu_subs):
        raise IndexError(f"--subgraph-index {args.subgraph_index} out of range (found {len(dpu_subs)} DPU subgraphs)")

    sg = dpu_subs[args.subgraph_index]
    print(f"Using DPU subgraph: {sg.get_name()}  (index {args.subgraph_index} / total {len(dpu_subs)})")
    r = vart.Runner.create_runner(sg, "run")

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
