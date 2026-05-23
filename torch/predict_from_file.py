import os
import sys
import argparse
import numpy as np
import torch

# Project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import resources.models as models

import librosa


def pad_or_trim(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)), mode="constant")
    return x[:target_len]


def multi_crop_1d(x: np.ndarray, input_len: int, n_crops: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if len(x) < input_len:
        x = np.pad(x, (0, input_len - len(x)), mode="constant")

    if n_crops <= 1:
        return pad_or_trim(x, input_len)[None, :]

    if len(x) == input_len:
        return np.tile(x[None, :], (n_crops, 1))

    max_start = len(x) - input_len
    if max_start <= 0:
        return np.tile(pad_or_trim(x, input_len)[None, :], (n_crops, 1))

    starts = np.linspace(0, max_start, num=n_crops).astype(int)
    return np.stack([x[s:s + input_len] for s in starts], axis=0)


def load_labels(path, n_classes):
    if not path:
        return [f"class_{i}" for i in range(n_classes)]
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    labels = (lines + [None] * n_classes)[:n_classes]
    return [lbl if lbl else f"class_{i}" for i, lbl in enumerate(labels)]


@torch.no_grad()
def predict(net, wav, input_len, n_crops, device):
    crops = multi_crop_1d(wav, input_len, n_crops)  # (B, L)
    x = torch.from_numpy(crops).to(device=device, dtype=torch.float32)

    # Try the two common shapes for your ACDNet variants
    try:
        x_in = x.unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
        logits = net(x_in)
    except Exception:
        x_in = x.unsqueeze(1).unsqueeze(3)  # (B,1,L,1)
        logits = net(x_in)

    probs = torch.softmax(logits, dim=-1)   # (B, C)
    probs_mean = probs.mean(dim=0)          # (C,)
    pred = int(torch.argmax(probs_mean).item())
    return pred, probs_mean.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="state_dict checkpoint (.pt/.pth) saved by torch.save({...})")
    ap.add_argument("--audio", required=True, help="Audio file (wav/mp3/m4a/ogg...)")
    ap.add_argument("--sr", type=int, default=20000, help="Target SR (match training)")
    ap.add_argument("--input-len", type=int, default=30225, help="Input length in samples (match training)")
    ap.add_argument("--n-crops", type=int, default=10, help="Number of crops to average")
    ap.add_argument("--labels", default=None, help="Optional labels file (one label per line)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-sec", type=float, default=5.0, help="Use only first N seconds (phone clips can be long)")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device = "cpu"

    print("Loading checkpoint:", args.model)
    state = torch.load(args.model, map_location=device)

    # Infer classes from checkpoint config if possible
    n_classes = 10
    print("n_classes:", n_classes)

    print("Building model...")
    net = models.GetACDNetModel(
        input_len=args.input_len,
        nclass=n_classes,
        sr=args.sr,
        channel_config=state["config"]
    ).to(device)

    net.load_state_dict(state["weight"])
    net.eval()

    print("Loading audio:", args.audio)
    wav, _ = librosa.load(args.audio, sr=args.sr, mono=True)  # float32
    print("Audio loaded:", wav.shape, "seconds:", len(wav) / args.sr)

    # Limit duration (recommended)
    if args.max_sec and args.max_sec > 0:
        max_len = int(args.max_sec * args.sr)
        if len(wav) > max_len:
            wav = wav[:max_len]
            print("Trimmed to:", args.max_sec, "seconds")

    # RMS normalization (helps phone recordings with AGC)
    eps = 1e-8
    rms = float(np.sqrt(np.mean(wav ** 2) + eps))
    target_rms = 0.05
    wav = wav * (target_rms / max(rms, eps))
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)

    #labels = load_labels(args.labels, n_classes)
    labels = [
    "cat",            # 0
    "chirpingbirds", # 1
    "cow",           # 2
    "dog",           # 3
    "frog",          # 4
    "hen",           # 5
    "insects",       # 6
    "pig",           # 7
    "rooster",       # 8
    "sheep"          # 9
    ]


    print("Running inference...")
    pred, probs = predict(net, wav, args.input_len, args.n_crops, device)

    topk = min(5, len(probs))
    idx = np.argsort(probs)[::-1][:topk]

    print("\nPrediction:")
    print(f"  top-1: {labels[pred]} ({probs[pred]*100:.1f}%)")
    print("  top-5:")
    for i in idx:
        print(f"    {labels[i]}: {probs[i]*100:.1f}%")

if __name__ == "__main__":
    main()

