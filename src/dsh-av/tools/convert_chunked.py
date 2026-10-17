# -*- coding: utf-8 -*-
"""长音频音色转换（分块）—— dsh-av 0.5.6

为什么需要它：
  原版 convert_pth.py 把整段音频一次性过 hubert，注意力矩阵是 T² 的。
  4 分钟的歌 = 12000 帧 → 单次申请 6.5 GB，直接
  "Failed to allocate memory for requested buffer of size 6955845376"。
  23 秒的短片没事，4 分钟的歌必炸。

做法：
  按 --chunk 秒分块（默认 20 秒），块间留 --overlap 秒重叠（默认 0.6），
  逐块转换后按重叠区线性交叉淡入拼起来。每块注意力只有 1000² 量级。

变调：
  --key 指定升/降几个半音（默认 0）。音高在送进声码器之前乘 2^(key/12)，
  所以是"用这个音高重新合成"，不是事后变速（不会改变时长）。
"""
import sys, io, os, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--pth', required=True)
p.add_argument('--output', required=True)
p.add_argument('--models', required=True, help='模型目录（找 hubert_fp16.onnx）')
p.add_argument('--rvc', required=True, help='RVC 仓库根目录')
p.add_argument('--chunk', type=float, default=20.0)
p.add_argument('--overlap', type=float, default=0.6)
p.add_argument('--key', type=float, default=0.0, help='升/降半音数')
a = p.parse_args()

sys.path.insert(0, a.rvc)
os.chdir(a.rvc)

import numpy as np, librosa, soundfile as sf
import torch, onnxruntime as ort
from infer.module.models import SynthesizerTrnMs768NSFsid

SR, SR_OUT = 16000, 40000
HUBERT = os.path.join(a.models, 'hubert_fp16.onnx')
if not os.path.exists(HUBERT):
    raise SystemExit(f'找不到 hubert: {HUBERT}')

y, _ = librosa.load(a.input, sr=SR, mono=True)
dur = len(y) / SR
print(f'  输入 {dur:.2f} 秒 · {SR} Hz · 变调 {a.key:+.1f} 半音')

hu = ort.InferenceSession(HUBERT, providers=['CPUExecutionProvider'])
ck = torch.load(a.pth, map_location='cpu')
net = SynthesizerTrnMs768NSFsid(*ck['config'], is_half=False)
miss, _ = net.load_state_dict(ck['weight'], strict=False)
bad = [k for k in miss if not k.startswith('enc_q.')]
if bad:
    raise SystemExit(f'权重不匹配: {bad[:4]}')
net.eval()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(dev).float()
print(f'  net_g 设备 {dev}')

M_MIN = 1127 * np.log(1 + 50.0 / 700)
M_MAX = 1127 * np.log(1 + 1100.0 / 700)


def conv(y16):
    n = len(y16)
    if n < SR * 0.5:
        return np.zeros(0, dtype=np.float32)
    feats = hu.run(None, {'input_values': y16[None, :].astype(np.float32)})[0]
    f2 = np.repeat(feats, 2, axis=1)
    p_len = min(f2.shape[1], n // 160)
    phone = f2[:, :p_len, :].astype(np.float32)
    f0 = librosa.pyin(y16, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                      sr=SR, frame_length=1024, hop_length=160)[0]
    f0 = np.nan_to_num(f0, nan=0.0)
    idx = np.arange(len(f0))
    good = f0 > 0
    f0 = np.interp(idx, idx[good], f0[good]) if good.sum() >= 2 else np.full(len(f0), 200.0)
    f0 = f0[:p_len].astype(np.float32) * (2.0 ** (a.key / 12.0))
    if len(f0) < p_len:
        f0 = np.pad(f0, (0, p_len - len(f0)), mode='edge')
    fm = (1127 * np.log(1 + f0 / 700) - M_MIN) * 254 / (M_MAX - M_MIN) + 1
    pitch = np.clip(np.round(fm), 1, 255).astype(np.int64)[None, :]
    with torch.no_grad():
        o, _, _ = net.infer(torch.from_numpy(phone).to(dev),
                            torch.tensor([p_len], dtype=torch.int64).to(dev),
                            torch.from_numpy(pitch).to(dev),
                            torch.from_numpy(f0)[None, :].to(dev),
                            torch.tensor([0], dtype=torch.int64).to(dev))
    return np.nan_to_num(o[0, 0].cpu().numpy().astype(np.float32))


step = int((a.chunk - a.overlap) * SR)
win = int(a.chunk * SR)
ov = int(a.overlap * SR)
out = np.zeros(int(dur * SR_OUT) + SR_OUT, dtype=np.float32)
pos = 0
nchunk = 0
t0 = time.time()
while pos < len(y):
    seg = y[pos:pos + win]
    a_ = conv(seg)
    if len(a_) == 0:
        break
    o0 = int(pos / SR * SR_OUT)
    if o0 > 0 and len(a_) > ov:
        k = min(ov, len(a_), len(out) - o0)
        if k > 0:
            w = np.linspace(0, 1, k, dtype=np.float32)
            out[o0:o0 + k] = out[o0:o0 + k] * (1 - w) + a_[:k] * w
            rest = a_[k:]
            e = min(len(out), o0 + k + len(rest))
            out[o0 + k:e] = rest[:e - (o0 + k)]
    else:
        e = min(len(out), o0 + len(a_))
        out[o0:e] = a_[:e - o0]
    nchunk += 1
    print(f'  第 {nchunk} 块 · 源 {pos/SR:.1f}s · 已用 {time.time()-t0:.0f}s')
    pos += step
    if pos + ov >= len(y):
        break

out = out[:int(len(y) / SR * SR_OUT)]
peak = float(np.max(np.abs(out))) or 1.0
out = out / peak * 0.9
sf.write(a.output, out, SR_OUT)
print(f'  完成 {len(out)/SR_OUT:.2f} 秒 · {nchunk} 块 · {time.time()-t0:.0f} 秒')
print(f'  → {a.output}')
