# -*- coding: utf-8 -*-
"""音色转换 · 走 .pth 的那条路（ONNX 走不通时的备胎）

为什么需要它：
  RVC 这个分支的 attentions.py 把 length 当 python int 用
      x_flat = x.view([batch, heads, int(length**2) + int(length * (length - 1))])
  → 导出成 ONNX 时长度被写死，换个长度就报 Reshape 失败。
  kizhi.onnx 是别的工具导的（它那份 attention 没这个问题），能跑；
  我们自己训的音色导出来就跑不了 —— 所以留这条路：**直接跑 .pth**。

用法：
  python convert_pth.py --input a.wav --pth voice.pth --output b.wav [--models DIR]

前处理照 dsh-av 的 netg.mjs：
  hubert(50fps) → 上采样 ×2 → 100fps
  p_len = min(特征帧×2, 采样点//160)
  f0 → pitch(1~255)： fm = (1127·ln(1+f0/700) − fmMin)·254/(fmMax−fmMin) + 1
  无音高的帧用邻近浊音帧补（留 0 会出电流声）
"""
import sys, io, os, json, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True)
ap.add_argument('--pth', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--models', default=os.path.join(os.path.expanduser('~'), '.dsh', 'dsh-av', 'models'))
ap.add_argument('--rvc', default='D:/RVC')
a = ap.parse_args()

sys.path.insert(0, a.rvc)
os.chdir(a.rvc)

import numpy as np, librosa, soundfile as sf
import torch, onnxruntime as ort
from infer.module.models import SynthesizerTrnMs768NSFsid

SR_OUT = 40000
HUBERT = os.path.join(a.models, 'hubert_fp16.onnx')

y, sr = librosa.load(a.input, sr=16000, mono=True)
if len(y) < 1600:
    print(json.dumps({'ok': False, 'message': '输入太短，不到 0.1 秒'}, ensure_ascii=False))
    sys.exit(1)

# ① hubert 特征
hu = ort.InferenceSession(HUBERT, providers=['CPUExecutionProvider'])
feats = hu.run(None, {'input_values': y[None, :].astype(np.float32)})[0]
# ② ×2 上采样 → 100fps
f2 = np.repeat(feats, 2, axis=1)
p_len = min(f2.shape[1], len(y) // 160)
phone = f2[:, :p_len, :].astype(np.float32)

# ③ f0（pyin，100fps）
f0 = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                  sr=16000, frame_length=1024, hop_length=160)[0]
f0 = np.nan_to_num(f0, nan=0.0)
idx = np.arange(len(f0))
good = f0 > 0
if good.sum() >= 2:
    f0 = np.interp(idx, idx[good], f0[good])
else:
    f0[:] = 200.0
f0 = f0[:p_len].astype(np.float32)
if len(f0) < p_len:
    f0 = np.pad(f0, (0, p_len - len(f0)), mode='edge')

# ④ f0 → pitch
m_min = 1127 * np.log(1 + 50.0 / 700)
m_max = 1127 * np.log(1 + 1100.0 / 700)
fm = (1127 * np.log(1 + f0 / 700) - m_min) * 254 / (m_max - m_min) + 1
pitch = np.clip(np.round(fm), 1, 255).astype(np.int64)[None, :]

# ⑤ 跑 net_g
ckpt = torch.load(a.pth, map_location='cpu')
net = SynthesizerTrnMs768NSFsid(*ckpt['config'], is_half=False)
miss, _ = net.load_state_dict(ckpt['weight'], strict=False)
bad = [k for k in miss if not k.startswith('enc_q.')]
if bad:
    print(json.dumps({'ok': False, 'message': '权重和模型对不上：' + ','.join(bad[:4])},
                     ensure_ascii=False)); sys.exit(1)
net.eval().float()

with torch.no_grad():
    o, _, _ = net.infer(
        torch.from_numpy(phone),
        torch.tensor([p_len], dtype=torch.int64),
        torch.from_numpy(pitch),
        torch.from_numpy(f0)[None, :],
        torch.tensor([0], dtype=torch.int64),
    )
audio = np.nan_to_num(o[0, 0].numpy().astype(np.float32))
audio44 = librosa.resample(audio, orig_sr=SR_OUT, target_sr=44100)
peak = float(np.max(np.abs(audio44))) or 1.0
audio44 = audio44 / peak * 0.9
sf.write(a.output, audio44, 44100)

print(json.dumps({'ok': True, 'out': a.output, 'seconds': round(len(audio44) / 44100, 2),
                  'peak': round(float(np.max(np.abs(audio44))), 3),
                  'p_len': int(p_len), 'f0_median': round(float(np.median(f0[f0 > 0])), 1)
                  if (f0 > 0).any() else 0}, ensure_ascii=False))
