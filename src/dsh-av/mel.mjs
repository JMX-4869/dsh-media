// mel + rmvpe 的纯 JS 实现 —— 不带 Python、不带 torch。
//
// ══════════════════════════════════════════════════════════════════════
//  配方是从 `infer/rmvpe.py` 的 `MelSpectrogram.forward` 读出来的（不是猜的）：
//    n_fft=1024 · win_length=1024 · hop=160 · window=hann · center=True
//    fft = torch.stft(audio, ..., center=True, return_complex=True)   ← 两端各 pad 512
//    magnitude = sqrt(re² + im²)
//    mel = mel_basis @ magnitude                                       ← [128,513] @ [513,T]
//    log_mel = log(clamp(mel, min=1e-5))                               ← **自然对数**
//
//  而 `mel_basis` 我没在 JS 里重算 —— 从 Python 那边导出来了（`mel_basis.f32`，256.5 KB）。
//  理由：librosa 那个 mel 滤波器有一堆边界约定，重算一遍很容易差一点点，
//        而那点差会被后面的 log 放大。**导出来是 256 KB，省下的是几小时的调试。**

import { readFileSync } from 'node:fs';

// ── mel 滤波器 ─────────────────────────────────────────────────────
export function loadMelBasis(binPath, metaPath) {
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  const buf = readFileSync(binPath);
  const f32 = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const rows = meta.mel_rows, cols = meta.mel_cols;
  if (f32.length !== rows * cols) {
    throw new Error(`mel_basis 大小不符：${f32.length} vs ${rows}×${cols}`);
  }
  return { basis: f32, rows, cols, meta };
}

// ── FFT（radix-2，就地的）──────────────────────────────────────────
// ⚠️ 只支持 2 的幂 —— 1024 是，所以够。**别拿它当通用 FFT 用。**
export function fftInPlace(re, im) {
  const n = re.length;
  if ((n & (n - 1)) !== 0) throw new Error('fft: 长度必须是 2 的幂');
  // 位反转置换
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

// ── hann 窗 ────────────────────────────────────────────────────────
// ⚠️ torch.hann_window(1024) 是**周期性的**（periodic），不是对称的：
//    它除的是 N，不是 N-1。而 numpy.hanning(1024) 除的是 N-1。
//    **差一个样本的相位 —— 而那点差会一路传到 mel 上。**
export function hannPeriodic(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
  return w;
}

// ── reflect pad（torch.stft 的 center=True 默认用 reflect）────────
function reflectPad(x, pad) {
  const n = x.length;
  const out = new Float64Array(n + 2 * pad);
  for (let i = 0; i < pad; i++) out[i] = x[pad - i];                 // 左侧镜像（不含端点）
  out.set(x, pad);
  for (let i = 0; i < pad; i++) out[pad + n + i] = x[n - 2 - i];     // 右侧镜像
  return out;
}

// ── mel 谱 ─────────────────────────────────────────────────────────
export function logMel(audio, mel) {
  const { basis, rows, cols, meta } = mel;
  const N = meta.n_fft, HOP = meta.hop, CLAMP = meta.clamp;
  const win = hannPeriodic(N);                    // ⚠️ 周期性 hann
  const pad = meta.center ? N / 2 : 0;
  const p = reflectPad(audio, pad);
  const nFrames = 1 + Math.floor((p.length - N) / HOP);
  const out = new Float64Array(rows * nFrames);

  const re = new Float64Array(N), im = new Float64Array(N);
  const mag = new Float64Array(cols);
  for (let f = 0; f < nFrames; f++) {
    const off = f * HOP;
    for (let i = 0; i < N; i++) { re[i] = p[off + i] * win[i]; im[i] = 0; }
    fftInPlace(re, im);
    for (let k = 0; k < cols; k++) mag[k] = Math.hypot(re[k], im[k]);
    for (let m = 0; m < rows; m++) {
      let s = 0;
      const base = m * cols;
      for (let k = 0; k < cols; k++) s += basis[base + k] * mag[k];
      out[m * nFrames + f] = Math.log(Math.max(s, CLAMP));
    }
  }
  return { spec: out, rows, nFrames };
}

// ── rmvpe 的 cents → Hz ────────────────────────────────────────────
// 这一段是照 `infer/rmvpe.py` 的 `to_local_average_cents` 和 `decode` 抄的：
//   · **不是直接取 argmax** —— 而是取 argmax 周围 9 个分箱做**加权平均**
//   · 最大概率 <= thred 的帧 → 判为无音高（0）
//   · f0 = 10 * 2**(cents/1200)
const CENTS_BASE = 1997.3794084376191;

export function decodeF0(hidden, nFrames, nBins, thred = 0.03) {
  // hidden: [1, T, 360] 的 Float32Array
  const cents = new Float64Array(nFrames);
  const padded = new Float64Array(nBins + 8);   // 前后各补 4（跟 Python 那边一样）
  for (let f = 0; f < nFrames; f++) {
    const row = f * nBins;
    let best = -1, bestI = 0;
    for (let i = 0; i < nBins; i++) {
      const v = hidden[row + i];
      if (v > best) { best = v; bestI = i; }
    }
    if (best <= thred) { cents[f] = 0; continue; }
    // 取 9 个邻域，跟 cents_mapping 做加权平均
    let ps = 0, ws = 0;
    for (let d = -4; d <= 4; d++) {
      let idx = bestI + d;
      // 越界就按 Python 的 pad 语义取 0（salience 那边补的是 0）
      const s = (idx < 0 || idx >= nBins) ? 0 : hidden[row + idx];
      const c = 20 * idx + CENTS_BASE;
      ps += s * c; ws += s;
    }
    cents[f] = ws > 0 ? ps / ws : 0;
  }
  const f0 = new Float64Array(nFrames);
  for (let f = 0; f < nFrames; f++) f0[f] = cents[f] > 0 ? 10 * Math.pow(2, cents[f] / 1200) : 0;
  return f0;
}
