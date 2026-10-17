// hubert 那一段：波形 → 768 维内容特征。
//
// ══════════════════════════════════════════════════════════════════════
//  接口是从 `infer/hubert.py` 读出来的：
//    · v2 用**第 12 层**（`last_hidden_state`），不是 v1 那条 `final_proj(hidden_states[9])`
//      —— 后者出来是 256 维，而 v2 要 768。
//    · `do_normalize` 是 **false**（在 `assets/hubert_base/preprocessor_config.json` 里），
//      所以**不用归一化音频**。
//    · 下采样是 320：16000 Hz → 50 帧/秒；而 6 秒音频给的是 **299 帧**（不是 300 ——
//      卷积的边界会吃掉一帧，别按"T/320"算）。
//
//  ⚠️ 重采样这一层要小心：
//      RVC 默认用 **ffmpeg**（`AUDIO_LOAD_BACKEND = "ffmpeg"`）解码加重采样，
//      而 ffmpeg 的那套是**带抗混叠的**。
//      → 所以这里**不能只做线性插值** —— 那会把 8k 以上的东西折回有用频段，
//        而混叠出来的东西会被 hubert 当成"内容"，一路传到音色上。
//      → 做法：降采样前先低通到目标 Nyquist，再用窗函数 sinc 插值。

/** 窗函数 sinc 插值（Kaiser 窗）。半宽 16 个输入样本 —— 够用，而且不至于太慢。 */
const KAISER_HALF = 16

function besselI0(x) {
  let s = 1, t = 1
  for (let k = 1; k < 25; k++) {
    t *= (x / (2 * k)) * (x / (2 * k))
    s += t
    if (t < 1e-16 * s) break
  }
  return s
}

function kaiserWindow(n, half) {
  // 生成 [0, 2*half] 上的窗（2*half+1 个点）
  const beta = 8.0
  const w = new Float64Array(n)
  const d = besselI0(beta)
  const M = n - 1
  for (let i = 0; i < n; i++) {
    const r = (2 * i) / M - 1
    w[i] = besselI0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / d
  }
  return w
}

function sinc(x) {
  if (x === 0) return 1
  const px = Math.PI * x
  return Math.sin(px) / px
}

/**
 * 带抗混叠的重采样。
 * 降采样时截止频率取**目标的** Nyquist（不是源的）—— 那正是抗混叠的关键。
 */
export function resample(x, fromRate, toRate, half = KAISER_HALF) {
  if (fromRate === toRate) return Float32Array.from(x)
  const ratio = toRate / fromRate
  const cut = ratio < 1 ? ratio : 1          // 截止（相对 Nyquist 的比例）
  const win = kaiserWindow(2 * half + 1, half)
  const nOut = Math.floor(x.length * ratio)
  const out = new Float32Array(nOut)
  for (let i = 0; i < nOut; i++) {
    const center = i / ratio
    const i0 = Math.ceil(center - half)
    let acc = 0, wsum = 0
    for (let k = i0; k <= Math.floor(center + half); k++) {
      if (k < 0 || k >= x.length) continue
      const d = center - k
      const w = win[k - i0] * sinc(cut * d) * cut
      acc += x[k] * w
      wsum += w
    }
    out[i] = wsum !== 0 ? acc / wsum : 0
  }
  return out
}

/** 立体声 → 单声道（取平均，跟 RVC 的 force_mono 一致） */
export function toMono(channels) {
  if (channels.length === 1) return channels[0]
  const n = channels[0].length
  const out = new Float32Array(n)
  for (let c = 0; c < channels.length; c++) {
    for (let i = 0; i < n; i++) out[i] += channels[c][i] / channels.length
  }
  return out
}

/**
 * 跑 hubert。
 * @param audio16k 16k 单声道波形（Float32Array）
 * @param sess     onnxruntime 的 InferenceSession（hubert_fp16.onnx 或 fp32）
 * @returns Float32Array，[1, T_out, 768] 展平
 */
export async function hubertFeatures(audio16k, sess, ort) {
  const dims = [1, audio16k.length]
  const feeds = {}
  feeds[sess.inputNames[0]] = new ort.Tensor('float32', audio16k, dims)
  const out = await sess.run(feeds)
  const t = out[sess.outputNames[0]]
  return { data: t.data, frames: t.dims[1], dim: t.dims[2], dims: t.dims }
}
