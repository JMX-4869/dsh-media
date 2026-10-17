// net_g 那一段：内容特征 + 音高 + 噪声 → 波形。
//
// ══════════════════════════════════════════════════════════════════════
//  这一步的算法全部是从 `infer/vc/pipeline.py` 读出来的（不是猜的），
//  而里面有三处**光看名字绝对猜不对**的地方：
//
//  ① **帧率不是靠插值 f0 对齐的，而是把 hubert 特征上采样 2 倍**：
//       feats = F.interpolate(feats.permute(0,2,1), scale_factor=2).permute(0,2,1)
//     50fps → 100fps，正好对上 f0 的 100fps。
//     ⚠️ `F.interpolate` **没写 mode**，所以是默认的 **nearest**（每个值复制两份）。
//
//  ② **p_len 是 min(特征帧数, 音频采样点 // 160)**，然后 pitch/pitchf 都裁到它。
//     6 秒音频：f0 601 帧 → 特征 299×2 = 598 → **p_len = 598**。
//
//  ③ **f0 → pitch 那一步有个 `+1`**：
//       fm = (1127·log(1+f0/700) − fm_min)·254/(fm_max − fm_min) + 1
//     ⚠️ 而 `infer/rtrvc.py` 那个实时版本**没有 +1** —— 两份代码不一样，离线这条有。
//
//  而还有一处是"不补会难听"的：
//     **无音高的帧（f0 == 0）要用邻近的浊音帧线性插值补上**，不能留 0。
//     留 0 会让 NSF 那一段没有音高 → 输出会怪。
//
// ⚠️ onnxruntime-node 那边：**int64 输入要用 BigInt64Array**，不是 Int32Array 也不是普通数组。
//    传错类型它会报错（或者更糟：静默按别的类型解释）。

/** f0 → pitch（1~255 的整数）和 pitchf（连续 f0，Hz） */
export function f0ToPitch(f0, pLen, f0Min = 50, f0Max = 1100) {
  const f = new Float64Array(pLen)
  for (let i = 0; i < pLen; i++) f[i] = f0[i]

  // ① 无音高的地方用线性插值补（照 pipeline.py 那段）
  const voiced = []
  for (let i = 0; i < pLen; i++) if (f[i] > 0) voiced.push(i)
  if (voiced.length && voiced.length < pLen) {
    for (let i = 0; i < pLen; i++) {
      if (f[i] > 0) continue
      // 找左右最近的浊音帧
      let lo = -1, hi = -1
      for (const v of voiced) { if (v < i) lo = v; else { hi = v; break } }
      if (lo < 0) f[i] = f[hi]
      else if (hi < 0) f[i] = f[lo]
      else f[i] = f[lo] + (f[hi] - f[lo]) * (i - lo) / (hi - lo)
    }
  }

  const pitchf = Float32Array.from(f)

  // ② 1127·log(1+f0/700) → 归一化到 1~255
  const mMin = 1127 * Math.log(1 + f0Min / 700)
  const mMax = 1127 * Math.log(1 + f0Max / 700)
  const pitch = new BigInt64Array(pLen)
  for (let i = 0; i < pLen; i++) {
    let fm = 1127 * Math.log(1 + f[i] / 700)
    if (fm > 0) fm = (fm - mMin) * 254 / (mMax - mMin) + 1     // ⚠️ 那个 +1
    if (fm <= 1) fm = 1
    if (fm > 255) fm = 255
    pitch[i] = BigInt(Math.round(fm))
  }
  return { pitch, pitchf }
}

/** 特征上采样 2 倍（nearest —— 每个值复制两份），照 `F.interpolate(scale_factor=2)` */
export function upsampleNearest2(feats, frames, dim) {
  const out = new Float32Array(frames * 2 * dim)
  for (let f = 0; f < frames; f++) {
    const src = f * dim
    const d0 = (2 * f) * dim
    out.set(feats.subarray(src, src + dim), d0)
    out.set(feats.subarray(src, src + dim), d0 + dim)
  }
  return out
}

/**
 * 跑 net_g。
 * @param feats   hubert 特征（Float32Array，[1, T_in, 768]）
 * @param frames  特征帧数
 * @param f0      f0（Float32Array，长度 ≥ 目标帧数）
 * @param audioLen 源音频的采样点数（16k）—— 用来算 p_len
 * @param rnd     噪声（Float32Array，[1, 192, pLen]）
 * @param sess    net_g 的 InferenceSession
 * @param ort     onnxruntime
 */
export async function netgInfer({ feats, frames, f0, audioLen, rnd, rndNsf = null, sess, ort, sid = 0 }) {
  const up2 = upsampleNearest2(feats, frames, 768)
  const upFrames = frames * 2

  let pLen = Math.floor(audioLen / 160)          // self.window = 160
  if (upFrames < pLen) pLen = upFrames
  const featsCut = up2.subarray(0, pLen * 768)

  const { pitch, pitchf } = f0ToPitch(f0, pLen)
  const phoneLengths = new BigInt64Array([BigInt(pLen)])
  const ds = new BigInt64Array([BigInt(sid)])

  // ⚠️ **按 inputNames 适配两种接口** —— 而不是写死一种：
  //
  //   六输入（onnxlib 导的）：phone / phone_lengths / pitch / **pitchf** / **ds** / rnd
  //   七输入（rvc-onnx-web 转的 + 我手工把噪声提成输入）：
  //        phone / phone_lengths / pitch / **nsff0** / **sid** / noise_1527 / noise_random_2110
  //
  //   而**能跑的是七输入那个** —— 六输入那张图里有个写死的 Reshape 形状，
  //   Python 和 Node 两个 runtime 都崩（requested {1,2,599,1195}）。
  //
  // ⚠️ 而七输入那个有两处噪声，**NSF 那处的形状是 [1, 音频长度, 1]**（不是 [1,1,L]）——
  //    音频长度 = p_len × 400。
  const names = sess.inputNames
  const has = (n) => names.includes(n)
  const pLenSamples = pLen * 400

  // ⚠️ **那个除法看着多余，而删掉它结果就错。**
  //    RVC 原版写的是 `z_p = (m_p + exp(logs_p) * randn_like(m_p) * **0.66666**) * x_mask`，
  //    而噪声被提成图输入之后，**那个 ×0.66666 仍然留图里** ——
  //    所以这里喂的会被图再乘一次。参考实现那边（onnxlib 的 forward）**不乘**。
  //    实测（同一段音频、同一组噪声）：
  //      喂同一个 rnd      相关 **0.97513601**
  //      喂 rnd / 0.66666  相关 **0.99973032**   ← 对上了
  //      喂 rnd * 0.66666  相关 0.94988034     ← 更差，方向也印证了
  const rndFeed = Float32Array.from(rnd.subarray(0, 192 * pLen), (v) => v / 0.66666)

  let feeds
  if (has('rnd')) {
    feeds = {
      phone: new ort.Tensor('float32', featsCut, [1, pLen, 768]),
      phone_lengths: new ort.Tensor('int64', phoneLengths, [1]),
      pitch: new ort.Tensor('int64', pitch, [1, pLen]),
      [has('pitchf') ? 'pitchf' : 'nsff0']: new ort.Tensor('float32', pitchf, [1, pLen]),
      [has('ds') ? 'ds' : 'sid']: new ort.Tensor('int64', ds, [1]),
      rnd: new ort.Tensor('float32', rndFeed, [1, 192, pLen]),
    }
  } else {
    // 七输入：主噪声和 NSF 噪声分开给
    let nsf = rndNsf
    if (!nsf) {
      // 没给就造一个（**而那样每次结果不同** —— 要比对时必须显式传）
      nsf = new Float32Array(pLenSamples)
      for (let i = 0; i < pLenSamples; i++) {
        // Box-Muller，避免依赖别的库
        const u1 = Math.max(Math.random(), 1e-12), u2 = Math.random()
        nsf[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
      }
    }
    feeds = {
      phone: new ort.Tensor('float32', featsCut, [1, pLen, 768]),
      phone_lengths: new ort.Tensor('int64', phoneLengths, [1]),
      pitch: new ort.Tensor('int64', pitch, [1, pLen]),
      [has('pitchf') ? 'pitchf' : 'nsff0']: new ort.Tensor('float32', pitchf, [1, pLen]),
      [has('ds') ? 'ds' : 'sid']: new ort.Tensor('int64', ds, [1]),
      noise_1527: new ort.Tensor('float32', rndFeed, [1, 192, pLen]),
      noise_random_2110: new ort.Tensor('float32', nsf, [1, pLenSamples, 1]),
    }
  }
  const out = await sess.run(feeds)
  const a = out[sess.outputNames[0]]
  return { audio: a.data, samples: a.data.length, pLen }
}
