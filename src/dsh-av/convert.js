/**
 * convert.js — 整条链：一个音频进去，换了音色的音频出来。
 *
 * ══════════════════════════════════════════════════════════════════════
 * 顺序（每一步的算法都在各自的文件里，这里只管串）：
 *
 *   音频文件
 *     → 读 wav（wav.js，按 chunk 走）
 *     → 立体声取平均 + 重采样到 16k（hubert.mjs 的 resample，带抗混叠）
 *     → hubert  →  768 维内容特征（50 帧/秒）
 *     → mel → rmvpe  →  每帧音高（100 帧/秒）
 *     → 特征上采样 2 倍 → 对齐到 100 帧/秒；p_len = min(特征帧数, 音频采样点//160)
 *     → net_g（音色 + 内容 + 音高 + 噪声）  →  波形
 *     → 写 wav
 *
 * ⚠️ **噪声是要显式给的** —— 那正是这个模型"每次不一样"的来源。
 *    不给就每次随机造一组（那样两遍跑的结果不同，没法比对）。
 *    给同一组，两遍就一模一样。
 *
 * ⚠️ 而**帧率对齐不是靠插值 f0** —— 是把 hubert 特征上采样 2 倍（50→100）。
 *    这条是从 `infer/vc/pipeline.py` 读出来的，而它反直觉：直觉上会去插值 f0。
 */
import { readWav, writeWav, toMono } from './wav.js'
import { loadMelBasis, logMel, decodeF0 } from './mel.mjs'
import { hubertFeatures, resample } from './hubert.mjs'
import { netgInfer } from './netg.mjs'
import { existsSync } from 'node:fs'

/** 造一组高斯噪声（Box-Muller）—— 不给噪声时用 */
function gauss(n, seed = 12345) {
  // 简单的 xorshift，让"同一次不给噪声"也能复现
  let s = seed >>> 0
  const rnd = () => {
    s ^= s << 13; s >>>= 0
    s ^= s >> 17
    s ^= s << 5; s >>>= 0
    return s / 4294967296
  }
  const out = new Float32Array(n)
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(rnd(), 1e-12), u2 = rnd()
    const r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2
    out[i] = r * Math.cos(th)
    if (i + 1 < n) out[i + 1] = r * Math.sin(th)
  }
  return out
}

/**
 * 跑一整条链。
 *
 * @param {object} o
 * @param {string} o.input         输入音频路径（wav）
 * @param {string} o.output        输出音频路径
 * @param {string} o.voiceOnnx     音色的 net_g onnx
 * @param {string} o.modelsDir     三件模型所在的目录（找 hubert / rmvpe）
 * @param {object} o.ort           onnxruntime
 * @param {object} o.sessions      { hubert, rmvpe, netg } —— 复用会话，别每次重建
 * @param {object} o.melBasis      由 loadMelBasis 载入的表
 * @param {Float32Array} [o.rnd]   主噪声（不给就造）
 * @param {Float32Array} [o.rndNsf] NSF 噪声（不给就造）
 * @param {number} [o.sid]         说话人 id
 * @param {string} [o.out16k]      额外写一份 16k 单声道 wav，供 dsh-mood/ASR 读取
 * @param {function} [o.log]       进度回调
 */
export async function convert(o) {
  const log = o.log || (() => {})
  const t0 = Date.now()
  const t = {}

  // ① 读 + 单声道 + 重采样
  let a = t.read = Date.now()
  const w = readWav(o.input)
  let mono = toMono(w.channels)
  if (w.sr !== 16000) mono = resample(mono, w.sr, 16000)
  log('读入 ' + (w.sr / 1000).toFixed(1) + 'k → 16k · ' + mono.length + ' 点'
    + '（' + (mono.length / 16000).toFixed(2) + ' 秒）')
  t.read = Date.now() - t.read

  // ② hubert
  t.hubert = Date.now()
  const { data: feats, frames, dim } = await hubertFeatures(mono, o.sessions.hubert, o.ort)
  if (dim !== 768) throw new Error('hubert 出来的维度是 ' + dim + '，而 net_g 要 768')
  log('hubert ' + frames + '×' + dim)
  t.hubert = Date.now() - t.hubert

  // ③ mel → rmvpe → f0
  t.rmvpe = Date.now()
  const { spec, rows, nFrames } = logMel(mono, o.melBasis)
  const nPad = 32 * (Math.floor((nFrames - 1) / 32) + 1) - nFrames
  const Tp = nFrames + nPad
  const padded = new Float32Array(rows * Tp)
  for (let m = 0; m < rows; m++) {
    for (let f = 0; f < nFrames; f++) padded[m * Tp + f] = spec[m * nFrames + f]
  }
  const outR = await o.sessions.rmvpe.run({
    mel: new o.ort.Tensor('float32', padded, [1, rows, Tp]),
  })
  const hidden = outR[o.sessions.rmvpe.outputNames[0]].data
  const f0 = decodeF0(hidden, nFrames, 360, 0.03)
  let voiced = 0
  for (let i = 0; i < f0.length; i++) if (f0[i] > 0) voiced++
  log('rmvpe ' + nFrames + ' 帧 · 有音高 ' + voiced + ' 帧'
    + '（' + (100 * voiced / nFrames).toFixed(1) + '%）')
  t.rmvpe = Date.now() - t.rmvpe

  // ④ 噪声（p_len 得先算出来才知道多大）
  let pLen = Math.floor(mono.length / 160)
  if (frames * 2 < pLen) pLen = frames * 2
  const rnd = o.rnd || gauss(192 * pLen)
  const rndNsf = o.rndNsf || gauss(pLen * 400, 987654321)

  // ⑤ net_g
  t.netg = Date.now()
  const { audio } = await netgInfer({
    feats, frames, f0, audioLen: mono.length,
    rnd, rndNsf, sess: o.sessions.netg, ort: o.ort, sid: o.sid || 0,
  })
  // ⚠️ **输出的采样率不是 16000** —— net_g 出来的是**音色训练时的采样率**
  //    （这个音色是 40k 训的，所以输出 40000 Hz）。
  //    我一开始在这儿写了 16000，于是 239200 点被当成"14.95 秒" —— 而它其实是 5.98 秒。
  //    ⚠️ 而那个错**在数值上完全看不出来**（样本数一样、波形一样），只在"时长"上露馅：
  //       是跟 RVC 自己的 cli 一比才发现的（它 5.98 秒，我 14.95 秒）。
  //    → 所以"跟原版比"这件事的价值不只在精度，还在**这种一眼看不出的单位错**。
  const sr = o.outSampleRate || 40000
  log('net_g 出来 ' + audio.length + ' 点（' + (audio.length / sr).toFixed(2) + ' 秒 @ ' + sr + ' Hz）')
  t.netg = Date.now() - t.netg

  // ⑥ 写
  const info = writeWav(o.output, audio, sr)
  log('→ ' + o.output + '（峰值 ' + info.peak.toFixed(4) + ' · 增益 ' + info.gain.toFixed(3) + '）')

  // dsh-mood 的 ASR 合同是 16 kHz；主输出仍保持音色训练使用的 40 kHz。
  let compat = null
  if (o.out16k) {
    const audio16k = sr === 16000 ? audio : resample(audio, sr, 16000)
    const compatInfo = writeWav(o.out16k, audio16k, 16000)
    compat = { output: o.out16k, sampleRate: 16000, seconds: audio16k.length / 16000,
      peak: compatInfo.peak, gain: compatInfo.gain }
    log('→ 16k ASR 兼容音频 ' + o.out16k)
  }

  return {
    output: o.output,
    sampleRate: sr,
    seconds: audio.length / sr,
    compatOutput: compat,
    pLen, frames, nFrames, voiced,
    ms: { ...t, total: Date.now() - t0 },
  }
}

/** 建三个会话（一次建好，重复用 —— 建会话是有开销的） */
export async function openSessions(modelsDir, voiceOnnx, ort) {
  const j = (f) => modelsDir.replace(/[\\/]$/, '') + '/' + f
  // ⚠️ **rmvpe 这里用 fp32，不用 fp16 —— 而这不是偷懒，是必须的。**
  //    `rmvpe_fp16.onnx` 在 onnxruntime-node 下**加载就 segfault**（不是抛异常，是原生崩溃）。
  //    诊断过：fp32 / int8动态 / int8静态 / hubert_fp16 四个都能加载，
  //    **只有 rmvpe_fp16 那一个崩** —— 所以是那个文件特有的，不是"fp16 都不行"。
  //    而 segfault **没法 try/catch** → 只能在这儿避开它。
  //    代价：多 172 MB（172.5 → 344.9），而质量一样。
  //    ⚠️ 而"重新转一个 fp16"我没试 —— 换个 opset 或者关掉 shape_infer 也许就不崩了。
  // ⚠️ **只认 fp32 那份 —— 不做"没有就退回 fp16"这种回退。**
  //    理由：rmvpe_fp16.onnx 在 onnxruntime-node 下**加载就 segfault**（原生崩溃，try/catch 拦不住），
  //    所以"悄悄退回"的后果是**整个宿主进程挂掉** —— 而那比"报一句缺文件"严重得多。
  //    ⚠️ 这一条是外部优化方案（GPT 那份）里唯一一条真 bug，它说对了：
  //       "缺少 fp32 模型时应清楚报错，而不是自动回退到 fp16"。
  //       而原来那行 `existsSync(rmvpe.onnx) ? 'rmvpe.onnx' : 'rmvpe_fp16.onnx'` 正是在干这事。
  const rmvpeFile = 'rmvpe.onnx'
  if (!existsSync(j(rmvpeFile))) {
    throw new Error('缺 ' + j(rmvpeFile) + '（音高网络**只能用 fp32 那份**）'
      + (existsSync(j('rmvpe_fp16.onnx'))
        ? '\n⚠️ 目录里那个 rmvpe_fp16.onnx **不能用** —— 它在 onnxruntime-node 下加载就原生崩溃'
          + '（segfault，没法 try/catch），退回它会把整个宿主搞挂。请转一份 fp32 的。'
        : ''))
  }
  if (!existsSync(j('hubert_fp16.onnx'))) throw new Error('缺 ' + j('hubert_fp16.onnx'))
  if (!existsSync(voiceOnnx)) throw new Error('缺音色：' + voiceOnnx)
  const netg = await ort.InferenceSession.create(voiceOnnx)
  // ⚠️ **一句诊断，值一小时**：六输入那张图（onnxlib 导的，输入里有 `rnd`）
  //    里面写死了一个 Reshape 形状 {1,2,599,1195} —— 那是**导出时那段 5.98 秒音频**的形状。
  //    于是它对别的长度就崩，而崩出来的话是 onnxruntime 的
  //    "The input tensor cannot be reshaped to the requested shape" ——
  //    **读起来像模型坏了，其实是那份导出只对一段长度有效。**
  //    → 所以在这里就说清楚，别让它到 net_g 那一步才炸。
  const sixInput = netg.inputNames.includes('rnd')
  return {
    hubert: await ort.InferenceSession.create(j('hubert_fp16.onnx')),
    rmvpe: await ort.InferenceSession.create(j(rmvpeFile)),
    netg,
    rmvpeFile,
    netgSixInput: sixInput,
    netgInputs: netg.inputNames,
  }
}

export { loadMelBasis }
