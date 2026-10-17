/**
 * render.js — 把音频渲染成片子。纯 JS 出 PNG，ffmpeg 拼 mp4。
 *
 * ══════════════════════════════════════════════════════════════════════
 * 画什么（三层，每层一个时间尺度）：
 *   ① **整段轮廓**（铺满宽度，淡）—— 整段的样子，已经过去的那部分更亮
 *   ② **"你在这"**（一道亮竖线）—— 把整段和下面那块连起来
 *   ③ **窗口波形 + 频谱**（只占中间 45%，亮）—— **这 0.35 秒**
 *
 * ⚠️ **三层的时间尺度必须分得开，这一版才改的。**
 *    上一版我把窗口拉伸到**整幅宽度**，于是它跟"整段轮廓"形状几乎一样 ——
 *    两条几乎重合的曲线叠在一起，读不出"局部 vs 整体"。
 *    而频谱（"这一刻"）也铺满整幅，跟波形（"这 0.35 秒"）的时间尺度对不上，
 *    读起来像"这块频谱属于整段"。
 *    → 现在：整段铺满、局部只占中间、频谱跟波形共用同一个 x 范围。
 *
 * ⚠️ **不画字** —— 那要字体渲染（FreeType 那套）。
 *    要字幕的话两条路：调 ffmpeg 的 `drawtext`，或者生成 SVG 让浏览器/ffmpeg 转。
 *    **这一版故意不碰它** —— 先把"能出片子"这条跑通。
 */
import { writeFileSync, mkdirSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { Canvas } from './png.js'
import { readWav, toMono } from './wav.js'
import { resample } from './hubert.mjs'
import { fftInPlace } from './mel.mjs'

const DEFAULT_RENDER_STYLE = {
  colors: {
    bg: [10, 11, 14], dim: [46, 54, 66], wave: [150, 214, 200],
    waveDim: [72, 96, 102], waveFuture: [40, 46, 55], spectrum: [232, 181, 106],
    accent: [255, 186, 110], boxFill: [16, 18, 23], boxEdge: [32, 38, 46],
    thumb: [26, 31, 38], progress: [34, 40, 50],
  },
  geometry: {
    pad: 40, midY: 0.52, wholeAmp: 0.26, windowAmp: 0.22,
    spectrumTop: 0.10, spectrumHeight: 0.18, localWidth: 0.45,
    windowSeconds: 0.35, spectrumBars: 64, spectrumDb: 70,
  },
}

/**
 * 可调的画面风格。它只提供参数和预设，品味/情绪层由调用方决定，避免和 dsh-mood 耦合。
 * 旧调用不传参数时使用 default，保持原来的画面。
 */
export const RENDER_PRESETS = Object.freeze({
  default: DEFAULT_RENDER_STYLE,
  warm: {
    colors: { bg: [18, 13, 14], dim: [74, 48, 52], wave: [246, 185, 145],
      waveDim: [132, 82, 78], waveFuture: [57, 37, 43], spectrum: [255, 177, 102],
      accent: [255, 218, 135], boxFill: [28, 19, 21], boxEdge: [70, 45, 47],
      thumb: [48, 29, 31], progress: [58, 36, 38] },
  },
  cool: {
    colors: { bg: [8, 14, 22], dim: [42, 65, 86], wave: [139, 211, 244],
      waveDim: [66, 107, 133], waveFuture: [32, 52, 72], spectrum: [159, 178, 255],
      accent: [113, 224, 255], boxFill: [12, 23, 35], boxEdge: [35, 69, 91],
      thumb: [23, 47, 65], progress: [29, 57, 76] },
  },
  violet: {
    colors: { bg: [15, 12, 23], dim: [63, 52, 82], wave: [220, 190, 255],
      waveDim: [121, 91, 151], waveFuture: [47, 37, 66], spectrum: [246, 170, 226],
      accent: [196, 151, 255], boxFill: [24, 18, 36], boxEdge: [66, 52, 89],
      thumb: [43, 32, 61], progress: [54, 41, 73] },
  },
})

function color(value, fallback) {
  if (!Array.isArray(value) || value.length < 3) return [...fallback]
  return value.slice(0, 3).map((v, i) => Number.isFinite(Number(v))
    ? Math.max(0, Math.min(255, Math.round(Number(v)))) : fallback[i])
}

function ratio(value, fallback, lo = 0.01, hi = 0.95) {
  const n = Number(value)
  return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : fallback
}

/** 合并一个预设或 JSON 风格对象，并把尺寸参数限制在可渲染范围内。 */
export function resolveRenderStyle(input) {
  let raw = input
  if (typeof raw === 'string') raw = { preset: raw }
  if (!raw || typeof raw !== 'object') raw = {}
  const preset = typeof raw.preset === 'string' ? RENDER_PRESETS[raw.preset] : null
  const base = preset ? {
    ...DEFAULT_RENDER_STYLE, ...preset,
    colors: { ...DEFAULT_RENDER_STYLE.colors, ...(preset.colors || {}) },
    geometry: { ...DEFAULT_RENDER_STYLE.geometry, ...(preset.geometry || {}) },
  } : DEFAULT_RENDER_STYLE
  const colors = { ...base.colors, ...(raw.colors || {}) }
  const geometry = { ...base.geometry, ...(raw.geometry || {}) }
  return {
    // ⚠️ 什么都没传时得是 'default'，不是 'custom' ——
    //    否则工具那边会打出「preset custom」，而画面其实就是默认那份。
    //    （这两行唯一的区别是"报出来的名字"，但报错了名字会让人以为画面被改过。）
    preset: preset ? raw.preset : (Object.keys(raw).length ? 'custom' : 'default'),
    colors: Object.fromEntries(Object.entries(DEFAULT_RENDER_STYLE.colors)
      .map(([k, v]) => [k, color(colors[k], v)])),
    geometry: {
      pad: Math.max(4, Math.min(200, Math.round(Number(geometry.pad) || base.geometry.pad))),
      midY: ratio(geometry.midY, base.geometry.midY, 0.2, 0.8),
      wholeAmp: ratio(geometry.wholeAmp, base.geometry.wholeAmp, 0.03, 0.45),
      windowAmp: ratio(geometry.windowAmp, base.geometry.windowAmp, 0.03, 0.45),
      spectrumTop: ratio(geometry.spectrumTop, base.geometry.spectrumTop, 0.01, 0.5),
      spectrumHeight: ratio(geometry.spectrumHeight, base.geometry.spectrumHeight, 0.03, 0.45),
      localWidth: ratio(geometry.localWidth, base.geometry.localWidth, 0.15, 0.9),
      windowSeconds: Math.max(0.05, Math.min(2, Number(geometry.windowSeconds) || base.geometry.windowSeconds)),
      spectrumBars: Math.max(8, Math.min(128, Math.round(Number(geometry.spectrumBars) || base.geometry.spectrumBars))),
      spectrumDb: Math.max(20, Math.min(100, Number(geometry.spectrumDb) || base.geometry.spectrumDb)),
    },
  }
}

/** 整段波形轮廓（每列取 min/max）—— 一次算好，每帧只查 */
function outline(mono, cols) {
  const lo = new Float32Array(cols), hi = new Float32Array(cols)
  const per = mono.length / cols
  for (let c = 0; c < cols; c++) {
    const a = Math.floor(c * per), b = Math.min(mono.length, Math.floor((c + 1) * per))
    let mn = 1e9, mx = -1e9
    for (let i = a; i < b; i++) { const v = mono[i]; if (v < mn) mn = v; if (v > mx) mx = v }
    if (a >= b) { mn = 0; mx = 0 }
    lo[c] = mn; hi[c] = mx
  }
  return { lo, hi }
}

/** 某一刻的频谱（512 点 FFT，周期性 hann） */
function spectrumAt(mono, sr, t, nFft = 512) {
  const start = Math.max(0, Math.min(mono.length - nFft, Math.round(t * sr) - nFft / 2))
  const re = new Float64Array(nFft), im = new Float64Array(nFft)
  for (let i = 0; i < nFft; i++) {
    const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nFft)
    re[i] = mono[start + i] * w
    im[i] = 0
  }
  fftInPlace(re, im)
  const bins = nFft / 2
  const mag = new Float32Array(bins)
  for (let k = 0; k < bins; k++) mag[k] = Math.hypot(re[k], im[k])
  return mag
}

/**
 * 渲染整段。
 * @param {object} o
 * @param {string} o.input      音频（wav）
 * @param {string} o.outDir     帧目录
 * @param {number} [o.fps]      帧率（默认 24）
 * @param {number} [o.w] [o.h]  尺寸（默认 1280×720）
 * @param {function} [o.log]
 */
export function renderFrames(o) {
  const log = o.log || (() => {})
  const style = resolveRenderStyle(o.style || o.preset)
  const { colors, geometry } = style
  const fps = o.fps || 24
  const W = o.w || 1280, H = o.h || 720
  const t0 = Date.now()

  const wav = readWav(o.input)
  let mono = toMono(wav.channels)
  if (wav.sr !== 16000) mono = resample(mono, wav.sr, 16000)
  const sr = 16000
  const dur = mono.length / sr
  // ⚠️ **这里是 ceil，不是 floor —— 差一帧会吃掉音频的尾巴。**
  //    实测（5.98 秒 @12fps）：floor → 71 帧 = **5.9167 秒**，
  //    而拼 mp4 那条命令带 `-shortest` → **音频被截到视频长度**：
  //      源 wav 5.980 s → 成片音频流 **5.8979 s（少 82 ms）**。
  //    → 改 ceil：72 帧 = 6.0 秒 ≥ 音频，`-shortest` 就切在音频上了。
  //    （这条是 GPT 那份方案里"音画漂移"那句话引出来的 —— 它没给数，但方向对：
  //      **尾部确实被吃掉了一截。**）
  const nFrames = Math.max(1, Math.ceil(dur * fps))
  log('音频 ' + dur.toFixed(2) + ' 秒 → ' + nFrames + ' 帧 @ ' + fps + 'fps · ' + W + '×' + H)

  if (!existsSync(o.outDir)) mkdirSync(o.outDir, { recursive: true })

  // ① 整段轮廓（一次算好）
  const COLS = W
  const { lo, hi } = outline(mono, COLS)
  let gmax = 1e-9
  for (let i = 0; i < mono.length; i++) { const a = Math.abs(mono[i]); if (a > gmax) gmax = a }

  // ② 频谱：两遍。第一遍收原始幅度并记下**整段**最大，第二遍用它做 dB 参考。
  //    ⚠️ 不能每帧各自归一化 —— 那样"绝对音量"整个丢了（大声和安静的两帧看起来一样高），
  //       而频谱柱的用途**恰恰**是让人看出音量在哪儿起伏。
  const NB = geometry.spectrumBars
  const specFrames = []
  const rawFrames = []
  const NF = 512
  let gmaxSpec = 1e-9
  for (let f = 0; f < nFrames; f++) {
    const t = (f + 0.5) / fps
    const mag = spectrumAt(mono, sr, t, NF)
    const bars = new Float32Array(NB)
    for (let b = 0; b < NB; b++) {
      const f0 = Math.floor(Math.pow(mag.length, b / NB))
      const f1 = Math.max(f0 + 1, Math.floor(Math.pow(mag.length, (b + 1) / NB)))
      let s = 0
      for (let k = f0; k < f1 && k < mag.length; k++) s += mag[k]
      bars[b] = s / (f1 - f0)
      if (bars[b] > gmaxSpec) gmaxSpec = bars[b]
    }
    rawFrames.push(bars)
  }
  for (const bars of rawFrames) {
    for (let b = 0; b < NB; b++) {
      const db = 20 * Math.log10((bars[b] + 1e-9) / gmaxSpec)
      bars[b] = Math.max(0, Math.min(1, (db + geometry.spectrumDb) / geometry.spectrumDb))
    }
    specFrames.push(bars)
  }
  log('频谱算完了（' + nFrames + ' × ' + NB + ' 根柱 · 按整段归一化）')

  // ══ 布局 ══════════════════════════════════════════════════════════
  const PAD = geometry.pad
  const midY = Math.round(H * geometry.midY)
  const halfAmp = Math.round(H * geometry.wholeAmp)       // 整段轮廓的振幅
  const wHalfAmp = Math.round(H * geometry.windowAmp)      // 窗口波形的振幅（略小，免得跟整段等高）
  const specTop = Math.round(H * geometry.spectrumTop)
  const specH = Math.round(H * geometry.spectrumHeight)
  // ⚠️ **局部那一块的横向范围** —— 只占中间 45%。
  //    铺满整幅的话它跟整段轮廓形状趋同，就分不出"局部/整体"了。
  const LW = Math.round(W * geometry.localWidth)
  const LX0 = Math.round((W - LW) / 2)
  const LX1 = LX0 + LW

  for (let f = 0; f < nFrames; f++) {
    const c = new Canvas(W, H, colors.bg)
    const t = (f + 0.5) / fps
    const pos = Math.min(1, t / dur)

    // ① 整段轮廓（铺满）—— 已经过去的那段更亮，没到的也有个底（不然第 1 帧一片空）
    for (let x = 0; x < COLS; x++) {
      const y0 = midY - hi[x] / gmax * halfAmp
      const y1 = midY - lo[x] / gmax * halfAmp
      const past = x / COLS <= pos
      c.vline(x, y0, y1, past ? colors.waveDim : colors.waveFuture, 1)
    }
    c.line(0, midY, W - 1, midY, colors.dim, 0.5)

    // ② ⚠️ **这里原来有一道"你在这"的橙竖线 —— 我去掉了。**
    //    因为它跟底部那条进度条**标的是同一件事**（整段里的位置），
    //    而它跟下面那个"局部框"的 x 坐标**不是一回事**（框是固定的，位置在滑动）。
    //    于是结尾那帧里竖线跑到画面最右边、框还在中间 —— 读起来像"框没跟上"。
    //    → 位置只留两处：底部进度条（整段）+ 框下方那条（这块在整段的哪儿）。
    //    **同一个信息给两个不一致的表示，比不给更糟。**

    // ③ 局部那一块的底（在曲线之前画 —— 它是底，不是盖）
    //    ⚠️ 框要**刚好包住里面的东西**。我上一版给的是 H*0.06 ~ H*0.84 ——
    //       比内容大出一圈，于是安静的段落里**框几乎是空的而边框很清楚**，
    //       读起来像"有个东西没加载出来"，而不像"这里现在很安静"。
    //       → 收窄到刚好包住频谱+波形，边框也调暗一档。
    const boxTop = specTop - 8
    const boxBot = Math.round(midY + wHalfAmp + 10)
    c.rect(LX0, boxTop, LW, boxBot - boxTop, colors.boxFill, 1)
    c.line(LX0, boxTop, LX0, boxBot, colors.boxEdge, 1)
    c.line(LX1, boxTop, LX1, boxBot, colors.boxEdge, 1)

    // ③b **这块对应整段的哪一段** —— 局部框下面一条缩略。
    //     ⚠️ 它得够粗才看得见：第一版我给的是 2px、而且离框 9px，
    //        在 960 宽的画面上几乎找不到（我得盯着看才看见）。
    const thumbY = boxBot + 10
    const wFrac = Math.min(1, geometry.windowSeconds / dur)
    const ta = Math.max(0, Math.min(1 - wFrac, pos - wFrac / 2))
    c.rect(LX0, thumbY, LW, 5, colors.thumb)
    c.rect(Math.round(LX0 + LW * ta), thumbY,
      Math.max(3, Math.round(LW * wFrac)), 5, colors.accent)

    // ④ 频谱柱（同样只占局部那块）
    const bars = specFrames[f]
    const bw = LW / NB
    for (let b = 0; b < NB; b++) {
      const h = bars[b] * specH
      if (h < 1) continue
      c.rect(Math.round(LX0 + b * bw), Math.round(specTop + specH - h),
        Math.max(1, Math.floor(bw) - 2), Math.round(h), colors.spectrum, 0.88)
    }

    // ⑤ 窗口波形（亮的，同样只占局部那块）—— 窗口 0.35 秒
    const halfWindow = geometry.windowSeconds / 2
    const winS = Math.max(0, Math.round((t - halfWindow) * sr))
    const winN = Math.min(mono.length - winS, Math.round(geometry.windowSeconds * sr))
    if (winN > 1) {
      for (let x = LX0; x < LX1; x++) {
        const i0 = winS + Math.floor((x - LX0) / LW * winN)
        const i1 = winS + Math.floor((x + 1 - LX0) / LW * winN)
        let mn = 1e9, mx = -1e9
        for (let i = i0; i < Math.max(i0 + 1, i1); i++) {
          const v = mono[i]; if (v < mn) mn = v; if (v > mx) mx = v
        }
        if (mn > mx) { mn = 0; mx = 0 }
        c.vline(x, midY - mx / gmax * wHalfAmp, midY - mn / gmax * wHalfAmp, colors.wave, 1)
      }
    }

    // ⑥ 进度条（整段）
    const py = H - PAD
    c.rect(PAD, py, W - PAD * 2, 3, colors.progress)
    c.rect(PAD, py, Math.round((W - PAD * 2) * pos), 3, colors.accent)

    writeFileSync(join(o.outDir, 'f' + String(f + 1).padStart(5, '0') + '.png'), c.toPNG())
    if (f % 24 === 0 || f === nFrames - 1) log('  画到 ' + (f + 1) + '/' + nFrames)
  }

  return {
    frames: nFrames, fps, w: W, h: H, seconds: dur,
    outDir: o.outDir, preset: style.preset, ms: Date.now() - t0,
  }
}
