/**
 * wav.js — 读和写 wav。纯 JS，不依赖任何库。
 *
 * ══════════════════════════════════════════════════════════════════════
 * ⚠️ **别假设"头就是 44 字节"** —— 那是新手最常见的错。wav 是一串 chunk：
 *      RIFF
 *        fmt  （格式：位深、声道、采样率）
 *        LIST / fact / data ……（**可能有别的 chunk 插在中间**）
 *        data （采样数据）
 *    → 所以**要按 chunk 走**，读到一个认不出的就跳过它（chunk 头上写着长度）。
 *      直接跳到 44 字节开始读，在带 LIST 的文件上会读到元数据当音频。
 *
 * ⚠️ 位深要分开处理：16（最常见）/ 24 / 32 整数 / 32 浮点 / 8 无符号。
 *    而 float32 的 wav 是把样本直接存成 IEEE 浮点，**没有那个 /32768 的换算** ——
 *    拿整数那套去读它，出来是一堆极小的数（听不见）。
 */
import { readFileSync, writeFileSync } from 'node:fs'

/** 读 wav → { sr, channels: [Float32Array, ...] } */
export function readWav(path) {
  const buf = readFileSync(path)
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('不是 wav：' + path)
  }
  let off = 12
  let fmt = null
  let data = null
  while (off + 8 <= buf.length) {
    const id = buf.toString('ascii', off, off + 4)
    const size = buf.readUInt32LE(off + 4)
    const body = off + 8
    if (id === 'fmt ') {
      fmt = {
        format: buf.readUInt16LE(body),          // 1 = PCM, 3 = IEEE float, 0xFFFE = 扩展
        channels: buf.readUInt16LE(body + 2),
        sampleRate: buf.readUInt32LE(body + 4),
        bits: buf.readUInt16LE(body + 14),
      }
      // ⚠️ 0xFFFE（WAVE_FORMAT_EXTENSIBLE）真正的格式码在扩展块里 —— 而 ffmpeg 转出来的
      //    多声道/float 常常是这个。子格式的头两个字节就是格式码。
      if (fmt.format === 0xFFFE && size >= 40) {
        fmt.format = buf.readUInt16LE(body + 24)
      }
    } else if (id === 'data') {
      data = { start: body, size: Math.min(size, buf.length - body) }
    }
    off = body + size + (size % 2)      // chunk 是偶数字节对齐的
  }
  if (!fmt || !data) throw new Error('wav 缺 fmt 或 data chunk')

  const bytes = fmt.bits / 8
  const n = Math.floor(data.size / (bytes * fmt.channels))
  const ch = []
  for (let c = 0; c < fmt.channels; c++) ch.push(new Float32Array(n))

  const isFloat = fmt.format === 3
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < fmt.channels; c++) {
      const p = data.start + (i * fmt.channels + c) * bytes
      let v
      if (isFloat && fmt.bits === 32) v = buf.readFloatLE(p)
      else if (fmt.bits === 16) v = buf.readInt16LE(p) / 32768
      else if (fmt.bits === 32) v = buf.readInt32LE(p) / 2147483648
      else if (fmt.bits === 24) {
        const lo = buf.readUInt16LE(p), hi = buf.readInt8(p + 2)
        v = ((hi << 16) | lo) / 8388608
      } else if (fmt.bits === 8) v = (buf.readUInt8(p) - 128) / 128
      else throw new Error('不支持的位深：' + fmt.bits)
      ch[c][i] = v
    }
  }
  return { sr: fmt.sampleRate, channels: ch, bits: fmt.bits, format: fmt.format }
}

/** 写 wav（16 位整数，单声道或给多声道） */
export function writeWav(path, channels, sr) {
  const chs = Array.isArray(channels) ? channels : [channels]
  const n = chs[0].length
  const nch = chs.length
  // ⚠️ 先找峰值再归一化 —— 而**不归一化到底**（0.9 留一点余量，免得量化时削顶）
  let peak = 0
  for (const c of chs) {
    for (let i = 0; i < n; i++) { const a = Math.abs(c[i]); if (a > peak) peak = a }
  }
  const g = peak > 0 ? 0.9 / peak : 1
  const size = 44 + n * nch * 2
  const buf = Buffer.alloc(size)
  buf.write('RIFF', 0); buf.writeUInt32LE(size - 8, 4); buf.write('WAVE', 8)
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20)
  buf.writeUInt16LE(nch, 22); buf.writeUInt32LE(sr, 24)
  buf.writeUInt32LE(sr * nch * 2, 28); buf.writeUInt16LE(nch * 2, 32); buf.writeUInt16LE(16, 34)
  buf.write('data', 36); buf.writeUInt32LE(n * nch * 2, 40)
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < nch; c++) {
      let v = chs[c][i] * g
      v = v > 1 ? 1 : (v < -1 ? -1 : v)
      buf.writeInt16LE(Math.round(v * 32767), 44 + (i * nch + c) * 2)
    }
  }
  writeFileSync(path, buf)
  return { path, samples: n, channels: nch, sr, peak, gain: g }
}

/** 立体声 → 单声道（取平均，跟 RVC 的 force_mono 一致） */
export function toMono(channels) {
  if (channels.length === 1) return channels[0]
  const n = channels[0].length
  const out = new Float32Array(n)
  for (let c = 0; c < channels.length; c++) {
    const ch = channels[c]
    for (let i = 0; i < n; i++) out[i] += ch[i] / channels.length
  }
  return out
}
