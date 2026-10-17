/**
 * png.js — 纯 JS 写 PNG。不依赖任何图像库。
 *
 * ══════════════════════════════════════════════════════════════════════
 * PNG 其实很简单，就三个 chunk：
 *     IHDR   宽 / 高 / 位深 8 / 颜色类型 2（真彩 RGB）
 *     IDAT   **zlib 压缩过的扫描行** —— 而每一行前面要有一个 filter 字节
 *     IEND
 *
 * ⚠️ **那个 filter 字节最容易忘** —— 每行开头一个 0（表示"不过滤"）。
 *    忘了它，出来的图会整体错位一列，而且**看起来像是"画错了"**，不像"格式错了"。
 *    它也不会报错 —— 图就是花的。
 *
 * ⚠️ 每个 chunk 后面要 4 字节 CRC32。Node 20+ 的 `zlib.crc32` 有；
 *    没有的话下面那个手写版也能用（用的是 PNG 规定的那个多项式）。
 */
import { deflateSync, crc32 as zlibCrc32 } from 'node:zlib'

// CRC32（PNG 用的是标准那个）。Node 有就用手写的兜底只在没有时用。
const CRC_TABLE = (() => {
  const t = new Int32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1)
    t[n] = c
  }
  return t
})()

function crc32(buf) {
  if (typeof zlibCrc32 === 'function') return zlibCrc32(buf) >>> 0
  let c = -1
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xFF] ^ (c >>> 8)
  return (c ^ -1) >>> 0
}

function chunk(type, data) {
  const len = Buffer.alloc(4)
  len.writeUInt32BE(data.length, 0)
  const t = Buffer.from(type, 'ascii')
  const crcBuf = Buffer.alloc(4)
  crcBuf.writeUInt32BE(crc32(Buffer.concat([t, data])), 0)
  return Buffer.concat([len, t, data, crcBuf])
}

/**
 * 把一块 RGB 像素写成 PNG。
 * @param {Uint8Array} rgb  长度 width*height*3，逐行
 * @param {number} width
 * @param {number} height
 */
export function encodePNG(rgb, width, height) {
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0)
  ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8        // 位深
  ihdr[9] = 2        // 颜色类型 2 = 真彩 RGB
  ihdr[10] = 0       // 压缩方法
  ihdr[11] = 0       // 滤波方法
  ihdr[12] = 0       // 隔行：0 = 不隔行
  // ⚠️ 每行前面加一个 filter 字节（0 = None）。**忘了它图就花。**
  const stride = width * 3
  const raw = Buffer.alloc((stride + 1) * height)
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0
    Buffer.from(rgb.buffer, rgb.byteOffset + y * stride, stride)
      .copy(raw, y * (stride + 1) + 1)
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 6 })),
    chunk('IEND', Buffer.alloc(0)),
  ])
}

/** 一块画布 —— 就是个 RGB 数组，带几个画线/画矩形的函数 */
export class Canvas {
  constructor(width, height, bg = [10, 11, 14]) {
    this.w = width
    this.h = height
    this.px = new Uint8Array(width * height * 3)
    if (bg) this.fill(bg)
  }

  fill([r, g, b]) {
    for (let i = 0; i < this.px.length; i += 3) {
      this.px[i] = r; this.px[i + 1] = g; this.px[i + 2] = b
    }
  }

  set(x, y, [r, g, b], alpha = 1) {
    x = x | 0; y = y | 0
    if (x < 0 || y < 0 || x >= this.w || y >= this.h) return
    const i = (y * this.w + x) * 3
    if (alpha >= 1) {
      this.px[i] = r; this.px[i + 1] = g; this.px[i + 2] = b
    } else {
      this.px[i] += (r - this.px[i]) * alpha
      this.px[i + 1] += (g - this.px[i + 1]) * alpha
      this.px[i + 2] += (b - this.px[i + 2]) * alpha
    }
  }

  /** 竖直的一条线（画波形/频谱用这个最省事） */
  vline(x, y0, y1, color, alpha = 1) {
    if (y1 < y0) { const t = y0; y0 = y1; y1 = t }
    for (let y = Math.floor(y0); y <= Math.ceil(y1); y++) this.set(x, y, color, alpha)
  }

  /** 按下面的"抗锯齿"画一条任意方向的线（Bresenham + 端点柔化） */
  line(x0, y0, x1, y1, color, alpha = 1) {
    x0 = Math.round(x0); y0 = Math.round(y0)
    x1 = Math.round(x1); y1 = Math.round(y1)
    const dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0)
    const sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1
    let err = dx - dy
    for (;;) {
      this.set(x0, y0, color, alpha)
      if (x0 === x1 && y0 === y1) break
      const e2 = 2 * err
      if (e2 > -dy) { err -= dy; x0 += sx }
      if (e2 < dx) { err += dx; y0 += sy }
    }
  }

  rect(x, y, w, h, color, alpha = 1) {
    for (let j = y; j < y + h; j++) for (let i = x; i < x + w; i++) this.set(i, j, color, alpha)
  }

  toPNG() { return encodePNG(this.px, this.w, this.h) }
}
