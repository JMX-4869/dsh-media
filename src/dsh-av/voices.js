/**
 * voices.js — 音色目录的管理。
 *
 * ══════════════════════════════════════════════════════════════════════
 * "音色"是什么：
 *    一个目录，里面至少有两样：
 *      <名字>.onnx         net_g —— **音色就在这一层**（53.5 MB 那个）
 *      <名字>.index        Faiss 检索索引（可选 —— 不装就是 index_rate=0 那条路）
 *    外加一份 `model_info.json`（这一版是怎么练出来的）。
 *
 * ⚠️ 而它是**可换的** —— 那也是"可持续训练"的意思：
 *    换个录音、重跑一遍 train_voice.py，就多一个目录。
 *    而底模和 rmvpe / hubert 那两个网络是共用的，不用重下。
 *
 * ⚠️ 而"检索"那一步我在 JS 里**还没做** —— 它是 net_g 外面的一层：
 *    拿 hubert 特征去索引里找最像的若干条，加权平均之后再喂给 net_g。
 *    影响的是"像不像"，不是"算得对不对"。**所以数值一致性不受它影响。**
 *    而 index_rate=0（不用检索）也是一条合法的路 —— 现在就是那样。
 */
import { existsSync, readdirSync, statSync, readFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'

/** 扫一个音色目录，看它有什么 */
export function readVoice(dir, name) {
  if (!existsSync(dir)) return null
  const files = readdirSync(dir)
  const onnx = files.filter((f) => f.endsWith('.onnx'))
  const index = files.filter((f) => f.endsWith('.index'))
  const infoPath = join(dir, 'model_info.json')
  let info = null
  if (existsSync(infoPath)) {
    try {
      info = JSON.parse(readFileSync(infoPath, 'utf8'))
    } catch (e) {
      // ⚠️ 这里**不能静默吞掉** —— 一个坏掉的 model_info.json 会让界面上
      //    "这个音色是拿什么练的"整栏空着，而那个空着看起来像"没记"。
      info = { error: String(e.message || e) }
    }
  }
  let mb = 0
  for (const f of files) {
    try { mb += statSync(join(dir, f)).size / 1048576 } catch { /* 目录项可能已经没了 */ }
  }
  return {
    name,
    dir,
    onnx,
    index,
    hasIndex: index.length > 0,
    info,
    mb: +mb.toFixed(1),
    // 首选哪个 onnx：**名字跟音色名一样的那个**（train_voice.py 就是那么命名的）
    preferred: onnx.find((f) => f === `${name}.onnx`)
      || onnx.find((f) => f.includes('fp16'))
      || onnx[0] || null,
  }
}

/** 列出所有音色 */
export function listVoices(voicesDir) {
  if (!existsSync(voicesDir)) return []
  return readdirSync(voicesDir)
    .filter((n) => !n.startsWith('.') && statSync(join(voicesDir, n)).isDirectory())
    .map((n) => readVoice(join(voicesDir, n), n))
    .filter(Boolean)
}

/** 按名字找一个 */
export function findVoice(voicesDir, name) {
  if (!name) return null
  const dir = join(voicesDir, String(name))
  return readVoice(dir, String(name))
}

/**
 * 三件网络的现状 —— 不猜，全从磁盘读。
 * ⚠️ 而"该多大"那几个数是**实测的**（不是估的），所以能拿它判"是不是下了一半"。
 *    （那个坑踩过：curl 下了一半不报错，一直到 torch.load 才炸。）
 */
export function modelStatus(modelsDir, spec) {
  return spec.map((m) => {
    const p = join(modelsDir, m.file)
    const has = existsSync(p)
    const actual = has ? +(statSync(p).size / 1048576).toFixed(1) : 0
    return {
      ...m,
      path: p,
      present: has,
      actualMB: actual,
      // 跟实测大小差 5% 以上就标出来（"下了一半"那种）
      suspect: has && Math.abs(actual - m.mb) > m.mb * 0.05,
    }
  })
}

export function ensureDirs(dirs) {
  for (const d of dirs) {
    if (!existsSync(d)) mkdirSync(d, { recursive: true })
  }
}
