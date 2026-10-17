/**
 * dsh-av — DeepSeek Harness 声音与视频插件
 *
 * ══════════════════════════════════════════════════════════════════════
 * 它做两件事，而这两件是同一件事的两头：
 *
 *   **声音**：把一段音频的"音色"换成某个人的（音色转换）。
 *            而它**不依赖 Python，也不依赖 torch** —— 三个网络都用 ONNX 在 Node 里跑：
 *              hubert   波形 → 768 维内容特征（不知道是谁在唱）
 *              rmvpe    mel  → 音高（每帧）
 *              net_g    内容特征 + 音高 + 噪声 → 波形（**音色在这一层**）
 *            而 mel 那一步（STFT + mel 滤波器）是纯 JS 写的 —— 见 `mel.mjs`。
 *
 *   **视频**：把音频渲染成片子（频谱 / 波形 / 歌词 / 曲线）。
 *
 * ── 为什么"不依赖 Python"这条值得单独说 ──
 * 第一次练音色时用的是 RVC 那一整套（torch + CUDA，装完 11 GB）。
 * 而后来量出来：**转换那一步根本不需要 torch** ——
 * `rvc-onnx-web` 那个纯 TS 库 1.12 秒就把 .pth 转成了 onnx。
 * 而推理这边，onnxruntime 的实际二进制（CPU）是 28 MB。
 * → 所以"用得上"和"拖得动"是两件事。
 *
 * ── 而体积这件事，我量过，不是猜的 ──
 *   net_g    fp16   53.5 MB
 *   hubert   fp16  180.3 MB    ← int8 量化后相关只剩 0.2032（**算错，不是有损**）
 *   rmvpe    fp16  172.5 MB    ← int8 量化后 20.6% 的帧音高错超半个半音
 *   onnxruntime（CPU）  ~28 MB
 * → **合计约 434 MB**。int8 那条路我试过、量过、放弃了。
 *   而"用自己的 YIN 换掉 rmvpe"也试过 —— 73% 的共同帧差超过半个半音。
 *   **这两条都在 README 里记着，免得以后再试一遍。**
 *
 * ══════════════════════════════════════════════════════════════════════
 * 数据放哪：
 *   模型（三件 onnx）在 `<数据目录>/dsh-av/models/`
 *   音色（每个音色一份 net_g + index）在 `<数据目录>/dsh-av/voices/<名字>/`
 *   而 `<数据目录>` 默认是 `$DSH_HOME`（即 ~/.dsh），可以用环境变量 DSH_AV_HOME 改。
 */
import { existsSync, statSync, mkdirSync, appendFileSync } from 'node:fs'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import os from 'node:os'

// ⚠️ 目录扫描那套放在 voices.js 里，**不在这里再写一遍** ——
//    两处各写一份，早晚会有一处不同步（而那种 bug 很安静）。
import { listVoices, modelStatus, ensureDirs } from './voices.js'

// ⚠️⚠️ **`@deepseek-ai/dsh-tools` 不用静态 import 了** —— 两个理由，都是血的：
//
//  ① 静态 import 失败 = **整个模块加载失败** = 插件 fiber 直接 failed（工具一个都不出现）。
//     而"宿主到底给不给我这份依赖"我在本地**验不了** —— 真的那份藏在 app.asar 里
//     （磁盘上找不到；dsh-mood 的 profile 副本连 node_modules 都没有，却能跑，
//      说明是宿主注入的）。→ 改成动态 import + 兜底，失败可捕获。
//  ② 我这边 `src/node_modules/@deepseek-ai/dsh-tools` 里放的是**测试替身**——
//     它只做参数校验，**不会把 parameters 规格编译成 JSON Schema**（真身会）。
//     它很可能一直**shadow 着宿主注入的真身**，于是工具注册了却不出现在工具表里。
//     → 现在：能拿到哪份就记哪份，而且**回查一次注册结果**（tools.get / tools.schemas）。

const HOME = process.env.DSH_AV_HOME
  || join(process.env.DSH_HOME || join(os.homedir(), '.dsh'), 'dsh-av')

const MODELS = join(HOME, 'models')
const VOICES = join(HOME, 'voices')

/**
 * 两张网络各自该在哪、多大、干什么（**音色不在这张表里** —— 它是 voices/<名字>/ 下的东西）。
 *
 * ⚠️ 这里按 **group** 判"缺没缺"，不按文件 —— 因为 rmvpe 有两份而**只需要一份**：
 *    `rmvpe.onnx`（fp32，344.9 MB）是**能跑的那份**；
 *    `rmvpe_fp16.onnx`（172.5 MB）在 onnxruntime-node 下**加载就 segfault**
 *    （不是抛异常，是原生崩溃，try/catch 拦不住）。
 *    convert.js 也是这么挑的：fp32 优先、fp16 兜底。
 *    ⚠️ 而这条我踩过 —— 表里原来只登记着 fp16 那份，于是
 *    **删掉 fp16 之后 status 报"缺件"、convert 直接拒跑**，而磁盘上东西是齐的。
 */
const MODELS_SPEC = [
  { file: 'hubert_fp16.onnx', mb: 180.3, group: 'hubert',
    what: '波形 → 768 维内容特征（不知道是谁在唱）' },
  { file: 'rmvpe.onnx', mb: 344.9, group: 'rmvpe',
    what: 'mel → 每帧音高（**fp32 —— 能用的就是这一份**）' },
  { file: 'rmvpe_fp16.onnx', mb: 172.5, group: 'rmvpe', optional: true,
    what: 'mel → 每帧音高（fp16 —— 在 Node 下加载就崩，只留档）' },
  { file: 'mel_basis.f32', mb: 0.3, group: 'mel',
    what: 'mel 滤波器表（STFT → 125 维 mel）—— 纯 JS 那一步要用' },
  { file: 'mel_meta.json', mb: 0.001, group: 'mel',
    what: 'mel 表的元信息（帧长 / hop / 采样率 / 频点数）' },
]

/**
 * 缺的是"哪组"，不是"哪个文件"。
 * 判据：一组里**任何一件必填的（非 optional）不在**，这组就算缺。
 * @returns {string[]} 缺的那些 group
 */
export function missingGroups(models) {
  const groups = [...new Set(MODELS_SPEC.map((m) => m.group))]
  return groups.filter((g) => models.some((m) => m.group === g && !m.optional && !m.present))
}

/** 扫一遍现状 —— 不猜，全从磁盘读 */
export function scan() {
  ensureDirs([HOME, MODELS, VOICES])
  return {
    home: HOME, modelsDir: MODELS, voicesDir: VOICES,
    models: modelStatus(MODELS, MODELS_SPEC),
    voices: listVoices(VOICES),
  }
}

/**
 * 把一份**任意** schema 修成合法的 JSON Schema —— 只修"类型名不合法"这一类。
 *
 * ⚠️⚠️ 这一条是 0.4.3 重启后**宿主亲口报出来的**：
 *     JsonSchemaError: unsupported JSON schema:
 *       schema.type must be one of object/array/string/number/integer/boolean/null
 *     而触发它的是我自己写的这一句（**照 dsh-mood 抄的**）：
 *         output: { schema: { type: 'json' }, render: … }     ← 'json' 不是合法类型
 *     真身 defineTool 会替你规范化，**兜底不会** —— 所以我这边一注册就抛。
 *     ⚠️ 教训：**照抄一个能跑的例子，得先问"它凭什么能跑"。**
 *        dsh-mood 能跑不是因为这个写法对，是因为它拿到的是真身。
 */
const LEGAL_TYPES = new Set(['object', 'array', 'string', 'number', 'integer', 'boolean', 'null'])
function fixSchema(node) {
  if (!node || typeof node !== 'object') return node
  if (Array.isArray(node)) return node.map(fixSchema)
  const out = {}
  for (const [k, v] of Object.entries(node)) {
    if (k === 'type' && typeof v === 'string' && !LEGAL_TYPES.has(v)) {
      out.type = 'object'   // 'json' 这类自造名字 → 退成"任意对象"（我们返回的就是个对象）
      continue
    }
    out[k] = (k === 'properties' || k === 'items' || k === 'oneOf' || k === 'anyOf' || k === 'allOf')
      ? fixSchema(v) : v
  }
  return out
}

/**
 * defineTool 的**兜底**实现 —— 它比那份测试替身多做两件事：
 *   ① 把 `parameters` 规格**编译成 JSON Schema**；
 *   ② 把 `output.schema` 里不合法的类型名**规范化**。
 * ⚠️ ② 是 0.4.3 才补的 —— 少了它，`register` 直接抛 JsonSchemaError（而客户端什么都看不见）。
 */
function fallbackDefineTool({ name, description, parameters, output, execute, ...rest }) {
  const properties = {}
  const required = []
  for (const [key, spec] of Object.entries(parameters || {})) {
    const s = spec && typeof spec === 'object' ? { ...spec } : { type: 'string' }
    if (s.required === true) required.push(key)
    delete s.required
    properties[key] = fixSchema(s)
  }
  return {
    name,
    description,
    parameters: { type: 'object', properties, required, additionalProperties: false },
    output: output && output.schema ? { ...output, schema: fixSchema(output.schema) } : output,
    execute,
    ...rest,
  }
}

/** 拿 defineTool：真身优先（含直接指到 app.asar 里那份），拿不到才用兜底 —— 并记下用了哪份 */
async function resolveDefineTool(log) {
  // ① 正常解析
  try {
    const m = await import('@deepseek-ai/dsh-tools')
    if (m && typeof m.defineTool === 'function') {
      const isStub = String(m.defineTool).includes('本地测试桩')
      log('defineTool：拿到了 · ' + (isStub
        ? '**是本地替身（被 shadow 了 —— 它不编译 schema、也不规范化 output.schema）**'
        : '是宿主注入的真身'))
      return m.defineTool
    }
    log('defineTool：模块里没有这个导出')
  } catch (e) {
    log('defineTool：import 失败（' + (e && e.message) + '）')
  }
  // ② 直接指到宿主自己那份 —— **它藏在 app.asar 里，而宿主 register 的报错栈证明了那个路径能 import**
  if (process.resourcesPath) {
    const p = join(process.resourcesPath, 'app.asar', 'dsh', 'node_modules',
      '@deepseek-ai', 'dsh-tools', 'lib', 'index.js')
    try {
      const m = await import(pathToFileURL(p).href)
      if (m && typeof m.defineTool === 'function') {
        log('defineTool：**从 app.asar 直接拿到了真身** → ' + p)
        return m.defineTool
      }
      log('defineTool：app.asar 里那份没有 defineTool 导出')
    } catch (e) {
      log('defineTool：直接指 app.asar 也没成（' + (e && e.message) + '）')
    }
  } else {
    log('defineTool：没有 process.resourcesPath（不在 Electron 里）')
  }
  log('defineTool：**用兜底**（自己编译 parameters + 规范化 output.schema）')
  return fallbackDefineTool
}

/**
 * 插件入口。
 *
 * ⚠️ **本体是同步的、只负责挂上一个 async 的 boot** —— 这样注册里出的任何错
 *    都只进日志，**不会把整个插件搞成 fiber failed**（上一轮就是这么死的）。
 * ⚠️ 宿主里的报错我看不见，所以 boot 的每一步都写进 `<数据目录>/load.log`。
 *    这是这台机器上我唯一能读到的"宿主视角"。
 */
export default function dshAv(ctx) {
  const lines = []
  const log = (m) => lines.push(new Date().toISOString() + '  ' + m)
  // ⚠️ **把 promise 返回出去** —— 两个理由：
  //   ① 宿主会 await 它，装载的先后顺序确定（而不是"插件 active 了、工具还没注册"）；
  //   ② 本地自测能 `await 插件(ctx)`，否则自测跑在注册之前，看着像"没注册上"。
  //   而 catch 把一切错误都吃掉 → **永远不会因为注册失败把 fiber 搞挂**。
  return boot(ctx, log)
    .catch((e) => log('boot 抛错：' + (e && (e.stack || e.message) || e)))
    .finally(() => {
      try {
        ensureDirs([HOME])
        appendFileSync(join(HOME, 'load.log'), lines.join('\n') + '\n')
      } catch { /* 日志写不进去也不能影响装载 */ }
    })
}

async function boot(ctx, log) {
  log('── 装载 ──  ctx.get=' + typeof ctx.get + ' · ctx.effect=' + typeof ctx.effect)
  const defineTool = await resolveDefineTool(log)
  // ⚠️⚠️ **服务必须用 `ctx.get('名字')` 取，不能 `ctx.tools` 直接读。**
  //
  //    cordis 对"没声明 inject 就想读服务"是**直接抛错**的：
  //        cannot get property "tools" without inject
  //    而抛在插件体里 = **整个插件挂载失败**（fiberPhase: failed）——
  //    界面上是"装上了、启用了、然后什么都没有"，工具一个都不出现。
  //
  //    2026-10-16 就栽在这：装进 profile、重启、插件列表里 208 条有它，
  //    而那一条写着 `fiberPhase: failed`。
  //    这条写法是从 dsh-mood 抄的 —— 它是这台机器上唯一一个跑着的参照，
  //    **而它用的是 `ctx.get`，不是解构。**
  const tools = ctx.get ? ctx.get('tools') : ctx.tools
  const effect = ctx.effect
  log('tools 服务：' + (tools ? '有' : '**没有**')
    + ' · register=' + typeof (tools && tools.register)
    + ' · get=' + typeof (tools && tools.get)
    + ' · schemas=' + typeof (tools && tools.schemas))

  // ── 工具 ────────────────────────────────────────────────────────────
  if (tools && typeof tools.register === 'function') {
    const def = defineTool({
      name: 'av',
      description:
        '声音与视频。声音那一层是**纯 JS 的音色转换** —— 三个 ONNX 网络在 Node 里跑，'
        + '不依赖 Python、不依赖 torch。\n'
        + 'action=status 看现状（模型在不在、多大、有几个音色）+ 数据目录；'
        + 'action=setup 打印缺哪几件、各自该多大、从哪来；'
        + 'action=convert 换音色（input/voice/output）；'
        + 'action=video 把音频渲染成片子（input/output/fps/width/preset，拼 mp4 那步要 ffmpeg）；'
        + 'action=listen 把音频整理成 dsh-mood ASR 可读的 16k 单声道 wav。\n'
        + '⚠️ 检索（Faiss index）还没接 —— 那是 net_g 外面一层，影响"像不像"而不是"算得对不对"。',
      parameters: {
        action: {
          type: 'string',
          enum: ['status', 'setup', 'convert', 'video', 'listen'],
          description: 'status=看现状（默认）；setup=打缺失清单和来源；'
            + 'convert=换音色；video=把音频渲染成片子；listen=输出 16k ASR 音频',
        },
        input: { type: 'string', description: 'convert / video / listen：输入音频路径（wav）' },
        voice: { type: 'string', description: 'convert：音色名（voices/ 下的目录名）' },
        output: { type: 'string', description: 'convert 出 wav；video 出 mp4；listen 出 16k wav' },
        compatOutput: { type: 'string', description: 'convert：可选，额外输出 16k ASR 兼容 wav' },
        fps: { type: 'number', description: 'video：帧率（默认 24）' },
        width: { type: 'number', description: 'video：宽（默认 1280。高度按 16:9 算）' },
        preset: { type: 'string', enum: ['default', 'warm', 'cool', 'violet'], description: 'video：画面品味预设' },
        taste: { type: 'string', description: 'video：可选 JSON 风格参数，覆盖 preset 的颜色/布局' },
      },
      output: {
        schema: { type: 'json' },
        render: (_args, value) => [{ type: 'text', text: value.message }],
      },
      async execute(args) {
        const action = args && args.action ? String(args.action) : 'status'
        // 不涉及模型的两条路径不要扫描模型和音色目录，减少启动等待。
        if (action === 'listen') {
          const input = args && args.input ? String(args.input) : ''
          const output = args && args.output ? String(args.output)
            : input.replace(/\.wav$/i, '') + '.16k.wav'
          if (!input) return { ok: false, message: 'listen 要 input（wav）' }
          try {
            const { readWav, toMono, writeWav } = await import('./wav.js')
            const { resample } = await import('./hubert.mjs')
            const wav = readWav(input)
            const mono = toMono(wav.channels)
            const audio = wav.sr === 16000 ? mono : resample(mono, wav.sr, 16000)
            writeWav(output, audio, 16000)
            return { ok: true, message: `已生成 16k 单声道 ASR 音频：${output}\n`
              + 'dsh-mood/ASR 可以按 16000 Hz 读取；当前 av 工具接口没有音频附件注入能力，'
              + '所以不能直接把声音塞进模型上下文。' }
          } catch (e) {
            return { ok: false, message: 'listen 失败：' + String(e && e.message || e).slice(0, 400) }
          }
        }
        // ── convert：真干活的那条 ─────────────────────────────────
        // ⚠️ 而它**故意把 onnxruntime 的加载放在函数里面**（不是文件顶部）：
        //    那个包很大，而 status / setup 根本用不到它。
        // ── video：把音频渲染成片子 ───────────────────────────────
        // ⚠️ 它**依赖外部的 ffmpeg**（拼 mp4 那一步）——
        //    而画帧那部分是纯 JS 的（PNG 自己编码，见 png.js）。
        //    不内嵌 ffmpeg 的原因：那个二进制 80~200 MB，而它是台通用机器上常见的。
        //    **所以这一步要先探一下它在不在，不在就直说。**
        if (action === 'video') {
          const input = args && args.input ? String(args.input) : ''
          const output = args && args.output ? String(args.output) : ''
          if (!input || !output) {
            return { ok: false, message: 'video 要两个参数：input（wav）/ output（mp4）' }
          }
          // ⚠️ 钳位**挡不住 NaN**：`Number('abc')` 是 NaN，而 Math.max/min 会把它原样传下去 ——
          //    后果是 ffmpeg 收到 `-framerate NaN`，报一句跟参数无关的错。
          //    → 所以先判有限性，再钳位。
          const num = (v, def, lo, hi) => {
            const n = Number(v)
            return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : def
          }
          const fps = num(args && args.fps, 24, 1, 60)
          const W = num(args && args.width, 1280, 320, 3840)
          const HH = Math.round(W * 9 / 16)
          const { spawnSync } = await import('node:child_process')
          const probe = spawnSync('ffmpeg', ['-version'], { encoding: 'utf8' })
          if (probe.error || probe.status !== 0) {
            return { ok: false, message:
              '这一步要 ffmpeg，而这个环境里找不到它。\n'
              + '（画帧那半是纯 JS 的，只有"帧拼成 mp4"要它。）' }
          }
          const { tmpdir } = await import('node:os')
          const { rmSync } = await import('node:fs')
          const frameDir = join(tmpdir(), 'dsh-av-frames-' + Date.now())
          const lines = []
          try {
            const { renderFrames } = await import('./render.js')
            let style = args && args.taste ? args.taste : (args && args.preset ? args.preset : undefined)
            if (typeof style === 'string' && style.trim().startsWith('{')) {
              try { style = JSON.parse(style) } catch {
                return { ok: false, message: 'taste 必须是合法 JSON' }
              }
            }
            const r = renderFrames({ input, outDir: frameDir, fps, w: W, h: HH, style,
              log: (m) => lines.push('  ' + m) })
            const vout = spawnSync('ffmpeg', [
              '-y', '-hide_banner', '-loglevel', 'error',
              '-framerate', String(fps), '-i', join(frameDir, 'f%05d.png'),
              '-i', input,
              '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p',
              '-c:a', 'aac', '-b:a', '192k', '-shortest', output,
            ], { encoding: 'utf8' })
            if (vout.status !== 0) {
              return { ok: false, message: 'ffmpeg 拼片子失败：\n'
                + String(vout.stderr || '').slice(-500) }
            }
            return { ok: true, message:
              `渲染好了：${output}\n${lines.join('\n')}\n`
              + `  ${r.frames} 帧 @ ${fps}fps · ${W}×${HH} · 音频 ${r.seconds.toFixed(2)} 秒\n`
              + `  画帧 ${r.ms} ms（纯 JS） · preset ${r.preset}` }
          } catch (e) {
            return { ok: false, message: '渲染失败：' + String(e && e.message || e).slice(0, 400) }
          } finally {
            // 帧是中间物 —— 片子出来了就不用留（143 帧约 2.6 MB，长片子会大得多）
            try { rmSync(frameDir, { recursive: true, force: true }) } catch { /* 删不掉不影响产出 */ }
          }
        }

        const s = scan()

        if (action === 'convert') {
          const input = args && args.input ? String(args.input) : ''
          const voice = args && args.voice ? String(args.voice) : ''
          const output = args && args.output ? String(args.output) : ''
          if (!input || !voice || !output) {
            return { ok: false, message: 'convert 要三个参数：input / voice / output' }
          }
          const v = s.voices.find((x) => x.name === voice)
          if (!v) {
            return { ok: false, message: '没找到音色「' + voice + '」。现有：'
              + (s.voices.map((x) => x.name).join(' / ') || '（一个都没有）') }
          }
          if (!v.preferred) return { ok: false, message: '音色「' + voice + '」里没有 .onnx' }
          const miss = missingGroups(s.models)
          if (miss.length) {
            return { ok: false, message: '缺：' + miss.join(' / ')
              + '（用 action=setup 看该放哪、多大）' }
          }
          try {
            const ort = (await import('onnxruntime-node')).default
            const { convert: run, openSessions, loadMelBasis } = await import('./convert.js')
            const lines = []
            const sessions = await openSessions(s.modelsDir, join(v.dir, v.preferred), ort)
            // ⚠️ 提前把"这份音色跑不跑得了"说出来 —— 否则要等到 net_g 那一步
            //    才炸出一句 onnxruntime 的 Reshape 报错（读起来像模型坏了）。
            if (sessions.netgSixInput) {
              lines.push('  ⚠️ 这个音色是**六输入**那张图（输入名里有 rnd）—— 它里面那个'
                + ' Reshape 的形状是导出时那段音频写死的，只有 5.98 秒的输入不崩。'
                + '\n     换一份**七输入**的（输入名里有 noise_1527）才是通用的。')
            }
            const melBasis = loadMelBasis(join(s.modelsDir, 'mel_basis.f32'),
              join(s.modelsDir, 'mel_meta.json'))
            const r = await run({
              input, output, out16k: args && args.compatOutput ? String(args.compatOutput) : undefined,
              ort, sessions, melBasis,
              log: (m) => lines.push('  ' + m),
            })
            // ⚠️ 这里原来是用 `+ '\n' +` 拼的 —— 而我在生成这段代码时把转义写错，
            //    结果**裸换行**落在 JS 字符串里（那不是合法字符串）。
            // → 改模板字符串：多行本来就合法，而且读起来是一整块。
            return {
              ok: true,
              message: `换好了：${output}\n${lines.join('\n')}\n`
                + `  耗时 ${r.ms.total} ms（读 ${r.ms.read} + hubert ${r.ms.hubert}`
                + ` + rmvpe ${r.ms.rmvpe} + net_g ${r.ms.netg}）\n`
                + `  时长 ${r.seconds.toFixed(2)} 秒（p_len ${r.pLen} · 有音高 ${r.voiced}/${r.nFrames} 帧）`
                + (r.compatOutput ? `\n  已附加 16k ASR 音频：${r.compatOutput.output}` : ''),
            }
          } catch (e) {
            const msg = String(e && e.message || e).slice(0, 400)
            // 那句 onnxruntime 的 "cannot be reshaped" 看起来很吓人，而它其实只说明
            // **这份导出里写死了形状**（多半是六输入那张图）—— 换文件就好，不用改代码。
            const hint = /Reshape|reshaped/i.test(msg)
              ? '\n\n⚠️ 这句是"图里写死的形状"，不是模型坏了 ——'
                + '换一份音色（七输入那份，输入名里有 noise_1527）通常就好了。'
              : ''
            return { ok: false, message: '转换失败：' + msg + hint }
          }
        }
        if (action === 'setup') {
          const miss = missingGroups(s.models)
          if (!miss.length) {
            return { ok: true, message:
              `要的东西都齐了（音色在 voices/ 里，按需要加）。数据目录：${s.home}` }
          }
          const lines = s.models
            .filter((m) => miss.includes(m.group) && !m.optional)
            .map((m) => `  ${m.file}  （约 ${m.mb} MB）—— ${m.what}\n      放：${m.path}`)
          return {
            ok: false,
            message: `缺 ${miss.length} 组：${miss.join(' / ')}\n${lines.join('\n')}\n\n`
              + '⚠️ 而这几件**都不是下载来的现成文件** —— 它们是从 RVC 的权重转出来的：\n'
              + '  hubert ← assets/hubert_base（transformers 格式）\n'
              + '  rmvpe  ← assets/rmvpe/rmvpe.pt（转 **fp32** —— fp16 那份在 Node 下加载就崩）\n'
              + '  mel 表 ← 跟着 mel.mjs 一起生成的（257 KB）\n'
              + '  而音色（net_g）不在这儿 —— 它是 voices/<名字>/<名字>.onnx（**七输入**那份，106 MB）。\n'
              + '  转换脚本和踩过的坑在 README 里。',
          }
        }
        // 用量先把"在同一组里被后一份顶掉的那件"排除掉 —— 否则 rmvpe 两份会被算两次。
        const inUse = s.models.filter((m) => m.present
          && !(m.optional && s.models.some((x) => x.group === m.group && !x.optional && x.present)))
        const total = inUse.reduce((a, m) => a + m.actualMB, 0)
        const lines = s.models.map((m) => {
          const mark = !m.present ? '❌' : (inUse.includes(m) ? '✅' : '▫️')
          const size = m.present ? `${m.actualMB} MB` : `（缺，约 ${m.mb} MB）`
          return `  ${mark} ${m.file.padEnd(20)} ${size.padEnd(16)} ${m.what}`
        })
        const vlines = s.voices.length
          ? s.voices.map((v) => `  ${v.name.padEnd(16)} ${v.mb} MB  ${v.onnx.join(', ')}`)
          : ['  （还没有音色）']
        return {
          ok: true,
          message:
            `声音与视频 · 数据目录 ${s.home}\n\n`
            + `模型（在用的 ${total.toFixed(1)} MB）：\n${lines.join('\n')}\n\n`
            + `音色（${s.voices.length} 个）：\n${vlines.join('\n')}\n\n`
            + '⚠️ 而"每件多大"和"能不能再小"我都量过：\n'
            + '  int8 量化 hubert → 相关 0.2032（算错，不是有损）\n'
            + '  int8 量化 rmvpe  → 20.6% 的帧音高错超半个半音\n'
            + '  自己的 YIN 换 rmvpe → 73% 的共同帧差超半个半音\n'
            + '  rmvpe 换 fp16    → **加载即崩**（原生 segfault，拦不住）\n'
            + `  → 所以在用这些是"不阉割"的代价。加运行时（实机用到的那几份约 66 MB）`
            + (s.voices.length
              ? `，再加音色 ${s.voices.reduce((a, v) => a + v.mb, 0).toFixed(1)} MB`
              : '')
            + ` ≈ ${(total + 66 + s.voices.reduce((a, v) => a + v.mb, 0)).toFixed(0)} MB。\n`
            + '  ⚠️ 这一行原来写的是"音色每个 54 MB / 共 608 MB" —— 那是 **fp16 六输入那份**的数，\n'
            + '     而那份**跑不起来**；真用的是 106 MB 的七输入那份（+21 MB 索引）。',
        }
      },
    })
    log('def 的 parameters 是 JSON Schema 吗 = '
      + (def.parameters && def.parameters.type === 'object' && def.parameters.properties
        ? '是（有 type/properties）' : '**不是** —— 那替换身没编译'))
    try {
      const reg = () => {
        const r = tools.register(def)
        log('register 调用成功 · 返回 ' + typeof r)
        return r
      }
      if (effect) effect(reg, 'dsh-av tool')
      else reg()
    } catch (e) {
      log('**register 抛错**：' + (e && (e.stack || e.message) || e))
    }
    // 回查一次 —— "注册调用没报错"和"真进了工具表"是两件事。
    try {
      if (typeof tools.get === 'function') {
        log('回查 tools.get("av")：' + (tools.get('av') ? '**在**' : '**不在**'))
      }
      if (typeof tools.schemas === 'function') {
        const all = tools.schemas() || []
        log('回查 tools.schemas()（全局视角）：' + all.length + ' 个工具 · 含 av 吗 = '
          + all.some((s) => s && s.name === 'av'))
      }
    } catch (e) {
      log('回查失败：' + (e && e.message))
    }
  } else {
    log('→ 没有可用的 tools 服务，跳过注册')
  }

  // ── 前端 ────────────────────────────────────────────────────────────
  // ⚠️ 这里原来有一段 `web.register({ id, title, view: null })` —— **删掉了**。
  //
  //    那是照"插件大概该这样写"编出来的：宿主里**没有一个"注册视图"的 web 服务**，
  //    真正有的是 `webServer`（dsh-mood 那条路是 `ctx.get('webServer')` + 注册一条路由）。
  //    而一个编出来的 API 比"没有前端"更糟 —— 它会让**整个插件起不来**。
  //
  //    → 等真做前端（client.js）时，照 dsh-mood 那条路重写，别猜。
}

export { HOME, MODELS, VOICES }
