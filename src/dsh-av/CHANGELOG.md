# 变更记录

## 0.4.5 —— 外部又给了一份"人声优化方案"，逐条核过，修了它指出的两个

那份方案在 `dsh-av/音视频人声优化方案.md`，**我的逐条核对写在它结尾**（"核对"那一节）。
这里只记**落进代码的两条**：

```
① convert.js：**不再"没有 fp32 就退回 fp16"**
   原来：const rmvpeFile = existsSync('rmvpe.onnx') ? 'rmvpe.onnx' : 'rmvpe_fp16.onnx'
   ⚠️ 而 fp16 那份在 onnxruntime-node 下**加载就 segfault**（原生崩溃，try/catch 拦不住）——
   所以"悄悄退回"的后果是**整个宿主进程挂掉**，比"报一句缺文件"严重得多。
   → 现在缺 fp32 就直接报错，并把"为什么不能用 fp16"写在报错里。
   （这条是那份方案里**唯一一个真 bug**，它说对了。）

② render.js：帧数 **floor → ceil**
   实测：5.98 秒 @12fps，floor 给 71 帧 = 5.9167 秒，而拼 mp4 带 `-shortest`
   → **音频被切到视频长度：5.980 → 5.8979 秒（少 82 ms）**。
   改 ceil（72 帧）之后，成片音频流 **5.980000 秒**，跟源一模一样。
```

**顺手做的一次实测（回答它"长音频会崩"那条）：**

```
59.8 秒整段（不分块）→ 跑通 · 2391200 点 @40000 Hz · **34901 ms**（net_g 占 25106）
尾部差 20 ms —— 是 p_len 取整，不是被裁
→ **分块不是现在的问题**；以后要分块的真正理由是 enc_p 注意力随帧数平方长
  （60 秒 → p_len 5978 → 约 3570 万），不是"内存压力/接缝"。
```

**它说对、我数过了的：** 训练切片 filelist 36 行 / `3_feature768` 37 个 ✅ ·
`train.log` 末行 `轮次：200` 而目标是 20000 ✅ · 导出是六输入 ✅。
**它漏掉的一条承重：** 喂噪声前的那个 `/0.66666`（0.9751 / **0.9997** / 0.9499）。

## 0.4.4 —— 宿主把凶手报出来了：`{ type: 'json' }` 不是合法 JSON Schema

**0.4.3 加的 `load.log` 直接交出了那句话**（这一轮的意义全在这几行）：

```
defineTool：import 失败（Cannot find package '@deepseek-ai/dsh-tools' …）→ 用兜底
tools 服务：有 · register=function · get=function · schemas=function
**register 抛错**：JsonSchemaError: unsupported JSON schema:
    schema.type must be one of object/array/string/number/integer/boolean/null
    at assertSupportedJsonSchema (…/@deepseek-ai/dsh-tools/lib/index.js:326:35)
    at Proxy.register                    (…/@deepseek-ai/dsh-tools/lib/index.js:2882:3)
回查 tools.get("av")：**不在**
回查 tools.schemas()（全局视角）：33 个工具 · 含 av 吗 = false
```

**凶手是我自己写的一行 —— 而且是照 dsh-mood 抄的：**

```js
output: { schema: { type: 'json' }, render: … }      // ← 'json' 不是 JSON Schema 的合法类型
```

```
· 真身 defineTool **会替你规范化**这一句 → 所以 dsh-mood 一路没事
· 而我这边 import 失败、走的是**兜底**（只编译 parameters，不管 output.schema）
  → register 抛 JsonSchemaError → **工具进不了工具表**（客户端什么都看不见）
```

**⚠️ 教训（比这一行代码值钱）：照抄一个能跑的例子，得先问"它凭什么能跑"。**
dsh-mood 能跑不是因为这个写法对，是因为**它拿到的是真身**。我把"能用"当成了"写法对"。

**改了什么：**

```
① 兜底新增 fixSchema()：把不合法的类型名修掉（'json' → 'object'），
   `output.schema` 和 `parameters` 都过一遍
② resolveDefineTool 加第二条路：**直接 import app.asar 里那份真身**
   （宿主 register 的报错栈证明了那条路径能 import）——
   拿不到才用兜底。日志会写明走的是哪条。
③ **本地自测加了一道"schema 合法性"检查**，而且是在注册**之前**跑：
   遍历 parameters + output.schema，任何一个 type 不在那七个合法值里就报错退出。
   → 判据：**本地该比宿主更严，不能更宽。**
     （之前两次"本地一路绿灯、装进宿主就死"——"假 ctx 比真 ctx 宽容"和这次的
      "替身不做校验"——是同一个毛病：**我的替身验的是替身，不是宿主。**）
```

**本地验过：**

```
✓ 注册上了：av · ✓ schema 合法（parameters + output.schema 都过）· ✓ status ok
load.log：import 失败 → 没有 process.resourcesPath（不在 Electron 里）→ 用兜底 → register 成功
```

✅ **真机验证通过（这一夜唯一一次"宿主自己开口"之后就干净的）** —— 重启后 `load.log`：

```
defineTool：**从 app.asar 直接拿到了真身** → …app.asar/dsh/node_modules/@deepseek-ai/dsh-tools/lib/index.js
tools 服务：有 · register=function · get=function · schemas=function
register 调用成功 · 返回 function
回查 tools.get("av")：**在**
回查 tools.schemas()（全局视角）：34 个工具 · 含 av 吗 = **true**
```
→ 走的是第 ② 条路（直接 import app.asar 里那份真身），**连兜底都没用上**。
会话里 `av` 进工具表了，真活也跑通：
`av {action:'convert', …}` → 40000 Hz · 239200 点 · 5.98 秒 · 4507 ms，并附带那份 16k 给 dsh-mood。

## 0.4.3 —— 起来了，但工具没进工具表：本地替身 shadow 了宿主

**0.4.2 修完之后重启，结果分两半：**

```
✅ `fiberPhase: **active**` —— 插件真的挂上了（上一轮是 failed，那行 ctx 解构是致命的）
❌ 工具表里**没有 `av`** —— 我自己的清单没有，**新开的子代理（全新 scope）也没有**
   → 所以不是"会话的工具表被冻结"，是**注册根本没成**
```

**查到的原因（两条，前一条是主因）：**

```
① `src/node_modules/@deepseek-ai/dsh-tools` 里放的是**测试替身**（734 字节，自己写着"本地测试桩"）。
   ⚠️ 而**宿主会注入真的那份** —— 证据：
      · 真的 `@deepseek-ai/dsh-tools` 在磁盘上**根本找不到**（只在 app.asar 里，字符串出现 197 次）
      · dsh-mood 在 profile 里的副本**连 node_modules 都没有**，而它的工具全都在 ✅
   → 于是我这份替身**shadow 了宿主注入的真身**。
   ⚠️ 而替身比真身**少做一件事**：它只做参数校验，
      **不会把 `parameters` 规格编译成 JSON Schema**（真身会，替身自己的注释就写着）。
   → 注册出去的东西 `parameters` 不是合法 JSON Schema —— **很可能因此进不了工具表，而且不报错。**

② 顺带把"回查"加上了：`register()` 不抛错 ≠ 真进了工具表。
   （工具表的 Service 契约里有 `get(name, scope)` 和 `schemas(scope)` —— 那才是能问的地方。）
```

**改了什么：**

```js
// ① 不再静态 import —— 静态 import 失败 = 整个模块加载失败 = fiber failed
const m = await import('@deepseek-ai/dsh-tools')   // 动态 + try/catch
// ② 拿不到就用**兜底实现**，而兜底**会编译 JSON Schema**（替身缺的就是这一步）
// ③ 入口改成"同步外壳 + async boot"，**注册里出的任何错都不再能搞挂 fiber**
// ④ 整个装载过程写进 <数据目录>/load.log —— 宿主里的报错我看不见，这是唯一能读的出口
```

**load.log 里会写清楚这几件事**（重启后一看就知道卡在哪）：

```
defineTool：拿到了 · 是本地替身吗 / 是宿主注入的真身 / import 失败 → 用兜底
tools 服务：有吗 · register / get / schemas 各自的类型
def 的 parameters 是 JSON Schema 吗 = 是 / 不是
register 调用成功 · 返回 …
回查 tools.get("av")：在 / 不在
回查 tools.schemas()（全局视角）：N 个工具 · 含 av 吗 = true/false
```

**本地验过（替身已挪到 `_stub-backup/`，不在解析路径上了）：**

```
node test-smoke.mjs status  → 注册上了 ✅ · status ok ✅
load.log → import 失败 → 用兜底 → parameters 是 JSON Schema ✅ → register 成功 ✅
（本地当然 import 失败 —— 真身在 app.asar 里，那不归 Node 的解析管）
```

⚠️ **而"宿主里到底走哪条路、最后进没进工具表"，仍然要下一次重启才知道** ——
但这次不管成不成，`load.log` 都会说出原因，不用再猜。

## 0.4.2 —— 装上之后起不来，原因是一行 `const { tools } = ctx`

**症状：** 装进 profile、重启宿主 —— 插件列表里 208 条**有它**，`enabled: true`，
而那一行写着 **`fiberPhase: failed`**；工具一个都不出现（会话里没有 `av`）。
界面上完全看不出为什么（`list_plugins` 只给状态，不给错误）。

**原因：** 拿到错误的那一下才明白 —— 是 `plugin_manager` 的 set_plugin 把诊断带出来了：

```
dsh-av (dsh-av): Error: cannot get property "tools" without inject
    at new dshAv (file:///D:/DSHzuoye/dsh-av/src/index.js:101:11)
```

```
cordis 不让插件**直接读服务属性**（`ctx.tools`）—— 那要求你声明 `inject`。
而这一行：
    const { tools, web, effect } = ctx
不但读了 `ctx.tools`，还顺手读了 `ctx.web`（**这个服务根本不存在**）——
**抛在插件体里 = 整个插件挂载失败。**
```

**修法**（照 dsh-mood 抄的，它是这台机器上唯一一个跑着的参照）：

```js
const tools = ctx.get ? ctx.get('tools') : ctx.tools   // ← ctx.get 不需要 inject
const effect = ctx.effect
```
**并且把那段 web 注册整个删掉** —— 那是照"插件大概该这样写"编的：
宿主里没有一个"注册视图"的 web 服务，真正有的是 `webServer`
（dsh-mood 那条路是 `ctx.get('webServer')` + 注册一条路由前缀）。
**一个编出来的 API 比"没有前端"更糟：它会让整个插件起不来。**

**⚠️ 而这一条最该记住的是"为什么自测没抓到"：**

```
我的 test-smoke.mjs 那时候给假 ctx 是这样的：
    mod.default({ tools: { register }, web: null })
——**比真 ctx 宽容**：它直接给了 `tools` 属性，于是那行解构不会报错。
→ 自测一路绿灯，装进宿主就 failed。
→ 现在那个假的 ctx 改成**跟宿主一个形状**：只给 `get` / `effect`，**故意不给 `tools` 属性**。
   （判据：**自测的替身若比真的宽容，它验的就只是替身。**）
```

**还有一个坑，是这次才撞见的：**

```
**宿主不会因为你改了源码就重新 import 那个模块。**
实测：改完之后 disable → enable 一遍，报的还是**同一个行号（101:11）**——
而我改完那句早就不在 101 了（现在是注释）。
→ 说明它跑的是进程里缓存的那份 ESM，engine 层的 disable/enable 不会重新 import。
→ **只能重启宿主。**（这也解释了为什么"改完没生效"看起来像"我的修改没写对"。）
```

**验证方式：** 本地用**宿主形状**的假 ctx 跑通（注册 + execute 都过），
真机上要等下一次重启才算数 —— **这条到现在还没有真机证据**。

## 0.4.1 —— 源码交给 GPT 优化了一遍，然后我逐条把的关

**动了三个文件**：`convert.js` · `index.js` · `render.js`。**其余 11 个文件逐字节没动**
（`netg.mjs` / `mel.mjs` / `hubert.mjs` / `voices.js` / `wav.js` / `png.js` / `package.json` / `cordis.patch.yml` …）——
**第三节那 8 条"承重"的东西一条都没被碰**，这是我拿到改动后先看的地方。

### ① 新增 `listen` action —— 输出 16k 单声道，给 dsh-mood 的耳朵

```
dsh-mood 的 ASR 只收 16000 Hz（不是就直接退出），
而主输出是 40000 Hz —— **它是对的，改了会变调**（音色是 40k 训的）。
→ 所以补的不是"改采样率"，而是"多一个出口"：

av { action: 'listen',  input: 'xxx.wav', output: 'xxx.16k.wav' }
av { action: 'convert', input, voice, output, compatOutput: 'xxx.16k.wav' }   ← 转换时顺手多落一份
```
走的是已有的**抗混叠** resample（`hubert.mjs` 那个 Kaiser-sinc），不是线性插值。

### ② `video` 的配色 / 布局变成可调 —— 原来硬编码的七个常量 + 一组几何比例

```
preset: 'default' | 'warm' | 'cool' | 'violet'
taste:  '{"preset":"cool","colors":{…},"geometry":{"localWidth":0.7}}'    ← JSON 字符串，覆盖 preset
```

**参数化后的两组名字**（这就是"品味 → 生成"要接的那排旋钮）：

```
colors    bg · dim · wave · waveDim · waveFuture · spectrum · accent · boxFill · boxEdge · thumb · progress
geometry  pad · midY · wholeAmp · windowAmp · spectrumTop · spectrumHeight · localWidth
          · windowSeconds · spectrumBars · spectrumDb
```

⚠️ **不传 = 原来的画面，逐字节相同。** 这条验过：同一段音频、同一组参数，新旧 `render.js` 各画 23 帧，
**23/23 帧 md5 一致**。所以"可调"没有顺手改掉 0.3.0 里逐版调出来的东西（整段归一化的频谱、
收窄的局部框、去掉的那道"你在这"竖线）。
⚠️ 这里是**旋钮**，不是"心情 / 品味"本身 —— 那一层归 dsh-mood；本插件**不 import 它**。

### ③ 两个小修（不是 GPT 改的，是我把关时补的）

```
· fps / width 钳位：原来 `Number('abc')` 是 NaN，而 Math.max/min **挡不住 NaN** ——
  它会一路传到 ffmpeg 的 `-framerate NaN`，报一句跟参数无关的错。
  → 先判有限性再钳位，非法值退回默认（24 / 1280）。fps 1~60 · width 320~3840。
· `preset` 什么都没传时报的是 'custom' → 修成 'default'（画面没变，是名字报错了）。
```

### 要跟 dsh-mood 说一声：那条断掉的接口现在接上了

```
dsh-av 主输出 40000 Hz —— **没改，也不该改**
新增 16k 出口：av { action:'listen' } 或 av { action:'convert', compatOutput }
```

### 验过的

```
status · setup · listen · convert · convert+compatOutput · video（4 个 preset + 坏 JSON）—— 全跑过
convert 端到端：跟 torch 那份 **仍然是 0.99799903**（改了 convert.js 之后同一个数）
  → 说明那条 out16k 分支没碰到主路
video 默认：71 帧 640×360 → mp4 **404 KB**（跟改动前同尺寸）
默认画面：23/23 帧逐字节相同
坏 taste JSON：干净地报「taste 必须是合法 JSON」，不吞
```

**⚠️ 这次外部优化只改了代码，没写文档** —— 上面这一节是补的。

## 0.4.0 补记 —— 回头同步旧对话那一遍（同一晚）

**把上一轮留下的产物对了一遍，找出三处"文件跟结论不一致"的地方：**

```
① **`重采样对照-我的版.wav` / `重采样对照-ffmpeg版.wav` / `插件产出-试跑.wav`
   的头写的是 16000 Hz，而它们装的是 239200 点（40000 Hz · 5.98 秒）。**
   —— 正好是报告里那条"单位错"的残留：**代码修了，文件没重写。**
   证据：`重采样对照-我的版.wav` 与 `chain_mine.f32` 逐样本相关 **1.00000000**。
   → 用 `fix-wav-rate.mjs` 把头改成 40000（只动 8 个字节，样本一个没碰）。
   ⚠️ 在那之前，这两段"拿耳朵判重采样"的对照，放出来是**慢 2.5 倍**的。

② **"逐样本比"在这一层到底有没有意义 —— 现在有数了。**
   把同一段输出跟它自己错开几个样本：
     错开  2 个样本（50 µs） → 相关 0.85
     错开 21 个样本（0.5 ms）→ 相关 0.57
     错开 55 个样本（1.4 ms）→ 相关 **−0.20**
   → 40 kHz 下这段波形的**相干长度约 1 ms**。所以两条链的输出相关 0.44 / −0.20
     **不是"谁算错了"**，是相位滑开了 —— 逐样本相关量的正是相位对齐。
   → **它只在输入逐位相同时才有意义**（那才是 0.998~1.0 那一档的来由）。

③ **golden 里那两组 16k 其实是两段不同的内容**（别交叉比）：
     `audio16k` + `audio16k_ffmpeg2`（互相关 0.9878）是一组，
     `audio16k_mine` + `audio16k_ffmpeg`（互相关 0.9990）是另一组，
     而两组之间相关 ≈ 0（最佳 lag 上也只有 0.036）。
   → 这一轮我先拿 B 组的音频配 A 组的噪声跑了一遍，追了半小时才发现"参考根本不属于这个输入"。
```

**新增的工具（都在 `dsh-av/`）：**

```
test-smoke.mjs           不走宿主，直接把工具定义拿出来跑（status/setup/convert/video）
src/probe-layers.mjs     一层一层跟 golden 对 —— 哪一层掉了，差异就在那一层
src/probe-netg.mjs       单独验 net_g，含那个 /0.66666 该乘还是该除
src/verify-chain.mjs     整条链（可选喂 golden 噪声）→ 跟 torch 那份比
src/verify-resample.mjs  重采样的 A/B：只换重采样器，噪声钉死
compare-wav.mjs          wav vs f32/wav —— 相关 + 绝对差 + 相对最大差
probe-phase.mjs          输出的"相位相干长度"（逐样本比有没有意义，看它）
probe-resample.mjs       两组 16k 是不是同一段内容（最佳 lag 相关）
fix-wav-rate.mjs         只改 wav 头里的采样率，样本不碰
```

## 0.4.0 — 真的装进宿主了（而这一轮栽的两次都是"参考不是为这个输入算的"）

**新加的：**
- `cordis.patch.yml` + package.json 里的 `dsh.bundle.patch` —— **宿主判"这是不是一个 bundle"看的就是它**。
  少了这个字段，`install_bundle` 报的是 `not-bundle`（而 pnpm 那一步是成功的，所以很容易读成"装上了但没生效"）。
- 装进 `desktop` profile：**用 `link:` 不用 `file:`**。
  ⚠️ pnpm 对 `file:` 的**目录**依赖是**复制**一份进 node_modules ——
  于是"改了源码、node_modules 里那份没变"，而症状正好是上面那个 `not-bundle`。
  `link:` 是软链，改一处就是一处（这也是 profile 里 `@dsh-pet/bridge` 用的那种）。
- 数据目录 `~/.dsh/dsh-av`（**不复制模型**）：
  ```
  models             → junction 到 D:\DSHzuoye\dsh-av\models（699 MB 原地不动）
  voices\kizhi\      → kizhi.onnx（106 MB · 七输入）+ kizhi.index（21 MB）
  ```
- 插件自己的 `node_modules/onnxruntime-node` → junction 到 `D:\rvconnx\node_modules\onnxruntime-node`。
  **一条 junction 就够** —— Node 默认按 **realpath** 解析模块，
  所以它自己的依赖（onnxruntime-common / global-agent / adm-zip…）会从 `D:\rvconnx\node_modules` 里找。

**⚠️ 音色文件一开始装错了 —— 而那个错值得记住：**

```
装的是 kizhi_fp16.onnx（54 MB）→ 一跑就崩：
  Non-zero status code returned while running Reshape node ...
  Input shape:{1,2,715407}, requested shape:{1,2,599,1195}
而它是**六输入**那张图（输入名里有 rnd —— 那是 onnxlib 导的），
那个 Reshape 的形状是**导出时那段 5.98 秒音频**写死的 → 只有那一种长度不崩。
换成 kizhi_det.onnx（**七输入**：phone/phone_lengths/pitch/nsff0/sid/noise_1527/noise_random_2110）就好了。
```
→ 现在开音色会话时就查一次 `inputNames`：**有 `rnd` 就提前说出来**，
   并且把 `cannot be reshaped` 那句话在结果里翻译成"图里写死的形状，换一份音色"。
   （原来那句报错读起来像"模型坏了"，而它其实只说明"这份导出只对一段长度有效"。）

**验证（这一轮的数，全部可复现）：**

| 层 | 跟 golden 比（**同一段输入** `audio16k.f32`） |
|---|---|
| 波形 | **1.00000000** |
| hubert 特征 | **0.99999917** ← fp16 的代价，跟 0.2.0 记的同一个数 |
| mel | **1.00000000** |
| rmvpe hidden | **1.00000000** |
| f0（601 帧） | **1.00000000** |

```
net_g 单独（kizhi_det + golden 的 feats/pitch/pitchf/rnd）：
  喂 rnd 原样        → 对 netg_audio 0.975136 · 对 netg_audio_det **1.000000**
  喂 rnd / 0.66666   → 对 netg_audio **0.999730** ← 复现 0.2.0 那个数（torch 那份是它）
  喂 rnd × 0.66666   → 0.949880（更差，方向也印证了）→ **那个 /0.66666 是对的，不用改**

整条链（golden 16k 输入 + golden 噪声）→ vs torch 那份：**相关 0.99799903**
```
⚠️ 0.998 而不是 0.9997，差在端到端多出来的两处：**输入被量化成 16 位**、**hubert 用 fp16**。

**⚠️ 这一轮栽的两次，是同一类：拿"不是为这个输入算的参考"去比。**

```
① 先拿 `wav16k_for_plugin.wav` 当输入比 → **连波形那一层都只有 0.9878**。
   那个 wav 是 19:55 转出来的，比 golden 那批参考（19:39~19:52）晚 —— 是另一次重采样的产物。
   **判据很短：波形层就对不上 → 问题在输入，不在任何一层网络。**
② `chain_mine.f32` / `chain_ffmpeg.f32` 我当成"整条链的参考"追了很久，只到 0.41 / 0.31。
   后来把它们**互相比了一下：它们俩之间也只有 0.4386** ——
   那就说明它们不是同一组噪声跑出来的，**拿它们当参考本身就不成立**（不是"我算错了"）。
   （顺带看清了 golden 里的重采样关系：`audio16k` vs `audio16k_ffmpeg` 0.9878、
     而 `audio16k_ffmpeg` vs `audio16k_mine` **0.9990** ——
     也就是 README 里那条"基准是错的：golden 那个 16k 是我用线性插值转的"。）
③ 还有一次是**工具自己栽的**：`Math.max(...arr)` 在 24 万个元素上爆调用栈 ——
   报的是 `RangeError: Maximum call stack size exceeded`，位置在"算参考幅度"那一行。
   → 展开成实参的写法，**长度一上去就不是同一个函数了**。
```

**四个 action 都真跑过一遍：**

```
status   5 ms      三件模型 525.5 MB（在用的）+ 1 个音色
setup    —         缺什么、该放哪、多大
convert  5293 ms   16k wav → 40k wav（纯算 3050：hubert 230 + rmvpe 345 + net_g 2445）
                   239200 点 @ 40000 Hz · p_len 598 · 有音高 197/601 帧
video    793 ms    71 帧 @ 12fps · 640×360 → mp4 404 KB（h264 + aac · 5.92 秒）
```

**体积（这一版实测）：**

```
插件本体（src/）               **99.8 KB**
models（hubert 180.3 + rmvpe 344.9 + mel 0.3）  525.5 MB
音色（kizhi.onnx 106 + index 21）               127 MB
onnxruntime（那个目录 288 MB —— 里面是三个平台的二进制，
             实机只用到 win32\x64 那几份 **66 MB**，其中一半是 DirectML/dxcompiler 那两坨）  ~288 MB
──────────────────────────────────────────
装上要占                                   **约 940 MB**
```
⚠️ 上一版记的"434 MB"是 **rmvpe 用 fp16** 那笔账 —— 而 fp16 那份在 Node 下加载就崩，
   所以真实的账是 fp32 那份（+172 MB）。**那个 434 是错的口径，这次改过来。**

## 0.3.0 — 视频那块接进来了

**新加的：**
- `png.js` —— 纯 JS 写 PNG（`zlib` + 三个 chunk）。
  ⚠️ 每行前面那个 **filter 字节**最容易忘 —— 忘了图会整体错位，
  **而且不报错**（看起来像"画错了"，不像"格式错了"）。
- `render.js` —— 音频 → 逐帧 PNG。三层：整段轮廓 / 局部波形 + 频谱 / 进度条。
- `av` 工具加了 **`video`** action（wav → mp4）。

**⚠️ 画面改了三版 —— 而每一版都是"看了才知道"的：**

```
① **频谱按"整段"归一化**（原来是每帧按自己的最大值）
    → 每帧都撑满同一个 70dB 范围，**"绝对音量"整个丢了**：
      大声和安静的两帧柱形一样高，而频谱柱的用途恰恰是让人看出音量起伏。
    → 改成两遍：先收原始幅度记下全局最大，再统一做 dB 参考。
       （用"全局一个最大"而不是"每根柱各自最大"—— 后者会把"低频比高频强"也抹掉。）

② **局部那个框收窄到刚好包住内容**（原来是 H*0.06~H*0.84，比内容大一圈）
    → 安静的段落里**框几乎是空的而边框很清楚**，读起来像"有个东西没加载出来"。

③ **去掉那道"你在这"竖线**
    → 它跟底部进度条标的是同一件事（整段里的位置），而它跟局部框的 x 坐标
      **不是一回事**（框是固定的、位置在滑动）。结尾那帧里竖线跑到最右边、框还在中间，
      读起来像"框没跟上"。**同一个信息给两个不一致的表示，比不给更糟。**
```

**验过的：**
- PNG 编码器：写一张，用 PIL 读回来对四个点的颜色 —— 全中 ✅
- 端到端：143 帧 · 1903 ms（纯 JS 画）→ mp4 **1.03 MB** · 1280×720 · 24fps · 5.958 秒 ✅

**⚠️ 那一轮栽的两次 —— 症状都像"模型的问题"，而两次都是环境：**

```
· 把 bash 的 `/d/rvconnx/...` 路径喂给 Node —— 它不认那个（ESM 要 file:// 或者相对路径）。
  于是六个"加载失败"全是路径错，而我第一反应是"这个 fp16 模型不行"。
· 修了路径，又栽在 `node -e` 的 cwd 上（不在 /d/rvconnx → 找不到包）。症状还是像模型的问题。
→ **fp16 重转六种配置全失败**（op17 那些是 segfault，op13/14/15 是"我改错了版本号"）。
  **结论：rmvpe 在 onnxruntime-node 下只能用 fp32。**
```

**清理：**
- 删了 `models/rmvpe_fp16.onnx`（172.5 MB）—— 删之前又确认了一遍它确实是加载就崩的那个。

## 0.2.0 — 三段 JS 搬进来，整条链在 Node 里跑通

**新加的：**
- `hubert.mjs` —— 波形 → 768 维特征；含**带抗混叠的**重采样（Kaiser 窗 sinc 插值）
  —— 因为 RVC 用的是 ffmpeg，而线性插值会把高频折回来
- `netg.mjs` —— 特征 + 音高 + 噪声 → 波形；按 `inputNames` **自动适配两种接口**
- `voices.js` —— 音色目录管理；`modelStatus()` 会拿实测大小判"是不是下了一半"

**整条链跑通了 —— 6 秒音频、纯 CPU、纯 JS：**

```
ffmpeg 版 16k：特征 299×768 · f0 601 帧 · p_len 598 · 波形 239200 点 · **3714 ms**
我的 sinc 版：特征 299×768 · f0 601 帧 · p_len 598 · 波形 239200 点 · **4060 ms**

耗时拆分：hubert 282 + rmvpe 707 + net_g 2725  →  **net_g 占 73%**
```

**各段分别验过的（跟 Python 比）：**

| 段 | 差 | 说明 |
|---|---|---|
| mel | 相对 **4.50e-05** | 拿绝对差判是错的（log 域里小值上它天然大） |
| rmvpe | **音高判定零差异 · 共同帧 0 帧差超半音** | f0 最大差 9.86e-05 Hz |
| hubert | fp32 下相关 **1.00000000** | fp16 下 0.99999917（那 0.3% 是 fp16 的代价） |
| net_g | 相关 **0.99973032** | 系数对齐之后；之前是 0.97513601 |

**⚠️ 两个坑，都花了时间：**

```
① **那个 0.66666** —— RVC 原版写的是
     z_p = (m_p + exp(logs_p) * randn_like(m_p) * **0.66666**) * x_mask
   而噪声被提成图输入之后，**那个系数仍然留在图里**。
   参考实现（onnxlib 的 forward）**不乘** —— 于是两边"实际作用在 z_p 上的噪声"差 1.5 倍。
   实测：喂同一个 rnd → 相关 0.9751；喂 rnd/0.66666 → **0.9997**；喂 rnd*0.66666 → 0.9499（更差，方向也印证了）

② **两种接口不能混** ——
   六输入那张图（onnxlib 导的）里有个写死的 Reshape：requested {1,2,599,1195}，
   **Python 和 Node 两个 runtime 都崩**；
   能跑的是七输入那张（rvc-onnx-web 转的 + 噪声提成输入）。
   → 现在按 `inputNames` 自动适配。
```

**⚠️ 而"重采样"那一环还没定论：**

```
波形层：我的 sinc vs ffmpeg，相关 0.99895 · 相对最大差 10.2%（**最佳 lag = 0，所以不是错位**）
特征层：同一对音频过 hubert，相对最大差 **45%**（个别点）
最终层：整条链跑两遍，相对最大差 130% —— **而那个数不能这么读**：
        两遍各自算 f0，f0 差一点会让波形相位错开，**逐样本比"两段不同的音频"本来就会出大数**。
        跟"onnx vs torch"那次不一样 —— 那次两边**喂的是同一组输入**，所以 6.65% 有意义。
→ 判据只能是听：`品味-临摹\实验\重采样对照-ffmpeg版.wav` / `重采样对照-我的版.wav`（各 14.95 秒）
```

**⚠️ 这一轮栽的那几次（全是同一类）：**

```
· 重采样的第一个比对里，我拿"golden 那个 16k"当基准 —— 而那个**不是 ffmpeg 转的**，
  是我在 golden.py 里用**线性插值**转的。**基准是错的，那两版结果全部不算数。**
  （还有一个迹象我当时没读出来：加大滤波器半宽 8→64 **完全不改善** ——
   那不是"抽头不够"，是"跟一个错的基准比"。）
· 更早：`golden_netg.py` 里我**没钉 NSF 那处噪声**，torch 自己抽的，而 onnx 那边喂的是我的
  → 两个用的不是同一组噪声，**逐样本比根本没意义**。
· `console.log` 里用 `%-10s`（Python 的格式符）—— Node 不认，参数整体错位，
  打出"特征 NaN×768 · f0 299 帧 · p_len 601 · 波形 598 点 239200"那种两个数粘在一起的东西。
→ **共同点：看到数就往下走，没先确认"这个数是从什么来的"。**
```

**还没做的：**
- 检索（Faiss index）—— 它在 net_g **外面**，影响"像不像"而不是"算得对不对"。
  现在没接，等于 index_rate=0 那条路。
- 前端（`client.js`）
- 视频渲染（还在 Python 那边）

**体积：**
```
插件本体（这一层）            **41.3 KB**
三件 onnx（fp16 那档）        406.3 MB
onnxruntime（CPU）            ~28 MB
─────────────────────────────
装上要占                    约 434 MB
```
