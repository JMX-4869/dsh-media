# dsh-av — 声音与视频

DeepSeek Harness 插件。**纯 JS 的音色转换**（不依赖 Python、不依赖 torch），加上把音频渲染成片子。

---

## 一、它做两件事，而这两件是同一件事的两头

```
**声音**   把一段音频的"音色"换成某个人的
**视频**   把音频渲染成片子（频谱 / 波形 / 歌词 / 曲线）
```

而"不依赖 Python"这条不是洁癖，是一次实测的结果：

```
第一次练音色用的是 RVC 那一整套 —— 装完 **11 GB**（torch + CUDA 占 7.4 GB）。
而后来量出来：**转换那一步根本不需要 torch**。
   `rvc-onnx-web`（纯 TypeScript，MIT）**1.12 秒**把一个 52.6 MB 的 .pth 转成了 onnx。
而推理这边，onnxruntime 的实际二进制（CPU）是 **28 MB**。
→ **"用得上"和"拖得动"是两件事。**
```

---

## 二、三件模型：各自多大、干什么、从哪来

| 文件 | 大小 | 干什么 |
|---|---|---|
| `hubert_fp16.onnx` | 180.3 MB | 波形 → 768 维**内容特征**（它不知道是谁在唱） |
| `rmvpe.onnx` | 344.9 MB | mel → 每帧**音高**（**fp32 —— 能跑的就是这一份**，见下面第 ⑪ 条） |
| `mel_basis.f32` + `mel_meta.json` | 0.3 MB | mel 滤波器表（纯 JS 那一步用） |
| `voices/<名字>/<名字>.onnx` | 106 MB | 内容特征 + 音高 + 噪声 → 波形（**音色在这一层**） |
| `voices/<名字>/<名字>.index` | 21 MB | Faiss 检索索引（**还没接** = `index_rate 0` 那条路） |
| onnxruntime（CPU） | ~288 MB | 跑上面的运行时（三个平台的二进制都在里面；实机用到的那几份 66 MB） |

**合计约 940 MB。**
⚠️ 这里原来写的是"三件模型（含 fp16 的 rmvpe）共 434 MB" —— **那个口径是错的**，
因为 fp16 那份 rmvpe 在 Node 下加载就崩，真用的是 fp32 那份（+172 MB）。

**它们都不是下载来的现成文件，是从 RVC 的权重转出来的：**

```
hubert   ← assets/hubert_base（transformers 格式，v2 要第 12 层，见下面那个坑）
rmvpe    ← assets/rmvpe/rmvpe.pt（**转 fp32**）
mel 表   ← 跟着 mel.mjs 一起生成的（257 KB）
```
⚠️ **音色（net_g）不在 `models/` 里** —— 它是 `voices/<名字>/<名字>.onnx`，一个 106 MB，
   换来换去的是它（"可持续训练"也是指它）。`models/` 那两张网络是共用的。

---

## 三、⚠️ 试过而**失败**的路 —— 别再试第二遍

这三条我都量过，不是推的。

### ① int8 量化 → **算错，不是有损**

```
hubert  fp32 360.2 MB → int8 90.8 MB（25%）  但：**相对差 1.4981（149%），相关 0.2032**
rmvpe   fp32 344.9 MB → int8 89.5 MB（26%）  但：**argmax 一致率 64.7%，20.6% 的帧差 > 半个半音**
```

**相关 0.20 不是"精度低"，是算错了。** 而动态量化更早就失败：
onnxruntime 的 CPU 端**没实现 `ConvInteger`** —— 动态量化只适合 MatMul（全连接 / Transformer），
**卷积要走静态量化**（那要校准数据，我用了真实人声校准 8 段）。

### ② 用自己的 YIN 换掉 rmvpe → **差得更多**

```
rmvpe 判有音高 910 帧 · YIN 判有音高 1721 帧（多一倍）
两边都有音高的 788 帧里：**中位差 509.6 cents（5 个半音）**
**差 > 50 cents（半个半音）的占 73.35%**  → 摊到全部帧是 29%
而速度也没赢：rmvpe 1.35 秒 vs YIN 1.48 秒
```

### ③ fp16 在 CPU 上**更慢**

```
rmvpe:  fp32 0.117 s · fp16 0.206 s · int8 0.054 s
hubert: fp32 0.160 s · fp16 0.307 s · int8 0.104 s
→ **"fp16 = 更快"是错的** —— 它省一半内存，但 CPU 不做 fp16 加速，还要来回转换。
```

### ④ 而 fp16 本身是可用的（这是留下的那条）

```
hubert  fp32 vs fp16：相对差 **1.28e-03** · 相关 **0.99999915**
rmvpe   fp32 vs fp16：argmax 一致率 **99.375%** · 差 > 50 cents 的 **1 帧（0.31%）**
```

---

## 四、验证过的数字（不是"应该行"）

### rmvpe 那条链，纯 JS vs Python —— **完全一致**

```
mel     128 × 601  ·  71 ms     最大相对差 **4.50e-05**（log 域里绝对差不能用，见下面那个坑）
pad     601 → 608 帧（32 的倍数）
rmvpe   hidden 608 × 360  ·  358 ms
decodeF0 601 帧  ·  5 ms

f0 那一站：两边都有音高 197 帧 · 只有 JS 有 **0 帧** · 只有 Python 有 **0 帧**
          **共同帧里差 > 50 cents 的：0 帧**
          f0 最大差 **9.86e-05 Hz**
→ 整条链 434 ms（6 秒音频）。而 Python 那边 1.35 秒（含模型加载）。
```

### net_g 那一层（onnx vs torch，噪声钉死之后）

```
✅ m_p（enc_p 第一输出）    2.62e-06
✅ logs_p（第二输出）       1.19e-06
✅ sine_waveforms（解码激励）1.49e-08
✅ noise                   0.0
⚠️ audio（最终）           1.2368e-02   相对峰值 6.65%
```

**而那个 6.65% 主人听过那两段对照音频，说「基本没有差异」，再听之后说「或者基本没有」。**

⚠️ **两个判据不矛盾，它们在量不同的东西**：
- **逐样本**：最大差 6.65%（那是**示波器**的判据，成立）
- **听感**：基本没有（那是**耳朵**的判据，也成立）—— 证据在同一份数据里：**平均差只有 0.71%**
  那 6.65% 是**个别瞬态点**上的，耳朵不逐个样本比。
- **我一开始拿"信噪比 23.5 dB"推"能听出来"—— 那个推错了**，错在拿最坏的单点代表整体。

---

## 五、踩过的坑（每个都花了时间，记下来免得重踩）

### 装环境那一串

```
① **uv 建的 venv 没有 pip** —— 而 RVC 的说明写的是 `python -m pip install` → 全空跑，
   而**退出码是 0**（PowerShell 管道吞了错误）。**退出码不是证据。**
② **`huggingface_hub.commands.huggingface_cli` 那个入口没了**（新版改成 `hf` 命令）。
③ **`torch==2.7.1+cu128` 在 nju 镜像上已经没有了**（镜像里有 2.10/2.11），而官方源还有。
   → 而 uv 默认**只认第一个含该包的索引** —— 要加 `--index-strategy unsafe-best-match`。
④ **sr=40k 要用 `configs/v1/40k.json`**，不是 v2 —— 那条判断埋在 `webui.py` 里：
   `if version19 == "v1" or sr2 == "40k": config_path = "v1/%s.json" % sr2`
⑤ **从 `train/` 里跑脚本会循环导入** —— `sys.path[0]` 变成 train/，`import train` 撞上 train.py 自己。
   → 要用 `python -m train.xxx` 从仓库根跑。
⑥ **`filelist.txt` 要自己生成**（WebUI 才帮你写）。格式从 `webui.py:1225~1262` 读出来的：
   f0 模式下每行 5 段用 `|` 分。
⑦ **curl 下了一半不报错** —— f0D40k.pth 只有 67 MB（该 143 MB），一直到 `torch.load` 才炸。
   → **一个不校验完整性的下载工具，等于没有错误处理。**
⑧ **训练检查点不能直接推理** —— 报 `Model does not contain weight/emb_g.weight`，
   要用 `process_ckpt.extract_small_model` 提出"只有 weight"的那份。
```

### 模型那一边

```
⑨ **hubert v2 要第 12 层，不是 v1 那条路**。`infer/hubert.py` 的 docstring 写着：
   「RVC v1 uses layer 9 followed by final_proj; **RVC v2 uses the final (12th) encoder layer directly.**」
   → 我第一次导了 v1 那条（`final_proj(hidden_states[9])`），**出来 256 维，而 v2 要 768**。
⑩ **int8 量化要从 fp32 转**，喂 fp16 给它图就非法（`Type 'tensor(float16)' ... invalid model`）。
   → 也意味着：**走 int8 的话，fp16 那一步是多余的。**
⑪ **`rmvpe_fp16.onnx` 在 onnxruntime-node 下加载就 segfault** ——
   而 fp32 / int8动态 / int8静态 / hubert_fp16 四个都能加载。
   所以是那个文件特有的，不是"fp16 都不行"。
⑫ **rmvpe 的最终 f0 不是"argmax → Hz"** —— 它取 argmax 周围 9 个分箱**加权平均**，
   最大概率 ≤ 0.05 判为无音高，然后 `f0 = 10 · 2^(cents/1200)`。
   → **拿 argmax 去比会人为造出差异。**
⑬ **`torch.hann_window(1024)` 是周期性的**（除 N，不是 N−1）——
   跟 `numpy.hanning` 差一个样本的相位，而那点差会一路传到 mel 上。
```

### 我自己反复栽的那两类

```
⑭ **Windows 路径别走 heredoc + 字符串替换** —— `\\R` 和 `\R` 对不上，
    `SyntaxWarning` 一闪而过，**替换没生效而我不知道**。改文件用编辑工具。
⑮ **`echo "exit=$?"` 在管道后面拿的是 `tail` 的退出码** ——
    一个崩溃的进程照样打印 `exit=0`。**差点被那个 0 骗过去。**
⑯ **在 log 域里用"最大绝对差"当判据是错的** —— `log(x+δ) − log(x) ≈ δ/x`，
    x 小的时候 δ/x 就大。同一份数据里**相对差是 4.5e-05，而绝对差是 2.3e-04**。
⑰ **看到差就下结论，而没先确认"我比的是不是同一个东西"** ——
    有过两次：拿 `sine_scaled`（sin×0.1 的半成品）去比 torch 的 `sine_waves`（sin×uv+noise），
    得出一个**根本不存在的** 1.5e-2；还有一次拿随机 mel 判 fp16，得出"760 cents"。
```

---

## 六、怎么再练一个音色

```bash
py tools/train_voice.py <录音目录> <音色名> [--epochs 200] [--resume]
```

**九步它都会跑**：切分 → 提音高 → 提 hubert 特征 → 写 filelist → 抄 config →
训练 → 建索引 → 提小模型 → 导 onnx → 转 fp16。

**而"在一个声音的基础上训练出其他的声音"：** `-pg` 那个参数是"生成器起点"，
指向另一个音色的**检查点**（`G_*.pth`，不是小模型）就是增量训练。
→ 所以训练脚本要**保留一份可续训的检查点** —— 那是"可持续训练"的实体。

```
底模 → 练出 A（200 轮）
A 的检查点 → 继续练 → 练出 B（几十轮就够，因为音色已经接近）
→ 而 B 可以再长 C……
```

---

## 七、视频那一半

音频 → 片子。**画帧是纯 JS 的**（PNG 自己编码，见 `png.js`），只有"帧拼成 mp4"用外部的 ffmpeg。

**三层，每层一个时间尺度 —— 而这三层必须分得开：**

```
① **整段轮廓**（铺满，淡）        —— 整段的样子，已过去那侧更亮
② **局部波形 + 频谱**（中间 45%，亮）—— 这 0.35 秒
③ **进度条**（铺满）              —— 整段里的位置
   而局部框下面还有一条窄的，标的是"框里这 0.35 秒在整段的哪一段"
```

**我第一版把局部拉伸到**整幅宽度** —— 于是它跟整段轮廓形状几乎一样，
两条几乎重合的曲线叠在一起，`读不出"局部/整体"`。**

**⚠️ 而画面改了三版，每一版都是"看了才知道"的：**

```
① **频谱按整段归一化**（原来每帧按自己的最大值）
   → 每帧都撑满同一个 70dB 范围，**"绝对音量"整个丢了** —— 大声和安静的两帧柱形一样高。
     而频谱柱的用途**恰恰**是让人看出音量在哪儿起伏。
② **局部那个框收窄到刚好包住内容**（原来 H*0.06~H*0.84，比内容大一圈）
   → 安静的段落里框几乎是空的而边框很清楚，读起来像"有个东西没加载出来"。
③ **去掉那道"你在这"竖线** —— 它跟底部进度条标的是同一件事，
   而它跟局部框的 x 坐标**不是一回事**（框固定、位置在滑）。
   结尾那帧竖线跑到最右边、框还在中间 —— 读起来像"框没跟上"。
```

**⚠️ 不内嵌 ffmpeg** —— 那个二进制 80~200 MB，而它是台通用机器上常见的东西。
所以这一步先探它在不在，不在就直说。

### 画面是可调的（0.4.1 起）—— 那排旋钮的名字

原来顶上硬编码的七个颜色常量 + 几个几何比例，现在是参数：

```
av { action:'video', input, output, preset: 'warm' }
av { action:'video', input, output, taste: '{"preset":"cool","geometry":{"localWidth":0.7}}' }
```

```
preset    default · warm · cool · violet
taste     可选 JSON（字符串），覆盖 preset 的 colors / geometry

colors    bg · dim · wave · waveDim · waveFuture · spectrum · accent
          · boxFill · boxEdge · thumb · progress
geometry  pad · midY · wholeAmp · windowAmp · spectrumTop · spectrumHeight
          · localWidth · windowSeconds · spectrumBars · spectrumDb
```

⚠️ **不传 = 原来的画面，逐字节相同**（同一段音频新旧各画 23 帧，23/23 帧一致）。
⚠️ 这里是**旋钮**，不是"心情/品味"本身 —— 那一层归 dsh-mood，本插件不 import 它。
（`render.js` 导出 `RENDER_PRESETS` 和 `resolveRenderStyle()`，那边可以直接读。）

## 八、和 dsh-mood 的三条接口

**代码上零耦合**（这边对 dsh-mood 的提及全是注释，没有一个 import），但有三处口头约定：

```
① 采样率 —— **这条以前是断的，0.4.1 接上了**
   dsh-mood 的耳朵（asr.js）只收 16000 Hz，不是就直接退出；
   而主输出必须是 **40000**（音色是 40k 训的，改了会变调）。
   → 主输出没动，补的是**出口**：
       av { action:'listen',  input, output }              单独转一份 16k 单声道
       av { action:'convert', input, voice, output, compatOutput }  转换时顺手多落一份
     用的是抗混叠 resample（hubert.mjs 那个），不是线性插值。

② `render.js` 的那排旋钮 —— 那是"品味 → 生成"的接入点（见上一节的名字表）。
   对面两种接法都能接：**读导出的 RENDER_PRESETS**（零依赖）或**工具传参**（可追溯）。

③ `voices.js` 的音色目录 —— **别现在加标签**（那是品味判断，不是工程活）。
   只要保持"对象数组"而不是裸字符串数组，将来能挂 `tone:['克制','手作']` 而不用改格式。
   现在返回的已经是对象（`{name, dir, onnx, index, mb, preferred…}`）✅
```

⚠️ 这三条改了**不会报错**，正因为不会报错才写在这儿；动了就在 CHANGELOG 里记一行。

## 九、没做 / 不确定的

```
⚠️ **"按需的 onnx 运行时"只做到一半。**
   现在的形态是"本体 82.9 KB + 模型放在数据目录里"，而**"要用时才去取"那个机制没有** ——
   因为那三件是我自己从 RVC 的权重转出来的，**没有一个可下载的源**（也不该往上传）。
   → 要真做到"按需"，得带一个**纯 JS 的转换器**，让用户拿自己那份上游权重就地转：
     · `net_g`   —— **能**：`rvc-onnx-web` 那个纯 TS 库 **1.12 秒**就把 .pth 转成了 onnx（验过）
     · `hubert`  —— **大致能**：上游是 safetensors，格式简单，纯 JS 读得了
     · `rmvpe`   —— **不能**：上游 `.pt` 是 pickle 格式，纯 JS 里读不了
   → 所以完整那条路是：**net_g 和 hubert 本地转，rmvpe 得另想办法**（或者别人给一份）。

⚠️ **字幕** —— 那要字体渲染（FreeType 那套），纯 JS 里没有。
   两条路：调 ffmpeg 的 `drawtext`，或者生成 SVG 让浏览器 / ffmpeg 转。**都没做。**
⚠️ **检索（Faiss index）** —— 它在 net_g **外面**，影响"像不像"而不是"算得对不对"。
   现在没接 = `index_rate 0` 那条路。
⚠️ **重采样的最终影响没定** —— 我的 sinc vs ffmpeg：波形层相关 0.99895、
   最佳 lag = 0（所以**不是错位**），特征层相对最大差 45%，
   而**最终影响只能靠听**（两段对照 wav 在 `品味-临摹\实验\`）。
⚠️ **端到端"逐样本比"在这个场景不适用** —— 两边噪声都是随机的（VITS 系模型的固有随机性）。
   同噪声下相关 **0.9997**；各自跑一遍是 **0.3** 左右 —— 后者量的是"两次独立生成差多少"，
   不是精度。**所以判据只能是听。**
   ⚠️ 2026-10-16 补了"为什么"：**这段波形的相干长度只有约 1 ms** ——
      把同一段输出跟它自己错开：2 个样本 → 0.85 · 21 个 → 0.57 · **55 个 → −0.20**。
      → 所以"逐样本相关"量的是**相位对齐**，不是精度；**它只在输入逐位相同时才有意义**
        （0.998~1.0 那一档就是这么来的）。两条链的输出相关 0.44 **不是谁算错了**。
      → 工具：`probe-phase.mjs`（看相干长度）、`probe-resample.mjs`（两组音频是不是同一段内容）。
⚠️ **golden 里的 16k 音频有两组、而且不是同一段内容**：
   `audio16k` + `audio16k_ffmpeg2`（互相关 0.9878）是一组、
   `audio16k_mine` + `audio16k_ffmpeg`（互相关 0.9990）是另一组，**两组之间相关 ≈ 0**。
   → 别拿一组的参考/噪声去套另一组（这一轮栽过一次）。
⚠️ **写 wav 的时候采样率要回头看** —— `重采样对照-*.wav` 三份曾经装的是 239200 点（40 kHz）
   而头里写着 16000（= 14.95 秒）—— **代码修了，文件没重写**，而那正好是拿来"用耳朵判"的两段。
   已经用 `fix-wav-rate.mjs` 改回来（只动头里 8 个字节）。
⚠️ **rmvpe 只能用 fp32** —— fp16 那个在 onnxruntime-node 下**加载就 segfault**，
   六种配置全试过（op17 那些崩，op13/14/15 是我改错了版本号）。
   那 172 MB 省不掉 —— 除非换 onnxruntime 版本，或者逐算子找是哪个让它崩。
⚠️ **视频只画波形 / 频谱** —— 没有分镜、没有段名、没有歌词。
⚠️ **"在一个声音的基础上训练出其他的声音"那一半没做** ——
   路子（`-pg` 指向另一个音色的**检查点**而不是小模型）写在第六节，**而没有真跑过一遍**。
⚠️ **这一整条链只在一段音频上验过**（6 秒 K 歌人声）。换纯音乐、极安静的、很长的 —— 没试。
```

## 十、怎么装进 DSH（这一版真装过一遍）

```
① 插件得先"像"一个 bundle —— package.json 里要有：
     "dsh": { "bundle": { "patch": "./cordis.patch.yml" } }
   而 cordis.patch.yml 里是一段 insert：
     - insert: [{ id: dsh-av, name: dsh-av }]
   ⚠️ **少了这个字段，宿主报的是 `not-bundle`** —— 而 pnpm 那一步是成功的，
      所以很容易读成"装上了但没生效"。

② 挂进 profile（~/.dsh/profiles/desktop/package.json）：
     dependencies          "dsh-av": "link:D:/DSHzuoye/dsh-av/src"
     dsh.profile.bundles   [ … , "dsh-av" ]
   ⚠️ **用 `link:` 不要用 `file:`** —— pnpm 对 `file:` 的**目录**依赖是**复制**一份，
      于是"改了源码、node_modules 里那份没变"，而症状又正好是上面那个 not-bundle。

③ 数据目录（~/.dsh/dsh-av）—— **699 MB 不复制**：
     models         → junction 到 D:\DSHzuoye\dsh-av\models
     voices\kizhi\  → kizhi.onnx（106 MB · **七输入**）+ kizhi.index（21 MB）+ model_info.json

④ 插件要 onnxruntime，而它（288 MB）不在这个仓库里：
     src/node_modules/onnxruntime-node → junction 到 D:\rvconnx\node_modules\onnxruntime-node
   **一条 junction 就够** —— Node 默认按 **realpath** 解析模块，
   它自己的依赖（onnxruntime-common / global-agent / adm-zip…）会从 D:\rvconnx\node_modules 里找。

⑤ ⚠️ **插件体里不许 `const { tools } = ctx`** —— 服务必须走 `ctx.get('tools')`。
   cordis 对"没声明 inject 就直接读服务属性"是**直接抛错**的：
       cannot get property "tools" without inject
   而抛在插件体里 = **整个插件挂载失败**（`fiberPhase: failed`，工具一个都不出现）。
   （这一条是 0.4.2 才撞出来的：装上了、重启了、然后静静地 failed。）
   写法照 dsh-mood：`const tools = ctx.get ? ctx.get('tools') : ctx.tools`

⑥ ⚠️ **改了源码要重启宿主，disable/enable 不会重新 import。**
   实测：改完之后 disable → enable 一遍，报的还是**改之前那个行号** ——
   说明它跑的是进程里缓存的 ESM。**看到"改了没生效"，先确认它有没有重新加载，
   别先怀疑自己没写对。**

⑦ ⚠️⚠️ **别在插件里放 `@deepseek-ai/dsh-tools` 的替身 —— 它会 shadow 宿主注入的真身。**
   真的那份**只在 app.asar 里**（磁盘上找不到；dsh-mood 在 profile 的副本连 node_modules
   都没有却能跑，说明是宿主注入的）。
   而**替身比真身少做一件事**：真身会把 `parameters` 规格**编译成 JSON Schema**，
   替身只是原样返回 —— 于是注册出去的东西 `parameters` 不合法，
   **工具进不了工具表，而且不报错**（最难查的那种）。
   → 现在的写法：**动态 import + 兜底**（兜底自己会编译），并且把用了哪份写进日志。
   ⚠️ 静态 `import { defineTool } from '@deepseek-ai/dsh-tools'` 也有风险：
   **它失败 = 整个模块加载失败 = fiber failed**，连日志都留不下。

⑧ **宿主里的报错，插件自己的日志是唯一出口。**
   插件的加载过程会写进 `<数据目录>/load.log`（默认 `~/.dsh/dsh-av/load.log`）：
   拿到了哪份 defineTool / tools 服务有没有 / `parameters` 是不是合法 JSON Schema /
   `register` 有没有抛错 / **回查 `tools.get('av')` 和 `tools.schemas()` 在不在**。
   ⚠️ 后两条是关键：**"register 没报错"和"真进了工具表"是两件事。**
   （判据是"程序知道但不告诉我"，那就让它写下来。）
```

**⚠️ 而"装上去"和"能用"之间还隔着一件事：音色文件必须是七输入的。**

```
六输入（输入名里有 rnd，onnxlib 导的）→ 一跑就崩：
   Non-zero status code returned while running Reshape node …
   Input shape:{1,2,715407}, requested shape:{1,2,599,1195}
   那个形状是**导出时那段 5.98 秒音频**写死的 → 只有那一种长度不崩。
七输入（phone/phone_lengths/pitch/nsff0/sid/noise_1527/noise_random_2110）→ 通用。
```
现在开音色会话时就查一次 `inputNames`：**有 `rnd` 就提前说出来**，
并且把 `cannot be reshaped` 那句话翻译成"图里写死的形状，换一份音色"——
原来那句报错读起来像"模型坏了"，而它其实只说明"这份导出只对一段长度有效"。

**自测（不走宿主，直接把工具定义拿出来跑）：**

```bash
node test-smoke.mjs status
node test-smoke.mjs convert '{"input":"…wav","voice":"kizhi","output":"…wav"}'
node test-smoke.mjs video   '{"input":"…wav","output":"…mp4","fps":12,"width":640}'
node src/probe-layers.mjs    # 一层一层跟 golden 对，定位差异在哪一层
node src/probe-netg.mjs      # 单独验 net_g，含那个 /0.66666 该乘还是该除
node src/verify-chain.mjs    # 整条链（喂 golden 噪声）→ 跟 torch 那份比
node compare-wav.mjs <我的.wav> <参考.f32>
```

## 十一、现在到哪一步了

```
✅ 三件 onnx 都转出来了，都验过
✅ 纯 JS 的 rmvpe 链路跟 Python **完全一致**（音高判定零差异，共同帧 0 帧差超半音）
✅ **hubert 和 net_g 都接进 JS 了**（fp32 下相关 1.00000000 / 0.9997）
✅ **整条链在 Node 里跑通**（44.1k wav → 40000 Hz wav，**3745 ms / 6 秒**）
✅ **跟 RVC 原版端到端对齐**（40000 Hz · 5.98 秒 · 形状一致）
✅ **视频那一半也能干活**（wav → mp4，143 帧 1903 ms + ffmpeg 拼）
✅ **装进 desktop profile 了**（bundle + link: + junction；四个 action 都真跑过）
   ⚠️ 而**真机上还没起来过**：重启后插件列表里 208 条有它、`fiberPhase: failed`
   —— 原因是 `const { tools } = ctx`（见第十节 ⑤），已修，**等下一次重启验证**。
✅ **端到端复现 torch 那份**（喂同一段输入 + 同一组噪声 → 相关 **0.99799903**）
   而分层看：波形 1.0 · hubert 0.99999917（fp16 的代价）· mel 1.0 · rmvpe 1.0 · f0 1.0
✅ **0.4.1：源码过了一遍外部优化**（`listen` / `compatOutput` / 画面可调），我逐条验过 ——
   改动只落在 3 个文件，承重那 8 条一条没碰；默认画面 **23/23 帧逐字节相同**；
   convert 端到端**还是 0.99799903**
⚠️ 上面"没做 / 不确定"那几条
```
