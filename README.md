# DSH 插件（打包版）

DeepSeek Harness 用的两个本地插件，**压缩包直接放在 `packages/` 里**，解压即装。

| 包 | 版本 | 体积 | 说明 |
| --- | --- | --- | --- |
| **音乐台** `@local/music-tab` | 1.0.1 | 43.5 KB | 设置页里的一页 Web Audio 作曲台：钢琴卷帘 + 步进网格 + 30 种音色 + 导出 WAV/MIDI/工程 |
| **dsh-av** | 0.4.5 | 65.2 KB | 音色转换 / 音频渲染，纯 JS（ONNX 在 Node 里跑，不依赖 Python） |

---

## 安装

1. 解压到任意目录，例如
   - `packages/dsh-music-1.0.1.zip` → `<某处>/dsh-music`
   - `packages/dsh-av-0.4.5.zip` → `<某处>/dsh-av`
2. 在 DSH 里用插件管理器装**那个目录**（不是 zip）：

```
plugin_manager { action: "install_bundle", target: "<绝对路径>/dsh-music" }
plugin_manager { action: "install_bundle", target: "<绝对路径>/dsh-av" }
```

3. 装完**刷新页面**。音乐台在：设置 → 音乐台。

---

## 音乐台长什么样

单文件 `www/音乐台.html`（约 97 KB，零依赖、全中文），插件只做一件事：把这一页原样送出去。

- **钢琴卷帘**：横向是时间、纵向是音高；点一下放音符，拖右边缘改长度，拖身体挪位置
- **步进网格**：一格一个十六分音符，8 条轨各一行
- **音色** 30 种（键盘/弦乐/管乐/电声/打击/氛围）+ 你自己录的采样
- **导出** WAV（手写 WAV 头，分段渲染）/ MIDI / 工程 JSON / 采样包
- **撤销** Ctrl+Z，**全屏** 一键铺满，**缩放** 4–64px 每格

---

## 已知限制

- **dsh-av 的模型（`models/`，约 700 MB）不在包里** —— 需要单独下载，否则只能跑不需要模型的路径
- 音乐台的高倍率缩放（>2×）下画布文字会略微软化
- 音量 / 声像 / 音色改动**不进撤销栈**（滑杆拖动太频繁）

---

## 目录

```
packages/
  dsh-music-1.0.1.zip    音乐台插件（自带 www/音乐台.html）
  dsh-av-0.4.5.zip       dsh-av 源码（不含 models/）
CHANGELOG.md
```

---

## 许可

自用插件，未选许可证。第三方依赖（dsh-av 的 onnxruntime 等）各自遵循其原始许可。
