# -*- coding: utf-8 -*-
"""dsh-av · 一次性语音合成 —— 拉起、合成、退出。**不常驻服务。**

用法（由 index.js 调用）：
    python tts_oneshot.py <text> <ref.wav> <ref.txt> <out.wav>

为什么不做常驻服务：
    常驻要多养一个进程、占着显存、还得有人管它什么时候死。
    一次性调用慢在"加载模型"那 20 秒，但换来的是**平时零占用** ——
    对"偶尔说一句话"的用法，这个交易是对的。
    （要快就设 DSH_AV_TTS_MODE=server，用 9880 那个常驻服务。）

⚠️ 这一段跟插件里那三个 ONNX 网络**不是一层**：
    ONNX 那条是纯 JS；这条必须 Python + torch（原版的全部能力都在这里，包括文本前端）。
"""
import sys, os, io, time, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

T0 = time.time()

if len(sys.argv) < 5:
    print(json.dumps({'ok': False, 'error': '参数：text ref.wav ref.txt out.wav'}, ensure_ascii=False))
    sys.exit(2)

TEXT, REF, REF_TXT, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

G = os.environ.get('DSH_AV_TTS_DIR') or r'D:\GPT-SoVITS-v2pro-20250604'
if not os.path.isdir(G):
    print(json.dumps({'ok': False, 'error': '找不到 GPT-SoVITS：' + G}, ensure_ascii=False))
    sys.exit(3)

os.chdir(G)
sys.path.insert(0, G)
sys.path.insert(0, os.path.join(G, 'GPT_SoVITS'))

try:
    import torch
    import numpy as np
    import soundfile as sf
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    cfg = TTS_Config(os.path.join(G, 'GPT_SoVITS', 'configs', 'tts_infer.yaml'))
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_load = time.time() - T0

    tts = TTS(cfg)
    t_ready = time.time() - T0

    req = {
        'text': TEXT, 'text_lang': 'zh',
        'ref_audio_path': REF, 'prompt_text': REF_TXT, 'prompt_lang': 'zh',
        'text_split_method': 'cut5', 'streaming_mode': False,
    }
    chunks = [c for c in tts.run(req)]
    sr = chunks[0][0]
    buf = np.concatenate([c[1] for c in chunks]) if len(chunks) > 1 else chunks[0][1]
    sf.write(OUT, buf, sr)

    print(json.dumps({
        'ok': True, 'out': OUT, 'sr': int(sr), 'seconds': round(len(buf) / sr, 2),
        'load_s': round(t_ready, 1), 'total_s': round(time.time() - T0, 1),
    }, ensure_ascii=False))
except Exception as e:
    print(json.dumps({'ok': False, 'error': '%s: %s' % (type(e).__name__, e)}, ensure_ascii=False))
    sys.exit(4)
