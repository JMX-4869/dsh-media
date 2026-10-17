# -*- coding: utf-8 -*-
"""打 dsh-av 的发布包（排除 node_modules / .git / *.log）。"""
import os
import sys
import io
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = r'D:\DSHzuoye\_gh-upload'
SRC = os.path.join(BASE, 'src')
OUT = os.path.join(BASE, 'packages', 'dsh-av-0.5.6.zip')
SKIP = {'node_modules', '.git'}

n = 0
total = 0
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(os.path.join(SRC, 'dsh-av')):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            if f.endswith('.log'):
                continue
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, SRC))
            n += 1
            total += os.path.getsize(full)

print(f'  {n} 个文件 · {os.path.getsize(OUT)/1024:.0f} KB（原始 {total/1024:.0f} KB）')
