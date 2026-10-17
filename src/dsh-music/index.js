/**
 * @local/music-tab —— 宿主半边
 *
 * 只做一件事：注册 `/dsh-music` 前缀路由，把工作区里那个 56 KB 的单文件音乐台**原样**读出来送走。
 * 不加工、不打包 —— 改音乐台就等于改这一页，不用重新构建插件。
 *
 * 为什么单独成包（而不是塞进 dsh-mood）：
 *   插件升级会重写自己的 index.js，手打的补丁会被冲掉；独立成包之后两边互不影响 ——
 *   dsh-mood 随便升，这一页还在；这个包删掉，dsh-mood 也照常跑。
 */
import { readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'

/* ⚠️ 不能只靠 process.cwd()：宿主进程的 cwd 是哪一层并不确定
   （实测环境变量里只有 DSH_HOME / DSH_PROFILE / DSH_PROFILE_DIR / DSH_WEB_URL，没有工作区变量）。
   所以**包内自带的那一份排第一**（www/音乐台.html）—— 这样这个 zip 拷到任何机器上都直接能用；
   显式路径 / 环境变量 / 相对路径只当覆盖和兜底。 */
const CANDIDATES = () => [
  join(import.meta.dirname, 'www', '音乐台.html'),
  process.env.DSH_MUSIC_HTML,
  'D:/DSHzuoye/音乐台/音乐台.html',
  join(process.cwd(), '音乐台', '音乐台.html'),
  join(process.cwd(), '..', '音乐台', '音乐台.html'),
]

const findHtml = () => {
  for (const p of CANDIDATES()) {
    if (!p) continue
    try { if (existsSync(p)) return p } catch { /* 忽略 */ }
  }
  return null
}

export const inject = ['webServer']

export function apply(ctx) {
  const web = ctx.webServer || (typeof ctx.get === 'function' ? ctx.get('webServer') : null)
  if (!web || typeof web.register !== 'function') {
    ctx.logger?.warn?.('music-tab: webServer 不可用，音乐台这一页打不开')
    return
  }
  ctx.effect(() => web.register({
    kind: 'prefix',
    path: '/dsh-music',
    handler(req, res) {
      const p = findHtml()
      if (!p) {
        res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' })
        res.end('没找到 音乐台.html。\n设环境变量 DSH_MUSIC_HTML 指到它，或把它放到 <工作区>/音乐台/音乐台.html\n试过：\n' + CANDIDATES().filter(Boolean).join('\n'))
        return
      }
      try {
        res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' })
        res.end(readFileSync(p, 'utf8'))
      } catch (e) {
        res.writeHead(500, { 'content-type': 'text/plain; charset=utf-8' })
        res.end('读不出来：' + (e && e.message ? e.message : e) + '\n路径：' + p)
      }
    },
  }))
}
