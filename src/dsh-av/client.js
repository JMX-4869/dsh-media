/**
 * dsh-av —— 客户端半边（音乐台这一页）
 *
 * dsh-av 是**一体**的：宿主那边是 av 工具 + /dsh-music 路由，这边是设置页里的「音乐台」。
 * 内容是 iframe 指向 `/dsh-music` —— 音乐台是写曲子的输入端，导出的 WAV 正好喂给 av convert。
 *
 * ⚠️ 必须是 iframe，不能把 HTML 塞 innerHTML —— 按 HTML 规范，innerHTML 插进去的 <script>
 *    **不会执行**，音乐台的 AudioContext 就起不来。
 */
window.__ModuleLoader__.load({
  id: 'dsh-av',
  factory(require) {
    const React = require('react')
    const h = React.createElement
    const API = '/dsh-music'
    const LABEL = '音乐台'          // 导航格里用来定位的文字

    /* ── 导航图标 ────────────────────────────────────────────────────────────
       宿主的设置导航图标是**硬编码**的（function navIcon(id){ if (id==='account') … return 齿轮 }），
       未知 id 一律回退齿轮，插件没有参与的机会 —— package.json 的 icon 只影响插件页的展示图。
       所以只能等导航渲染出来后自己换掉那个 <svg>。这是 hack，宿主改结构就失效，
       所以全包在 try 里，失败也不影响别处。

       画的是**音乐台自己的步进网格**：3×3 里四个格子填实，走一条上扬的动机
       —— 不是音符也不是音箱，因为这里是"写曲子"的地方。 */
    function makeNavIcon() {
      const NS = 'http://www.w3.org/2000/svg'
      const svg = document.createElementNS(NS, 'svg')
      svg.setAttribute('viewBox', '0 0 24 24')
      svg.setAttribute('width', '20')
      svg.setAttribute('height', '20')
      svg.setAttribute('fill', 'none')
      svg.setAttribute('stroke', 'currentColor')
      svg.setAttribute('stroke-width', '1.5')
      svg.setAttribute('stroke-linejoin', 'round')
      const lit = new Set([6, 4, 2, 7])   // 3×3 的下标：左下 → 中 → 右上，再补一格
      let i = 0
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++, i++) {
          const rect = document.createElementNS(NS, 'rect')
          rect.setAttribute('x', String(3.5 + c * 6))
          rect.setAttribute('y', String(3.5 + r * 6))
          rect.setAttribute('width', '5')
          rect.setAttribute('height', '5')
          rect.setAttribute('rx', '1.3')
          if (lit.has(i)) {
            rect.setAttribute('fill', 'currentColor')
            rect.setAttribute('stroke', 'none')
          }
          svg.appendChild(rect)
        }
      }
      return svg
    }

    function swapNavIcon(root) {
      let n = 0
      try {
        for (const btn of (root || document).querySelectorAll('button')) {
          const svg = btn.firstElementChild
          if (!svg || svg.tagName.toLowerCase() !== 'svg') continue
          if (svg.dataset.musicTab === '1') continue
          const span = svg.nextElementSibling
          if (!span || (span.textContent || '').trim() !== LABEL) continue
          const fresh = makeNavIcon()
          fresh.dataset.musicTab = '1'
          btn.replaceChild(fresh, svg)
          n++
        }
      } catch { /* 宿主改结构就放弃，不抛 */ }
      return n
    }

    function installNavIcon() {
      try {
        swapNavIcon(document)
        let scheduled = false
        const mo = new MutationObserver(() => {
          if (scheduled) return
          scheduled = true
          requestAnimationFrame(() => { scheduled = false; swapNavIcon(document) })
        })
        mo.observe(document.body, { childList: true, subtree: true })
        return () => { try { mo.disconnect() } catch { /* 忽略 */ } }
      } catch { return () => {} }
    }

    /** 一页「音乐台」：说明一行 + 填满面板的 iframe。
     *
     *  ⚠️ **不做"在新标签打开"** —— 宿主是 Electron 壳子，window.open 默认被拒
     *  （返回 null），而 location.href 兜底会把整个 DSH 导航走（用户看到的就是"直接退了"）。
     *  既然要的是内置，就只留内置这一条路：把面板本身撑大，而不是往浏览器里甩。 */
    function MusicTab() {
      return h('div', { style: { display: 'flex', flexDirection: 'column', gap: 8 } },
        h('span', {
          style: { fontSize: 12, lineHeight: 1.6, color: 'var(--dsw-alias-label-secondary)' },
        }, 'Web Audio 实时合成 · 点网格格子出声 · 第一次要先按「播放」或弹一个键，浏览器才允许发声音'),
        /* 高度按视口给满（面板自己会滚）—— 内置是唯一入口，小气就不好用了 */
        h('iframe', {
          src: API + '/',
          title: LABEL,
          /* ⚠️ `fullscreen` 必须显式允许：音乐台里那个「全屏」按钮调 requestFullscreen，
             没有这个权限会被直接拒（iframe 默认不给）。 */
          allow: 'autoplay; fullscreen',
          style: {
            width: '100%', height: 'calc(100vh - 250px)', minHeight: 420,
            border: 0, borderRadius: 12, background: '#0d1117', display: 'block',
            boxShadow: '0 1px 3px rgba(0,0,0,.18)',
          },
        }),
      )
    }

    return {
      inject: ['slots'],
      apply(ctx) {
        ctx.slots.inject('settings.section', () => ctx.slots.register(
          { name: 'settings.section', id: 'musictab', order: 32, label: LABEL },
          MusicTab,
        ))
        return installNavIcon()
      },
    }
  },
})
