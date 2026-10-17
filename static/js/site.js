/* S-Agent 2.0 project page. Numbers come from the paper's tables (cited next to each block). */
(function () {
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  /* ---- tooltip ---- */
  const tip = el('div', 'tip');
  tip.hidden = true;
  document.body.appendChild(tip);
  function showTip(target, x, y) {
    tip.textContent = '';
    tip.append(el('b', null, target.dataset.tipV || ''), target.dataset.tip);
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  document.addEventListener('pointermove', (e) => {
    const t = e.target.closest && e.target.closest('[data-tip]');
    if (t) showTip(t, e.clientX, e.clientY); else tip.hidden = true;
  });
  document.addEventListener('focusin', (e) => {
    const t = e.target.closest && e.target.closest('[data-tip]');
    if (t) { const r = t.getBoundingClientRect(); showTip(t, r.right, r.top); }
  });
  document.addEventListener('focusout', () => { tip.hidden = true; });

  /* ---- grow animations replay on first sight; bars are full width at rest ---- */
  const io = 'IntersectionObserver' in window ? new IntersectionObserver((es) => {
    es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('play'); io.unobserve(e.target); } });
  }, { threshold: 0.25 }) : null;
  function watch(n) { if (io && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) io.observe(n); }

  /* ---- leaderboards (paper Tables 1 and 2, overall averages) ---- */
  const MMSI = [
    ['S-Agent 2.0', 'Gemini-3.8-Flash', 74.1, 'ours'], ['GPT-6', '', 71.0, ''], ['SpatialClaw', 'Gemini-3.8-Flash', 67.0, 'agent'],
    ['S-Agent 2.0', 'Qwen3.8-27B', 66.5, 'ours'], ['SpatialClaw', 'Qwen3.8-27B', 62.6, 'agent'], ['Gemini 3.8 Flash', '', 60.0, ''],
    ['GPT-5.6', '', 53.2, ''], ['Qwen3.8-27B', '', 50.4, ''], ['S-Agent', 'Qwen3.8-27B', 48.3, 'agent'], ['Gemini 3 Pro', '', 45.2, ''], ['GPT-5.4', '', 41.9, ''],
  ];
  const REVSI = [
    ['S-Agent 2.0', 'Gemini-3.8-Flash', 90.5, 'ours'], ['GPT-6', '', 84.2, ''], ['S-Agent 2.0', 'Qwen3.8-27B', 82.3, 'ours'],
    ['SpatialClaw', 'Gemini-3.8-Flash', 79.7, 'agent'], ['Gemini 3.8 Flash', '', 77.0, ''], ['SpatialClaw', 'Qwen3.8-27B', 74.5, 'agent'],
    ['GPT-5.6', '', 72.0, ''], ['Qwen3.8-27B', '', 61.3, ''], ['Gemini 3 Pro', '', 60.9, ''], ['S-Agent', 'Qwen3.8-27B', 59.5, 'agent'], ['GPT-5.4', '', 54.0, ''],
  ];
  function leaderboard(node, rows, lo, hi, bench) {
    const g = el('div', 'lb');
    rows.forEach(([name, back, v, cls]) => {
      const n = el('div', 'name' + (cls === 'ours' ? ' ours' : ''));
      n.textContent = name;
      if (back) n.appendChild(el('small', null, back));
      const tr = el('div', 'track');
      const b = el('div', 'bar' + (cls ? ' ' + cls : ''));
      b.style.width = ((v - lo) / (hi - lo) * 100).toFixed(2) + '%';
      tr.appendChild(b);
      tr.dataset.tip = name + (back ? ' (' + back + ')' : '') + ' · ' + bench;
      tr.dataset.tipV = v.toFixed(1);
      tr.tabIndex = 0;
      const val = el('div', 'val' + (cls === 'ours' ? ' ours' : ''), v.toFixed(1));
      g.append(n, tr, val);
    });
    const ax = el('div', 'axis');
    for (let t = lo; t <= hi; t += 10) ax.appendChild(el('span', null, String(t)));
    g.append(el('div', 'spacer'), ax, el('div', 'spacer'));
    node.appendChild(g);
    watch(node);
  }
  const lbM = $('#lb-mmsi'), lbR = $('#lb-revsi');
  if (lbM) leaderboard(lbM, MMSI, 40, 80, 'MMSI-Bench avg.');
  if (lbR) leaderboard(lbR, REVSI, 50, 100, 'ReVSI avg.');

  /* ---- five benchmarks vs agentic systems (paper Table 3, Qwen3.8-27B) ---- */
  const T3 = {
    cols: ['MMSI-Bench', 'MindCube', 'SPAR-Bench', 'MMSI-Video', 'ReVSI'],
    rows: [
      ['pySpatial', [36.3, 66.3, 42.1, 26.9, 39.3]], ['SpaceTools', [43.2, 58.6, 55.3, 39.3, 64.4]],
      ['S-Agent', [48.3, 75.5, 63.0, 37.9, 59.5]], ['SpatialClaw', [62.6, 83.2, 70.2, 53.1, 74.5]],
      ['S-Agent 2.0', [66.5, 88.3, 72.4, 56.4, 82.3], 'ours'],
    ],
  };
  const t3 = $('#t3');
  if (t3) {
    const tb = el('table', 'bench');
    const thead = el('thead'); const hr = el('tr');
    hr.appendChild(el('th', null, 'System'));
    T3.cols.forEach((c) => hr.appendChild(el('th', null, c)));
    thead.appendChild(hr); tb.appendChild(thead);
    const body = el('tbody');
    const sc = T3.rows[3][1];
    T3.rows.forEach(([name, vals, cls]) => {
      const tr = el('tr', cls || '');
      tr.appendChild(el('td', null, name));
      vals.forEach((v, i) => {
        const td = el('td', 'num');
        const s = el('span', 'v', v.toFixed(1)); td.appendChild(s);
        if (cls) td.appendChild(el('span', 'd', '+' + (v - sc[i]).toFixed(1)));
        const m = el('div', 'mini'); const bar = el('i'); bar.style.width = v + '%'; m.appendChild(bar); td.appendChild(m);
        td.dataset.tip = name + ' · ' + T3.cols[i]; td.dataset.tipV = v.toFixed(1);
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    tb.appendChild(body);
    t3.appendChild(tb);
  }

  /* ---- tool gain dumbbell (paper Table 5) ---- */
  const T5 = [
    ['Qwen3.8-27B', 'ReVSI', [61.3, 74.5], [61.8, 82.3]],
    ['Qwen3.8-27B', 'MMSI-Bench', [50.4, 62.6], [52.6, 66.5]],
    ['Gemini-3.8-Flash', 'ReVSI', [77.0, 79.7], [80.9, 90.5]],
    ['Gemini-3.8-Flash', 'MMSI-Bench', [60.0, 67.0], [62.9, 74.1]],
  ];
  const db = $('#dumbbell');
  if (db) {
    const NS = 'http://www.w3.org/2000/svg';
    const W = 520, padL = 8, padR = 58, rowH = 24, labH = 22, groupGap = 16, top = 4;
    const lo = 45, hi = 95;
    const H = top + T5.length * (labH + rowH * 2 + groupGap) + 18;
    const x = (v) => padL + (v - lo) / (hi - lo) * (W - padL - padR);
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Score without and with the toolkit for SpatialClaw and S-Agent 2.0 on two backbones and two benchmarks');
    const mk = (tag, attrs, text) => { const n = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v)); if (text != null) n.textContent = text; svg.appendChild(n); return n; };
    const axisY = H - 18;
    for (let t = 50; t <= 90; t += 10) {
      mk('line', { x1: x(t), x2: x(t), y1: top, y2: axisY, class: 'grid' });
      mk('text', { x: x(t), y: axisY + 15, 'text-anchor': 'middle' }, String(t));
    }
    let y = top;
    T5.forEach(([bb, bench, sc, us]) => {
      const lab = mk('text', { x: padL, y: y + 15, class: 'ylab' }, bench + ' ');
      const sub = document.createElementNS(NS, 'tspan');
      sub.setAttribute('style', 'fill:var(--ink3);font-size:12px');
      sub.textContent = '\u00b7 ' + bb;
      lab.appendChild(sub);
      y += labH;
      [['SpatialClaw', sc, 'var(--base)'], ['S-Agent 2.0', us, 'var(--ours)']].forEach(([name, [a, b], col], k) => {
        const cy = y + k * rowH + rowH / 2;
        mk('line', { x1: x(a), x2: x(b), y1: cy, y2: cy, stroke: col, 'stroke-width': 2, 'stroke-linecap': 'round' });
        mk('circle', { cx: x(a), cy, r: 5, fill: 'var(--panel)', stroke: col, 'stroke-width': 2 });
        mk('circle', { cx: x(b), cy, r: 5.5, fill: col, stroke: 'var(--panel)', 'stroke-width': 2 });
        mk('text', { x: x(b) + 11, y: cy + 4.5, class: 'gain' }, '+' + (b - a).toFixed(1));
        const hit = mk('rect', { x: x(a) - 9, y: cy - 11, width: x(b) - x(a) + 18, height: 22, fill: 'transparent', tabindex: 0 });
        hit.dataset.tip = name + ' \u00b7 ' + bench + ' \u00b7 ' + bb + ': ' + a.toFixed(1) + ' without the toolkit, ' + b.toFixed(1) + ' with it';
        hit.dataset.tipV = '+' + (b - a).toFixed(1);
      });
      y += rowH * 2 + groupGap;
    });
    db.appendChild(svg);
  }

  /* ---- toolkit usage heat table (paper Table 8) ---- */
  const T8 = [
    ['recon', 'tool', [1.00, 100], [1.00, 100], [1.00, 100]], ['segment', 'tool', [1.75, 93], [1.47, 92], [2.25, 100]],
    ['add_instance', 'model', [1.47, 36], [0.72, 26], [1.48, 48]], ['drop', 'model', [0.37, 26], [0.27, 21], [0.53, 40]],
    ['list_instance', 'readout', [0.29, 24], [0.52, 45], [1.11, 91]], ['bev', 'readout', [1.66, 94], [1.67, 94], [1.74, 98]],
    ['distance', 'readout', [1.53, 46], [0.37, 17], [2.86, 75]], ['depth', 'readout', [0.18, 12], [0.02, 2], [0.01, 1]],
    ['side', 'readout', [1.66, 62], [1.06, 53], [0.50, 29]], ['elevation', 'readout', [0.17, 7], [0.11, 7], [0.11, 8]],
    ['motion', 'readout', [1.08, 73], [0.91, 66], [0.06, 3]], ['compass', 'readout', [0.55, 25], [0.69, 25], [0.01, 0]],
  ];
  const heat = $('#heat');
  if (heat) {
    const cols = ['Qwen3.8-27B · MMSI', 'Gemini-3.8-Flash · MMSI', 'Gemini-3.8-Flash · ReVSI'];
    const tb = el('table', 'heat');
    const hr = el('tr'); hr.appendChild(el('th', null, 'Command'));
    cols.forEach((c) => hr.appendChild(el('th', null, c)));
    const th = el('thead'); th.appendChild(hr); tb.appendChild(th);
    const body = el('tbody');
    T8.forEach(([cmd, kind, ...cells]) => {
      const tr = el('tr');
      const c0 = el('td'); c0.appendChild(el('span', 'chip ' + kind, cmd)); tr.appendChild(c0);
      cells.forEach(([calls, share], i) => {
        const td = el('td');
        td.style.background = `rgba(47,158,99,${(0.06 + share / 100 * 0.52).toFixed(3)})`;
        td.style.color = 'var(--ink)';
        td.append(calls.toFixed(2));
        td.appendChild(el('small', null, share + '% of episodes'));
        td.dataset.tip = cmd + ' · ' + cols[i] + ': ' + calls.toFixed(2) + ' calls per episode';
        td.dataset.tipV = share + '%';
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    tb.appendChild(body);
    heat.appendChild(tb);
  }

  /* ---- hero trajectory sample (MMSI_58, real run) ---- */
  const hero = window.SAGENT_TRAJ && window.SAGENT_TRAJ.id === 'MMSI_58' ? window.SAGENT_TRAJ : null;
  if (hero && window.SAgentTraj) window.SAgentTraj.remember(hero);
  const stage = $('#stage'), rail = $('#rail');
  if (hero && stage && rail) {
    const B = 'trajectories/';
    const S = hero.steps;
    const find = (pred) => S.find(pred) || {};
    const recon = find((s) => s.cmd === 'recon');
    const seg = find((s) => s.cmd === 'segment');
    const bev1 = find((s) => s.cmd === 'bev');
    const adds = S.filter((s) => s.cmd === 'add_instance');
    const bev2 = S.filter((s) => s.cmd === 'bev')[1] || bev1;
    const side = find((s) => s.cmd === 'side');
    const gtText = (hero.options.find((o) => o.key === hero.gt) || {}).text || '';
    const slides = [
      { k: 'q', label: 'Question', title: hero.question, q: true, imgs: hero.inputs },
      { k: 'tool', label: 'Tool perception', title: 'Reconstruct and segment the scene', cmds: [recon.line, seg.line], out: '→ parking lot · bed · headboard registered', imgs: seg.images },
      { k: 'readout', label: 'Readout', title: 'Inspect the top-down map', cmds: [bev1.line], imgs: (bev1.images || []).slice(0, 1), one: true },
      { k: 'model', label: 'Model perception', title: 'Place the observer in bed', cmds: adds.map((a) => a.line), out: '→ lying_person: top toward the headboard, forward toward the ceiling', imgs: (bev2.images || []).slice(0, 1), one: true },
      { k: 'readout', label: 'Readout', title: 'Read from the observer’s frame', cmds: [side.line], out: side.conclusion, imgs: hero.inputs },
      { k: 'answer', label: 'Answer', title: gtText, answer: true },
    ];
    const short = (line) => line.replace(/ --source "[^"]*"/, ' --source "…"');
    slides.forEach((sl, i) => {
      const d = el('div', 'slide');
      d.id = 'slide-' + i;
      const kc = { q: 'var(--accent)', tool: 'var(--tool)', model: 'var(--model)', readout: 'var(--readout)', answer: 'var(--accent)' }[sl.k];
      const kind = el('div', 'kind'); kind.style.color = kc;
      kind.append(el('span', 'chip', sl.label), el('span', 'n', String(i).padStart(2, '0') + ' / ' + String(slides.length - 1).padStart(2, '0')));
      d.appendChild(kind);
      if (sl.answer) {
        const a = el('div', 'answer');
        a.appendChild(el('div', 'big', hero.gt));
        const t = el('div');
        t.appendChild(el('h3', null, sl.title));
        t.appendChild(el('div', 'q', 'Side readout · Ground truth: ' + hero.gt));
        t.appendChild(el('span', 'ok', '✓ correct'));
        a.appendChild(t);
        d.appendChild(a);
      } else {
        d.appendChild(el('h3', sl.q ? 'q' : null, sl.title));
        if (sl.q) {
          const o = el('div', 'opts');
          hero.options.forEach((op) => o.appendChild(el('span', null, op.key + '  ' + op.text)));
          d.appendChild(o);
        }
        if (sl.cmds) {
          const pre = el('pre');
          sl.cmds.filter(Boolean).forEach((c, j) => { if (j) pre.append('\n'); pre.appendChild(el('span', 'c', '$ ' + short(c))); });
          if (sl.out) { pre.append('\n'); pre.appendChild(el('span', 'r', sl.out)); }
          d.appendChild(pre);
        }
        if (sl.imgs && sl.imgs.length) {
          const g = el('div', 'imgs' + (sl.one ? ' one' : ''));
          sl.imgs.forEach((im) => {
            const f = el('figure');
            const img = el('img'); img.src = B + im.src; img.alt = im.cap; f.appendChild(img);
            f.appendChild(el('figcaption', null, im.cap));
            g.appendChild(f);
          });
          d.appendChild(g);
        }
      }
      stage.appendChild(d);
      const b = el('button', null, ['Q', 'Tool', 'Map', 'Model', 'Read', 'Answer'][i] || sl.label);
      b.type = 'button'; b.dataset.kind = sl.k; b.setAttribute('aria-controls', d.id);
      b.setAttribute('aria-label', 'Step ' + i + ': ' + sl.label);
      b.addEventListener('click', () => { go(i); stop(); });
      rail.appendChild(b);
    });
    let cur = 0, timer = null;
    function go(i) {
      cur = (i + slides.length) % slides.length;
      $$('.slide', stage).forEach((s, j) => s.classList.toggle('on', j === cur));
      $$('button', rail).forEach((b, j) => { b.classList.toggle('on', j === cur); b.setAttribute('aria-current', j === cur ? 'step' : 'false'); });
    }
    function start() { if (!timer && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) timer = setInterval(() => go(cur + 1), 4200); }
    function stop() { clearInterval(timer); timer = null; }
    go(0); start();
    const card = $('#sample');
    card.addEventListener('pointerenter', stop);
    card.addEventListener('pointerleave', start);
    card.addEventListener('focusin', stop);
  }

  /* ---- trajectory cards + modal ---- */
  const KINDCLS = { recon: 'tool', segment: 'tool', add_instance: 'model', drop: 'model' };
  const READ = ['list_instance', 'bev', 'side', 'distance', 'depth', 'elevation', 'motion', 'compass'];
  const grid = $('#cards');
  const cases = window.SAGENT_CASES || [];
  const meta = {};
  cases.forEach((c) => { meta[c.id] = c; });
  if (grid && cases.length) {
    cases.forEach((c) => {
      const b = el('button', 'tcard');
      b.type = 'button';
      b.dataset.bench = c.bench;
      b.setAttribute('aria-label', 'Open trajectory ' + c.id + ': ' + c.short_label);
      const img = el('img'); img.src = 'trajectories/' + c.thumb; img.alt = ''; img.loading = 'lazy';
      b.appendChild(img);
      const body = el('div', 'b');
      const top = el('div', 'top');
      top.append(el('span', 'tag lab', c.short_label), el('span', 'tag', c.bench), el('span', 'tag', c.backbone));
      body.appendChild(top);
      body.appendChild(el('div', 'qq', c.question_short));
      const res = el('div', 'res');
      res.append('GT ' + c.gt + ' · S-Agent 2.0 ', el('b', null, c.pred + ' ✓'));
      body.appendChild(res);
      const pipe = el('div', 'pipe');
      c.pipeline.filter((p) => p !== 'view' && p !== 'answer' && p !== 'python').forEach((p) => {
        pipe.appendChild(el('span', KINDCLS[p] || (READ.includes(p) ? 'readout' : ''), p));
      });
      body.appendChild(pipe);
      body.appendChild(el('div', 'open', 'View trajectory →'));
      b.appendChild(body);
      b.addEventListener('click', () => openCase(c.id));
      grid.appendChild(b);
    });
    $$('#filters button').forEach((f) => f.addEventListener('click', () => {
      $$('#filters button').forEach((x) => x.setAttribute('aria-pressed', String(x === f)));
      const v = f.dataset.f;
      $$('.tcard', grid).forEach((c) => { c.hidden = v !== 'all' && c.dataset.bench !== v; });
    }));
  }

  const modal = $('#modal'), mbody = $('#modal-body');
  let lastFocus = null;
  function openCase(id) {
    if (!modal || !window.SAgentTraj) return;
    lastFocus = document.activeElement;
    mbody.textContent = '';
    mbody.appendChild(el('div', 'tv-loading', 'Loading ' + id + ' …'));
    modal.hidden = false;
    document.body.classList.add('locked');
    $('.close', modal).focus();
    try { history.replaceState(null, '', '#traj-' + id); } catch (e) { /* sandboxed */ }
    window.SAgentTraj.load(id, 'trajectories/').then((t) => {
      window.SAgentTraj.render(mbody, t, 'trajectories/', meta[id]);
      modal.scrollTop = 0;
    }).catch((err) => { mbody.textContent = ''; mbody.appendChild(el('div', 'tv-loading', err.message)); });
  }
  function closeCase() {
    if (!modal || modal.hidden) return;
    modal.hidden = true;
    document.body.classList.remove('locked');
    try { history.replaceState(null, '', location.pathname + location.search); } catch (e) { /* sandboxed */ }
    if (lastFocus) lastFocus.focus();
  }
  if (modal) {
    $('.close', modal).addEventListener('click', closeCase);
    modal.addEventListener('click', (e) => { if (e.target === modal) closeCase(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeCase(); });
  }
  $$('[data-open]').forEach((b) => b.addEventListener('click', () => openCase(b.dataset.open)));
  const m = /^#traj-([A-Za-z0-9_]+)$/.exec(location.hash);
  if (m && meta[m[1]]) openCase(m[1]);

})();
