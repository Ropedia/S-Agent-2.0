/* Trajectory viewer shared by the project page modal and trajectories/index.html.
   Case files (trajectories/cases/<ID>.js) set window.SAGENT_TRAJ; image paths inside them are
   relative to trajectories/. All case text goes into the DOM as text, never as markup. */
(function () {
  const KIND = {
    tool: ['k-tool', 'Tool perception'],
    model: ['k-model', 'Model perception'],
    readout: ['k-readout', 'Readout'],
    view: ['k-view', 'Agent looks'],
    code: ['k-code', 'Shell'],
    answer: ['k-answer', 'Answer'],
  };
  const cache = {};
  let queue = Promise.resolve();

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  // Case files all assign the same global, so loads run one at a time.
  function load(id, base) {
    if (cache[id]) return Promise.resolve(cache[id]);
    queue = queue.then(() => new Promise((resolve, reject) => {
      if (cache[id]) return resolve(cache[id]);
      const s = document.createElement('script');
      s.src = base + 'cases/' + encodeURIComponent(id) + '.js';
      s.onload = () => { cache[id] = window.SAGENT_TRAJ; s.remove(); resolve(cache[id]); };
      s.onerror = () => { s.remove(); reject(new Error('Could not load trajectory ' + id)); };
      document.head.appendChild(s);
    }));
    return queue.then(() => cache[id]);
  }

  function remember(data) { if (data && data.id) cache[data.id] = data; }

  /* lightbox */
  let lb;
  function lightbox(src, cap) {
    if (!lb) {
      lb = el('div', 'lightbox');
      lb.hidden = true;
      lb.setAttribute('role', 'dialog');
      lb.setAttribute('aria-label', 'Image preview');
      lb.append(el('img'), el('div', 'c'));
      lb.addEventListener('click', () => { lb.hidden = true; });
      document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !lb.hidden) { lb.hidden = true; e.stopPropagation(); } }, true);
      document.body.appendChild(lb);
    }
    lb.querySelector('img').src = src;
    lb.querySelector('img').alt = cap || '';
    lb.querySelector('.c').textContent = cap || '';
    lb.hidden = false;
  }

  function figure(src, cap) {
    const f = el('figure');
    const b = el('button');
    b.type = 'button';
    b.setAttribute('aria-label', 'Enlarge ' + (cap || 'image'));
    const img = el('img');
    img.src = src; img.alt = cap || ''; img.loading = 'lazy';
    b.appendChild(img);
    b.addEventListener('click', () => lightbox(src, cap));
    f.appendChild(b);
    if (cap) f.appendChild(el('figcaption', null, cap));
    return f;
  }

  function verdictText(t) {
    if (!t.options || !t.options.length) {
      return (t.correct ? '✓ ' : '') + t.pred + ' · GT ' + t.gt + (t.metric_note ? ' · ' + t.metric_note : '');
    }
    return (t.correct ? '✓ correct · ' : '') + 'answer ' + t.pred + ' · GT ' + t.gt;
  }

  function pipeline(steps) {
    const out = [];
    steps.forEach((s) => {
      const last = out[out.length - 1];
      if (!last || last.cmd !== s.cmd) out.push({ cmd: s.cmd, kind: s.kind });
    });
    return out;
  }

  function render(root, t, base, meta) {
    base = base || '';
    root.textContent = '';
    const wrap = el('div', 'tv');

    const head = el('div', 'tv-head');
    const left = el('div');
    const eb = el('div', 'eyebrow');
    eb.append('Real trajectory · ', el('b', null, t.id));
    left.appendChild(eb);
    left.appendChild(el('h2', null, (meta && meta.short_label) || t.category));
    const tags = el('div', 'tv-tags');
    [t.bench, t.category, t.backbone, t.harness].filter(Boolean).forEach((x, i) => tags.appendChild(el('span', 'tag' + (i === 0 ? ' lab' : ''), x)));
    left.appendChild(tags);
    head.appendChild(left);
    const v = el('div', 'verdict');
    v.textContent = verdictText(t);
    head.appendChild(v);
    wrap.appendChild(head);

    const q = el('div', 'tv-q');
    const qa = el('div');
    qa.appendChild(el('div', 'eyebrow', 'Question'));
    const stem = el('div', 'stem', t.question);
    stem.style.marginTop = '10px';
    qa.appendChild(stem);
    if (t.options && t.options.length) {
      const o = el('div', 'tv-opts');
      t.options.forEach((op) => {
        const d = el('div', op.key === t.gt ? 'gt' : '');
        d.append(el('b', null, op.key), el('span', null, op.text));
        o.appendChild(d);
      });
      qa.appendChild(o);
    } else {
      qa.appendChild(el('div', 'tv-foot', 'Numerical answer. Ground truth ' + t.gt + (t.metric_note ? ' (' + t.metric_note + ')' : '') + '.'));
    }
    q.appendChild(qa);
    const qi = el('div');
    qi.appendChild(el('div', 'eyebrow', 'Input'));
    const ins = el('div', 'tv-inputs' + (t.inputs.length === 1 ? ' sheet' : ''));
    if (t.inputs.length > 1 && t.inputs.length <= 4) ins.style.gridTemplateColumns = 'repeat(2,minmax(0,1fr))';
    ins.style.marginTop = '10px';
    t.inputs.forEach((im) => ins.appendChild(figure(base + im.src, im.cap)));
    qi.appendChild(ins);
    q.appendChild(qi);
    wrap.appendChild(q);

    const pipe = el('div', 'tv-pipe');
    pipe.appendChild(el('span', 'lbl', 'Pipeline'));
    pipeline(t.steps).forEach((p, i) => {
      if (i) pipe.appendChild(el('span', 'arr', '→'));
      pipe.appendChild(el('span', 'chip ' + (p.kind === 'view' || p.kind === 'code' || p.kind === 'answer' ? 'generic' : p.kind), p.cmd));
    });
    wrap.appendChild(pipe);

    const legend = el('div', 'tv-legend');
    [['tool', 'Tool perception: recon, segment'], ['model', 'Model perception: add_instance, drop'], ['readout', 'Readouts'], ['view', 'Agent looks at an image'], ['code', 'Shell']].forEach(([k, label]) => {
      const s = el('span');
      const i = el('i', KIND[k][0]);
      i.style.background = 'currentColor';
      s.append(i, label);
      legend.appendChild(s);
    });
    wrap.appendChild(legend);

    const steps = el('div', 'tv-steps');
    const shown = new Set();
    t.steps.forEach((s, idx) => {
      const [kc, kl] = KIND[s.kind] || KIND.code;
      const row = el('div', 'tv-step ' + kc);
      row.appendChild(el('div', 'num', String(idx + 1)));
      const box = el('div', 'box');
      const hd = el('div', 'hd');
      hd.append(el('span', 'kind', kl), el('span', 'cmd', s.cmd));
      box.appendChild(hd);
      if (s.note) box.appendChild(el('div', 'note', s.note));
      if (s.line) box.appendChild(el('pre', 'line', s.line));
      if (s.conclusion) {
        box.appendChild(el('div', 'concl', s.conclusion));
        if (s.reply) {
          const d = el('details');
          d.append(el('summary', null, 'reply excerpt'), el('pre', null, s.reply));
          box.appendChild(d);
        }
      } else if (s.reply) {
        const n = s.reply.split('\n').length;
        if (n > 8) {
          const d = el('details');
          d.append(el('summary', null, 'reply excerpt \u00b7 ' + n + ' lines'), el('pre', null, s.reply));
          box.appendChild(d);
        } else {
          box.appendChild(el('pre', 'reply', s.reply));
        }
      }
      if (s.images && s.images.length) {
        const seen = s.images.every((im) => shown.has(im.src));
        const g = el('div', 'tv-imgs' + (seen ? ' small' : ''));
        s.images.forEach((im) => { g.appendChild(figure(base + im.src, im.cap)); shown.add(im.src); });
        box.appendChild(g);
      }
      row.appendChild(box);
      steps.appendChild(row);
    });
    wrap.appendChild(steps);

    const fin = el('div', 'tv-final');
    fin.appendChild(el('div', 'big', t.final && t.final.answer != null ? String(t.final.answer) : t.pred));
    const ft = el('div');
    const opt = (t.options || []).find((o) => o.key === t.pred);
    ft.appendChild(el('div', 't', (opt ? opt.text + ' · ' : '') + (t.correct ? 'matches the ground truth' : 'ground truth ' + t.gt)));
    if (t.final && t.final.text) ft.appendChild(el('div', 's', t.final.text));
    fin.appendChild(ft);
    wrap.appendChild(fin);

    wrap.appendChild(el('div', 'tv-foot', 'Run excerpts: "python3 /app/perception.py" is shortened to "perception.py"; … marks omissions. Step notes are commentary; shell detours are marked.'));
    root.appendChild(wrap);
  }

  window.SAgentTraj = { load, render, remember, lightbox };
})();
