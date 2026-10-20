'use strict';
/* LogiX-GIN Explorer (explorer/__init__.py). Plain JS + SVG, no dependency.
 * Every value shown comes from the symbolic reading computed in explorer/reader.py; the only
 * computation done here is drawing the step functions phi(w*x + b) the literals read. */

const TWO_PI = 2 * Math.PI;
const SVGNS = 'http://www.w3.org/2000/svg';

// ------------------------------------------------------------------ DOM helpers

function setAttrs(e, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else if (k === 'text') e.textContent = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
}
function append(e, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k === null || k === undefined || k === false) continue;
    e.appendChild(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return e;
}
function el(tag, attrs, ...kids) { const e = document.createElement(tag); setAttrs(e, attrs); return append(e, kids); }
function sv(tag, attrs, ...kids) { const e = document.createElementNS(SVGNS, tag); setAttrs(e, attrs); return append(e, kids); }
function clear(e) { while (e.firstChild) e.removeChild(e.firstChild); return e; }
const $ = id => document.getElementById(id);

const f2 = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? '–' : Number(x).toFixed(d);
const pct = (x, d = 1) => (x === null || x === undefined) ? '–' : (100 * x).toFixed(d) + '%';
const sci = x => (x === null || x === undefined) ? '> 1e308' : x < 1e4 ? String(Math.round(x)) : x.toExponential(1).replace('e+', 'e');
const sgn = x => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(2);

// tooltip
const tip = {
  show(ev, text) { const t = $('tip'); t.textContent = text; t.hidden = false; tip.move(ev); },
  move(ev) {
    const t = $('tip'); if (t.hidden) return;
    const x = Math.min(ev.clientX + 14, window.innerWidth - t.offsetWidth - 8);
    const y = ev.clientY + 16 + t.offsetHeight > window.innerHeight ? ev.clientY - t.offsetHeight - 10 : ev.clientY + 16;
    t.style.left = x + 'px'; t.style.top = y + 'px';
  },
  hide() { $('tip').hidden = true; },
};
function hoverTip(node, text) {
  node.addEventListener('mouseenter', ev => tip.show(ev, typeof text === 'function' ? text() : text));
  node.addEventListener('mousemove', tip.move);
  node.addEventListener('mouseleave', tip.hide);
  return node;
}

// ------------------------------------------------------------------ the literal step function

/* phi_in of a LogicalLayer: step(z) = clamp((F(z) / F(pi/2) + 1) / 2, 0, 1), F the partial
 * Fourier series of a square wave with tau + 1 terms. The literal is TRUE iff
 * z mod 2pi lies in [0, pi] (min_covers.literal_true_intervals), for every tau. */
function fourier(z, tau) { let r = 0; for (let i = 0; i <= tau; i++) r += Math.sin((2 * i + 1) * z) / (2 * i + 1); return r; }
function phi(z, tau) { return Math.min(1, Math.max(0, (fourier(z, tau) / fourier(Math.PI / 2, tau) + 1) / 2)); }
function litTrue(z) { const m = ((z % TWO_PI) + TWO_PI) % TWO_PI; return m <= Math.PI; }

// ------------------------------------------------------------------ data sources

class ServerSource {
  constructor() { this.kind = 'server'; }
  async get(route, q = {}) {
    const r = await fetch(`api/${route}?` + new URLSearchParams(q));
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  }
  runs() { return this.get('runs'); }
  open(run) { return this.get('open', { run }); }
  index(run, split) { return this.get('index', { run, split }); }
  trace(run, split, i) { return this.get('trace', { run, split, i }); }
  has() { return true; }
  splits() { return ['test', 'val', 'train']; }
}

class BundleSource {
  constructor(b) { this.b = b; this.kind = 'bundle'; }
  async runs() { return [{ id: 0, label: this.b.meta.path, dataset: this.b.meta.dataset, level: this.b.card.level }]; }
  async open() { return { meta: this.b.meta, card: this.b.card }; }
  async index(run, split) {
    if (!this.b.index[split]) throw new Error(`split ${split} is not in this export`);
    return this.b.index[split];
  }
  async trace(run, split, i) {
    const t = this.b.traces[`${split}:${i}`];
    if (!t) throw new Error(`graph ${split} #${i} is not in this export (python -m explorer export --graphs N, or serve the run)`);
    return t;
  }
  has(split, i) { return `${split}:${i}` in this.b.traces; }
  splits() { return Object.keys(this.b.index); }
}

// ------------------------------------------------------------------ state

const S = {
  src: null, runs: [], run: null, meta: null, card: null, split: 'test', index: null,
  trace: null, adj: null, node: 0, sel: null, hoverLit: null, step: null, timer: null,
  tab: 'predict', filter: 'all', query: '', unitLayer: 0, unitSel: null, unitSort: 'u',
  protoSort: 'cites', hideTrivial: true, showAllLits: false, showConst: false,
  base: null, edits: null, editMode: false, pending: null, hoverNodes: null,
};

async function busy(text, fn) {
  $('busy-text').textContent = text; $('busy').hidden = false;
  try { return await fn(); } finally { $('busy').hidden = true; }
}
function fail(e) { console.error(e); alert(e.message || String(e)); }

// ------------------------------------------------------------------ model accessors

const card = () => S.card;
const T = () => S.trace;
const outBit = (l, v, u) => T().layers[l].out[v][u] === '1';
const litBit = (l, v, i) => T().layers[l].lit[v][i] === '1';
const xBit = (v, i) => T().x[v][i] === '1';
const className = c => card().classes[c];
const hasCare = (li, k) => card().protos[li].care[k].includes('1');
const caredBits = (li, k) => { const s = card().protos[li].care[k], out = []; for (let d = 0; d < s.length; d++) if (s[d] === '1') out.push(d); return out; };
const nodeProtoLayer = () => card().protos.findIndex(p => p.kind === 'node');
const unitName = (l, u) => `L${l}u${u}`;
const protoName = (li, k) => (card().protos[li].kind === 'graph' && card().level === 'both' ? 'G' : 'P') + k;
function nodeName(v) { const lab = T().labels[v]; return lab ? `${lab}${v}` : `node ${v}`; }
function nbh(v) { return [v, ...S.adj[v]]; }

function protoSim(li, k, v = S.node) {
  const p = T().protos[li];
  return card().protos[li].kind === 'node' ? p.sim[v][k] : p.sim[k];
}
function protoMism(li, k, v = S.node) {
  const p = T().protos[li];
  return card().protos[li].kind === 'node' ? p.mism[v][k] : p.mism[k];
}
function bestNode(k, li = nodeProtoLayer()) {
  if (li < 0) return 0;
  const sim = T().protos[li].sim; let b = 0;
  for (let v = 1; v < sim.length; v++) if (sim[v][k] > sim[b][k] + 1e-9) b = v;
  return b;
}
/* the (proto layer, k) a head literal reads, or null for a classic model */
function litProto(i) {
  const col = card().readout[card().head.literals[i].col];
  return col.k === undefined ? null : { li: col.proto_layer, k: col.k };
}
function defaultNode() {
  for (const i of T().evidence || []) {
    const lit = card().head.literals[i], p = litProto(i);
    if (p && lit.pos && card().protos[p.li].kind === 'node' && hasCare(p.li, p.k)) return bestNode(p.k, p.li);
  }
  return 0;
}
/* does the subject of conv literal i of layer l hold at node w (what the count counts) */
function subjectHolds(l, i, w) {
  const n = card().conv[l].n_in, base = i % n, neg = i >= n;
  const h = l === 0 ? xBit(w, base) : outBit(l - 1, w, base);
  return neg ? !h : h;
}
function protoBitsAt(li, v) {
  /* the input the prototype layer compares at node v (node kind) or for the graph (graph kind) */
  const p = card().protos[li];
  if (p.kind === 'graph') return T().protos[li].input;
  return p.bit_unit.map(unit => outBit(Math.floor(unit / card().hidden), v, unit % card().hidden) ? 1 : 0);
}

// ------------------------------------------------------------------ boot and header

async function boot() {
  if (window.EXPLORER_BUNDLE) S.src = new BundleSource(window.EXPLORER_BUNDLE);
  else if (location.protocol.startsWith('http')) S.src = new ServerSource();
  else return renderOpenFile();
  try {
    S.runs = await S.src.runs();
    const h = new URLSearchParams(location.hash.slice(1));
    const id = h.has('run') ? +h.get('run') : S.runs[0].id;
    await openRun(id, h.get('split'), h.has('i') ? +h.get('i') : null, h.has('node') ? +h.get('node') : null);
    applyHash(h);
  } catch (e) { fail(e); }
}

/* #sel=unit.1.12 | proto.0.18 | class.0 | headlit.5 | readout.3, #tab=protos, #step=2 */
function applyHash(h) {
  const sel = h.get('sel');
  if (sel) {
    const [type, a, b] = sel.split('.');
    const n = [a, b].map(Number);
    S.sel = { unit: { type, l: n[0], u: n[1] }, proto: { type, li: n[0], k: n[1] }, class: { type, c: n[0] }, headlit: { type, i: n[0] }, readout: { type, j: n[0] } }[type] || S.sel;
  }
  if (h.has('tab')) S.tab = h.get('tab');
  if (h.has('step')) S.step = +h.get('step');
  if (h.has('layer')) { S.unitLayer = +h.get('layer'); S.unitSel = h.has('unit') ? +h.get('unit') : null; }
  renderTabs(); renderMain();
}

function renderOpenFile() {
  const input = el('input', { type: 'file', accept: '.json', onchange: async ev => {
    const f = ev.target.files[0]; if (!f) return;
    window.EXPLORER_BUNDLE = JSON.parse(await f.text()); boot();
  } });
  append(clear($('main')), el('div', { class: 'drop panel' },
    el('h2', { text: 'Open an explorer export' }),
    el('p', { class: 'muted', text: 'python -m explorer export <run> --json out.json, then pick the file. Or run python -m explorer serve.' }),
    input));
}

async function openRun(id, split = null, gi = null, node = null) {
  S.run = id;
  const { meta, card: c } = await busy('Reading the model: the first open measures literal ranges, rule counts and prototype matches on the reference split…',
    () => S.src.open(id));
  S.meta = meta; S.card = c;
  const splits = S.src.splits();
  S.split = split && splits.includes(split) ? split : splits[0];
  await loadIndex();
  const rows = S.index.rows;
  const first = gi !== null && S.src.has(S.split, gi) ? gi : (rows.find(r => S.src.has(S.split, r.i)) || rows[0]).i;
  await openGraph(S.split, first, { node });
}

async function loadIndex() {
  S.index = await busy(`Reading the ${S.split} split…`, () => S.src.index(S.run, S.split));
  renderHeader();
}

async function openGraph(split, i, opts = {}) {
  try {
    const t = await S.src.trace(S.run, split, i);
    setTrace(t);
    S.base = null; S.edits = { x: {}, del: [], add: [] }; S.editMode = false; S.pending = null;
    S.node = opts.node !== null && opts.node !== undefined && opts.node < t.n ? opts.node : defaultNode();
    S.sel = opts.sel || { type: 'class', c: t.rule };
    stopPlay();
    S.tab = 'predict';
    renderTabs(); renderMain();
    location.hash = new URLSearchParams({ run: S.run, split, i, node: S.node }).toString();
  } catch (e) { fail(e); }
}

function setTrace(t) {
  S.trace = t;
  S.adj = Array.from({ length: t.n }, () => []);
  for (const [a, b] of t.edges) { S.adj[a].push(b); S.adj[b].push(a); }
}

// ------------------------------------------------------------------ what-if edits

const edgeKey = (a, b) => a < b ? `${a}-${b}` : `${b}-${a}`;
const editCount = () => S.edits ? Object.keys(S.edits.x).length + S.edits.del.length + S.edits.add.length : 0;
function toggleEdit() {
  if (S.src.kind !== 'server') { alert('What-if edits need the model: run python -m explorer serve.'); return; }
  S.editMode = !S.editMode; S.pending = null;
  if (S.editMode && !S.base) S.base = S.trace;
  stopPlay(); refresh();
}
async function applyEdits() {
  if (!editCount()) { setTrace(S.base); refresh(); return; }
  try {
    const t = await S.src.get('whatif', { run: S.run, split: S.base.split, i: S.base.i, edits: JSON.stringify(S.edits) });
    t.pos = S.base.pos;                    // keep the drawing still
    setTrace(t);
    if (S.sel.type === 'class') S.sel = { type: 'class', c: t.rule };
    refresh();
  } catch (e) { fail(e); }
}
function editNodeFeatures(v, bitsStr) {
  if (bitsStr === S.base.x[v]) delete S.edits.x[v]; else S.edits.x[v] = bitsStr;
  applyEdits();
}
function clickEdge(a, b) {
  const k = edgeKey(a, b), ia = S.edits.add.findIndex(([x, y]) => edgeKey(x, y) === k);
  if (ia >= 0) S.edits.add.splice(ia, 1); else S.edits.del.push([a, b]);
  applyEdits();
}
function shiftNode(v) {
  if (S.pending === null || S.pending === v) { S.pending = S.pending === v ? null : v; renderGraph(); return; }
  const a = S.pending, k = edgeKey(a, v);
  S.pending = null;
  const id = S.edits.del.findIndex(([x, y]) => edgeKey(x, y) === k);
  if (id >= 0) S.edits.del.splice(id, 1);
  else if (!T().edges.some(([x, y]) => edgeKey(x, y) === k)) S.edits.add.push([a, v]);
  applyEdits();
}
function resetEdits() { S.edits = { x: {}, del: [], add: [] }; S.pending = null; applyEdits(); }

function editorBar() {
  const c = card(), v = S.node, t = T();
  const bar = el('div', { class: 'editor' });
  bar.append(el('div', { class: 'muted small' }, 'What-if: click a node, then change its features below · click an edge to delete it · shift-click two nodes to add an edge. The rules are recomputed exactly.'));
  const cur = t.x[v], one = cur.split('').filter(b => b === '1').length === 1 && S.base.x.every(r => r.split('').filter(b => b === '1').length === 1);
  const row = el('div', { class: 'editrow' }, el('b', {}, nodeName(v)));
  if (one) {
    row.append(el('select', { onchange: ev => editNodeFeatures(v, c.features.map((_, j) => j === +ev.target.value ? '1' : '0').join('')) },
      c.features.map((f, j) => el('option', { value: j, selected: cur[j] === '1' }, f))));
  } else {
    c.features.forEach((f, j) => row.append(el('label', { class: 'chip' }, el('input', { type: 'checkbox', checked: cur[j] === '1',
      onchange: ev => editNodeFeatures(v, cur.split('').map((b, jj) => jj === j ? (ev.target.checked ? '1' : '0') : b).join('')) }), ' ' + f)));
  }
  bar.append(row);
  const list = el('div', { class: 'chips' });
  for (const [vv, b] of Object.entries(S.edits.x)) list.append(el('span', { class: 'chip' }, `${S.base.labels[vv] || 'node '}${vv} → ${c.features.filter((_, j) => b[j] === '1').join('+') || '∅'}`));
  for (const [a, b] of S.edits.del) list.append(el('span', { class: 'chip' }, `− edge ${a}–${b}`));
  for (const [a, b] of S.edits.add) list.append(el('span', { class: 'chip' }, `+ edge ${a}–${b}`));
  if (S.pending !== null) list.append(el('span', { class: 'chip' }, `edge from ${S.pending}… shift-click the other end`));
  if (editCount()) list.append(el('button', { onclick: resetEdits }, 'reset'));
  bar.append(list);
  return bar;
}

/* units whose value changed (any node) between the original graph and the edited one */
function changedUnits() {
  if (!S.base || S.base === S.trace) return null;
  return S.trace.layers.map((L, l) => { let n = 0; L.out.forEach((r, v) => { const o = S.base.layers[l].out[v]; for (let u = 0; u < r.length; u++) if (r[u] !== o[u]) n++; }); return n; });
}

function renderHeader() {
  const bar = clear($('runbar'));
  if (S.runs.length > 1) {
    const sel = el('select', { title: 'run', onchange: ev => openRun(+ev.target.value).catch(fail) },
      S.runs.map(r => el('option', { value: r.id, selected: r.id === S.run }, `${r.label}${r.finetunes && !r.finetunes.length ? ' / base' : ''}${r.level ? ` · ${r.level}` : ''}`)));
    bar.append(sel);
  }
  const c = card(), m = S.meta;
  const splitSel = el('select', { title: 'split', onchange: async ev => { S.split = ev.target.value; await loadIndex(); renderMain(); } },
    S.src.splits().map(s => el('option', { value: s, selected: s === S.split }, s)));
  const sum = S.index ? S.index.summary : {};
  bar.append(
    el('span', { class: 'chip', title: `${c.model_class}\n${m.path}` }, el('b', {}, c.dataset), ` · ${c.level}`,
      (m.path.match(/\/fold(\d+)\//) || [])[1] !== undefined ? ` · fold ${m.path.match(/\/fold(\d+)\//)[1]}` : '',
      c.protos.length ? ` · ${c.protos.map(p => `${p.K} protos`).join(' + ')}` : '',
      ` · ${m.finetunes.length ? m.finetunes.join('→') : 'base'}`),
    splitSel,
    el('div', { class: 'chips' },
      el('span', { class: 'chip', title: 'network accuracy on this split' }, 'acc ', el('b', {}, pct(sum.acc))),
      el('span', { class: 'chip', title: 'accuracy of the symbolic (rule) model' }, 'rules ', el('b', {}, pct(sum.logic_acc))),
      el('span', { class: 'chip', title: 'share of graphs where rules and network predict the same class' }, 'fidelity ', el('b', {}, pct(sum.logic_fidelity))),
      el('span', { class: 'chip', title: 'share of predictions whose class rule actually fires' }, 'rule-backed ', el('b', {}, pct(sum.rule_backed)))));
  renderTabs();
}

const TABS = [['predict', 'Prediction'], ['protos', 'Prototypes'], ['units', 'Trunk units'], ['head', 'Head rules'], ['metrics', 'Metrics'], ['about', 'How to read']];
function renderTabs() {
  const nav = clear($('tabs'));
  for (const [k, label] of TABS) {
    if (k === 'protos' && card() && !card().protos.length) continue;
    nav.append(el('button', { class: S.tab === k ? 'active' : '', onclick: () => { S.tab = k; stopPlay(); renderTabs(); renderMain(); } }, label));
  }
}

function renderMain() {
  const main = clear($('main'));
  if (!S.card) return;
  ({ predict: renderPredict, protos: renderProtos, units: renderUnits, head: renderHead, metrics: renderMetrics, about: renderAbout })[S.tab](main);
}

// ------------------------------------------------------------------ prediction tab

function renderPredict(main) {
  main.append(el('div', { class: 'predict' },
    el('aside', { class: 'panel list', id: 'list' }),
    el('section', { class: 'center' },
      el('div', { class: 'row1' }, el('div', { class: 'panel', id: 'graphp' }), el('div', { class: 'panel decision', id: 'decision' })),
      el('div', { class: 'panel', id: 'netp' }),
      el('div', { class: 'panel', id: 'treep' })),
    el('aside', { class: 'panel inspector', id: 'inspector' })));
  renderList(); refresh();
}

function refresh() {
  if (S.tab !== 'predict') return;
  renderGraph(); renderDecision(); renderNetwork(); renderInspector(); renderTree();
}

function select(sel, node = null) {
  S.sel = sel;
  if (node !== null) { S.node = node; location.hash = new URLSearchParams({ run: S.run, split: T().split, i: T().i, node }).toString(); }
  S.hoverLit = null;
  refresh();
}
function setNode(v) { select(S.sel, v); }

const FILTERS = [['all', 'all', r => true], ['faithful', 'rule = net', r => r.rule === r.net], ['unfaithful', 'rule ≠ net', r => r.rule !== r.net],
  ['right', 'rule right', r => r.rule === r.y], ['wrong', 'rule wrong', r => r.rule !== r.y], ['unbacked', 'no rule fires', r => !r.backed],
  ['exported', 'exported', r => S.src.has(S.split, r.i)]];
function rowKind(r) { return r.rule !== r.net ? 'unfaithful' : !r.backed ? 'unbacked' : r.rule === r.y ? 'correct' : 'wrong'; }

function renderList() {
  const box = clear($('list'));
  const filters = el('div', { class: 'filters' }, FILTERS.filter(([k]) => k !== 'exported' || S.src.kind === 'bundle').map(([k, label, keep]) =>
    el('button', { class: S.filter === k ? 'on' : '', onclick: () => { S.filter = k; renderList(); } }, label, el('span', { class: 'n' }, ` ${S.index.rows.filter(keep).length}`))));
  const q = el('input', { type: 'search', placeholder: 'graph # or class', value: S.query, oninput: ev => { S.query = ev.target.value; drawRows(); } });
  const rows = el('div', { class: 'rows' });
  box.append(el('h2', {}, `${S.split} graphs`), q, filters, rows);
  function drawRows() {
    clear(rows);
    const qq = S.query.trim().toLowerCase();
    let shown = 0;
    for (const r of S.index.rows) {
      const kind = rowKind(r);
      if (!(FILTERS.find(f => f[0] === S.filter) || FILTERS[0])[2](r)) continue;
      if (qq && !(String(r.i) === qq || className(r.y).toLowerCase().includes(qq))) continue;
      if (++shown > 1500) break;
      const sel = T() && T().split === S.split && T().i === r.i;
      const mark = kind === 'unfaithful' ? el('span', { class: 'mark uf', title: 'rules and network disagree' }, '≠net')
        : r.rule === r.y ? el('span', { class: 'mark ok', title: 'rules correct' }, '✓') : el('span', { class: 'mark ko', title: 'rules wrong' }, '✗');
      const row = el('div', { class: 'grow' + (sel ? ' sel' : '') + (S.src.has(S.split, r.i) ? '' : ' missing'),
        onclick: () => openGraph(S.split, r.i) },
        el('span', { class: 'mono' }, '#' + r.i),
        el('span', { class: 'cls', title: `true ${className(r.y)} · rules ${className(r.rule)} · net ${className(r.net)} (${f2(r.p)})` },
          className(r.y), el('span', { class: 'muted' }, ` · ${r.n} nodes`)),
        mark);
      rows.append(row);
    }
    if (!shown) rows.append(el('div', { class: 'muted small', style: { padding: '8px' } }, 'no graph'));
  }
  drawRows();
}

// ---- graph view

function graphLens() {
  /* what the node fills show: depends on the play step, then on the selection */
  const c = card(), t = T(), sel = S.sel;
  if (S.step !== null) {
    if (S.step >= 1 && S.step <= c.num_layers) {
      const l = S.step - 1;
      return { title: `conv ${l}: share of the ${c.hidden} units on at each node`, value: v => t.layers[l].out[v].split('').filter(b => b === '1').length / c.hidden, pulse: true };
    }
    const k = (t.evidence || []).map(litProto).find(p => p && c.protos[p.li].kind === 'node');
    if (S.step === c.num_layers + 1 && k) return { title: `similarity to ${protoName(k.li, k.k)}`, value: v => protoSim(k.li, k.k, v) };
    return { title: '' };
  }
  if (sel.type === 'unit') return { title: `${unitName(sel.l, sel.u)} at every node (dark = 1)`, bit: v => outBit(sel.l, v, sel.u) };
  let p = null;
  if (sel.type === 'proto') p = { li: sel.li, k: sel.k };
  if (sel.type === 'headlit') p = litProto(sel.i);
  if (sel.type === 'readout' && card().readout[sel.j].k !== undefined) p = { li: card().readout[sel.j].proto_layer, k: card().readout[sel.j].k };
  if (p && c.protos[p.li].kind === 'node') {
    const best = protoSim(p.li, p.k, bestNode(p.k, p.li)) || 1e-9;
    return { title: `similarity to ${protoName(p.li, p.k)}, relative to the best node (${f2(best, 3)}) · rings: exact matches`,
      value: v => protoSim(p.li, p.k, v) / best, exact: v => protoSim(p.li, p.k, v) >= 1 - 1e-6 };
  }
  return { title: 'click a node to see the network at that node' };
}

function renderGraph() {
  const box = clear($('graphp')), t = T(), c = card();
  const lens = graphLens();
  box.append(el('div', { class: 'graph-head' },
    el('h2', {}, `${t.split} #${t.i}`),
    el('span', { class: 'muted' }, `${t.n} nodes · true ${t.y === null ? '?' : className(t.y)}`),
    el('span', { class: 'lens' }, lens.title),
    S.src.kind === 'server' ? el('button', { class: S.editMode ? 'on' : '', title: 'edit the graph and see how the decision changes', onclick: toggleEdit },
      S.editMode ? '✎ editing' : '✎ what-if') : null));
  const W = Math.max(box.clientWidth - 24, 260), H = 330, pad = 22;
  const xs = t.pos.map(p => p[0]), ys = t.pos.map(p => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const sc = Math.min((W - 2 * pad) / Math.max(x1 - x0, 1e-6), (H - 2 * pad) / Math.max(y1 - y0, 1e-6));
  const X = x => (W - sc * (x1 - x0)) / 2 + (x - x0) * sc, Y = y => (H - sc * (y1 - y0)) / 2 + (y - y0) * sc;
  const r = t.n > 120 ? 4 : t.n > 50 ? 6 : 9;
  const svg = sv('svg', { class: 'gsvg', width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  if (t.region) t.region.forEach((g, v) => { if (g) svg.append(sv('circle', { class: 'halo', cx: X(t.pos[v][0]), cy: Y(t.pos[v][1]), r: r * 2 })); });
  const added = new Set((S.edits ? S.edits.add : []).map(([a, b]) => edgeKey(a, b)));
  const line = (a, b, cls, style) => sv('line', { class: cls, x1: X(t.pos[a][0]), y1: Y(t.pos[a][1]), x2: X(t.pos[b][0]), y2: Y(t.pos[b][1]), style });
  if (S.base && S.base !== t) {                       // edges the what-if deleted
    const now = new Set(t.edges.map(([a, b]) => edgeKey(a, b)));
    for (const [a, b] of S.base.edges) if (!now.has(edgeKey(a, b))) svg.append(line(a, b, 'edge deleted'));
  }
  for (const [a, b] of t.edges) {
    const near = S.node === a || S.node === b;
    svg.append(line(a, b, 'edge' + (lens.pulse ? ' pulse' : '') + (added.has(edgeKey(a, b)) ? ' added' : ''),
      near && !lens.pulse ? { stroke: 'var(--ink2)', opacity: 1 } : null));
    if (S.editMode) {
      const hit = line(a, b, 'edgehit');
      hit.addEventListener('click', () => clickEdge(a, b));
      hoverTip(hit, `edge ${a}–${b}: click to delete`);
      svg.append(hit);
    }
  }
  const witness = new Set(S.hoverNodes || []);
  if (S.hoverLit) for (const w of nbh(S.node)) if (subjectHolds(S.hoverLit.l, S.hoverLit.i, w)) witness.add(w);
  for (let v = 0; v < t.n; v++) {
    const cx = X(t.pos[v][0]), cy = Y(t.pos[v][1]);
    let fill = 'var(--panel)';
    if (lens.bit) fill = lens.bit(v) ? 'var(--on)' : 'var(--off)';
    else if (lens.value) fill = `color-mix(in srgb, var(--blue) ${Math.round(100 * Math.max(0, Math.min(1, lens.value(v))))}%, var(--off))`;
    const circ = sv('circle', { class: 'node' + (v === S.node ? ' selnode' : '') + (S.pending === v ? ' pending' : ''), cx, cy, r, fill,
      onclick: ev => S.editMode && ev.shiftKey ? shiftNode(v) : setNode(v) });
    if (S.edits && S.edits.x[v] !== undefined) svg.append(sv('circle', { class: 'edited', cx, cy, r: r + 3.5 }));
    hoverTip(circ, () => nodeTip(v));
    svg.append(circ);
    if (lens.exact && lens.exact(v)) svg.append(sv('circle', { class: 'ring', cx, cy, r: r + 4, stroke: 'var(--blue)' }));
    if (witness.has(v)) svg.append(sv('circle', { class: 'witness', cx, cy, r: r + 6 }));
    const lab = t.labels[v] || (t.n <= 40 ? String(v) : '');
    if (lab && r >= 6) {
      const dark = lens.bit ? lens.bit(v) : lens.value ? lens.value(v) > 0.55 : false;
      svg.append(sv('text', { class: 'nlabel', x: cx, y: cy, style: dark ? { fill: 'var(--panel)' } : null }, lab));
    }
  }
  box.append(svg);
  if (S.editMode) box.append(editorBar());
  const leg = el('div', { class: 'legend' });
  if (t.region && c.ground_truth) leg.append(el('span', {}, el('i', { style: { background: 'var(--gt)' } }), `ground truth: ${c.ground_truth.shading}`));
  leg.append(el('span', {}, el('i', { style: { background: 'var(--panel)', border: '3px solid var(--accent)' } }), `selected: ${nodeName(S.node)}`));
  if (S.hoverLit) leg.append(el('span', {}, el('i', { style: { border: '2px dashed var(--accent)' } }), 'nodes the hovered literal counts'));
  box.append(leg);
}

function nodeTip(v) {
  const c = card(), t = T();
  const on = t.layers.map(L => L.out[v].split('').filter(b => b === '1').length);
  let s = `${nodeName(v)} · degree ${S.adj[v].length}\nunits on: ${on.map((n, l) => `L${l} ${n}/${c.hidden}`).join(' · ')}`;
  const li = nodeProtoLayer();
  if (li >= 0) {
    const sims = t.protos[li].sim[v].map((x, k) => [x, k]).filter(([, k]) => hasCare(li, k)).sort((a, b) => b[0] - a[0]).slice(0, 3);
    if (sims.length) s += `\nclosest prototypes: ${sims.map(([x, k]) => `${protoName(li, k)} ${f2(x)}`).join(', ')}`;
  }
  return s + '\nclick: show the network at this node';
}

// ---- decision panel

function renderDecision() {
  const box = clear($('decision')), t = T(), c = card();
  const ok = t.y === null ? null : t.rule === t.y;
  if (S.base && S.base !== t) {
    const b = S.base, flipped = b.rule !== t.rule, ch = changedUnits();
    box.append(el('div', { class: 'whatif ' + (flipped ? 'flip' : '') },
      el('b', {}, `What-if (${editCount()} edit${editCount() > 1 ? 's' : ''}): `),
      `original → ${className(b.rule)} (${sgn(b.margin[b.rule])}) · now → ${className(t.rule)} (${sgn(t.margin[t.rule])})`,
      flipped ? el('span', { class: 'badge warn', style: { marginLeft: '6px' } }, 'decision flipped') : el('span', { class: 'muted' }, ' · same decision'),
      el('div', { class: 'small muted' }, `units that changed value (all nodes): ${ch.map((n, l) => `L${l} ${n}`).join(' · ')} · network now → ${className(t.net.pred)} (${f2(t.net.prob[t.net.pred])})`)));
  }
  box.append(el('h2', {}, 'Decision'),
    el('div', { class: 'verdict' },
      el('span', { class: 'badge' }, 'true ', el('b', {}, t.y === null ? '?' : className(t.y))),
      el('span', { class: 'badge ' + (ok === null ? '' : ok ? 'ok' : 'ko') }, 'rules → ', el('b', {}, className(t.rule))),
      el('span', { class: 'badge ' + (t.net.pred === t.rule ? 'ok' : 'warn'), title: 'the network (soft) prediction and its probability' },
        'network → ', el('b', {}, className(t.net.pred)), ` (${f2(t.net.prob[t.net.pred])})`),
      t.backed ? null : el('span', { class: 'badge warn', title: 'no class rule fires: the class is the argmax of negative margins' }, 'no rule fires')));
  // class margins: labels left, bars around a zero line
  const mx = Math.max(...t.margin.map(Math.abs), 1e-6);
  const W = Math.max(box.clientWidth - 30, 240), rowH = 22, labW = 120, barW = W - labW - 60, zero = labW + barW / 2;
  const msvg = sv('svg', { width: W, height: rowH * c.classes.length + 4 });
  t.margin.forEach((m, k) => {
    const y = 2 + k * rowH, w = (barW / 2) * Math.abs(m) / mx;
    const g = sv('g', { style: { cursor: 'pointer' }, onclick: () => select({ type: 'class', c: k }) },
      sv('text', { x: labW - 8, y: y + 14, 'text-anchor': 'end', style: { fontWeight: k === t.rule ? 700 : 400 } }, className(k)),
      sv('rect', { x: m >= 0 ? zero : zero - w, y: y + 4, width: Math.max(w, 1), height: rowH - 9, rx: 2,
        fill: m >= 0 ? (k === t.rule ? 'var(--accent)' : 'var(--ink2)') : 'var(--muted)' }),
      sv('text', { x: m >= 0 ? zero + w + 4 : zero + 4, y: y + 14, class: 'mono', style: { fontSize: '11px' } }, sgn(m)));
    hoverTip(g, `class ${className(k)}: Σ w·lit = ${f2(m + c.head.S[k], 3)}, threshold S = ${f2(c.head.S[k], 3)}\nmargin = Σ − S = ${sgn(m)}; ≥ 0: its rule fires. Click: the threshold view.`);
    msvg.append(g);
  });
  msvg.append(sv('line', { x1: zero, x2: zero, y1: 0, y2: rowH * c.classes.length + 4, stroke: 'var(--line)' }));
  box.append(el('div', { class: 'small muted' }, 'margin = Σ w·lit − S: the bar is how far each class is above (right) or below (left) its own threshold, not the sum itself. ',
    c.classes.map((_, k) => `${className(k)}: Σ ${f2(t.margin[k] + c.head.S[k])} vs S ${f2(c.head.S[k])}`).join(' · ')), msvg);
  // explanation: constant literals counted as a bias, then the input-dependent evidence
  const W_ = c.head.W[t.rule], S_ = c.head.S[t.rule];
  const bias = c.head.literals.reduce((acc, lit, i) => acc + (lit.constant === true ? W_[i] : 0), 0);
  if (t.evidence) {
    const n = t.evidence.length;
    box.append(el('div', { class: 'why-line' },
      `Why ${className(t.rule)}: its rule needs Σ w·lit ≥ S = ${f2(S_)}. `,
      el('span', { title: 'head literals that are true on every possible input (exact, see Head rules)' }, `The always-true literals give ${f2(bias)} (a bias), `),
      n ? `so ${n} input-dependent literal${n > 1 ? 's' : ''} suffice${n > 1 ? '' : 's'} (the fewest that reach the rest, exact):`
        : 'which already reaches S: this class rule fires on every graph.'));
    const ul = el('ul', { class: 'expl' });
    for (const i of t.evidence) ul.append(explItem(i));
    box.append(ul);
    const nc = (t.expl || []).filter(i => c.head.literals[i].constant !== null).length;
    if (t.expl) box.append(el('div', { class: 'muted small', style: { marginTop: '6px' } },
      `interp_metrics' shortest explanation counts constants too: ${t.expl.length} literal${t.expl.length > 1 ? 's' : ''}, ${nc} of them constant.`));
  } else {
    box.append(el('div', { class: 'why-line' }, 'No rule of any class fires on this graph; the prediction is the class with the largest (negative) margin. Click a margin to see how far each class is from its threshold.'));
  }
}

function explItem(i) {
  const c = card(), t = T(), lit = c.head.literals[i], p = litProto(i);
  const go = p && c.protos[p.li].kind === 'node' && hasCare(p.li, p.k)
    ? el('button', { title: `jump to the node closest to ${protoName(p.li, p.k)} and show the prototype`, onclick: ev => { ev.stopPropagation(); select({ type: 'proto', li: p.li, k: p.k }, bestNode(p.k, p.li)); } }, 'trace ›')
    : null;
  return el('li', { onclick: () => select({ type: 'headlit', i }) },
    el('span', { class: 'w' }, '+' + f2(c.head.W[t.rule][i])),
    el('span', { class: 't' }, lit.text, el('span', { class: 'muted small' }, `  (${lit.what} = ${f2(t.s[lit.col])})`)),
    go);
}

// ---- network view (the MLP-style diagram)

function netColumns() {
  const c = card(), cols = [];
  cols.push({ key: 'in', title: 'input', sub: `${c.features.length} features`, n: c.features.length });
  for (let l = 0; l < c.num_layers; l++) cols.push({ key: 'L', l, title: `conv ${l}`, sub: `${c.hidden} units`, n: c.hidden });
  if (c.protos.length) {
    const items = [];
    c.protos.forEach((p, li) => { for (let k = 0; k < p.K; k++) items.push({ li, k }); });
    cols.push({ key: 'P', title: 'prototypes', sub: c.protos.map(p => `${p.K} ${p.kind}`).join(' + '), n: items.length, items });
  }
  cols.push({ key: 'R', title: 'readout', sub: `${c.readout.length} pooled`, n: c.readout.length });
  cols.push({ key: 'C', title: 'classes', sub: 'margin Σ − S', n: c.classes.length });
  return cols;
}
const colIndex = (cols, key, l) => cols.findIndex(c => c.key === key && (l === undefined || c.l === l));
function pIndex(cols, li, k) { const P = cols[colIndex(cols, 'P')]; return P.items.findIndex(it => it.li === li && it.k === k); }

/* edges [{from: [col, idx], to: [col, idx], w, kind, neg}] */
function unitFlows(cols, l, u, v, onlyWhy = false) {
  const c = card().conv[l], W = c.W[u], why = T().layers[l].why[v][u] || [], out = [];
  const mx = Math.max(...W);
  for (let i = 0; i < W.length; i++) {
    const inWhy = why.includes(i);
    if (onlyWhy ? !inWhy : W[i] < 0.03 * mx) continue;
    out.push({ from: [l === 0 ? 0 : colIndex(cols, 'L', l - 1), i % c.n_in], to: [colIndex(cols, 'L', l), u], w: W[i] / mx,
      kind: inWhy ? 'expl' : litBit(l, v, i) ? 'true' : 'false', neg: i >= c.n_in });
  }
  return out;
}
function protoFlows(cols, li, k, v) {
  const p = card().protos[li], h = card().hidden, inp = protoBitsAt(li, v), out = [];
  for (const d of caredBits(li, k)) {
    const unit = p.bit_unit[d], need = p.bits[k][d] === '1', has = inp[d] >= 0.5;
    out.push({ from: [colIndex(cols, 'L', Math.floor(unit / h)), unit % h], to: [colIndex(cols, 'P'), pIndex(cols, li, k)], w: 0.6,
      kind: has !== need ? 'miss' : need ? 'need1' : 'need0' });
  }
  return out;
}
function readoutSource(cols, j) {
  const col = card().readout[j];
  if (col.k === undefined) return [colIndex(cols, 'L', Math.floor(col.unit / card().hidden)), col.unit % card().hidden];
  return [colIndex(cols, 'P'), pIndex(cols, col.proto_layer, col.k)];
}
function headFlows(cols, lits, classes) {
  const c = card(), t = T(), out = [];
  for (const k of classes) {
    const mx = Math.max(...c.head.W[k]);
    for (const i of lits) {
      const w = c.head.W[k][i];
      if (w < 0.03 * mx) continue;
      const inExpl = k === t.rule && (t.evidence || []).includes(i);
      out.push({ from: [colIndex(cols, 'R'), c.head.literals[i].col], to: [colIndex(cols, 'C'), k], w: w / mx,
        kind: inExpl ? 'expl' : c.head.literals[i].constant !== null ? 'const' : t.head_lit[i] === '1' ? 'true' : 'false', neg: !c.head.literals[i].pos });
    }
  }
  return out;
}
const allLits = () => card().head.literals.map((_, i) => i);
const litsOfCol = j => allLits().filter(i => card().head.literals[i].col === j);
const allClasses = () => card().classes.map((_, k) => k);

function flowsForSel(cols) {
  const s = S.sel, v = S.node;
  switch (s.type) {
    case 'unit': return unitFlows(cols, s.l, s.u, v);
    case 'proto': return protoFlows(cols, s.li, s.k, v);
    case 'readout': return [{ from: readoutSource(cols, s.j), to: [colIndex(cols, 'R'), s.j], w: 0.8, kind: 'expl' }, ...headFlows(cols, litsOfCol(s.j), allClasses())];
    case 'headlit': return [{ from: readoutSource(cols, card().head.literals[s.i].col), to: [colIndex(cols, 'R'), card().head.literals[s.i].col], w: 0.8, kind: 'expl' }, ...headFlows(cols, [s.i], allClasses())];
    case 'class': return headFlows(cols, allLits(), [s.c]);
    default: return [];
  }
}

/* the edges that "carry" step s of the forward pass: the exact reasons at the selected node */
function flowsForStep(cols, s) {
  const c = card(), t = T(), v = S.node, key = cols[s].key, out = [];
  if (key === 'L') {
    const l = cols[s].l;
    for (let u = 0; u < c.hidden; u++) if (outBit(l, v, u)) out.push(...unitFlows(cols, l, u, v, true));
  } else if (key === 'P') {
    const cited = new Set((t.evidence || []).map(litProto).filter(Boolean).map(p => `${p.li}:${p.k}`));
    c.protos.forEach((p, li) => { for (let k = 0; k < p.K; k++) if (cited.has(`${li}:${k}`) && hasCare(li, k)) out.push(...protoFlows(cols, li, k, v)); });
  } else if (key === 'R') {
    const used = new Set((t.evidence || []).map(i => c.head.literals[i].col));
    for (const j of used) out.push({ from: readoutSource(cols, j), to: [colIndex(cols, 'R'), j], w: 0.8, kind: 'expl' });
  } else if (key === 'C') {
    out.push(...headFlows(cols, t.evidence || [], [t.rule]));
  }
  return out;
}

function renderNetwork() {
  const box = clear($('netp')), c = card(), t = T(), v = S.node, cols = netColumns();
  const playing = S.timer !== null;
  box.append(el('div', { class: 'net-head' },
    el('h2', {}, `Inside the network at ${nodeName(v)}`),
    el('span', { class: 'muted small' }, 'trunk columns: that node · prototypes: its similarity (node prototypes) · readout and classes: the whole graph'),
    el('span', { class: 'spacer' }),
    el('button', { class: playing ? 'on' : 'primary', title: 'animate the forward pass layer by layer (space)', onclick: togglePlay }, playing ? '❚❚ pause' : '▶ play forward pass'),
    el('button', { title: 'previous step (←)', onclick: () => stepBy(-1) }, '‹'),
    el('button', { title: 'next step (→)', onclick: () => stepBy(1) }, '›'),
    S.step !== null ? el('button', { onclick: () => { stopPlay(); refresh(); } }, 'show all') : null));
  const W = Math.max(box.clientWidth - 24, 640), H = 600, top = 42, bottom = 16;
  const plotH = H - top - bottom;
  const maxN = Math.max(...cols.map(col => col.key === 'C' ? 0 : col.n));
  const gap = 1.5, ch = Math.max(2.5, Math.min(14, plotH / maxN - gap)), cw = 16;
  const left = 70, right = 110;
  const colX = i => left + i * (W - left - right) / (cols.length - 1);
  const cellY = (ci, idx) => {
    const col = cols[ci];
    if (col.key === 'C') { const bh = 46; return top + (plotH - col.n * (bh + 10)) / 2 + idx * (bh + 10); }
    return top + (plotH - col.n * (ch + gap)) / 2 + idx * (ch + gap);
  };
  const cellH = ci => cols[ci].key === 'C' ? 46 : ch;
  const svg = sv('svg', { class: 'netsvg', width: W, height: H, viewBox: `0 0 ${W} ${H}` });

  // edges first (under the cells)
  const flows = S.step !== null ? flowsForStep(cols, S.step) : flowsForSel(cols);
  const dense = flows.length > 60;           // many reasons at once (a whole layer): lighter strokes
  const eg = sv('g', {});
  const chain = new Set();
  for (const f of flows) {
    const [a, i] = f.from, [b, j] = f.to;
    if (a < 0 || b < 0 || i < 0 || j < 0) continue;
    const x1 = colX(a) + cw / 2, y1 = cellY(a, i) + cellH(a) / 2, x2 = colX(b) - (cols[b].key === 'C' ? 40 : cw / 2), y2 = cellY(b, j) + cellH(b) / 2;
    const mx = (x1 + x2) / 2;
    eg.append(sv('path', { class: `e e-${f.kind}` + (f.neg ? ' neg' : '') + (S.step !== null ? ' flow' : ''),
      d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`, 'stroke-width': ((dense ? 0.4 : 0.6) + (dense ? 1.2 : 2.6) * f.w).toFixed(2),
      style: dense ? { opacity: 0.3 } : null }));
    if (f.kind === 'expl' || f.kind === 'need1' || f.kind === 'need0' || f.kind === 'miss') chain.add(`${a}:${i}`);
  }
  svg.append(eg);

  const focus = selCell(cols);
  cols.forEach((col, ci) => {
    const g = sv('g', { class: 'col' + (S.step !== null && ci > S.step ? ' future' : '') + (S.step === ci ? ' now' : '') });
    const x = colX(ci);
    g.append(sv('text', { class: 'ctitle', x, y: 14 }, col.title), sv('text', { class: 'csub', x, y: 27 }, col.sub));
    for (let idx = 0; idx < col.n; idx++) {
      const y = cellY(ci, idx);
      const info = cellInfo(col, idx);
      let node;
      if (col.key === 'C') {
        node = sv('g', { style: { cursor: 'pointer' } },
          sv('rect', { class: 'bigcell', x: x - 40, y, width: 80, height: 46, rx: 8,
            fill: idx === t.rule ? 'var(--accent)' : 'var(--off)', 'stroke-width': focus === `${ci}:${idx}` ? 2.5 : 1, stroke: focus === `${ci}:${idx}` ? 'var(--ink)' : 'var(--line)' }),
          sv('text', { x, y: y + 19, 'text-anchor': 'middle', style: { fontWeight: 700, fill: idx === t.rule ? '#fff' : 'var(--ink)', fontSize: '12px' } }, className(idx)),
          sv('text', { x, y: y + 35, 'text-anchor': 'middle', class: 'mono', style: { fill: idx === t.rule ? '#fff' : 'var(--ink2)', fontSize: '11px' } }, sgn(t.margin[idx])));
      } else {
        const changed = S.base && S.base !== t && col.key === 'L' && S.base.layers[col.l].out[v][idx] !== t.layers[col.l].out[v][idx];
        const cls = ['cell', info.bin === undefined ? '' : info.bin ? 'on' : 'off', focus === `${ci}:${idx}` ? 'focus' : chain.has(`${ci}:${idx}`) ? 'chain' : '', info.disagree ? 'disagree' : '', changed ? 'changed' : ''].join(' ');
        node = sv('rect', { class: cls, x: x - cw / 2, y, width: cw, height: ch, rx: Math.min(2, ch / 3),
          fill: info.bin === undefined ? `color-mix(in srgb, var(--blue) ${Math.round(100 * info.val)}%, var(--off))` : null });
      }
      node.addEventListener('click', () => clickCell(col, idx));
      hoverTip(node, () => info.tip);
      g.append(node);
      // labels for short columns
      if (col.key === 'in' && ch >= 7) g.append(sv('text', { class: 'clabel', x: x - cw / 2 - 4, y: y + ch / 2, 'text-anchor': 'end' }, c.features[idx]));
      if ((col.key === 'L' || col.key === 'R' || col.key === 'P') && idx % 8 === 0 && ci === 1)
        g.append(sv('text', { class: 'clabel', x: x - cw / 2 - 4, y: y + ch / 2, 'text-anchor': 'end' }, `u${idx}`));
    }
    svg.append(g);
  });
  box.append(svg);
  box.append(el('div', { class: 'legend' },
    el('span', {}, el('i', { style: { background: 'var(--on)' } }), 'unit = 1 (rule fires)'),
    el('span', {}, el('i', { style: { background: 'var(--off)', border: '1px solid var(--line)' } }), '0'),
    el('span', {}, el('i', { style: { border: '2px solid var(--bad)' } }), 'network value on the other side of 0.5'),
    el('span', {}, el('i', { style: { background: 'var(--accent)' } }), 'exact reason (shortest sufficient literals)'),
    el('span', {}, el('i', { style: { background: 'var(--ink2)' } }), 'true literal'),
    el('span', {}, el('i', { style: { background: 'var(--blue)' } }), 'prototype needs 1'),
    el('span', {}, el('i', { style: { background: 'var(--orange)' } }), 'needs 0'),
    el('span', {}, '- - - negated literal (counts NOT u)'),
    el('span', { class: 'muted' }, 'a unit\'s edge is a count over the node and its neighbours, not the lower unit at this node alone')));
}

function selCell(cols) {
  const s = S.sel;
  if (s.type === 'unit') return `${colIndex(cols, 'L', s.l)}:${s.u}`;
  if (s.type === 'proto') return `${colIndex(cols, 'P')}:${pIndex(cols, s.li, s.k)}`;
  if (s.type === 'readout') return `${colIndex(cols, 'R')}:${s.j}`;
  if (s.type === 'headlit') return `${colIndex(cols, 'R')}:${card().head.literals[s.i].col}`;
  if (s.type === 'class') return `${colIndex(cols, 'C')}:${s.c}`;
  return '';
}

function cellInfo(col, idx) {
  const c = card(), t = T(), v = S.node;
  if (col.key === 'in') {
    const on = xBit(v, idx);
    return { bin: on, tip: `${c.features[idx]} = ${on ? 1 : 0} at ${nodeName(v)}` };
  }
  if (col.key === 'L') {
    const l = col.l, on = outBit(l, v, idx), net = t.layers[l].net[v][idx], m = t.layers[l].margin[v][idx];
    const why = t.layers[l].why[v][idx];
    return { bin: on, disagree: (net >= 0.5) !== on,
      tip: `${unitName(l, idx)} at ${nodeName(v)} = ${on ? 1 : 0}\nmargin Σw·lit − S = ${sgn(m)} · network ${f2(net, 3)}\n${on ? `fires because of ${why.length} literal${why.length > 1 ? 's' : ''}` : 'true literals fall short of S'} · on at ${pct(c.conv[l].rate[idx])} of ${c.ref_split} nodes\nclick: open its rule` };
  }
  if (col.key === 'P') {
    const { li, k } = col.items[idx], p = c.protos[li], sim = protoSim(li, k), cared = caredBits(li, k).length;
    return { val: sim, tip: `${protoName(li, k)} (${p.kind} prototype, ${cared} cared bit${cared === 1 ? '' : 's'})\nsimilarity ${f2(sim, 3)}${p.masked ? ` · ${f2(protoMism(li, k), 1)} mismatches` : ''}${p.kind === 'node' ? ` at ${nodeName(v)}` : ' (graph)'}\n${cared ? p.rule[k] : 'cares about no bit: always similarity 1'}` };
  }
  if (col.key === 'R') {
    const r = c.readout[idx], s = t.s[idx];
    const name = r.k === undefined ? `${r.op}(${unitName(Math.floor(r.unit / c.hidden), r.unit % c.hidden)})` : `${r.op} sim(${protoName(r.proto_layer, r.k)})`;
    const lits = litsOfCol(idx).map(i => `${c.head.literals[i].pos ? '' : '¬'}lit ${i}: ${t.head_lit[i] === '1' ? 'true' : 'false'}`).join(' · ');
    return { val: r.op === 'sum' ? Math.min(1, s / Math.max(...t.s.filter((_, j) => c.readout[j].op === 'sum'), 1)) : s,
      tip: `${name} over the graph = ${f2(s, 3)}\nhead literals: ${lits}` };
  }
  return { tip: '' };
}

function clickCell(col, idx) {
  stopPlay();
  if (col.key === 'L') select({ type: 'unit', l: col.l, u: idx });
  else if (col.key === 'P') { const it = col.items[idx]; select({ type: 'proto', li: it.li, k: it.k }); }
  else if (col.key === 'R') select({ type: 'readout', j: idx });
  else if (col.key === 'C') select({ type: 'class', c: idx });
}

// play
function togglePlay() {
  if (S.timer !== null) { clearInterval(S.timer); S.timer = null; refresh(); return; }
  const n = netColumns().length;
  if (S.step === null || S.step >= n - 1) S.step = 0;
  S.timer = setInterval(() => {
    if (S.step >= n - 1) { clearInterval(S.timer); S.timer = null; refresh(); return; }
    S.step += 1; refresh();
  }, 1300);
  refresh();
}
function stepBy(d) {
  if (S.timer !== null) { clearInterval(S.timer); S.timer = null; }
  const n = netColumns().length;
  S.step = Math.max(0, Math.min(n - 1, (S.step === null ? (d > 0 ? -1 : n) : S.step) + d));
  refresh();
}
function stopPlay() { if (S.timer !== null) clearInterval(S.timer); S.timer = null; S.step = null; }

document.addEventListener('keydown', ev => {
  if (S.tab !== 'predict' || ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (ev.key === 'ArrowRight') { stepBy(1); ev.preventDefault(); }
  else if (ev.key === 'ArrowLeft') { stepBy(-1); ev.preventDefault(); }
  else if (ev.key === ' ') { togglePlay(); ev.preventDefault(); }
  else if (ev.key === 'Escape') { stopPlay(); refresh(); }
});
window.addEventListener('resize', () => { if (S.tab === 'predict' && T()) refresh(); });

// ------------------------------------------------------------------ readable rules of a unit

/* unpack_rules.unit_rules of a conv unit, fetched once per unit (server only) */
S.rules = {};
function rulesBlock(l, u, v = null) {
  const box = el('div', { class: 'rulesblock' });
  if (S.src.kind !== 'server') {
    box.append(el('div', { class: 'muted small' }, 'Readable rules of deep units are computed on demand: open the run with python -m explorer serve.'));
    return box;
  }
  const key = `${S.run}:${l}:${u}`;
  const draw = r => {
    clear(box);
    if (r.error) { box.append(el('div', { class: 'error' }, r.error)); return; }
    if (!r.pos) { box.append(el('div', { class: 'muted' }, r.dead ? 'the unit can never fire' : 'the unit always fires (S ≤ 0)')); return; }
    for (const [key2, title] of [['pos', 'fires when one of these holds'], ['neg', 'is 0 when one of these holds']]) {
      const rr = r[key2];
      box.append(el('div', { class: 'rhead' }, el('b', {}, title),
        el('span', { class: 'muted small' }, ` · ${rr.n_rules} rules cover ${pct(rr.coverage, 0)} of the ${r.split} nodes where it ${key2 === 'pos' ? 'fires' : 'is 0'}` +
          ` · ${rr.mode === 'all observed' ? `${rr.n_rules_observed} observed minimal rules` : 'read by the shortest rule of each observed literal pattern'}` +
          ` · ≈ ${sci(rr.n_rules_formal_est)} formal`)));
      const shown = S.allRules ? rr.rules : rr.rules.slice(0, 3);
      shown.forEach(rule => box.append(ruleLine(l, rule, key2 === 'pos', v)));
      if (rr.n_rules > shown.length) box.append(el('div', { class: 'muted small' }, `… ${rr.n_rules - shown.length} more rule${rr.n_rules - shown.length > 1 ? 's' : ''}` +
        (rr.rules.length > shown.length ? '' : ' (not listed)')));
    }
    box.append(el('button', { onclick: () => { S.allRules = !S.allRules; draw(r); } }, S.allRules ? 'show 3 rules per side' : 'show 8 rules per side'));
    box.append(el('div', { class: 'muted small' }, `Rules from unpack_rules.py on the ${r.split} split: a covering set of observed minimal rules, each shortened while its precision on the data does not drop (grey: the exact rule's length). ✓ = the exact rule holds at the selected node (it decides the unit); ~ = only the shortened one does.`));
  };
  if (S.rules[key]) draw(S.rules[key]);
  else {
    box.append(el('div', { class: 'muted small' }, `computing the rules of ${unitName(l, u)} (a few seconds the first time)…`));
    S.src.get('rules', { run: S.run, l, u }).then(r => { S.rules[key] = r; draw(r); }).catch(e => draw({ error: e.message }));
  }
  return box;
}

function ruleLine(l, rule, pos, v) {
  const t = T(), simple = rule.simple || rule.literals;
  const holds = (idx, w) => idx.every(i => litBit(l, w, i) === pos);
  const exact = v !== null && t && holds(rule.literals, v), loose = v !== null && t && holds(simple, v);
  const line = el('div', { class: 'rule' + (exact ? ' here' : '') },
    el('span', { class: 'sup' }, `${pct(rule.simple_support ?? rule.support, 1)} · prec ${pct(rule.simple_precision ?? rule.precision, 0)}`),
    v !== null && t ? el('span', { class: exact ? 'tok' : 'muted' }, exact ? '✓ ' : loose ? '~ ' : '  ') : null,
    (rule.simple_text || rule.text || []).join('  ∧  ') || 'TRUE',
    el('span', { class: 'muted' }, `  [${simple.length}/${rule.literals.length}]`));
  if (t) {
    const nodes = [...Array(t.n).keys()].filter(w => holds(simple, w));
    line.addEventListener('mouseenter', () => { S.hoverNodes = nodes; renderGraph(); });
    line.addEventListener('mouseleave', () => { S.hoverNodes = null; renderGraph(); });
    hoverTip(line, `holds at ${nodes.length} of ${t.n} nodes of this graph (ringed)`);
  }
  return line;
}

// ------------------------------------------------------------------ explanation tree

/* The prediction unfolded as a proof: every line is a fact of the rule model, its children
 * are its exact reason (the shortest one the reader computed), down to atom counts.
 * Children are built when a line is opened, so deep trees cost nothing until explored. */
function tItem({ text, sub = null, kids = null, open = false, nodes = null, onclick = null, cls = '' }) {
  const head = el('span', { class: 'tline' }, text, sub ? el('span', { class: 'tsub' }, ' ', sub) : null);
  if (onclick) head.append(el('a', { href: '#', class: 'tgo', onclick: ev => { ev.preventDefault(); ev.stopPropagation(); onclick(); } }, 'inspect'));
  const hover = target => {
    if (!nodes) return;
    target.addEventListener('mouseenter', () => { S.hoverNodes = nodes; renderGraph(); });
    target.addEventListener('mouseleave', () => { S.hoverNodes = null; renderGraph(); });
  };
  if (!kids) { const leaf = el('div', { class: 'tleaf ' + cls }, head); hover(leaf); return leaf; }
  const det = el('details', { class: 'titem ' + cls });
  const sum = el('summary', {}, head);
  hover(sum);
  det.append(sum);
  let filled = false;
  const fill = () => {
    if (filled) return;
    filled = true;
    const box = el('div', { class: 'tkids' });
    for (const k of kids()) box.append(k);
    det.append(box);
  };
  det.addEventListener('toggle', () => { if (det.open) fill(); });
  if (open) { det.open = true; fill(); }
  return det;
}
const mark = ok => el('span', { class: ok ? 'tok' : 'tko' }, ok ? ' ✓' : ' ✗');

function renderTree() {
  const box = clear($('treep')), t = T(), c = card();
  box.append(el('div', { class: 'net-head' }, el('h2', {}, 'Why: the prediction unfolded'),
    el('span', { class: 'muted small' }, 'every line is an exact fact of the rule model; open it for its reason (the fewest literals that settle it), down to atom counts · hover: the nodes involved')));
  const W = c.head.W[t.rule];
  const bias = c.head.literals.reduce((a, lit, i) => a + (lit.constant === true ? W[i] : 0), 0);
  const lits = t.evidence !== null && t.evidence !== undefined ? t.evidence : null;
  box.append(tItem({
    text: el('b', {}, `${className(t.rule)}`), sub: `margin ${sgn(t.margin[t.rule])} · ${t.backed ? 'its rule fires' : 'no class rule fires (largest margin)'}`, open: true,
    kids: () => [
      tItem({ text: `always-true head literals: +${f2(bias)} of S = ${f2(c.head.S[t.rule])}`, sub: '(constant on every input, a bias)', cls: 'muted' }),
      ...(lits || []).map(i => headLitTree(i, true)),
      ...(lits && !lits.length ? [tItem({ text: 'the bias alone reaches S: this class rule fires on every graph', cls: 'muted' })] : []),
    ],
  }));
}

function headLitTree(i, open = false) {
  const c = card(), t = T(), lit = c.head.literals[i], on = t.head_lit[i] === '1';
  const col = c.readout[lit.col];
  const sub = `${lit.what} = ${f2(t.s[lit.col], 3)} · weight ${f2(c.head.W[t.rule][i])}`;
  let kids = null;
  if (col.k !== undefined) {
    const p = c.protos[col.proto_layer];
    kids = () => p.kind === 'node' ? [protoTree(col.proto_layer, col.k, bestNode(col.k, col.proto_layer), col.op)] : [graphProtoTree(col.proto_layer, col.k)];
  } else {
    const l = Math.floor(col.unit / c.hidden), u = col.unit % c.hidden;
    kids = () => [...Array(t.n).keys()].filter(w => outBit(l, w, u)).slice(0, 30).map(w => unitTree(l, u, w))
      .concat([tItem({ text: `${unitName(l, u)} holds on ${[...Array(t.n).keys()].filter(w => outBit(l, w, u)).length} of ${t.n} nodes`, cls: 'muted' })]);
  }
  return tItem({ text: [lit.text, mark(on)], sub, kids, open, onclick: () => select({ type: 'headlit', i }) });
}

function protoTree(li, k, v, op = 'max') {
  const c = card(), p = c.protos[li], cared = caredBits(li, k), inp = protoBitsAt(li, v);
  const bad = cared.filter(d => (inp[d] >= 0.5) !== (p.bits[k][d] === '1')).length;
  const why = op === 'max' ? 'the closest node decides "max"' : op === 'mean' ? 'the closest node, one term of the mean' : '';
  return tItem({
    text: `${protoName(li, k)} at ${nodeName(v)}: ${bad} mismatch${bad === 1 ? '' : 'es'} on ${cared.length} cared bits`,
    sub: `similarity ${f2(protoSim(li, k, v), 3)} · ${why}`, nodes: [v], open: true,
    onclick: () => select({ type: 'proto', li, k }, v),
    kids: () => cared.map(d => {
      const unit = p.bit_unit[d], l = Math.floor(unit / c.hidden), u = unit % c.hidden, need = p.bits[k][d] === '1';
      return unitTree(l, u, v, need);
    }),
  });
}

function graphProtoTree(li, k) {
  const c = card(), p = c.protos[li], inp = T().protos[li].input;
  return tItem({
    text: `${protoName(li, k)} (graph prototype): similarity ${f2(protoSim(li, k), 3)}`, onclick: () => select({ type: 'proto', li, k }),
    kids: () => caredBits(li, k).map(d => tItem({ text: [`${p.bit_labels[d]} = ${f2(inp[d], 2)} over the graph, needs ${p.bits[k][d]}`, mark((inp[d] >= 0.5) === (p.bits[k][d] === '1'))],
      sub: 'pooled over all nodes', cls: 'muted' })),
  });
}

function unitTree(l, u, v, need = null) {
  const c = card(), conv = c.conv[l], t = T(), on = outBit(l, v, u), why = t.layers[l].why[v][u] || [];
  const W = conv.W[u], S_ = conv.S[u], total = W.reduce((a, b) => a + b, 0);
  const sub = on ? `fires: ${why.length} true literal${why.length === 1 ? '' : 's'} reach S = ${f2(S_)}`
    : total < S_ ? 'can never fire' : `stays 0: ${why.length} false literal${why.length === 1 ? '' : 's'} keep it below S`;
  return tItem({
    text: [`${unitName(l, u)} = ${on ? 1 : 0} at ${nodeName(v)}`, need === null ? null : el('span', { class: 'muted' }, ` (needs ${need ? 1 : 0})`), need === null ? null : mark(on === need)],
    sub, nodes: [v], onclick: () => select({ type: 'unit', l, u }, v),
    kids: why.length ? () => why.map(i => literalTree(l, i, v)) : null,
  });
}

function literalTree(l, i, v) {
  const c = card(), conv = c.conv[l], t = T(), n = conv.n_in, on = litBit(l, v, i), count = t.layers[l].count[v][i];
  const N = nbh(v), holds = N.filter(w => subjectHolds(l, i, w));
  const subj = conv.subjects[i];
  const text = [`#{n ∈ N[${nodeName(v)}] : ${subj}(n)} = ${count}`, el('span', { class: 'muted' }, on ? ' → true' : ' → false')];
  const sub = `true when: ${conv.literals[i].text.replace(/^#\{[^}]*\}\s*/, '')}`;
  if (l === 0) {
    return tItem({ text, sub: `${sub} · counted: ${holds.map(nodeName).join(', ') || 'none'}`, nodes: holds });
  }
  const base = i % n;
  return tItem({
    text, sub, nodes: holds,
    kids: () => N.map(w => unitTree(l - 1, base, w)),
  });
}

// ------------------------------------------------------------------ small plots

const STEP_KEY = 'curve: phi(w·x + b), the step the literal reads\ngreen band: where the literal is true (phi ≥ 0.5, dotted line)';
const STEP_KEY_INT = '\ndots: the counts that can occur (counts are integers, so only these points matter); filled = true, hollow = false';
/* the step function of one literal: phi(w * map(x) + b) over [lo, hi], green where the
 * literal is true; integer = counts (dots at each count); x = the current value */
function stepPlot({ w, b, tau, lo, hi, x = null, integer = false, map = z => z, width = 200, height = 58 }) {
  if (x !== null) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
  const pad = { l: 4, r: 4, t: 4, b: 13 };
  const a = integer ? lo - 0.5 : lo, c = integer ? hi + 0.5 : hi, span = Math.max(c - a, 1e-6);
  const X = u => pad.l + (u - a) / span * (width - pad.l - pad.r);
  const Y = y => pad.t + (1 - y) * (height - pad.t - pad.b);
  const g = sv('svg', { class: 'stepplot', width, height, viewBox: `0 0 ${width} ${height}` });
  hoverTip(g, STEP_KEY + (integer ? STEP_KEY_INT : '') + (x !== null ? `\npurple: the value here (${integer ? Math.round(x) : f2(x)}), ${litTrue(w * map(x) + b) ? 'true' : 'false'}` : ''));
  const N = 300;
  let start = null;
  for (let j = 0; j <= N; j++) {
    const u = a + span * j / N, on = litTrue(w * map(u) + b);
    if (on && start === null) start = u;
    if ((!on || j === N) && start !== null) { g.append(sv('rect', { class: 'truezone', x: X(start), y: pad.t, width: Math.max(X(u) - X(start), 1), height: height - pad.t - pad.b })); start = null; }
  }
  let d = '';
  for (let j = 0; j <= N; j++) { const u = a + span * j / N; d += (j ? 'L' : 'M') + X(u).toFixed(1) + ',' + Y(phi(w * map(u) + b, tau)).toFixed(1); }
  g.append(sv('line', { class: 'half', x1: pad.l, x2: width - pad.r, y1: Y(0.5), y2: Y(0.5) }), sv('path', { class: 'curve', d }));
  if (integer && hi - lo <= 60) for (let k = Math.ceil(lo); k <= hi; k++) g.append(sv('circle', { class: 'dot ' + (litTrue(w * map(k) + b) ? 'on' : 'off'), cx: X(k), cy: Y(phi(w * map(k) + b, tau)), r: 2.3 }));
  const fmt = u => integer ? String(Math.round(u)) : f2(u);
  g.append(sv('text', { x: pad.l, y: height - 2 }, fmt(lo)), sv('text', { x: width - pad.r, y: height - 2, 'text-anchor': 'end' }, fmt(hi)));
  if (x !== null) {
    const on = litTrue(w * map(x) + b);
    g.append(sv('line', { class: 'cur', x1: X(x), x2: X(x), y1: pad.t, y2: height - pad.b }),
      sv('circle', { class: 'curdot', cx: X(x), cy: Y(phi(w * map(x) + b, tau)), r: 3.6 }),
      sv('text', { x: Math.min(Math.max(X(x), 14), width - 14), y: height - 2, 'text-anchor': 'middle', style: { fill: on ? 'var(--good)' : 'var(--ink2)', fontWeight: 700 } }, fmt(x)));
  }
  return g;
}

/* Σ w·lit against the threshold S: true literals stacked (exact reason first), S marked */
function thresholdBar(items, S_, width = 360) {
  const rank = { expl: 0, const: 1, true: 2 };
  const tr = items.filter(it => it.state !== 'false').sort((p, q) => rank[p.state] - rank[q.state] || q.w - p.w);
  const sum = tr.reduce((s, it) => s + it.w, 0);
  const scale = (width - 20) / Math.max(sum, S_, 1e-6) / 1.08;
  const g = sv('svg', { class: 'thr', width, height: 44 });
  let x = 10;
  for (const it of tr) {
    const r = sv('rect', { class: 'seg ' + it.state, x, y: 6, width: Math.max(it.w * scale, 0.5), height: 16 });
    hoverTip(r, `${it.label}\nw = ${f2(it.w, 3)}`);
    if (it.onclick) { r.style.cursor = 'pointer'; r.addEventListener('click', it.onclick); }
    g.append(r); x += it.w * scale;
  }
  const sx = 10 + S_ * scale;
  g.append(sv('line', { class: 'sline', x1: sx, x2: sx, y1: 1, y2: 27 }),
    sv('text', { x: Math.min(sx, width - 60), y: 38 }, `S = ${f2(S_)}`),
    sv('text', { x: 10, y: 38, style: { fill: 'var(--ink2)' } }, `Σ true = ${f2(sum)} ${sum >= S_ ? '≥' : '<'} S`));
  return g;
}

/* a prototype's cared bits over the trunk (rows: layers, cols: units), with node values */
function bitGrid(li, k, v = null, cellPx = 6) {
  const c = card(), p = c.protos[li], h = c.hidden;
  const nOps = p.d / (c.num_layers * h);
  const rows = c.num_layers * nOps, rowH = Math.max(cellPx + 1, 10), W = h * cellPx + 44, H = rows * rowH + 2;
  const g = sv('svg', { class: 'bitgrid', width: W, height: H });
  const inp = v === null || !T() ? null : protoBitsAt(li, v);
  for (let d = 0; d < p.d; d++) {
    const r = Math.floor(d / h), u = d % h, x = 40 + u * cellPx, y = r * rowH + (rowH - cellPx) / 2;
    const cared = p.care[k][d] === '1', need = p.bits[k][d] === '1';
    const rect = sv('rect', { class: 'bc ' + (cared ? (need ? 'n1' : 'n0') : 'dc'), x, y, width: cellPx, height: cellPx });
    const unit = p.bit_unit[d], l = Math.floor(unit / h);
    hoverTip(rect, () => `${p.bit_labels[d]}${cared ? ` · ${protoName(li, k)} needs ${need ? 1 : 0}` : ' · not cared'}${inp ? ` · here ${f2(inp[d], 2)}` : ''}`);
    rect.addEventListener('click', () => { if (S.tab === 'predict') select({ type: 'unit', l, u: unit % h }); else { S.unitLayer = l; S.unitSel = unit % h; S.tab = 'units'; renderTabs(); renderMain(); } });
    g.append(rect);
    if (inp && inp[d] >= 0.5) g.append(sv('circle', { class: 'vb', cx: x + cellPx / 2, cy: y + cellPx / 2, r: cellPx / 4.5 }));
    if (inp && cared && (inp[d] >= 0.5) !== need) g.append(sv('rect', { class: 'mx', x: x - 0.5, y: y - 0.5, width: cellPx + 1, height: cellPx + 1, fill: 'none' }));
  }
  for (let r = 0; r < rows; r++) {
    const op = nOps > 1 ? p.bit_labels[r * h].split(':')[0] + ' ' : '';
    g.append(sv('text', { x: 0, y: r * rowH + rowH - 2, style: { fontSize: '9px' } }, `${op}L${r % c.num_layers}`));
  }
  return g;
}

/* grouped histogram, one series per class, each class normalised to its own total (classes
 * are imbalanced); x = bins; marker = this graph / node; trueAt(x) shades where the head
 * literal reading this distribution is true */
const CLASS_COLORS = ['var(--c0)', 'var(--c1)', 'var(--c2)', 'var(--c3)'];
function histChart({ series, labels, marker = null, markerText = '', trueAt = null, xTitle, width = 360, height = 120, normalise = true }) {
  const n = labels.length, padL = 30, padR = 6, padT = 8, padB = 30;
  const shares = series.map(sr => { const tot = sr.counts.reduce((a, b) => a + b, 0) || 1; return sr.counts.map(c => normalise ? c / tot : c); });
  const ymax = Math.max(...shares.flat(), 1e-9);
  const bw = (width - padL - padR) / n, inner = Math.max(bw - 2, 1), gw = inner / series.length;
  const X = i => padL + i * bw, Y = v => padT + (1 - v / ymax) * (height - padT - padB);
  const g = sv('svg', { width, height, class: 'hist' });
  if (trueAt) for (let i = 0; i < n; i++) if (trueAt(i)) g.append(sv('rect', { class: 'truezone', x: X(i), y: padT, width: bw, height: height - padT - padB }));
  for (const t of [0, ymax / 2, ymax]) {
    g.append(sv('line', { class: 'grid', x1: padL, x2: width - padR, y1: Y(t), y2: Y(t) }),
      sv('text', { x: padL - 4, y: Y(t) + 3, 'text-anchor': 'end' }, normalise ? `${Math.round(100 * t)}%` : t >= 1e3 ? `${(t / 1e3).toFixed(t >= 1e4 ? 0 : 1)}k` : String(Math.round(t))));
  }
  for (let i = 0; i < n; i++) {
    series.forEach((sr, j) => {
      const v = shares[j][i];
      if (!v) return;
      const h = Math.max(Y(0) - Y(v), 1), x = X(i) + 1 + j * gw;
      const bar = sv('path', { fill: sr.color, d: `M${x},${Y(0)} v${-h + Math.min(2, h)} q0,-${Math.min(2, h)} ${Math.min(2, gw / 2)},-${Math.min(2, h)} h${Math.max(gw - 2 * Math.min(2, gw / 2), 0)} q${Math.min(2, gw / 2)},0 ${Math.min(2, gw / 2)},${Math.min(2, h)} v${h - Math.min(2, h)} z` });
      hoverTip(bar, `${sr.name} · ${xTitle} ${labels[i]}\n${sr.counts[i]} (${pct(shares[j][i] * (normalise ? 1 : 1 / (sr.counts.reduce((a, b) => a + b, 0) || 1)))} of ${sr.name})`);
      g.append(bar);
    });
    const step = Math.ceil(n / 12);
    if (i % step === 0 || (i === n - 1 && i % step >= 0.8 * step)) g.append(sv('text', { x: X(i) + bw / 2, y: height - padB + 11, 'text-anchor': 'middle' }, labels[i]));
  }
  g.append(sv('line', { class: 'axis', x1: padL, x2: width - padR, y1: Y(0), y2: Y(0) }),
    sv('text', { x: padL + (width - padL - padR) / 2, y: height - 4, 'text-anchor': 'middle', class: 'xt' }, xTitle));
  if (marker !== null) {
    const mx = X(marker) + bw / 2;
    g.append(sv('line', { class: 'mark', x1: mx, x2: mx, y1: padT - 4, y2: Y(0) }),
      sv('text', { x: Math.min(Math.max(mx, 30), width - 30), y: padT - 0, 'text-anchor': 'middle', class: 'markt' }, markerText));
  }
  const legend = el('div', { class: 'legend' }, series.map(sr => el('span', {}, el('i', { style: { background: sr.color } }), sr.name)),
    trueAt ? el('span', {}, el('i', { style: { background: 'color-mix(in srgb, var(--good) 25%, transparent)' } }), 'head literal true') : null);
  return el('div', { class: 'histwrap' }, g, legend);
}

/* the head literal reading (proto li, k) through `op`, if one is not constant */
function literalOn(li, k, op) {
  const c = card();
  const i = c.head.literals.findIndex(lit => { const col = c.readout[lit.col]; return col.proto_layer === li && col.k === k && col.op === op && lit.constant === null; });
  return i < 0 ? null : i;
}
function literalTrueAt(i, s) {
  const lit = card().head.literals[i];
  return litTrue(lit.phi_w * (lit.pos ? s : 1 - s) + lit.phi_b);
}

/* agreement of the reference graphs with prototype (li, k), the current graph marked */
function protoAgreement(li, k, { width = 360, compact = false } = {}) {
  const c = card(), p = c.protos[li], dist = p.dist[k], out = el('div', {});
  const series = counts => counts.map((cs, cl) => ({ name: className(cl), color: CLASS_COLORS[cl % 4], counts: cs }));
  const here = !compact && !!T();
  if (p.kind === 'graph') {
    const lab = [...Array(dist.bins).keys()].map(b => f2(b / dist.bins, 2));
    const s = here ? protoSim(li, k) : null;
    const lit = literalOn(li, k, 'pooled');
    out.append(el('div', { class: 'small muted' }, `similarity of each ${c.ref_split} graph to ${protoName(li, k)}, share of each class`),
      histChart({ series: series(dist.graph), labels: lab, xTitle: 'similarity', width, height: compact ? 90 : 120,
        marker: s === null ? null : Math.min(dist.bins - 1, Math.floor(s * dist.bins)), markerText: 'this graph',
        trueAt: lit === null ? null : b => literalTrueAt(lit, (b + 0.5) / dist.bins) }));
    return out;
  }
  // bins are indexed by mismatches m = 0..d; the charts run by agreement a = 1 − m/d (utils/agreement.py), so bin i is m = d − i
  const d = dist.d, D = Math.max(d, 1), byAgree = counts => [...counts].reverse();
  const lab = [...Array(d + 1).keys()].map(i => f2(i / D, 2));
  const sim = m => p.masked ? Math.exp(-m / p.T) : 1 - m / D;
  const mmax = here && card().protos[li].kind === 'node' ? Math.round(Math.min(...T().protos[li].mism.map(r => r[k]))) : null;
  const litMax = literalOn(li, k, 'max');
  const sep = separation(dist.closest);
  const key = p.masked
    ? `agreement a = share of the ${d} cared bits a node matches (m mismatches: a = 1 − m/${d}), as in agreement.png of the training run. The model itself reads the similarity exp(−m / T), T = ${f2(p.T)}: a = 1 → 1, ${[1, 2, 3].filter(m => m <= d).map(m => `a = ${f2(1 - m / D, 2)} → ${f2(sim(m), 2)}`).join(', ')}; the green zone is where the head literal is true.`
    : `agreement a = share of the ${d} bits a node matches; for an unmasked prototype it is also the similarity the model reads.`;
  if (!compact) out.append(el('div', { class: 'small', style: { marginBottom: '4px' } }, key));
  out.append(el('div', { class: 'small muted' }, `${c.ref_split} graphs by the agreement of their best node (what "max sim" reads). 1: the graph contains an exact match. Share of each class${sep === null ? '' : `; separation ${f2(sep, 3)}`}`),
    histChart({ series: series(dist.closest.map(byAgree)), labels: lab, xTitle: `agreement of the best node (of ${d} cared bits)`, width, height: compact ? 90 : 120,
      marker: mmax === null ? null : d - mmax, markerText: 'this graph', trueAt: litMax === null ? null : i => literalTrueAt(litMax, sim(d - i)) }));
  if (compact) return out;
  const litMean = literalOn(li, k, 'mean');
  const meanHere = here ? T().s[c.readout.findIndex(r => r.proto_layer === li && r.k === k && r.op === 'mean')] : null;
  const [lo, hi] = dist.mean_range || [0, 1], bw = Math.max(hi - lo, 1e-9) / dist.bins;
  const at = v => Math.max(0, Math.min(dist.bins - 1, Math.floor((v - lo) / bw)));
  out.append(el('div', { class: 'small muted', style: { marginTop: '8px' } }, `${c.ref_split} graphs by the similarity averaged over all their nodes (what "mean sim" reads; observed range ${f2(lo, 3)}–${f2(hi, 3)}). It is 1 only if every node matches; a graph with one exact node among 20 gets about 0.05, so this stays far below 1 even where max sim is 1.`),
    histChart({ series: series(dist.mean), labels: [...Array(dist.bins).keys()].map(b => f2(lo + b * bw, 3)), xTitle: 'mean similarity', width,
      marker: meanHere === null || meanHere === undefined ? null : at(meanHere), markerText: 'this graph',
      trueAt: litMean === null ? null : b => literalTrueAt(litMean, lo + (b + 0.5) * bw) }));
  const mv = here ? Math.round(protoMism(li, k)) : null;
  out.append(el('div', { class: 'small muted', style: { marginTop: '8px' } }, `every ${c.ref_split} node by its own agreement with ${protoName(li, k)} (${c.ref_nodes} nodes; 1 is an exact match)`),
    histChart({ series: [{ name: 'nodes', color: 'var(--ink2)', counts: byAgree(dist.nodes) }], labels: lab, xTitle: 'agreement of the node', width, normalise: false,
      marker: mv === null ? null : d - mv, markerText: nodeName(S.node) }));
  return out;
}

/* utils/agreement.separation from per-class histograms over the same bins:
   max over classes of |2·AUC(class vs rest) − 1| (ties count one half), null with one class */
function separation(counts) {
  const present = counts.filter(cs => cs.some(n => n > 0));
  if (present.length < 2) return null;
  let best = 0;
  for (const cs of present) {
    const rest = cs.map((_, b) => counts.reduce((a, o) => a + o[b], 0) - cs[b]);
    const nc = cs.reduce((a, b) => a + b, 0), nr = rest.reduce((a, b) => a + b, 0);
    let below = 0, auc = 0;
    cs.forEach((n, b) => { auc += n * (below + rest[b] / 2); below += rest[b]; });
    best = Math.max(best, Math.abs(2 * auc / (nc * nr) - 1));
  }
  return best;
}

function mixText(mix) {
  const tot = mix.reduce((a, b) => a + b, 0);
  if (!tot) return '';
  const k = mix.indexOf(Math.max(...mix));
  return ` (${pct(mix[k] / tot, 0)} ${className(k)})`;
}
function mixBar(mix) {
  const tot = mix.reduce((a, b) => a + b, 0) || 1;
  return hoverTip(el('div', { class: 'mix' }, mix.map(n => el('div', { style: { width: `${100 * n / tot}%` } }))),
    mix.map((n, k) => `${className(k)}: ${n} (${pct(n / tot, 0)})`).join('\n'));
}

// ------------------------------------------------------------------ inspector

function renderInspector() {
  const box = clear($('inspector')), s = S.sel;
  if (S.step !== null) return renderStepInspector(box);
  ({ unit: unitInspector, proto: protoInspector, readout: readoutInspector, headlit: headLitInspector, class: classInspector })[s.type](box);
}

const STEP_TEXT = {
  in: 'Each node starts from its one-hot features. A conv literal does not read a feature at the node itself: it reads how many nodes of its closed neighbourhood N[v] = {v} ∪ neighbours carry it (GINConv sum, eps = 0).',
  L: 'Each unit is a threshold rule: it fires (1) iff the weights of its true literals sum to at least S. A literal is true iff its step function phi(w·count + b) ≥ 0.5. The purple edges are the shortest sufficient set of true literals of each unit that fires at this node: the exact reason, not an attribution score.',
  P: 'Prototypes are binary patterns over the trunk units. A masked prototype only cares about a few bits; its similarity is exp(−mismatches / T), so 1 means every cared bit matches. Blue: needs 1, orange: needs 0, red: mismatch at this node.',
  R: 'The readout pools each prototype\'s similarity over all nodes of the graph (mean, max): "some node matches P" is max = 1.',
  C: 'Each class is a threshold rule over the head literals (each one a step function of a readout value, or of 1 − value). The prediction is the class with the largest margin; the purple edges are the fewest input-dependent true literals that, with the always-true ones (a bias, dotted), reach its threshold.',
};
function renderStepInspector(box) {
  const cols = netColumns(), col = cols[S.step];
  box.append(el('div', { class: 'ihead' }, el('span', { class: 'big' }, `Step ${S.step + 1}/${cols.length}: ${col.title}`)),
    el('p', {}, STEP_TEXT[col.key]),
    el('p', { class: 'muted small' }, '← → step · space play/pause · Esc show all · click any cell to inspect it'));
  if (col.key === 'L') {
    const l = col.l, on = T().layers[l].out[S.node].split('').filter(b => b === '1').length;
    box.append(el('p', {}, `${on} of ${card().hidden} units fire at ${nodeName(S.node)}.`));
  }
  if (col.key === 'C') classInspector(box, T().rule, true);
}

function nodeHeader(extra) {
  return el('div', { class: 'muted small' }, `at ${nodeName(S.node)} of ${T().split} #${T().i}`, extra || '');
}

function unitInspector(box) {
  const { l, u } = S.sel, c = card(), conv = c.conv[l], t = T(), v = S.node;
  const on = outBit(l, v, u), W = conv.W[u], S_ = conv.S[u], why = t.layers[l].why[v][u] || [];
  const net = t.layers[l].net[v][u];
  box.append(el('div', { class: 'ihead' }, el('span', { class: 'big' }, unitName(l, u)), el('span', { class: 'state ' + (on ? 'on' : 'off') }, on ? '1' : '0'),
    el('span', { class: 'muted' }, `network ${f2(net, 3)}`)), nodeHeader());
  box.append(el('dl', { class: 'kv' },
    el('dt', {}, 'rule'), el('dd', {}, `fires iff Σ w·lit ≥ S = ${f2(S_, 3)}`),
    el('dt', {}, 'margin here'), el('dd', {}, sgn(t.layers[l].margin[v][u])),
    el('dt', {}, 'shortest rule'), el('dd', {}, conv.min_card === null ? '–' : conv.min_card[u] === null ? 'never fires' : `${conv.min_card[u]} literals (exact lower bound)`),
    el('dt', {}, 'minimal rules'), el('dd', {}, `≈ ${sci(conv.n_covers[u])}${conv.n_covers[u] > 1e5 ? ' (too many to list: read the per-node reason below)' : ''}`),
    el('dt', {}, 'fires on'), el('dd', {}, `${pct(conv.rate[u])} of ${c.ref_split} nodes`)));
  const items = [];
  for (let i = 0; i < W.length; i++) if (W[i] > 0) items.push({ w: W[i], state: on && why.includes(i) ? 'expl' : litBit(l, v, i) ? 'true' : 'false', label: conv.literals[i].text, onclick: null });
  box.append(el('h3', {}, 'threshold'), thresholdBar(items, S_));
  const total = W.reduce((a, b) => a + b, 0), held = why.reduce((a, i) => a + W[i], 0);
  box.append(el('div', { class: 'why-line' }, on
    ? `Fires because the ${why.length} purple literal${why.length > 1 ? 's' : ''} alone reach S (the fewest true literals that do: exact).`
    : total < S_ ? 'Can never fire: all its weights together stay below S.'
      : `Stays 0 because the ${why.length} purple literal${why.length > 1 ? 's are' : ' is'} false: they weigh ${f2(held)} > Σw − S = ${f2(total - S_)}, so even with every other literal true the sum stays below S (the fewest false literals that guarantee it: exact).`));
  box.append(el('h3', {}, `literals (${S.showAllLits ? 'all' : 'weight ≥ 3% of max'}) · hover: the nodes counted`),
    el('div', { class: 'muted small' }, 'Plots: the step phi(w·count + b); green = true. Counts are integers, so only the dots matter (filled = true, hollow = false); purple = the count at this node.'),
    litTable(l, u, v),
    el('button', { onclick: () => { S.showAllLits = !S.showAllLits; renderInspector(); } }, S.showAllLits ? 'only the heavy literals' : 'show every literal'));
  box.append(el('h3', {}, 'readable rules (data-simplified)'), rulesBlock(l, u, v));
  if (l === 0 && conv.rules && conv.rules[u]) {
    box.append(el('h3', {}, `exact minimal rules observed on ${c.ref_split}`));
    const rules = conv.rules[u];
    if (!rules.length) box.append(el('div', { class: 'muted' }, 'no minimal rule holds on any reference node'));
    for (const r of rules) {
      box.append(el('div', { class: 'rule' }, el('span', { class: 'sup' }, pct(r.support, 2)),
        r.literals.length ? r.literals.map(i => conv.literals[i].text).join('  ∧  ') : 'TRUE (S ≤ 0)'));
    }
    box.append(el('div', { class: 'muted small' }, 'Each line is a minimal set of literals reaching S; the unit is their OR. Support: share of reference nodes where the set holds.'));
  }
}

function litTable(l, u, v) {
  const c = card(), conv = c.conv[l], W = conv.W[u], t = T(), why = t.layers[l].why[v][u] || [];
  const mx = Math.max(...W);
  const order = W.map((w, i) => i).filter(i => S.showAllLits ? W[i] > 0 : W[i] >= 0.03 * mx).sort((a, b) => W[b] - W[a]);
  const tb = el('table', { class: 'lits' });
  for (const i of order) {
    const on = litBit(l, v, i), count = t.layers[l].count[v][i];
    const lower = l > 0 ? { l: l - 1, u: i % conv.n_in } : null;
    const tr = el('tr', { class: (why.includes(i) ? 'inexpl' : '') + (on ? '' : ' isfalse'),
      onmouseenter: () => { S.hoverLit = { l, i }; renderGraph(); }, onmouseleave: () => { S.hoverLit = null; renderGraph(); },
      onclick: () => { if (lower) select({ type: 'unit', l: lower.l, u: lower.u }); },
      title: lower ? `click: open ${unitName(lower.l, lower.u)} (then click a highlighted node to see why it holds there)` : '' },
      el('td', { style: { width: '54px' } }, el('div', { class: 'wbar', style: { width: `${Math.max(2, 50 * W[i] / mx)}px` } }), el('span', { class: 'mono small muted' }, f2(W[i]))),
      el('td', {}, el('div', { class: 'mono small' }, conv.literals[i].text), el('div', { class: 'small muted' }, `count here = ${count}`)),
      el('td', {}, el('span', { class: 'tf ' + (on ? 't' : 'f') }, on ? 'T' : 'F')),
      el('td', {}, stepPlot({ w: conv.phi_w[i], b: conv.phi_b[i], tau: conv.phi_tau, lo: conv.count_lo[i], hi: conv.count_hi[i], x: count, integer: true, width: 120, height: 40 })));
    tb.append(tr);
  }
  return tb;
}

function protoInspector(box) {
  const { li, k } = S.sel, c = card(), p = c.protos[li], v = S.node;
  const cared = caredBits(li, k), sim = protoSim(li, k), name = protoName(li, k);
  box.append(el('div', { class: 'ihead' }, el('span', { class: 'big' }, name), el('span', { class: 'muted' }, `${p.kind} prototype · ${cared.length} cared bit${cared.length === 1 ? '' : 's'} of ${p.d}`)),
    p.kind === 'node' ? nodeHeader() : el('div', { class: 'muted small' }, 'compared with the pooled trunk of the whole graph'));
  box.append(el('dl', { class: 'kv' },
    el('dt', {}, 'agreement'), el('dd', {}, cared.length ? `${f2(1 - protoMism(li, k) / cared.length, 3)} (${cared.length - Math.round(protoMism(li, k))} of ${cared.length} cared bits match)` : '1 (no cared bit)'),
    el('dt', {}, 'similarity'), el('dd', {}, `${f2(sim, 3)}${p.masked ? ` = exp(−${f2(protoMism(li, k), 2)} / T), T = ${f2(p.T)}: what the model reads` : ' (Hamming agreement, what the model reads)'}`),
    p.kind === 'node' ? el('dt', {}, 'best node') : null,
    p.kind === 'node' ? el('dd', {}, (() => { const b = bestNode(k, li); return el('a', { href: '#', onclick: ev => { ev.preventDefault(); setNode(b); } }, `${nodeName(b)} (${f2(protoSim(li, k, b), 3)})`); })()) : null,
    el('dt', {}, `exact on ${c.ref_split}`), el('dd', {}, `${p.exact[k].graphs} graphs contain an exact match${mixText(p.exact[k].class_mix)}`, mixBar(p.exact[k].class_mix))));
  if (p.dist) box.append(el('h3', {}, `agreement on ${c.ref_split}`), protoAgreement(li, k, { width: 360 }));
  box.append(el('h3', {}, 'rule (AND of its cared bits)'));
  const inp = protoBitsAt(li, v);
  const chips = el('div', { class: 'rulebox' });
  if (!cared.length) chips.append(el('div', { class: 'muted' }, 'cares about no bit: similarity is always 1, the prototype acts as a constant'));
  for (const d of cared) {
    const need = p.bits[k][d] === '1', has = inp[d] >= 0.5, unit = p.bit_unit[d];
    const chip = el('span', { class: `litchip ${need ? 'need1' : 'need0'}${has !== need ? ' miss' : ''}`,
      onclick: () => select({ type: 'unit', l: Math.floor(unit / c.hidden), u: unit % c.hidden }) }, (need ? '' : '¬') + p.bit_labels[d]);
    hoverTip(chip, `${p.bit_labels[d]} must be ${need ? 1 : 0}; here ${f2(inp[d], 2)}${has !== need ? ' → mismatch' : ' ✓'}\nclick: open the unit`);
    chips.append(chip);
  }
  box.append(chips, el('h3', {}, 'cared bits over the trunk · dot: value here · red: mismatch'), bitGrid(li, k, v, 5));
  if (p.top[k].length) {
    box.append(el('h3', {}, `best matches on ${c.ref_split}`));
    for (const m of p.top[k]) {
      const can = S.src.has(c.ref_split, m.graph);
      box.append(el('div', { class: 'small' }, el('a', { href: '#', style: can ? null : { opacity: 0.45 }, onclick: ev => { ev.preventDefault(); openGraph(c.ref_split, m.graph, { node: m.node === undefined ? null : m.node, sel: { type: 'proto', li, k } }); } },
        `${c.ref_split} #${m.graph}${m.node !== undefined ? `, node ${m.node}` : ''}`), ` · similarity ${f2(m.sim, 3)}`));
    }
  }
  const cites = S.index.proto_cites[li];
  if (cites) box.append(el('h3', {}, `cited by explanations on ${S.split}`), el('div', {}, cites[k].map((n, cl) => `${className(cl)}: ${n}`).join(' · ')));
}

function readoutInspector(box) {
  const j = S.sel.j, c = card(), r = c.readout[j], t = T();
  const name = r.k === undefined ? `${r.op}(${unitName(Math.floor(r.unit / c.hidden), r.unit % c.hidden)})` : `${r.op} sim(${protoName(r.proto_layer, r.k)})`;
  box.append(el('div', { class: 'ihead' }, el('span', { class: 'big' }, name), el('span', { class: 'muted' }, `= ${f2(t.s[j], 3)} on this graph`)));
  if (r.k !== undefined) box.append(el('button', { onclick: () => select({ type: 'proto', li: r.proto_layer, k: r.k }) }, `open ${protoName(r.proto_layer, r.k)}`));
  box.append(el('h3', {}, 'head literals reading it'));
  for (const i of litsOfCol(j)) box.append(headLitBlock(i));
}

function headLitBlock(i) {
  const c = card(), t = T(), lit = c.head.literals[i], s = t.s[lit.col], on = t.head_lit[i] === '1';
  const bounded = lit.domain[0] === 0 && lit.domain[1] === 1;
  const sdom = lit.pos || !bounded ? lit.domain : [1 - lit.domain[1], 1 - lit.domain[0]];
  return el('div', { class: 'rule', style: { cursor: 'pointer' }, onclick: () => select({ type: 'headlit', i }) },
    el('div', {}, el('span', { class: 'tf ' + (on ? 't' : 'f') }, on ? 'T ' : 'F '), lit.text, lit.constant !== null ? el('span', { class: 'chip', style: { marginLeft: '6px' } }, 'constant') : null),
    stepPlot({ w: lit.phi_w, b: lit.phi_b, tau: c.head.phi_tau, lo: sdom[0], hi: sdom[1], x: s, map: lit.pos ? (z => z) : (z => 1 - z), width: 300, height: 52 }),
    el('div', { class: 'muted small' }, `x-axis: ${lit.what}${lit.pos ? '' : ' (the literal reads 1 − value)'} · weights: ${c.classes.map((n, k) => `${n} ${f2(c.head.W[k][i])}`).join(', ')}`));
}

function headLitInspector(box) {
  const i = S.sel.i, c = card(), lit = c.head.literals[i], p = litProto(i);
  box.append(el('div', { class: 'ihead' }, el('span', { class: 'big' }, `head literal ${i}`)), headLitBlock(i));
  if (p) {
    const pr = c.protos[p.li];
    box.append(el('button', { onclick: () => select({ type: 'proto', li: p.li, k: p.k }, pr.kind === 'node' ? bestNode(p.k, p.li) : null) }, `open ${protoName(p.li, p.k)}${pr.kind === 'node' ? ' at its best node' : ''}`));
    if (hasCare(p.li, p.k)) box.append(el('div', { class: 'rule' }, pr.rule[p.k]));
  }
  const cites = S.index.literal_cites[i];
  box.append(el('h3', {}, `cited by explanations on ${S.split}`), el('div', {}, cites.map((n, k) => `${className(k)}: ${n}`).join(' · ')));
}

function classInspector(box, k = null, compact = false) {
  k = k === null ? S.sel.c : k;
  const c = card(), t = T(), W = c.head.W[k], S_ = c.head.S[k];
  const expl = k === t.rule ? (t.evidence || []) : [];
  if (!compact) box.append(el('div', { class: 'ihead' }, el('span', { class: 'big' }, `class ${className(k)}`),
    el('span', { class: 'muted' }, `Σ w·lit ${f2(t.margin[k] + S_, 3)} vs S ${f2(S_, 3)}: margin ${sgn(t.margin[k])}${k === t.rule ? ' · predicted' : ''}`)));
  box.append(el('dl', { class: 'kv' },
    el('dt', {}, 'rule'), el('dd', {}, `Σ w·lit ≥ S = ${f2(S_, 3)} over ${W.length} head literals`),
    el('dt', {}, 'shortest rule'), el('dd', {}, c.head.min_card[k] === null ? 'never fires' : `${c.head.min_card[k]} literals`),
    el('dt', {}, 'minimal rules'), el('dd', {}, `≈ ${sci(c.head.n_covers[k])}`)));
  const items = W.map((w, i) => ({ w, state: expl.includes(i) ? 'expl' : c.head.literals[i].constant === true ? 'const' : t.head_lit[i] === '1' ? 'true' : 'false', label: c.head.literals[i].text, onclick: () => select({ type: 'headlit', i }) })).filter(it => it.w > 0);
  box.append(el('h3', {}, 'threshold'), thresholdBar(items, S_),
    el('div', { class: 'muted small' }, 'purple: the evidence (fewest input-dependent literals) · hatched grey: always-true literals (a bias) · grey: other true literals'));
  const mx = Math.max(...W);
  const order = W.map((w, i) => i).filter(i => W[i] >= 0.03 * mx).sort((a, b) => W[b] - W[a]);
  box.append(el('h3', {}, 'its literals (weight ≥ 3% of max)'));
  const tb = el('table', { class: 'lits' });
  for (const i of order) {
    const on = t.head_lit[i] === '1';
    tb.append(el('tr', { class: (expl.includes(i) ? 'inexpl' : '') + (on && c.head.literals[i].constant === null ? '' : ' isfalse'), onclick: () => select({ type: 'headlit', i }) },
      el('td', { style: { width: '54px' } }, el('div', { class: 'wbar', style: { width: `${Math.max(2, 50 * W[i] / mx)}px` } }), el('span', { class: 'mono small muted' }, f2(W[i]))),
      el('td', {}, el('div', { class: 'small' }, c.head.literals[i].text), el('div', { class: 'small muted' }, `${c.head.literals[i].what} = ${f2(t.s[c.head.literals[i].col], 3)}`)),
      el('td', {}, el('span', { class: 'tf ' + (on ? 't' : 'f') }, on ? 'T' : 'F'))));
  }
  box.append(tb);
}

// ------------------------------------------------------------------ prototypes tab

function renderProtos(main) {
  const c = card();
  const bar = el('div', { class: 'toolbar' },
    el('span', {}, 'sort by'),
    el('select', { onchange: ev => { S.protoSort = ev.target.value; renderMain(); } },
      [['cites', `citations on ${S.split}`], ['exact', `exact matches on ${c.ref_split}`], ['bits', 'cared bits'], ['k', 'index']].map(([v, l]) => el('option', { value: v, selected: S.protoSort === v }, l))),
    el('label', {}, el('input', { type: 'checkbox', checked: S.hideTrivial, onchange: ev => { S.hideTrivial = ev.target.checked; renderMain(); } }), ' hide prototypes with no cared bit (constant similarity 1)'));
  main.append(bar);
  const grid = el('div', { class: 'pgrid' });
  const all = [];
  c.protos.forEach((p, li) => { for (let k = 0; k < p.K; k++) all.push({ li, k }); });
  const cites = ({ li, k }) => (S.index.proto_cites[li] || [])[k] ? S.index.proto_cites[li][k].reduce((a, b) => a + b, 0) : 0;
  const key = { cites: x => -cites(x), exact: x => -c.protos[x.li].exact[x.k].graphs, bits: x => -caredBits(x.li, x.k).length, k: x => x.li * 1000 + x.k }[S.protoSort];
  all.sort((a, b) => key(a) - key(b));
  let hidden = 0;
  for (const { li, k } of all) {
    const p = c.protos[li], cared = caredBits(li, k);
    if (S.hideTrivial && !cared.length) { hidden++; continue; }
    const cc = (S.index.proto_cites[li] || [])[k] || [];
    const chips = el('div', { class: 'rulebox' }, cared.map(d => el('span', { class: 'litchip ' + (p.bits[k][d] === '1' ? 'need1' : 'need0') }, (p.bits[k][d] === '1' ? '' : '¬') + p.bit_labels[d])));
    const top = p.top[k][0];
    grid.append(el('div', { class: 'panel pcard' },
      el('h2', {}, protoName(li, k), el('span', { class: 'muted small' }, `${p.kind} · ${cared.length} cared bit${cared.length === 1 ? '' : 's'} · cited ${cc.reduce((a, b) => a + b, 0)}× (${cc.map((n, cl) => `${className(cl)} ${n}`).join(', ')})`)),
      chips,
      bitGrid(li, k, null, 4),
      el('div', { class: 'small' }, `exact match in ${p.exact[k].graphs} ${c.ref_split} graphs${mixText(p.exact[k].class_mix)}`, mixBar(p.exact[k].class_mix)),
      p.dist ? protoAgreement(li, k, { width: 300, compact: true }) : null,
      top ? el('button', { onclick: () => openGraph(c.ref_split, top.graph, { node: top.node === undefined ? null : top.node, sel: { type: 'proto', li, k } }) },
        `open best match: ${c.ref_split} #${top.graph}${top.node !== undefined ? ` node ${top.node}` : ''}`) : null));
  }
  main.append(grid);
  if (hidden) main.append(el('p', { class: 'muted' }, `${hidden} prototype${hidden > 1 ? 's' : ''} with no cared bit hidden: their similarity is 1 on every node, so their head literals are constants.`));
}

// ------------------------------------------------------------------ trunk units tab

function renderUnits(main) {
  const c = card(), l = S.unitLayer, conv = c.conv[l];
  const bar = el('div', { class: 'toolbar' }, c.conv.map((_, j) => el('button', { class: j === l ? 'on' : '', onclick: () => { S.unitLayer = j; S.unitSel = null; renderMain(); } }, `conv ${j}`)),
    el('span', { class: 'muted' }, `${c.hidden} units · ${2 * conv.n_in} literals (${conv.n_in} counts of ${l === 0 ? 'input features' : `layer ${l - 1} units`} and of their negations)`));
  main.append(bar);
  const wrap = el('div', { class: 'units' });
  const left = el('div', { class: 'panel' });
  // weight heatmap: rows units, cols literals
  const nL = 2 * conv.n_in, mx = Math.max(...conv.W.flat());
  const cv = el('canvas', { class: 'heat', width: nL, height: c.hidden, style: { height: `${Math.min(520, c.hidden * 7)}px` } });
  const ctx = cv.getContext('2d'), img = ctx.createImageData(nL, c.hidden);
  const ink = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
  const rgb = hexRgb(ink) || [122, 75, 224], bg = hexRgb(getComputedStyle(document.documentElement).getPropertyValue('--off').trim()) || [235, 233, 226];
  for (let u = 0; u < c.hidden; u++) for (let i = 0; i < nL; i++) {
    const a = Math.sqrt(conv.W[u][i] / mx), o = 4 * (u * nL + i);
    for (let ch = 0; ch < 3; ch++) img.data[o + ch] = Math.round(bg[ch] + (rgb[ch] - bg[ch]) * a);
    img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const pick = ev => { const r = cv.getBoundingClientRect(); return [Math.floor((ev.clientY - r.top) / r.height * c.hidden), Math.floor((ev.clientX - r.left) / r.width * nL)]; };
  cv.addEventListener('mousemove', ev => { const [u, i] = pick(ev); if (u >= 0 && u < c.hidden && i >= 0 && i < nL) tip.show(ev, `${unitName(l, u)} ← ${conv.literals[i].text}\nw = ${f2(conv.W[u][i], 3)}`); });
  cv.addEventListener('mouseleave', tip.hide);
  cv.addEventListener('click', ev => { S.unitSel = pick(ev)[0]; renderMain(); });
  left.append(el('h2', {}, `weights of conv ${l}: units (rows) × literals (columns)`),
    el('div', { class: 'muted small' }, `left half: counts of ${l === 0 ? 'features' : 'units'}, right half: counts of their negations · colour ∝ √w · hover for the literal, click for the unit`), cv);
  // table
  const sortKey = { u: u => u, rate: u => -conv.rate[u], S: u => -conv.S[u], card: u => conv.min_card[u] ?? 1e9, covers: u => conv.n_covers[u] }[S.unitSort];
  const order = [...Array(c.hidden).keys()].sort((a, b) => sortKey(a) - sortKey(b));
  const th = (k, label) => el('th', { onclick: () => { S.unitSort = k; renderMain(); } }, label + (S.unitSort === k ? ' ▾' : ''));
  const tb = el('table', { class: 'utable' }, el('tr', {}, th('u', 'unit'), th('rate', `fires (${c.ref_split})`), th('S', 'threshold S'), th('card', 'shortest rule'), th('covers', '# minimal rules'), el('th', {}, 'heaviest literals')));
  for (const u of order) {
    const top = conv.W[u].map((w, i) => [w, i]).sort((a, b) => b[0] - a[0]).slice(0, 3).map(([, i]) => conv.subjects[i]).join(', ');
    tb.append(el('tr', { class: S.unitSel === u ? 'sel' : '', onclick: () => { S.unitSel = u; renderMain(); } },
      el('td', { class: 'mono' }, unitName(l, u)), el('td', {}, pct(conv.rate[u])), el('td', {}, f2(conv.S[u])),
      el('td', {}, conv.min_card[u] === null ? 'never' : String(conv.min_card[u])), el('td', {}, sci(conv.n_covers[u])), el('td', { class: 'mono small' }, top)));
  }
  left.append(el('h3', {}, 'units'), el('div', { class: 'scroll' }, tb));
  const right = el('div', { class: 'panel inspector' });
  if (S.unitSel === null) right.append(el('p', { class: 'muted' }, 'Pick a unit (row or heatmap) to see its literals and their step functions.'));
  else unitCard(right, l, S.unitSel);
  wrap.append(left, right);
  main.append(wrap);
}

function unitCard(box, l, u) {
  const c = card(), conv = c.conv[l], W = conv.W[u], mx = Math.max(...W);
  box.append(el('div', { class: 'ihead' }, el('span', { class: 'big' }, unitName(l, u)), el('span', { class: 'muted' }, `fires on ${pct(conv.rate[u])} of ${c.ref_split} nodes`)),
    el('dl', { class: 'kv' }, el('dt', {}, 'rule'), el('dd', {}, `Σ w·lit ≥ S = ${f2(conv.S[u], 3)}`),
      el('dt', {}, 'shortest rule'), el('dd', {}, conv.min_card[u] === null ? 'never fires' : `${conv.min_card[u]} literals`),
      el('dt', {}, 'minimal rules'), el('dd', {}, `≈ ${sci(conv.n_covers[u])}`)));
  if (T()) box.append(el('button', { onclick: () => { S.tab = 'predict'; renderTabs(); renderMain(); select({ type: 'unit', l, u }); } }, `show at ${nodeName(S.node)} of ${T().split} #${T().i}`));
  box.append(el('h3', {}, 'readable rules (data-simplified)'), rulesBlock(l, u, null));
  if (l === 0 && conv.rules) {
    box.append(el('h3', {}, `exact minimal rules observed on ${c.ref_split}`));
    for (const r of conv.rules[u]) box.append(el('div', { class: 'rule' }, el('span', { class: 'sup' }, pct(r.support, 2)), r.literals.length ? r.literals.map(i => conv.literals[i].text).join('  ∧  ') : 'TRUE'));
  }
  box.append(el('h3', {}, 'literals (weight ≥ 3% of max)'));
  const tb = el('table', { class: 'lits' });
  for (const i of W.map((w, i) => i).filter(i => W[i] >= 0.03 * mx).sort((a, b) => W[b] - W[a])) {
    tb.append(el('tr', {},
      el('td', { style: { width: '54px' } }, el('div', { class: 'wbar', style: { width: `${Math.max(2, 50 * W[i] / mx)}px` } }), el('span', { class: 'mono small muted' }, f2(W[i]))),
      el('td', { class: 'mono small' }, conv.literals[i].text),
      el('td', {}, stepPlot({ w: conv.phi_w[i], b: conv.phi_b[i], tau: conv.phi_tau, lo: conv.count_lo[i], hi: conv.count_hi[i], integer: true, width: 130, height: 40 }))));
  }
  box.append(tb);
}

function hexRgb(s) {
  const m = /^#?([0-9a-f]{6})$/i.exec(s.trim());
  return m ? [0, 2, 4].map(j => parseInt(m[1].slice(j, j + 2), 16)) : null;
}

// ------------------------------------------------------------------ head tab

function renderHead(main) {
  const c = card();
  const grid = el('div', { class: 'pgrid', style: { gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))' } });
  c.classes.forEach((name, k) => {
    const W = c.head.W[k], mx = Math.max(...W);
    const order = W.map((w, i) => i).filter(i => W[i] >= 0.03 * mx).sort((a, b) => W[b] - W[a]);
    const constant = order.filter(i => c.head.literals[i].constant !== null);
    const panel = el('div', { class: 'panel' }, el('h2', {}, `class ${name}`),
      el('dl', { class: 'kv' }, el('dt', {}, 'rule'), el('dd', {}, `Σ w·lit ≥ S = ${f2(c.head.S[k], 3)}`),
        el('dt', {}, 'shortest rule'), el('dd', {}, c.head.min_card[k] === null ? 'never' : `${c.head.min_card[k]} literals`),
        el('dt', {}, 'minimal rules'), el('dd', {}, `≈ ${sci(c.head.n_covers[k])}`),
        el('dt', {}, 'constant literals'), el('dd', {}, `${constant.length} of the ${order.length} listed are always / never true; the always-true ones add ${f2(c.head.literals.reduce((s, lit, i) => s + (lit.constant === true ? W[i] : 0), 0))} on every graph (a bias), so the input-dependent literals must reach ${f2(c.head.S[k] - c.head.literals.reduce((s, lit, i) => s + (lit.constant === true ? W[i] : 0), 0))}`)));
    const tb = el('table', { class: 'lits' });
    for (const i of order.filter(i => S.showConst || c.head.literals[i].constant === null)) {
      const lit = c.head.literals[i];
      const bounded = lit.domain[0] === 0 && lit.domain[1] === 1;
      const sdom = lit.pos || !bounded ? lit.domain : [1 - lit.domain[1], 1 - lit.domain[0]];
      tb.append(el('tr', { class: constant.includes(i) ? 'isfalse' : '' },
        el('td', { style: { width: '54px' } }, el('div', { class: 'wbar', style: { width: `${Math.max(2, 50 * W[i] / mx)}px` } }), el('span', { class: 'mono small muted' }, f2(W[i]))),
        el('td', { class: 'small' }, lit.text),
        el('td', {}, stepPlot({ w: lit.phi_w, b: lit.phi_b, tau: c.head.phi_tau, lo: sdom[0], hi: sdom[1], map: lit.pos ? (z => z) : (z => 1 - z), width: 130, height: 40 }))));
    }
    panel.append(el('h3', {}, S.showConst ? 'literals by weight' : 'input-dependent literals by weight'), tb,
      el('button', { onclick: () => { S.showConst = !S.showConst; renderMain(); } }, S.showConst ? 'hide constant literals' : `show the ${constant.length} constant literals too`));
    grid.append(panel);
  });
  main.append(grid);
}

// ------------------------------------------------------------------ metrics tab

const METRIC_DOC = {
  acc: 'network accuracy', balanced_acc: 'network balanced accuracy', auc: 'network ROC-AUC', f1: 'minority-class F1',
  logic_acc: 'accuracy of the symbolic model', logic_fidelity: 'rules and network agree', head_fidelity: 'network trunk, binarised head: agreement',
  trunk_bit_agreement: 'node bits where the symbolic trunk equals the network ≥ 0.5', rule_backed: 'the predicted class rule actually fires',
  expl_literals_median: 'head literals per explanation (median)', expl_bits_median: 'trunk bits cited per explanation (median)',
  protos_cited: 'distinct prototypes cited', bits_per_proto_median: 'cared bits of a cited prototype (median)', units_cited: 'distinct trunk units cited',
  units_cited_L0: 'share of cited units in conv layer 0', literal_purity: 'class purity of cited head literals',
  expl_len_median: 'head literals per explanation, constants included (median)', expl_len_mean: 'same, mean',
  evidence_len_median: 'input-dependent head literals per explanation (median)', evidence_len_mean: 'same, mean',
  constant_head_literals: 'head literals true or false on every input',
  gt_precision: 'best-matching nodes inside the ground-truth region', gt_chance: 'same for a random node', mask_temp: 'prototype mask temperature',
};
function renderMetrics(main) {
  const sum = S.index.summary, m = S.meta;
  const tiles = (obj, keys) => el('div', { class: 'metrics' }, keys.filter(k => obj[k] !== undefined && obj[k] !== null).map(k =>
    el('div', { class: 'metric' }, el('div', { class: 'd' }, k), el('div', { class: 'v' }, Number.isInteger(obj[k]) && obj[k] > 1 ? String(obj[k]) : typeof obj[k] === 'number' && obj[k] <= 1 && !/median|mean|cited|n$|temp|literals/.test(k) ? pct(obj[k]) : f2(obj[k])), el('div', { class: 'd' }, METRIC_DOC[k] || ''))));
  main.append(el('div', { class: 'panel' }, el('h2', {}, `${S.split} split, recomputed by the explorer (${sum.graphs} graphs)`),
    tiles(sum, ['acc', 'logic_acc', 'logic_fidelity', 'head_fidelity', 'rule_backed', 'trunk_bit_agreement', 'expl_len_median', 'evidence_len_median', 'expl_len_mean', 'evidence_len_mean', 'constant_head_literals'])));
  for (const [split, met] of Object.entries(m.stored_metrics || {})) {
    main.append(el('div', { class: 'panel', style: { marginTop: '12px' } }, el('h2', {}, `interp_metrics.py, ${split} split (interp_${split}.json next to the checkpoint)`),
      tiles(met, Object.keys(METRIC_DOC))));
  }
  main.append(el('div', { class: 'panel', style: { marginTop: '12px' } }, el('h2', {}, 'run'),
    el('dl', { class: 'kv' }, el('dt', {}, 'path'), el('dd', { class: 'mono small' }, m.path),
      el('dt', {}, 'checkpoint'), el('dd', {}, m.ckpt), el('dt', {}, 'fine-tunes'), el('dd', {}, m.finetunes.join(' → ') || 'none'),
      el('dt', {}, 'read at'), el('dd', {}, `${m.read_at} · commit ${(m.git_commit || '?').slice(0, 10)}${m.git_dirty === 'True' ? ' (dirty tree)' : ''}`),
      el('dt', {}, 'reference split'), el('dd', {}, `${card().ref_split}: ${card().ref_graphs} graphs, ${card().ref_nodes} nodes (literal ranges, firing rates, prototype matches)`)),
    el('h3', {}, 'arguments'), el('div', { class: 'mono small' }, Object.entries(m.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join('  '))));
}

// ------------------------------------------------------------------ about

function renderAbout(main) {
  main.append(el('div', { class: 'panel about' },
    el('h2', {}, 'How to read the explorer'),
    el('p', {}, 'Everything on screen is the symbolic model: the same weights, with every truth value binarised the way the rules read it (interp_metrics.py). Each number is an exact property of the rules, not an attribution score. The network itself is shown beside it (network values, network prediction) so you can see where the two differ.'),
    el('h3', {}, 'conv literals and step functions'),
    el('p', {}, 'A conv unit reads 2·d literals. Literal j counts the nodes of the closed neighbourhood N[v] (v and its neighbours) where feature / lower unit j holds (or, for the second half, does not hold). The literal is true iff phi(w·count + b) ≥ 0.5, phi being a periodic step (a Fourier square wave), so a literal can be true on a band of counts, e.g. "count ∈ {1, 2}". The small plots draw phi, shade where it is true and mark the count at the selected node.'),
    el('h3', {}, 'units are threshold rules'),
    el('p', {}, 'A unit fires iff the weights (all ≥ 0) of its true literals sum to at least S. Its rule set is the minimal sets of literals reaching S (their OR). Layer-0 units have few, listed under "observed rules". Deeper units have ~1e9 minimal rules, too many to list, so the explorer gives instead, at every node, the exact shortest sufficient reason: the heaviest true literals until S is reached (no rule using fewer literals exists for that node). "Shortest rule" is the same bound over all inputs.'),
    el('h3', {}, 'prototypes and head'),
    el('p', {}, 'A (masked) prototype is an AND over a few trunk bits. Its similarity is exp(−mismatches / T), and the readout pools it over the nodes: "max = 1" means some node matches exactly. The head is again threshold rules over literals of these pooled values. The decision panel lists the shortest sufficient explanation of the prediction. "trace ›" jumps to the best-matching node of a cited prototype, from where every bit can be followed down to atom counts by clicking.'),
    el('h3', {}, 'navigation'),
    el('p', {}, 'Click a node in the graph to see the network at that node. Click any cell of the network (unit, prototype, readout, class) to inspect it. Hover a literal to ring the nodes it counts. ▶ plays the forward pass column by column, drawing the reasons that carry each step. Keys: ← → step, space play/pause, Esc show all.')));
}

boot();
