
// // How to run it
// // First time only (installs dependencies; Puppeteer downloads a headless Chromium):
// cd C:\Users\srde\Documents\Study\LanguageModels\tools
// npm install

// // Every time you want to rebuild the PDF:
// cd C:\Users\srde\Documents\Study\LanguageModels\tools
// npm run build

// // That outputs LanguageModels.pdf in the LanguageModels folder.



const fs = require('fs');
const path = require('path');
const MarkdownIt = require('markdown-it');
const texmath = require('markdown-it-texmath');
const katex = require('katex');
const puppeteer = require('puppeteer');

// Source dir = parent folder (the LanguageModels folder). Output = LanguageModels.pdf there.
// Override with: node build.js <sourceDir> <outputPdf>
const srcDir = process.argv[2] || path.join(__dirname, '..');
const outPdf = process.argv[3] || path.join(srcDir, 'LanguageModels.pdf');

// Collect all .md files, sorted (01..09 first, README last). Ignore anything in tools/.
const order = fs
  .readdirSync(srcDir)
  .filter((f) => f.toLowerCase().endsWith('.md'))
  .sort((a, b) => a.localeCompare(b, 'en', { numeric: true }));

const md = new MarkdownIt({ html: true, linkify: true, typographer: false, breaks: false });
md.use(texmath, {
  engine: katex,
  delimiters: 'dollars',
  katexOptions: { throwOnError: false, strict: false },
});

// Render ```mermaid fenced blocks as <div class="mermaid"> so Mermaid.js can draw them.
const defaultFence = md.renderer.rules.fence.bind(md.renderer.rules);
md.renderer.rules.fence = (tokens, idx, options, env, self) => {
  const token = tokens[idx];
  if ((token.info || '').trim().toLowerCase() === 'mermaid') {
    return `<div class="mermaid">${md.utils.escapeHtml(token.content)}</div>`;
  }
  return defaultFence(tokens, idx, options, env, self);
};

// --- Build a clickable Table of Contents from h1/h2 headings -----------------
const usedIds = new Set();
const makeId = (text) => {
  let base =
    text
      .toLowerCase()
      .replace(/\$[^$]*\$/g, '')
      .replace(/[`*_~]/g, '')
      .replace(/[^\w\s-]/g, '')
      .trim()
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .slice(0, 50) || 'section';
  let id = base;
  let n = 1;
  while (usedIds.has(id)) id = `${base}-${n++}`;
  usedIds.add(id);
  return id;
};
const cleanLabel = (s) =>
  s
    .replace(/\$[^$]*\$/g, '')
    .replace(/[`*_~]/g, '')
    .replace(/\\[a-zA-Z]+/g, '')
    .trim();

const toc = [];
const sections = order.map((f) => {
  const raw = fs.readFileSync(path.join(srcDir, f), 'utf8');
  const tokens = md.parse(raw, {});
  for (let j = 0; j < tokens.length; j++) {
    const t = tokens[j];
    if (t.type === 'heading_open') {
      const level = Number(t.tag.slice(1));
      const inline = tokens[j + 1];
      const label = cleanLabel(inline ? inline.content : '');
      if (!label) continue;
      const id = makeId(label);
      t.attrSet('id', id);
      if (level <= 2) toc.push({ level, label, id });
    }
  }
  const html = md.renderer.render(tokens, md.options, {});
  return `<section class="section page-break">${html}</section>`;
});

const tocHtml = `
<section class="section toc">
  <h1>Contents</h1>
  <nav>
    ${toc
      .map(
        (e) =>
          `<a class="toc-l${e.level}" href="#${e.id}">${md.utils.escapeHtml(e.label)}</a>`
      )
      .join('\n')}
  </nav>
</section>`;

const katexCss = fs.readFileSync(require.resolve('katex/dist/katex.min.css'), 'utf8');

const styles = `
${katexCss}
* { box-sizing: border-box; }
body {
  font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.55;
  color: #1a1a1a;
  margin: 0;
  padding: 0;
}
.section { padding: 0 8px; }
.page-break { page-break-before: always; }
h1 { font-size: 22pt; border-bottom: 2px solid #2b6cb0; padding-bottom: 6px; color: #1a365d; margin-top: 0.2em; }
h2 { font-size: 16pt; border-bottom: 1px solid #cbd5e0; padding-bottom: 4px; color: #2a4365; margin-top: 1.2em; }
h3 { font-size: 13pt; color: #2c5282; margin-top: 1em; }
h4 { font-size: 11.5pt; color: #2d3748; }
p { margin: 0.6em 0; }
a { color: #2b6cb0; text-decoration: none; }
code {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  background: #f0f2f5;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 0.9em;
  color: #b83280;
}
pre {
  background: #1e293b;
  color: #e2e8f0;
  padding: 12px 14px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 0.85em;
  line-height: 1.45;
}
pre code { background: none; color: inherit; padding: 0; }
blockquote {
  border-left: 4px solid #2b6cb0;
  margin: 0.8em 0;
  padding: 0.2em 1em;
  color: #4a5568;
  background: #f7fafc;
}
table { border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 0.92em; }
th, td { border: 1px solid #cbd5e0; padding: 6px 10px; text-align: left; }
th { background: #edf2f7; }
tr:nth-child(even) td { background: #f7fafc; }
ul, ol { margin: 0.5em 0; padding-left: 1.6em; }
li { margin: 0.25em 0; }
img { max-width: 100%; }
hr { border: none; border-top: 1px solid #cbd5e0; margin: 1.5em 0; }
.katex-display { overflow-x: auto; overflow-y: hidden; padding: 4px 0; }
.katex { font-size: 1.05em; }
.mermaid { text-align: center; margin: 1em 0; }
.mermaid svg { max-width: 100%; height: auto; }
.toc nav { display: flex; flex-direction: column; margin-top: 0.5em; }
.toc a { text-decoration: none; color: #2b6cb0; padding: 4px 2px; border-bottom: 1px dotted #d7dee6; }
.toc-l1 { font-weight: 700; font-size: 12.5pt; color: #1a365d; margin-top: 10px; }
.toc-l2 { font-size: 10.5pt; color: #2c5282; padding-left: 22px; }
`;

const mermaidScript = `
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose', flowchart: { htmlLabels: true } });
  window.__renderMermaid = async () => {
    try { await mermaid.run({ querySelector: '.mermaid' }); }
    catch (e) { console.error('mermaid error', e); }
    window.__mermaidDone = true;
  };
</script>`;

const fullHtml = `<!doctype html><html><head><meta charset="utf-8"><style>${styles}</style>${mermaidScript}</head><body>${tocHtml}
${sections.join('\n')}</body></html>`;

(async () => {
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
  const page = await browser.newPage();
  await page.setContent(fullHtml, { waitUntil: 'networkidle0' });
  // Render all Mermaid diagrams and wait for completion.
  await page.evaluate(() => window.__renderMermaid && window.__renderMermaid());
  await page.waitForFunction(() => window.__mermaidDone === true, { timeout: 60000 });
  await new Promise((r) => setTimeout(r, 500));
  await page.pdf({
    path: outPdf,
    format: 'A4',
    printBackground: true,
    margin: { top: '18mm', bottom: '18mm', left: '16mm', right: '16mm' },
    displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="width:100%;font-size:8pt;color:#718096;text-align:center;">Language Models &nbsp;&middot;&nbsp; <span class="pageNumber"></span> / <span class="totalPages"></span></div>',
  });
  await browser.close();
  console.log('PDF written to', outPdf, `(${order.length} files)`);
})().catch((e) => { console.error(e); process.exit(1); });
