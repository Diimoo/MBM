const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');
const fs = require('fs');
const path = require('path');

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

const tex = new TeX({packages: AllPackages});
const svg = new SVG({fontCache: 'local'});
const html = mathjax.document('', {InputJax: tex, OutputJax: svg});

const formulas = {
    gate: '\\mathrm{gate} = \\sigma\\!\\left(W_{\\mathrm{sel}}\\,\\mathbf{s} + W_{\\mathrm{fb}}\\,\\mathbf{f} - \\mathbf{b}\\right)',
    gain: '\\mathrm{gain} = 1 + \\alpha_{\\mathrm{ACh}}\\cdot \\mathrm{ACh} + \\alpha_{\\mathrm{NE}}\\cdot \\mathrm{NE}',
    gated_input: '\\mathbf{x}_{\\mathrm{gated}} = \\mathbf{x} \\odot \\mathrm{gate} \\odot \\mathrm{gain}',
    tdrpe: '\\delta_t = r_t + (1-\\mathrm{done}_t)\\,\\gamma\\,V(s_{t+1}) - V(s_t)',
    novelty: '\\mathrm{novelty} = \\frac{1 - \\max_i \\cos(\\mathbf{z}, \\mathbf{m}_i)}{2}'
};

const assetsDir = path.join(__dirname, 'assets', 'formulas');
if (!fs.existsSync(assetsDir)) {
    fs.mkdirSync(assetsDir, {recursive: true});
}

Object.entries(formulas).forEach(([name, latex]) => {
    const node = html.convert(latex, {display: true});
    const mathSvg = adaptor.innerHTML(node);
    
    const vbMatch = mathSvg.match(/viewBox="([^"]+)"/);
    if (!vbMatch) return;
    const [vx, vy, vw, vh] = vbMatch[1].split(' ').map(parseFloat);
    
    // Add significant padding to ensure background is unmistakable
    const padW = vw * 0.2 + 500;
    const padH = vh * 0.4 + 500;
    const nvx = vx - padW;
    const nvy = vy - padH;
    const nvw = vw + 2 * padW;
    const nvh = vh + 2 * padH;

    const defsMatch = mathSvg.match(/<defs>.*?<\/defs>/s);
    const defs = defsMatch ? defsMatch[0] : '';
    let inner = mathSvg.replace(/<svg[^>]*>/, '').replace(/<\/svg>/, '').replace(/<defs>.*?<\/defs>/s, '');
    
    // Force black everywhere in the math content
    inner = inner.replace(/currentColor/g, '#000000');

    // Assembly with explicit white background and cache-busting suffix
    const finalSvg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${nvw/10}px" height="${nvh/10}px" viewBox="${nvx} ${nvy} ${nvw} ${nvh}">
  <rect x="${nvx}" y="${nvy}" width="${nvw}" height="${nvh}" fill="#ffffff" />
  ${defs}
  <g fill="#000000" stroke="#000000" stroke-width="0">
    ${inner}
  </g>
</svg>`;

    fs.writeFileSync(path.join(assetsDir, `${name}_v5.svg`), finalSvg);
    console.log(`Generated ${name}_v5.svg`);
});
