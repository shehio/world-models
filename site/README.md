# site/ — the wm-chess Hugo site

Dark monochromatic, JetBrains Mono, teal accent. Inspired by
[projectmontecarlo.com](https://projectmontecarlo.com/)
but redesigned around a single richer home page + four focused
detail pages, with inline SVG diagrams instead of a card-only grid.

Deployed to **https://shehio.github.io/world-models/** via the
`.github/workflows/site.yml` workflow on every push that touches
`site/`.

## Layout

```
site/
├── hugo.toml                 site config + baseURL + canonifyURLs
├── layouts/
│   ├── index.html              long-form home (hero, findings, SVGs, vs-az, links)
│   ├── _default/
│   │   ├── baseof.html         outer html shell
│   │   └── single.html         per-page template (detail pages)
│   └── partials/
│       ├── head.html           meta + font + css
│       ├── nav.html            top nav with 5 section links
│       ├── footer.html         repo link
│       ├── architecture-svg.html   inline SVG: 20×256 ResNet sketch
│       ├── mcts-svg.html           inline SVG: a small MCTS tree
│       └── search-curve-svg.html   inline SVG: Elo vs MCTS sims chart
├── content/
│   ├── _index.md               root stub (content lives in layouts/index.html)
│   ├── findings.md             the three ablations: capacity, search, data
│   ├── method.md               distillation loss, eval anchors, elo-bisect
│   ├── vs-alphazero.md         full comparison + "what's literally identical"
│   ├── next.md                 self-play loop status + roadmap
│   ├── infra.md                EKS + bare-EC2 + multi-region quota
│   └── failures.md             nine concrete bugs and their fixes
└── static/
    └── css/main.css            the entire stylesheet
```

## What changed from v1

- **Consolidated 11 pages → 5 nav items.** `capacity` / `data` /
  `search` merged into `/findings/`; `distillation` / `eval` /
  `elo-bisect` merged into `/method/`. The home page does much more
  work.
- **No more "Experiment A / C / E" labels.** The three ablations are
  named by what they tested: *capacity*, *search*, *data*.
- **Three inline SVGs** — the ResNet architecture sketch, an MCTS
  tree showing visit-count weighting, and a chart of Elo vs MCTS
  sims at eval (the 800 → 4,000 jump from the search ablation).
- **Better typography** — bigger hero, larger headline result box,
  more whitespace per section, sticky nav with backdrop blur, tighter
  card hover state.

## Build locally

```bash
brew install hugo
cd site
hugo server         # http://localhost:1313/world-models/
```

Production build (what the workflow does):

```bash
cd site && hugo --minify
# output -> site/public/
```

## When you buy a domain

Two changes:
1. `site/hugo.toml`: set `baseURL` to the new domain, drop
   `canonifyURLs = true` since you're no longer on a subpath.
2. Drop a `site/static/CNAME` file with the bare hostname (e.g.
   `chesslab.com`). Hugo copies static files verbatim, so this gets
   deployed as `/CNAME` next to `index.html`.
3. Configure A records at the registrar to point at GitHub Pages IPs
   (`185.199.108.153` / 109 / 110 / 111).
