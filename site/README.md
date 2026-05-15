# site/ — the wm-chess Hugo site

Mirrors the structure / style of [projectmontecarlo.com](https://projectmontecarlo.com/)
([repo](https://github.com/shehio/project-monte-carlo)). Dark
monochromatic, JetBrains Mono, teal accent, card-grid index.

## Layout

```
site/
├── hugo.toml                 site config + base URL
├── layouts/
│   ├── index.html              home page (hero + stats + 3 card grids)
│   ├── _default/
│   │   ├── baseof.html         outer html shell
│   │   └── single.html         per-page template
│   └── partials/
│       ├── head.html           meta + font + css
│       ├── nav.html            top nav with section links
│       └── footer.html         repo link + small print
├── content/
│   ├── _index.md               root stub (content lives in layouts/index.html)
│   ├── baseline.md             d15 20×256 trajectory
│   ├── capacity.md             Experiment A — 40×256
│   ├── data.md                 Experiment C — full 30M positions
│   ├── search.md               Experiment E — sims=4000
│   ├── selfplay.md             distill-then-RL loop
│   ├── distillation.md         multipv soft targets
│   ├── eval.md                 the auto-eval daemon, calibrated Stockfish
│   ├── elo-bisect.md           binary-search UCI_Elo
│   ├── alphazero.md            full comparison table
│   ├── infra.md                EKS + bare-EC2 + multi-region
│   └── failures.md             nine concrete bugs and their fixes
└── static/
    └── css/main.css            the entire stylesheet
```

## Build

```bash
brew install hugo                       # or: snap install hugo --classic
cd site
hugo server -D                          # http://localhost:1313
```

Production build:

```bash
hugo --minify                           # output -> site/public/
```

## Deploy

Two cheap paths:

- **GitHub Pages** — push `site/public/` to a `gh-pages` branch, or
  serve `site/public/` via a CNAME on the root domain. See
  `.github/workflows/site.yml` if it exists.
- **Cloudflare Pages / Netlify** — point at this repo, set the build
  command to `cd site && hugo --minify` and the output directory to
  `site/public/`. Free tier, sub-second builds.

## Content conventions

Match the PMC style: every page has a `title` + `subtitle` in
frontmatter, lowercase H1s with letter-spacing, teal accent on H2 /
links / strong, tables are first-class (Hugo renders pipe tables
natively).

If you add a new content page, add it to `layouts/partials/nav.html`
and to one of the card grids in `layouts/index.html`.
