# Deploying the demo

Two artefacts, two hosts, both free. They serve different purposes and it is
worth having both.

| | Static briefing | Interactive dashboard |
|---|---|---|
| Artefact | `outputs/report.html` staged into `_site/` | `dashboard/app.py` |
| Host | GitHub Pages | Streamlit Community Cloud |
| Runtime | none, it is a file | a Python process |
| Availability | always on | sleeps after about a week idle, wakes on visit |
| Cost | free | free |

Link the static page by default. It cannot break, cannot sleep, and loads on a
phone on a conference wifi. Link the dashboard when someone wants to move the
scenario sliders.

## Live

- Static briefing: <https://koutian.is-a.dev/texas-water-resilience-demo/>
- Dashboard: <https://texas-water-resilience-demo.streamlit.app/>

The repository is public, which GitHub Pages requires on the free plan (Pages on
a private repository needs a paid plan). Streamlit Community Cloud will deploy
from a private repository on the free tier, so the dashboard alone would still
have been possible either way.

## GitHub Pages

`.github/workflows/pages.yml` does the whole thing on every push to `main`:
install, run the test suite, run the pipeline, render the report, stage `_site/`,
publish. Nothing is committed; the page is rebuilt from a fixed seed each time.
That is deliberate. `outputs/` is gitignored so a synthetic CSV cannot end up in
git history and be mistaken for an observation later.

One-time setup, after the repository is public:

1. Settings, Pages, **Source: GitHub Actions**. Not "deploy from a branch".
2. Push to `main`, or run the workflow manually from the Actions tab.
3. The URL is `https://ktwu01.github.io/texas-water-resilience-demo/`.

Or from the CLI:

```bash
gh repo edit ktwu01/texas-water-resilience-demo --visibility public
gh api -X POST repos/ktwu01/texas-water-resilience-demo/pages \
  -f 'build_type=workflow'
gh workflow run pages.yml
```

To see exactly what will be published before pushing:

```bash
make site
open _site/index.html
```

The staged site is the report, plus every source CSV under `_site/data/`. A
reader who does not believe a table can open the file it came from. That is the
point of publishing them.

## Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io> with the GitHub account that owns the
   repository, and grant it repository access.
2. New app, pick this repository, branch `main`, **main file path
   `dashboard/app.py`**.
3. Deploy. First boot takes a few minutes: Streamlit Cloud installs the root
   `requirements.txt`, which is why `streamlit` is uncommented there rather
   than left as a `pyproject.toml` extra.

After changing anything the app reads, use **Reboot app** in the Streamlit Cloud
console. It does not redeploy on every push by itself, and a throttled or sleeping
container keeps serving the code it booted with.

`outputs/` is gitignored, so the hosted app starts with nothing to read.
`bootstrap_outputs()` in `dashboard/app.py` downloads the CSVs that CI already
published next to the static briefing (`twr/published.py`), which takes seconds
and no CPU. The dashboard and the published page then show byte-identical
numbers, because they are the same artefacts from the same run.

**Do not make the hosted app compute the pipeline.** The first version ran
`run_pipeline(["--scenario"])` on first request: leave-one-basin-out
cross-validation plus a scenario sweep, roughly two and a half minutes of
scikit-learn. Streamlit Community Cloud CPU-throttled the app for it, which is
the correct response from their side. If the download fails the app falls back to
a local `--fast` run, not `--scenario`, and says so in the sidebar; the
cross-validation and scenario tabs then show their own empty states.

A consequence worth knowing: a partial download is a normal state, so every tab
has to tolerate a missing CSV. `has_columns()` exists for that, and the
dashboard shows an empty state per panel rather than failing the page.

The app sleeps after about a week idle and re-runs that bootstrap on wake, since
the container filesystem does not survive. Waking is a download, not a
computation.

### Hugging Face Spaces, as an alternative

Also free, and it does not sleep as aggressively. Create a Space with the
Streamlit SDK, point it at this repository, and set the entrypoint to
`dashboard/app.py`. The same `requirements.txt` and the same bootstrap apply.

## What not to bother with

Render and Fly free tiers will host the dashboard but cold-start the container on
every visit, which is worse than Streamlit Cloud's sleep behaviour for no gain.
Vercel and Netlify are excellent for the static page and add nothing over Pages
here, since there is no custom domain and no preview-per-PR requirement.

## Before sharing a link

Everything published is synthetic, and the page says so in a banner at the top.
Keep that banner. The link will be read by people who did not clone the
repository and will not read [`LIMITATIONS.md`](LIMITATIONS.md) unless the page
puts it in front of them.
