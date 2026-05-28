# Cloudflare Pages deployment

This repo ships an mkdocs site built from `extensions/reports/` (the
research notes published to collaborators). Configuration lives in
[`mkdocs.extensions.yml`](mkdocs.extensions.yml); the landing page is
[`extensions/reports/index.md`](extensions/reports/index.md).

## One-time Cloudflare Pages setup

1. **Sign in** at <https://dash.cloudflare.com/> → **Pages** → **Create
   a project** → **Connect to Git**.
2. **Pick the fork**: `14H034160212/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning`
3. **Production branch**: `main`.
4. **Framework preset**: choose **None** (we provide the command).
5. **Build settings**:
   - **Build command**:

     ```
     pip install --no-cache-dir -r requirements-docs.txt && mkdocs build -f mkdocs.extensions.yml -d _site
     ```
   - **Build output directory**: `_site`
   - **Root directory**: leave blank (project root)
6. **Environment variables**:
   - `PYTHON_VERSION` = `3.11` (Cloudflare's default Python 2 install
     won't have mkdocs; 3.11 works with mkdocs-material 9.x)
7. Click **Save and Deploy**.

The first build takes ~2–3 minutes. Subsequent deploys auto-trigger on
each `git push` to `main`.

## Local preview

```bash
pip install -r requirements-docs.txt
mkdocs serve -f mkdocs.extensions.yml   # http://127.0.0.1:8000/
```

To rebuild the static site for inspection:

```bash
mkdocs build -f mkdocs.extensions.yml -d _site
```

The output directory `_site/` is git-ignored.

## What gets published

The site mirrors the structure of [`extensions/reports/`](extensions/reports/):

- A landing page with the headline numbers (ReClor +0.6 pp, LogiQA −2.0 pp, both seed-robust)
- The T5 fine-tune trajectory (v1 → v4) + the De Morgan rule fix
- v6 contrastive backbone + cross-eval matrix
- ReClor / LogiQA downstream (single + multi-seed)
- Diversity root-cause analysis + four failed corpus-level mitigations (v8, v10, v9, v11, v12)
- xxlarge robustness check

Editing or adding a markdown under `extensions/reports/` then `git push`-ing
re-deploys the site automatically.

## Custom domain (optional)

In the CF Pages project → **Custom domains** → **Set up a custom domain**.
Add a CNAME from your domain to `<your-pages-project>.pages.dev`. SSL
provisioning happens automatically.
