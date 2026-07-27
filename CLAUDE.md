# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Jeffrey Han's personal blog, hosted on GitHub Pages at `braveveigar.github.io`. It's a Jekyll static site built on the **Not Pure Poole** theme (a Poole/Pure-derived theme vendored directly into this repo, not pulled in as a remote theme — `_layouts/`, `_includes/`, `_sass/`, and `assets/` are all local, editable copies of the theme).

## Commands

Setup:
```sh
bundle install
```

Serve locally with drafts enabled (http://localhost:4000):
```sh
./scripts/serve
# equivalent to: bundle exec jekyll serve --draft --trace
```

Create a new draft post (via `jekyll-compose`):
```sh
./scripts/draft "Post Title"
```

Publish a draft (moves it from `_drafts/` into `_posts/` with today's date):
```sh
./scripts/publish _drafts/some-post.md
```

There is no test suite, linter, or build step beyond Jekyll's own build (`bundle exec jekyll build`). GitHub Pages builds and deploys the site automatically from the default branch.

## Structure and conventions

- **Posts** live in `_posts/`, named `YYYY-MM-DD-title.md`. Frontmatter pattern:
  ```yaml
  layout: post
  title: ...
  date: YYYY-MM-DD HH:MM:SS +ZZZZ
  last_modified_at: YYYY-MM-DD HH:MM:SS +ZZZZ   # bump when editing an existing post
  tags: [tag-one, tag-two]
  math: true   # enables MathJax (_includes/mathjax.html) for posts with LaTeX
  toc: true    # enables the right-side table of contents (_includes/toc.html)
  ```
- **Standalone pages** (`about.md`, `categories.md`, `tags.md`, `dates.md`) use `layout: page` or one of the archive layouts (`archive-dates`, `archive-taxonomies` with a `type:` of `categories`/`tags`). Archive/navigation entries are driven by `_data/archive.yml` and `_data/navigation.yml` — adding an archive page also means adding a corresponding entry there.
- **Site-wide config** is in `_config.yml`: title, description, author, cover image, plugins, SEO/social metadata. Social links (footer/sidebar icons) are in `_data/social.yml`, using Font Awesome icon classes.
- **Theme internals**: layouts in `_layouts/`, reusable partials in `_includes/`, styles in `_sass/*.scss` compiled via `assets/styles.scss` (sass compression is on, per `_config.yml`). Color theming is done via CSS custom properties in `_sass/_variables.scss` scoped under `[data-theme="..."]` selectors, not by forking the stylesheets.
- Optional integrations (Google Analytics, Disqus) are wired through `_includes/google-analytics.html` / `_includes/disqus.html` and toggled by uncommenting `google_analytics:` / `disqus:` keys in `_config.yml`.
- `_includes/custom-head.html` is the theme's designated extension point for injecting arbitrary `<head>` content (favicons, extra meta tags) without touching `_includes/head.html`.
