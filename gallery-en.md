---
layout: page
title: Gallery
lang: en
translation_key: gallery
permalink: /en/gallery/
---

{%- assign photos = site.static_files | where_exp: "f", "f.path contains '/assets/pictures/'" | sort: "name" -%}
<div class="gallery-grid">
{%- for photo in photos -%}
  <a class="gallery-item" href="#photo-{{ forloop.index }}">
    <img src="{{ photo.path | relative_url }}" alt="Photo {{ forloop.index }}" loading="lazy">
  </a>
{%- endfor -%}
</div>

{%- for photo in photos -%}
<div class="lightbox" id="photo-{{ forloop.index }}">
  <a href="#_" class="lightbox-backdrop" aria-label="Close"></a>
  <img src="{{ photo.path | relative_url }}" alt="Photo {{ forloop.index }}">
  <a href="#_" class="lightbox-close" aria-label="Close">&times;</a>
</div>
{%- endfor -%}
