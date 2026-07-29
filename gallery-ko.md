---
layout: page
title: 갤러리
lang: ko
translation_key: gallery
permalink: /gallery/
---

{%- assign photos = site.static_files | where_exp: "f", "f.path contains '/assets/pictures/'" | sort: "name" -%}
<div class="gallery-grid">
{%- for photo in photos -%}
  <a class="gallery-item" href="#photo-{{ forloop.index }}">
    <img src="{{ photo.path | relative_url }}" alt="사진 {{ forloop.index }}" loading="lazy">
  </a>
{%- endfor -%}
</div>

{%- for photo in photos -%}
<div class="lightbox" id="photo-{{ forloop.index }}">
  <a href="#_" class="lightbox-backdrop" aria-label="닫기"></a>
  <img src="{{ photo.path | relative_url }}" alt="사진 {{ forloop.index }}">
  <a href="#_" class="lightbox-close" aria-label="닫기">&times;</a>
</div>
{%- endfor -%}
