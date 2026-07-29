---
name: blog-post
description: Jekyll 기반 GitHub 블로그(braveveigar.github.io)에 새 포스팅을 작성하거나 수정할 때 사용. 파일 위치, 파일명 규칙, ko/en 이중 작성, front matter, 코드/이미지/영상/수식 삽입 방식 등 블로그 고유의 작성 규칙을 적용한다. "블로그 글 써줘", "포스트 작성해줘", "블로그 포스팅" 등의 요청에 사용.
---

# 블로그 포스트 작성 규칙

## 언어: 반드시 -ko / -en 두 파일을 쓴다

이 블로그는 모든 포스트를 한국어(ko)와 영어(en) 두 언어로 제공한다. **포스트를 하나 쓴다는 것은 파일 두 개(-ko.md, -en.md)를 쓴다는 뜻이다.** 번역은 직역이 아니라 해당 언어로 자연스럽게 쓴다.

**순서 규칙: -en은 -ko가 사용자 검수를 마친 뒤에 작성한다.** 즉 -ko 파일을 먼저 만들어 보여주고, 사용자가 내용을 확인/수정하고 "됐다", "이제 번역해줘" 등으로 승인하기 전까지는 -en 파일을 만들지 않는다. -ko 작성과 동시에 -en을 같이 만들지 않는다 (초안이 바뀔 수 있는 상태에서 미리 번역하면 다시 해야 하는 일이 생김).

두 파일은:
- 같은 `date`/`last_modified_at`을 쓴다.
- 같은 `translation_key`를 공유한다 (사이드바 언어 탭이 이 값으로 두 언어 버전을 서로 연결한다).
- `lang`만 `ko`/`en`으로 다르다.
- `permalink`은 언어별로 다르다 (아래 참고).

## 파일 위치 및 이름

- 저장 경로: `_posts/`
- 파일명 형식: `YYYY-MM-DD-<key>-<lang>.md`
  - `key`는 `translation_key`와 동일한 값 (소문자, 단어 사이 하이픈)
  - `lang`은 `ko` 또는 `en`
  - 예: `_posts/2026-07-29-cctv-robot-map-reid-ko.md`, `_posts/2026-07-29-cctv-robot-map-reid-en.md`

## Front matter

```yaml
---
layout: post
title: 제목                # en 파일은 자연스러운 영어 제목으로 별도 작성
date: 2026-07-29 00:00:00 +0900
last_modified_at: 2026-07-29 00:00:00 +0900
tags: [tag1, tag2]         # 공용 기술 용어(numpy, cnn, mlp 등)는 두 언어에서 동일하게 유지
math: true                 # 수식($, $$)이 있을 때만
toc: true                  # 목차가 필요할 때만 (보통 항상 켜둠)
lang: ko                   # 또는 en
translation_key: cctv-robot-map-reid   # 두 파일이 동일한 값을 가져야 함
permalink: /cctv-robot-map-reid/       # ko는 접두사 없음, en은 /en/ 접두사
---
```

- ko 파일 permalink: `/<key>/` (접두사 없음, 기본 언어)
- en 파일 permalink: `/en/<key>/`

## 코드 블록

- 일반 마크다운 코드펜스만 사용한다: ` ```language ` ~ ` ``` `
- Jekyll의 `{% highlight %}` 태그는 사용하지 않는다.

## 이미지

- 저장 위치: `/assets/images/` 고정 폴더
- 참조 방식: 마크다운 이미지 문법, 절대경로
  - 예: `![설명 텍스트](/assets/images/파일명.png)`
- 가운데 정렬/크기 조절이 필요하면 raw HTML도 가능: `<img src="/assets/images/파일명.png" width="70%" class="align-center">`

## 영상

- 저장 위치: `/assets/videos/` 고정 폴더
- 파일명에 특수문자(`&` 등)나 공백을 쓰지 않는다 (하이픈으로 대체)
- 삽입 방식: raw HTML5 `<video>` 태그 (마크다운 자체엔 영상 문법이 없지만 kramdown이 raw HTML을 그대로 통과시킴)
  ```html
  <video controls width="100%">
    <source src="/assets/videos/파일명.mp4" type="video/mp4">
  </video>
  ```

## 수식

- MathJax가 인라인 수식으로 `$...$`와 `\(...\)` 둘 다 지원한다 (사이트 커스텀 설정).
- 디스플레이 수식은 `$$ ... $$` 블록을 사용한다.
- `math: true`를 front matter에 켜야 렌더링된다.

## 작성 시 체크리스트

1. -ko를 먼저 만들어 사용자 검수/승인을 받았는가 — 승인 전에는 -en을 만들지 않는다
2. 두 파일이 같은 `translation_key`/`date`를 쓰는가, `lang`만 다른가
3. `permalink`이 언어별 규칙(ko: 접두사 없음, en: `/en/`)에 맞는가
4. `_posts/` 아래에 `YYYY-MM-DD-<key>-<lang>.md` 규칙에 맞는 파일명으로 생성했는가
5. 코드는 전부 마크다운 코드펜스로 작성했는가 (`{% highlight %}` 금지)
6. 이미지가 있다면 `/assets/images/`, 영상이 있다면 `/assets/videos/`에 저장하고 절대경로로 참조했는가
7. 수식이 있다면 `math: true`를 켰는가

## 참고

- 글 톤앤매너: 기존 포스트 스타일(담백한 기술 블로그 톤)을 참고해 자연스럽게 맞춘다.
- 태그: 소문자, 언어별로 자연스럽게 번역하되 numpy/cnn/mlp처럼 두 언어에서 관용적으로 그대로 쓰는 기술 용어는 통일한다 (그래야 `/tags/` 아카이브에서 언어별로 올바르게 묶인다).
- 홈(`/`, `/en/`), Tags(`/tags/`, `/en/tags/`), Dates(`/dates/`, `/en/dates/`)는 각각 `lang`으로 포스트를 필터링해서 보여주므로, 새 포스트를 두 언어로 다 만들어야 양쪽 목록에 정상적으로 노출된다.
