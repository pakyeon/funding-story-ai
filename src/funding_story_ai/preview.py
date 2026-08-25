# ruff: noqa: E501
# HTML/CSS literals intentionally keep individual rules intact for readable previews.
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


_ORDERED_ITEM = re.compile(r"^\s*\d+\.\s+(.+)$")


def _render_inline(value: str) -> str:
    escaped = _escape(value)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", escaped)
    return re.sub(r"==(.+?)==", r"<mark>\1</mark>", escaped)


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _render_markdown_body(value: str) -> str:
    """Render the small, escaped Markdown subset allowed by the generation prompt."""

    lines = value.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if (
            "|" in line
            and index + 1 < len(lines)
            and _is_table_separator(lines[index + 1])
        ):
            headers = _table_cells(line)
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(_table_cells(lines[index]))
                index += 1
            head = "".join(f"<th>{_render_inline(cell)}</th>" for cell in headers)
            body = "".join(
                "<tr>"
                + "".join(f"<td>{_render_inline(cell)}</td>" for cell in row)
                + "</tr>"
                for row in rows
            )
            blocks.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue
        if line.startswith("- "):
            items: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                items.append(lines[index].strip()[2:].strip())
                index += 1
            blocks.append("<ul>" + "".join(f"<li>{_render_inline(item)}</li>" for item in items) + "</ul>")
            continue
        if _ORDERED_ITEM.match(line):
            items = []
            while index < len(lines):
                match = _ORDERED_ITEM.match(lines[index].strip())
                if not match:
                    break
                items.append(match.group(1))
                index += 1
            blocks.append("<ol>" + "".join(f"<li>{_render_inline(item)}</li>" for item in items) + "</ol>")
            continue
        if line.startswith("> "):
            blocks.append(f"<blockquote>{_render_inline(line[2:].strip())}</blockquote>")
            index += 1
            continue
        blocks.append(f"<p>{_render_inline(line)}</p>")
        index += 1
    return "".join(blocks)


def render_story_html(
    *,
    story: dict[str, Any],
    template: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    fallback_image: str | None = None,
) -> str:
    assets = {
        asset["section_id"]: (asset["path"], asset.get("qa_status", "pending"))
        for asset in (manifest or {}).get("assets", [])
        if asset["status"] == "success"
        and asset.get("qa_status") != "fail"
    }
    template_by_id = {section["id"]: section for section in template["layout"]}
    colors = template["style"]["color_palette"]
    section_html: list[str] = []
    for index, section in enumerate(story["sections"], start=1):
        section_id = section["template_section_id"]
        spec = template_by_id[section_id]
        generated_asset = assets.get(section_id)
        image_source = generated_asset[0] if generated_asset else None
        qa_status = generated_asset[1] if generated_asset else None
        image_status = "generated"
        if not image_source and section["image_intent"]["required"] and fallback_image:
            image_source = fallback_image
            image_status = "fallback"
        image = ""
        if image_source:
            image = (
                f'<figure class="visual {image_status}">'
                f'<img src="{_escape(image_source)}" alt="{_escape(section["heading"])}">'
                f'<figcaption>{_escape(section["image_intent"]["purpose"])}'
                f'{" · 사람 검토 대기" if qa_status == "pending" else ""}</figcaption>'
                "</figure>"
            )
        body_html = _render_markdown_body(section["body"])
        source_fields = "".join(
            f"<code>{_escape(source)}</code>" for source in section["source_fields"]
        )
        section_html.append(
            f'<section id="{_escape(section_id)}" class="story-section type-{_escape(section["type"])}">'
            f'<div class="section-kicker">{index:02d} · {_escape(spec["label"])}</div>'
            f'<h2>{_escape(section["heading"])}</h2>{image}'
            f'<div class="body">{body_html}</div>'
            f'<details><summary>출처 필드</summary><div class="source-fields">{source_fields}</div></details>'
            "</section>"
        )
    titles = "".join(f"<li>{_escape(title)}</li>" for title in story["title_candidates"])
    warning_count = len(story["warnings"])
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(story['title_candidates'][0])}</title>
<style>
:root {{--ink:{colors[0]};--accent:{colors[1]};--surface:{colors[2]};}}
*{{box-sizing:border-box}} body{{margin:0;background:#eef1f4;color:var(--ink);font-family:Arial,'Noto Sans KR',sans-serif;line-height:1.7}}
.toolbar{{position:sticky;top:0;z-index:10;display:flex;gap:12px;align-items:center;justify-content:space-between;padding:12px 20px;background:#111827;color:white}}
.toolbar button{{border:0;border-radius:8px;padding:9px 14px;background:var(--accent);color:white;font-weight:700;cursor:pointer}}
.notice{{max-width:860px;margin:24px auto 0;padding:16px 20px;border:1px solid #f59e0b;background:#fffbeb;border-radius:12px}}
.story{{max-width:860px;margin:24px auto 80px;background:white;box-shadow:0 16px 48px #0f172a18}}
.title-panel{{padding:52px 48px;background:var(--ink);color:white}} .title-panel h1{{margin:0 0 16px;font-size:42px;line-height:1.2}}
.story-section{{padding:48px;border-bottom:1px solid #e5e7eb}} .section-kicker{{color:var(--accent);font-weight:800;letter-spacing:.06em}}
h2{{margin:8px 0 24px;font-size:30px;line-height:1.3}} .visual{{margin:0 0 28px}} .visual img{{display:block;width:100%;border-radius:14px}}
.visual.fallback::before{{content:'기준 이미지 대체 사용';display:inline-block;margin-bottom:8px;padding:3px 8px;background:#fef3c7;border-radius:999px;font-size:12px}}
figcaption{{margin-top:8px;color:#64748b;font-size:13px}} .body p{{white-space:pre-wrap}} .body li{{margin:6px 0}}
.body table{{width:100%;border-collapse:collapse;margin:18px 0}} .body th,.body td{{padding:12px;border:1px solid #dbe2ea;text-align:left;vertical-align:top}}
.body th{{background:var(--surface)}} .body blockquote{{margin:18px 0;padding:14px 18px;border-left:4px solid var(--accent);background:var(--surface)}}
.body mark{{background:#ccfbf1;padding:0 .15em}}
details{{margin-top:24px;color:#64748b}} .source-fields{{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}}
code{{padding:3px 7px;background:#f1f5f9;border-radius:6px}} @media(max-width:700px){{.title-panel,.story-section{{padding:28px 22px}}.title-panel h1{{font-size:32px}}}}
</style>
</head>
<body>
<div class="toolbar"><span>Funding Story AI · 검토 필수</span><button onclick="copyStory()">본문 HTML 복사</button></div>
<div class="notice"><strong>AI 생성 초안입니다.</strong> 게시 전 사실·정책·이미지를 사람이 확인해야 합니다. 자동 검증 경고: {warning_count}건.</div>
<main id="story" class="story">
<header class="title-panel"><div>제목 후보</div><h1>{_escape(story['title_candidates'][0])}</h1><ol>{titles}</ol></header>
{''.join(section_html)}
</main>
<script>
async function copyStory(){{await navigator.clipboard.writeText(document.getElementById('story').innerHTML);const b=document.querySelector('button');b.textContent='복사 완료';setTimeout(()=>b.textContent='본문 HTML 복사',1400)}}
</script>
</body>
</html>
"""


def write_story_preview(
    *,
    target: Path,
    story: dict[str, Any],
    template: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    fallback_image: str | None = None,
) -> None:
    target.write_text(
        render_story_html(
            story=story,
            template=template,
            manifest=manifest,
            fallback_image=fallback_image,
        ),
        encoding="utf-8",
    )


def render_editor_fragment(
    *,
    story: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    fallback_image: str | None = None,
) -> str:
    """Render conservative, editable HTML suitable for a Froala-style editor import."""

    assets = {
        asset["section_id"]: asset["path"]
        for asset in (manifest or {}).get("assets", [])
        if asset["status"] == "success" and asset.get("qa_status") != "fail"
    }
    sections = []
    for section in story["sections"]:
        section_id = section["template_section_id"]
        image_source = assets.get(section_id)
        if not image_source and section["image_intent"]["required"]:
            image_source = fallback_image
        image = (
            '<p style="text-align: center;">'
            f'<img src="{_escape(image_source)}" alt="{_escape(section["heading"])}" '
            'style="width: 100%; max-width: 860px;">'
            "</p>"
            if image_source
            else ""
        )
        sections.append(
            f'<div data-story-section="{_escape(section_id)}">'
            f'<h2 style="text-align: center;">{_escape(section["heading"])}</h2>'
            f'{image}{_render_markdown_body(section["body"])}'
            "</div>"
        )
    return "\n".join(sections)


def write_editor_fragment(
    *,
    target: Path,
    story: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    fallback_image: str | None = None,
) -> None:
    target.write_text(
        render_editor_fragment(
            story=story,
            manifest=manifest,
            fallback_image=fallback_image,
        ),
        encoding="utf-8",
    )
