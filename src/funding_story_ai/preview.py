# ruff: noqa: E501
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Literal


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
    lines = value.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if "|" in line and index + 1 < len(lines) and _is_table_separator(lines[index + 1]):
            headers = _table_cells(line)
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(_table_cells(lines[index]))
                index += 1
            head = "".join(f"<th>{_render_inline(cell)}</th>" for cell in headers)
            body = "".join(
                "<tr>" + "".join(f"<td>{_render_inline(cell)}</td>" for cell in row) + "</tr>"
                for row in rows
            )
            blocks.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue
        if line.startswith("- "):
            items: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                items.append(lines[index].strip()[2:].strip())
                index += 1
            blocks.append(
                "<ul>" + "".join(f"<li>{_render_inline(item)}</li>" for item in items) + "</ul>"
            )
            continue
        if _ORDERED_ITEM.match(line):
            items = []
            while index < len(lines):
                match = _ORDERED_ITEM.match(lines[index].strip())
                if not match:
                    break
                items.append(match.group(1))
                index += 1
            blocks.append(
                "<ol>" + "".join(f"<li>{_render_inline(item)}</li>" for item in items) + "</ol>"
            )
            continue
        if line.startswith("> "):
            blocks.append(f"<blockquote>{_render_inline(line[2:].strip())}</blockquote>")
            index += 1
            continue
        blocks.append(f"<p>{_render_inline(line)}</p>")
        index += 1
    return "".join(blocks)


def _media_html(
    *,
    section_id: str,
    media_plan: dict[str, Any],
    manifest: dict[str, Any],
    mode: Literal["draft", "publishable"],
) -> str:
    asset_by_slot = {
        asset["slot_id"]: asset
        for asset in manifest["assets"]
        if asset["status"] == "success" and asset["qa_status"] != "fail"
    }
    blocks: list[str] = []
    for slot in media_plan["slots"]:
        if slot["section_id"] != section_id:
            continue
        asset = asset_by_slot.get(slot["slot_id"])
        if asset and (mode == "draft" or asset["qa_status"] == "pass"):
            blocks.append(
                f'<figure data-media-slot="{_escape(slot["slot_id"])}">'
                f'<img src="{_escape(asset["path"])}" alt="{_escape(slot["scene"]["summary"])}">'
                "</figure>"
            )
        elif mode == "draft":
            blocks.append(
                f'<div class="media-placeholder" data-media-slot="{_escape(slot["slot_id"])}">'
                f"<strong>이미지 자리</strong><p>{_escape(slot['scene']['visual_direction'])}</p>"
                "</div>"
            )
    if mode == "draft":
        for placeholder in media_plan["placeholders"]:
            if placeholder["section_id"] != section_id:
                continue
            blocks.append(
                '<aside class="content-placeholder">'
                f"<strong>{_escape(placeholder['title'])}</strong>"
                f"<p>{_escape(placeholder['description'])}</p>"
                f"<small>{_escape(placeholder['input_format'])}</small>"
                "</aside>"
            )
    return "".join(blocks)


def content_placeholder_sections(
    *, brief: dict[str, Any], template: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    unknown_fields = {item["field"] for item in brief.get("unknowns", [])}
    layout_ids = {section["id"] for section in template["layout"]}
    result: dict[str, dict[str, Any]] = {}
    for section_id, guide in template.get("content_placeholders", {}).items():
        fields = set(guide["fields"])
        if guide.get("trigger", "any_unknown") == "all_unknown":
            triggered = fields.issubset(unknown_fields)
        else:
            triggered = bool(fields.intersection(unknown_fields))
        if triggered and section_id in layout_ids:
            result[section_id] = guide
    return result


def _content_placeholder_html(guide: dict[str, Any]) -> str:
    return (
        '<aside class="content-placeholder" data-content-status="example">'
        '<span class="example-badge">작성 예시 · 실제 정보 아님</span>'
        f"<strong>{_escape(guide['title'])}</strong>"
        '<p class="example-notice">입력된 정보가 없어 작성 형식만 보여드립니다. '
        "게시 전에 실제 메이커 정보로 교체해야 합니다.</p>"
        f'<p class="placeholder-example">{_escape(guide["example"])}</p>'
        f"<small>권장 입력 형식: {_escape(guide['input_format'])}</small>"
        "</aside>"
    )


def can_render_publishable(
    *,
    media_plan: dict[str, Any],
    manifest: dict[str, Any],
    story: dict[str, Any] | None = None,
    brief: dict[str, Any] | None = None,
    template: dict[str, Any] | None = None,
) -> bool:
    assets = manifest["assets"]
    content_placeholders = (
        content_placeholder_sections(brief=brief, template=template)
        if brief is not None and template is not None
        else {}
    )
    return bool(
        media_plan["publishable"]
        and not media_plan["placeholders"]
        and not content_placeholders
        and manifest["requested"] == manifest["succeeded"]
        and all(asset["qa_status"] == "pass" for asset in assets)
        and (story is None or not story.get("warnings"))
    )


def render_funding_story_html(
    *,
    story: dict[str, Any],
    template: dict[str, Any],
    media_plan: dict[str, Any],
    manifest: dict[str, Any],
    mode: Literal["draft", "publishable"] = "draft",
    brief: dict[str, Any] | None = None,
) -> str:
    if mode == "publishable" and not can_render_publishable(
        media_plan=media_plan,
        manifest=manifest,
        story=story,
        brief=brief,
        template=template,
    ):
        raise ValueError(
            "Publishable HTML requires complete facts, assets, and passed image review"
        )
    colors = template["style"]["color_palette"]
    placeholder_sections = (
        content_placeholder_sections(brief=brief, template=template) if brief is not None else {}
    )
    sections: list[str] = []
    for section in story["sections"]:
        section_id = section["template_section_id"]
        media = _media_html(
            section_id=section_id,
            media_plan=media_plan,
            manifest=manifest,
            mode=mode,
        )
        placeholder = placeholder_sections.get(section_id)
        copy = f'<div class="story-copy">{_render_markdown_body(section["body"])}</div>'
        if mode == "draft" and placeholder is not None:
            copy += _content_placeholder_html(placeholder)
        sections.append(
            f'<section data-story-section="{_escape(section_id)}">'
            f"<h2>{_escape(section['heading'])}</h2>"
            f"{media}{copy}"
            "</section>"
        )
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(story["title_candidates"][0])}</title>
<style>
:root{{--story-ink:{colors[0]};--story-accent:{colors[1]};--story-surface:{colors[2]}}}
*{{box-sizing:border-box}}body{{margin:0;background:#fff;color:var(--story-ink);font-family:Arial,'Noto Sans KR',sans-serif;line-height:1.72}}
.funding-story{{width:100%;max-width:740px;margin:0 auto;background:#fff;overflow:hidden}}
.story-hero{{padding:56px 36px;background:var(--story-ink);color:#fff;text-align:center}}
.story-hero h1{{margin:0;font-size:38px;line-height:1.25}}
section{{padding:44px 36px;border-bottom:1px solid #e9edf1}}h2{{margin:0 0 22px;font-size:28px;line-height:1.35;text-align:center}}
figure{{margin:0 0 26px}}img{{display:block;width:100%;height:auto}}.story-copy p{{white-space:pre-wrap}}
.story-copy table{{width:100%;border-collapse:collapse;margin:18px 0}}.story-copy th,.story-copy td{{padding:11px;border:1px solid #dbe2ea;text-align:left}}
.story-copy th{{background:var(--story-surface)}}blockquote{{margin:18px 0;padding:14px 18px;border-left:4px solid var(--story-accent);background:var(--story-surface)}}
mark{{background:#fff2a8}}.media-placeholder,.content-placeholder{{margin:0 0 24px;padding:24px;border:2px dashed #aab4c0;background:#f8fafc;text-align:center}}
.media-placeholder p,.content-placeholder p{{margin:8px 0}}.content-placeholder small{{color:#586474}}
.example-badge{{display:inline-block;margin:0 0 12px;padding:4px 10px;border-radius:999px;background:#fff3cd;color:#7a4b00;font-size:13px;font-weight:700}}
.content-placeholder strong{{display:block;font-size:19px}}.example-notice{{color:#475569}}.placeholder-example{{padding:14px;background:#fff;border-radius:10px;color:#334155}}
@media(max-width:768px){{.funding-story{{max-width:100%}}.story-hero,section{{padding:32px 20px}}.story-hero h1{{font-size:31px}}h2{{font-size:25px}}}}
</style>
</head>
<body><main class="funding-story" data-render-mode="{mode}">
<header class="story-hero"><h1>{_escape(story["title_candidates"][0])}</h1></header>
{"".join(sections)}
</main></body>
</html>
"""


def write_funding_story_html(
    *,
    target: Path,
    story: dict[str, Any],
    template: dict[str, Any],
    media_plan: dict[str, Any],
    manifest: dict[str, Any],
    mode: Literal["draft", "publishable"] = "draft",
    brief: dict[str, Any] | None = None,
) -> None:
    target.write_text(
        render_funding_story_html(
            story=story,
            template=template,
            media_plan=media_plan,
            manifest=manifest,
            mode=mode,
            brief=brief,
        ),
        encoding="utf-8",
    )
