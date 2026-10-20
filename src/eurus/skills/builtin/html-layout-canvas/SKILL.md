---
name: html-layout-canvas
description: "Design polished HTML layouts and canvas-backed visual previews for reports, dashboards, and presentations."
---
# HTML Layout and Canvas Skill

Use this skill when producing HTML previews, visual reports, or presentation companions. Prefer semantic HTML for content and CSS Grid/Flexbox for layout; use Canvas only for purposeful charts, diagrams, decorative accents, or interactive visual elements.

## Design workflow

1. Identify audience, purpose, and the one key takeaway before choosing a layout.
2. Define a compact visual system: one neutral background, one strong text color, and one restrained accent. Avoid generic gradients, excessive cards, and decoration without meaning.
3. Establish clear hierarchy through title, concise subtitle, section labels, and content density.
4. Vary composition to fit the information: timeline for sequence, comparison table for alternatives, chart for quantitative evidence, and open editorial layouts for narrative.
5. Use responsive CSS and readable type sizes; verify at narrow and wide viewports.
6. Canvas must have a clear purpose and a non-canvas text equivalent. Never place essential text only inside a canvas.
7. Escape all user/model-provided text before inserting it into HTML. Do not execute content from source documents as scripts.
8. Keep output self-contained when practical: inline CSS/JS, no remote fonts or third-party scripts unless explicitly requested.

## Presentation companion

When creating PowerPoint, use this skill to shape the HTML preview, not as a substitute for the editable `.pptx` file. Keep the preview and slide content aligned. Use one message per slide, consistent margins, a stable type scale, a restrained palette, and page numbers. Do not force every slide into the same layout; use a title slide, section breaks, comparison layouts, and callouts only when content justifies them.
