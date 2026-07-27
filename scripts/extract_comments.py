"""Extract PDF annotations (highlights + notes) into a review checklist.
Usage: python3 scripts/extract_comments.py <annotated.pdf>"""
import sys

import fitz

doc = fitz.open(sys.argv[1])
n = 0
for page in doc:
    for a in page.annots() or []:
        n += 1
        kind = a.type[1]
        note = (a.info.get("content") or "").strip()
        quoted = ""
        if kind in ("Highlight", "Underline", "StrikeOut", "Squiggly"):
            quads = a.vertices
            if quads:
                rects = [fitz.Quad(quads[i:i+4]).rect for i in range(0, len(quads), 4)]
                quoted = " ".join(page.get_textbox(r).strip() for r in rects)
        print(f"--- comment {n} (page {page.number + 1}, {kind})")
        if quoted:
            print(f'    on: "{quoted[:200]}"')
        if note:
            print(f"    note: {note}")
print(f"\n{n} annotations total")
