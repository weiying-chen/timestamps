#!/usr/bin/env python3
"""Insert SRT timestamp ranges above matching Chinese blocks in a DOCX.

Usage:
  python3 add_timestamps.py input.docx input.srt output.docx
"""
import argparse
import re
import zipfile
from difflib import SequenceMatcher
from xml.etree import ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})(?:[,:](\d{2,3}))?$")


def normalize(text: str) -> str:
    text = text.lower()
    # keep ASCII letters/digits + CJK Unified Ideographs
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def has_cjk(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def is_block_header(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if re.match(r"^\(.*\)$", s):
        return True
    if s.startswith("/*") or s.endswith("*/"):
        return True
    if re.match(r"^[A-Za-z0-9_]+$", s) and len(s) <= 12:
        return True
    return False


def find_block_start(index: int, paras: list[str]) -> int:
    i = index
    while i > 0 and is_block_header(paras[i - 1]):
        i -= 1
    return i


TSV_TIME_RE = re.compile(r"^(\d{2}:\d{2}:\d{2}:\d{2})\t+(\d{2}:\d{2}:\d{2}:\d{2})$")


def parse_srt(path: str):
    content = open(path, "r", encoding="utf-8").read().strip()
    if not content:
        return [], " --> "

    lines = content.splitlines()
    has_tsv = any(TSV_TIME_RE.match(l.strip()) for l in lines)
    if has_tsv:
        items = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            m = TSV_TIME_RE.match(line)
            if not m:
                i += 1
                continue
            start, end = m.group(1), m.group(2)
            i += 1
            text_lines = []
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line or TSV_TIME_RE.match(next_line):
                    break
                text_lines.append(next_line)
                i += 1
            text = "".join(text_lines)
            items.append(("", start, end, text))
            continue
        return items, "\t"

    blocks = re.split(r"\n\s*\n", content)
    items = []
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if len(lines) >= 3:
            idx = lines[0]
            ts = lines[1]
            text = "".join(lines[2:])
            start, end = [x.strip() for x in ts.split("-->")]
            items.append((idx, start, end, text))
    return items, " --> "


def extract_paragraphs(docx_path: str):
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    paras = []
    for p in root.findall(".//w:p", NS):
        texts = [t.text for t in p.findall(".//w:t", NS) if t.text]
        para_text = "".join(texts)
        paras.append(para_text)
    return root, paras


def best_match(paras_norm, srt_norm, candidate_mask=None):
    best_i = None
    best_score = 0.0
    for i, p in enumerate(paras_norm):
        if candidate_mask is not None and not candidate_mask[i]:
            continue
        if not p:
            continue
        if srt_norm and srt_norm in p:
            score = 1.0
        else:
            score = SequenceMatcher(None, srt_norm, p).ratio()
        if score > best_score:
            best_score = score
            best_i = i
    return best_i, best_score


def overlap_score(srt_norm: str, para_norm: str) -> float:
    if not srt_norm or not para_norm:
        return 0.0
    srt_chars = set(srt_norm)
    if not srt_chars:
        return 0.0
    return len(srt_chars & set(para_norm)) / len(srt_chars)


def best_match_overlap(paras_norm, srt_norm, candidate_mask=None):
    best_i = None
    best_score = 0.0
    for i, p in enumerate(paras_norm):
        if candidate_mask is not None and not candidate_mask[i]:
            continue
        if not p:
            continue
        score = overlap_score(srt_norm, p)
        if score > best_score:
            best_score = score
            best_i = i
    return best_i, best_score


def window_score(window_norm: str, para_norm: str) -> float:
    return overlap_score(window_norm, para_norm)


def parse_ts_range(ts: str):
    start, end = [x.strip() for x in ts.split("-->")]
    return start, end


def to_mmss(ts: str) -> str:
    m = TIME_RE.match(ts.strip())
    if not m:
        return "0000"
    hours = int(m.group(1))
    minutes = int(m.group(2))
    seconds = int(m.group(3))
    total_minutes = hours * 60 + minutes
    return f"{total_minutes:02d}{seconds:02d}"


def make_timestamp_paragraph(start_label: str, end_label: str, separator: str, highlight: bool):
    p = ET.Element(f"{{{W_NS}}}p")

    def append_run(text: str, highlighted: bool) -> None:
        r = ET.SubElement(p, f"{{{W_NS}}}r")
        r_pr = ET.SubElement(r, f"{{{W_NS}}}rPr")
        ET.SubElement(
            r_pr,
            f"{{{W_NS}}}rFonts",
            {
                f"{{{W_NS}}}ascii": "Calibri",
                f"{{{W_NS}}}cs": "Calibri",
                f"{{{W_NS}}}eastAsia": "Calibri",
                f"{{{W_NS}}}hAnsi": "Calibri",
            },
        )
        if highlighted:
            ET.SubElement(r_pr, f"{{{W_NS}}}highlight", {f"{{{W_NS}}}val": "green"})
        ET.SubElement(r_pr, f"{{{W_NS}}}rtl", {f"{{{W_NS}}}val": "0"})
        t = ET.SubElement(r, f"{{{W_NS}}}t")
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = text

    def append_tab_run() -> None:
        r = ET.SubElement(p, f"{{{W_NS}}}r")
        r_pr = ET.SubElement(r, f"{{{W_NS}}}rPr")
        ET.SubElement(
            r_pr,
            f"{{{W_NS}}}rFonts",
            {
                f"{{{W_NS}}}ascii": "Calibri",
                f"{{{W_NS}}}cs": "Calibri",
                f"{{{W_NS}}}eastAsia": "Calibri",
                f"{{{W_NS}}}hAnsi": "Calibri",
            },
        )
        ET.SubElement(r_pr, f"{{{W_NS}}}rtl", {f"{{{W_NS}}}val": "0"})
        ET.SubElement(r, f"{{{W_NS}}}tab")

    append_run(start_label, highlighted=highlight)
    if separator == "tab":
        append_tab_run()
    else:
        append_run(" ", highlighted=False)
    append_run(end_label, highlighted=highlight)
    if separator == "tab":
        append_tab_run()
    return p


def insert_timestamps(root, paras_text, para_to_range, label_style: str, separator: str, highlight: bool):
    body = root.find(".//w:body", NS)
    if body is None:
        return root

    ordered = sorted(para_to_range.items(), key=lambda x: x[0])
    para_to_label = {}
    for idx, (para_i, (start, end)) in enumerate(ordered, start=1):
        if label_style == "numbered":
            start_mmss = to_mmss(start)
            end_mmss = to_mmss(end)
            para_to_label[para_i] = (f"{idx}_{start_mmss}", f"{idx}_{end_mmss}")
        else:
            para_to_label[para_i] = (start, end)

    # rebuild body children with inserted timestamp paragraphs
    new_children = []
    para_index = -1
    for child in list(body):
        if child.tag == f"{{{W_NS}}}p":
            para_index += 1
            if para_index in para_to_label:
                start_label, end_label = para_to_label[para_index]
                new_children.append(
                    make_timestamp_paragraph(
                        start_label,
                        end_label,
                        separator=separator,
                        highlight=highlight,
                    )
                )
        new_children.append(child)

    body[:] = new_children
    return root


def main():
    parser = argparse.ArgumentParser(description="Insert SRT timestamps into DOCX above matching Chinese blocks.")
    parser.add_argument("docx", help="Input DOCX")
    parser.add_argument("srt", help="Input SRT")
    parser.add_argument("output", help="Output DOCX")
    parser.add_argument("--min-score", type=float, default=0.55, help="Minimum match score to accept")
    parser.add_argument("--report", default=None, help="Optional report path (txt)")
    parser.add_argument("--max-window", type=int, default=8, help="Max SRT lines per paragraph window")
    parser.add_argument("--lookahead", type=int, default=6, help="How far ahead (in SRT lines) to search")
    parser.add_argument(
        "--label-style",
        choices=["raw", "numbered"],
        default="raw",
        help="Timestamp label style: raw start/end values or numbered N_MMSS",
    )
    parser.add_argument(
        "--separator",
        choices=["space", "tab"],
        default="space",
        help="Separator between start/end labels (space or tab)",
    )
    parser.add_argument(
        "--no-highlight",
        action="store_true",
        help="Do not apply highlight to inserted timestamp labels",
    )
    args = parser.parse_args()

    srt_items, _ts_sep = parse_srt(args.srt)
    if not srt_items:
        raise SystemExit("SRT is empty or invalid.")

    root, paras = extract_paragraphs(args.docx)
    paras_norm = [normalize(p) for p in paras]
    paras_has_cjk = [has_cjk(p) for p in paras]

    # Precompute SRT fields
    srt_norms = []
    srt_ranges = []
    srt_texts = []
    for _idx, start, end, text in srt_items:
        srt_norms.append(normalize(text))
        srt_ranges.append((start, end))
        srt_texts.append(text)

    para_to_srt_indices = {}
    para_to_scores = {}
    matched_srt = set()

    # Pass 1: assign each SRT line to its best paragraph
    for j, srt_norm in enumerate(srt_norms):
        if not srt_norm:
            continue
        if not has_cjk(srt_texts[j]):
            continue
        best_i = None
        best_score = 0.0
        for i, p_norm in enumerate(paras_norm):
            if not p_norm or not paras_has_cjk[i]:
                continue
            score = overlap_score(srt_norm, p_norm)
            if score > best_score:
                best_score = score
                best_i = i

        min_score = 0.60 if len(srt_norm) > 6 else 0.50
        if best_i is None or best_score < min_score:
            continue

        para_to_srt_indices.setdefault(best_i, []).append(j)
        para_to_scores.setdefault(best_i, []).append((j, best_score))
        matched_srt.add(j)

    # Pass 2: for each paragraph, keep the best contiguous cluster of its assigned lines
    for i, indices in list(para_to_srt_indices.items()):
        if not indices:
            continue
        indices.sort()
        # build clusters
        clusters = []
        cur = [indices[0]]
        for idx in indices[1:]:
            if idx == cur[-1] + 1:
                cur.append(idx)
            else:
                clusters.append(cur)
                cur = [idx]
        clusters.append(cur)

        # choose cluster with most lines (tie-break by avg score)
        best_cluster = None
        best_len = -1
        best_avg = -1.0
        score_map = {j: s for j, s in para_to_scores.get(i, [])}
        for c in clusters:
            avg = sum(score_map.get(j, 0.0) for j in c) / max(len(c), 1)
            if len(c) > best_len or (len(c) == best_len and avg > best_avg):
                best_len = len(c)
                best_avg = avg
                best_cluster = c

        para_to_srt_indices[i] = set(best_cluster) if best_cluster else set()

    # Track unmatched SRT lines for reporting
    unmatched = []
    for j, srt_norm in enumerate(srt_norms):
        if j not in matched_srt:
            ts = " --> ".join(srt_ranges[j])
            unmatched.append((ts, srt_texts[j], 0.0))

    # Build paragraph ranges from matched SRT indices
    para_to_ts = {}
    for para_i, indices in para_to_srt_indices.items():
        if not indices:
            continue
        min_i = min(indices)
        max_i = max(indices)
        start, _ = srt_ranges[min_i]
        _, end = srt_ranges[max_i]
        para_to_ts[para_i] = (start, end)

    # Shift timestamps to the top of each block (e.g., above (NS) or IDs)
    adjusted_para_to_ts = {}
    for para_i, (start, end) in para_to_ts.items():
        insert_i = find_block_start(para_i, paras)
        if insert_i in adjusted_para_to_ts:
            cur_start, cur_end = adjusted_para_to_ts[insert_i]
            adjusted_para_to_ts[insert_i] = (min(cur_start, start), max(cur_end, end))
        else:
            adjusted_para_to_ts[insert_i] = (start, end)

    root = insert_timestamps(
        root,
        paras,
        adjusted_para_to_ts,
        args.label_style,
        args.separator,
        not args.no_highlight,
    )

    # write new docx
    with zipfile.ZipFile(args.docx) as zin:
        with zipfile.ZipFile(args.output, "w") as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                    zout.writestr(item, xml_bytes)
                else:
                    zout.writestr(item, zin.read(item.filename))

    print(f"Wrote {args.output}")
    print(f"Matched paragraphs: {len(para_to_ts)} / {len(paras)}")
    if unmatched:
        print(f"Unmatched SRT lines: {len(unmatched)}")

    if args.report:
        lines = []
        for i, p in enumerate(paras):
            if i in para_to_ts:
                start, end = para_to_ts[i]
                indices = sorted(para_to_srt_indices.get(i, []))
                lines.append(f"[{i:03d}] {start} --> {end} | SRT {indices}\n{p}\n")
        if unmatched:
            lines.append("\nUNMATCHED SRT:\n")
            for ts, text, score in unmatched:
                lines.append(f"{ts} | {score:.2f} | {text}\n")
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("".join(lines))


if __name__ == "__main__":
    main()
