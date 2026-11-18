"""Check our Stage 1 policy corpus. Standard library only; never edits files."""

from datetime import date
import json
from pathlib import Path
import re


POLICIES = {
    "remote-work": ("RW", "Remote Work Policy", []),
    "expense-reimbursement": ("EXP", "Expense Reimbursement Policy", ["EXP-T01", "EXP-T02"]),
    "pto": ("PTO", "PTO Policy", ["PTO-T01"]),
    "information-security": ("SEC", "Information Security Policy", ["SEC-T01"]),
}
SECTIONS = [
    "Purpose and scope", "Roles and eligibility", "Policy rules", "Procedures",
    "Exceptions and escalation", "Ownership and review",
]
FIELDS = {
    "document_id", "policy_id", "title", "version", "effective_date", "status",
    "source", "supersedes",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_metadata(manifest):
    """Check the five documents, their metadata, and the version relationship."""
    require(isinstance(manifest, dict), "Manifest must be a JSON object")
    require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
            "Expected schema_version 1")
    require(manifest.get("company") == "Northstar Analytics", "Incorrect company name")
    documents = manifest.get("documents")
    require(isinstance(documents, list) and len(documents) == 5, "Expected five documents")
    for document in documents:
        require(isinstance(document, dict) and set(document) == FIELDS, "Incorrect metadata fields")
        require(all(isinstance(document[key], str) and document[key].strip()
                    for key in FIELDS - {"supersedes"}), "Metadata values must be nonempty strings")
        policy, version = document["policy_id"], document["version"]
        identifier = document["document_id"]
        require(policy in POLICIES, f"{identifier}: unknown policy")
        require(document["title"] == POLICIES[policy][1], f"{identifier}: incorrect title")
        require(re.fullmatch(r"\d+\.\d+", version), f"{identifier}: invalid version")
        require(identifier == f"{policy}-v{version}", f"{identifier}: ID/version mismatch")
        require(document["source"] == f"data/policies/{identifier}.md", f"{identifier}: incorrect source")
        effective = date.fromisoformat(document["effective_date"])
        old = identifier == "remote-work-v1.0"
        require(effective.isoformat() == document["effective_date"], f"{identifier}: use YYYY-MM-DD")
        require(document["effective_date"] == ("2025-01-01" if old else "2026-01-01"),
                f"{identifier}: incorrect effective date")
        require(document["status"] == ("superseded" if old else "current"), f"{identifier}: incorrect status")
        previous = "remote-work-v1.0" if identifier == "remote-work-v2.0" else None
        require(document["supersedes"] == previous, f"{identifier}: incorrect supersedes reference")
    expected = {f"{policy}-v1.0" for policy in POLICIES} | {"remote-work-v2.0"}
    require({d["document_id"] for d in documents} == expected, "Missing or duplicate document IDs")
    by_id = {d["document_id"]: d for d in documents}
    old, current = by_id["remote-work-v1.0"], by_id["remote-work-v2.0"]
    require(old["effective_date"] < current["effective_date"], "Remote Work dates are out of order")
    return documents


def cells(line):
    return [cell.strip() for cell in line.strip().removeprefix("|").removesuffix("|").split("|")]


def count_words(text):
    """Count body text, including captions and table cells, without Markdown markers."""
    body = []
    for line in text.splitlines():
        if line.startswith("#") or re.fullmatch(r"[\s|:-]+", line):
            continue
        line = re.sub(r"^\s*(?:[-*+] |\d+\. )", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        body.append(re.sub(r"[*_`~]", "", line).replace("|", " "))
    return len("\n".join(body).split())


def check_structure(document, text):
    """Check body length, heading hierarchy, section IDs, and procedural steps."""
    prefix, title, _ = POLICIES[document["policy_id"]]
    words = count_words(text)
    require(500 <= words <= 800, f"Body has {words} words; expected 500–800")
    lines = text.splitlines()
    require(all(re.fullmatch(r"#{1,3} .+", line) for line in lines if line.lstrip().startswith("#")),
            "Headings must use H1/H2/H3 Markdown syntax")
    headings = [(i, len(m[1]), m[2]) for i, line in enumerate(lines)
                if (m := re.fullmatch(r"(#+) (.+)", line))]
    h1 = f"{title} v{document['version']}"
    require(lines[0] == f"# {h1}" and [h for _, level, h in headings if level == 1] == [h1],
            "Expected one matching H1 title on the first line")
    expected = [f"{prefix}-{i:02}: {name}" for i, name in enumerate(SECTIONS, 1)]
    require([h for _, level, h in headings if level == 2] == expected, "Incorrect H2 sections or order")
    require(any(level == 3 for _, level, _ in headings), "Expected an H3 subsection")
    parent, previous_level, identifiers = None, 0, []
    for position, (index, level, heading) in enumerate(headings):
        require(level <= 3 and level <= previous_level + 1, "Heading levels must follow H1/H2/H3")
        previous_level = level
        if level == 1:
            continue
        match = re.fullmatch(rf"({prefix}-\d{{2}}(?:\.\d+)?): (.+)", heading)
        require(match, f"Invalid section heading: {heading}")
        identifier = match[1]
        identifiers.append(identifier)
        if level == 2:
            parent = identifier
        else:
            require(identifier.rsplit(".", 1)[0] == parent, f"{identifier}: wrong parent section")
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        require(any(line.strip() for line in lines[index + 1:end]), f"{identifier}: empty section")
    require(len(identifiers) == len(set(identifiers)), "Duplicate section IDs")
    start = next(i for i, _, heading in headings if heading == expected[3])
    end = next(i for i, level, _ in headings if i > start and level == 2)
    steps = re.findall(r"(?m)^(\d+)\. ", "\n".join(lines[start + 1:end]))
    require(len(steps) >= 2 and list(map(int, steps)) == list(range(1, len(steps) + 1)),
            "Procedures need sequential numbered steps starting at 1")
    return words


def check_tables(document, text):
    """Check caption IDs, headers, delimiter rows, and rectangular table bodies."""
    prefix, _, expected = POLICIES[document["policy_id"]]
    found = []
    for table in re.finditer(r"(?m)^\|[^\n]*(?:\n\|[^\n]*)*", text):
        lines = table[0].splitlines()
        rows = [cells(line) for line in lines]
        caption = text[:table.start()].rstrip().split("\n")[-1]
        match = re.fullmatch(rf"({prefix}-T\d{{2}}): (\S.*)", caption)
        require(match, "Table needs an ID and caption directly above it")
        found.append(match[1])
        require(all(line.endswith("|") for line in lines), "Table rows need enclosing pipes")
        require(len(rows) >= 5 and len(rows[0]) >= 3, "Table needs three data rows and three columns")
        require(all(len(row) == len(rows[0]) for row in rows), "Unequal table column counts")
        require(all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]), "Invalid table delimiter row")
        require(all(cell for row in rows[:1] + rows[2:] for cell in row), "Empty table header or cell")
    captions = re.findall(rf"(?m)^({prefix}-T\d{{2}}): \S.*$", text)
    require(found == expected and captions == expected, f"Expected table IDs: {expected}")
    return len(found)


def check_conflict(texts):
    """Allow only the intended three-day versus two-day Remote Work difference."""
    normalized = []
    for version, days in (("1.0", "3"), ("2.0", "2")):
        body = " ".join(" ".join(texts[f"remote-work-v{version}"].splitlines()[1:]).split())
        allowances = re.findall(r"up to (\d+) days per calendar week", body)
        require(allowances == [days], f"Remote Work v{version}: expected one {days}-day allowance")
        normalized.append(body.replace(f"up to {days} days per calendar week", "<ALLOWANCE>", 1))
    require(normalized[0] == normalized[1], "Remote Work bodies have additional differences")


def validate_corpus(root):
    """Load metadata, check every document, then compare the Remote Work versions."""
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    documents = check_metadata(manifest)
    actual = {p.relative_to(root).as_posix() for p in (root / "data/policies").rglob("*.md")}
    require(actual == {d["source"] for d in documents}, "Missing or unregistered policy files")
    texts, summaries = {}, []
    for document in documents:
        identifier = document["document_id"]
        text = (root / document["source"]).read_text(encoding="utf-8")
        try:
            words = check_structure(document, text)
            tables = check_tables(document, text)
        except ValueError as error:
            raise ValueError(f"{identifier}: {error}") from error
        texts[identifier] = text
        summaries.append(f"PASS {identifier}: {words} body words, {tables} table(s)")
    check_conflict(texts)
    return summaries


if __name__ == "__main__":
    try:
        for summary in validate_corpus(Path(__file__).resolve().parents[1]):
            print(summary)
        print("PASS metadata, headings, tables, and version relationships")
        print("PASS planted conflict preserved: Remote Work v1.0 = 3 days; v2.0 = 2 days")
    except (OSError, ValueError) as error:
        print(f"FAIL {error}")
        raise SystemExit(1)
