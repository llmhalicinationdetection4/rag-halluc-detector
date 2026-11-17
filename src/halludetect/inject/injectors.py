"""Phase 2: controlled error injection.

Take a response that RAGTruth annotators judged faithful, change exactly one thing in it, and keep
the context the same. The edited response is then known to contain one specific kind of error, so we
can measure which kinds of error each verifier misses.

Five error types (all rule-based and seeded, so the same seed gives the same test set):
  number_swap          a number that appears in the context is replaced by a different number
  entity_swap          a name that appears in the context is replaced by another name from the context
  negation_flip        a verb is negated ("is" -> "is not") or a negation is removed
  date_change          a year or month that appears in the context is changed
  unsupported_addition a generic, unverifiable sentence is appended

Limits to state in the report: edits are synthetic, entity_swap relies on spaCy's small English NER model (with a
rough regex fallback used only in tests), and the appended sentences for unsupported_addition come from a fixed pool of topic-independent claims.
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

ERROR_TYPES = ("number_swap", "entity_swap", "negation_flip", "date_change", "unsupported_addition")

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]

_STOP = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "he", "she", "they", "we", "i", "you",
    "in", "on", "at", "of", "for", "to", "by", "with", "from", "however", "but", "and", "or", "also", "as",
    "after", "before", "while", "when", "where", "which", "who", "what", "there", "their", "his", "her", "our",
    "your", "if", "then", "so", "although", "according", "overall", "additionally", "furthermore", "moreover",
    "based", "some", "many", "most", "other", "one", "two", "three", "here", "yes", "no", "not", "unable",
    "please", "note", "review", "reviews", "customers", "customer", "despite", "since", "because", "both",
    "each", "every", "all", "any", "such", "these", "while", "during", "following", "several", "another",
}

UNSUPPORTED_SENTENCES = [
    "This was later confirmed by an independent audit.",
    "According to a recent survey, more than 70% of people agree.",
    "Experts have described this as a turning point.",
    "Several officials declined to comment on the matter.",
    "The same pattern was observed in earlier years.",
    "A spokesperson said further details will be released next month.",
    "Critics have long argued that this approach is flawed.",
    "Similar cases have been reported in neighbouring regions.",
    "The figures were revised downward after a second review.",
    "Local residents have mixed feelings about the development.",
    "The trend is expected to continue for at least a decade.",
    "A follow-up study is currently under way.",
]


@dataclass
class Injection:
    example_id: str
    task_type: str
    error_type: str
    context: str
    original: str
    edited: str
    description: str
    before: str
    after: str

    def to_dict(self) -> dict:
        return asdict(self)


def _window(text: str, start: int, end: int, pad: int = 40) -> str:
    s, e = max(0, start - pad), min(len(text), end + pad)
    snippet = text[s:e].replace("\n", " ")
    return ("..." if s > 0 else "") + snippet + ("..." if e < len(text) else "")


_PREAMBLE_RE = re.compile(r"^\s*(?:here\b|sure\b|certainly\b|of course\b|summary\b|overview\b|answer\b|output\b|response\b)[^\n]{0,150}?:", re.I)


def _blocked_spans(answer: str) -> list[tuple[int, int]]:
    """Lead-in text such as 'Here is a summary in 80 words:' or a line ending in ':' is not a claim, so we never edit it."""
    spans, pos = [], 0
    for line in answer.split("\n"):
        end = pos + len(line)
        m = _PREAMBLE_RE.match(line)
        if m:
            spans.append((pos, pos + m.end()))
        elif line.rstrip().endswith(":"):
            spans.append((pos, end))
        pos = end + 1
    return spans


def _is_blocked(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < b_end and end > b_start for b_start, b_end in spans)


# ------------------------------------------------------------------ numbers
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+(?:\.\d+)?%?")


def _number_candidates(text: str):
    """Yield (start, end, token) for plain numbers: not years, times, dates, ranges or things like 3rd / 5pm."""
    for m in _NUM_RE.finditer(text):
        s, e = m.span()
        prev = text[s - 1] if s > 0 else ""
        nxt = text[e] if e < len(text) else ""
        after = text[e + 1:e + 2]
        if prev.isalnum() or prev in ".:/,-":
            continue
        if nxt.isalnum() or nxt in ":/" or (nxt == "-" and after.isdigit()) or (nxt in ".," and after.isdigit()):
            continue
        core = m.group().rstrip("%")
        if "," not in core and "." not in core and len(core) == 4 and 1900 <= int(core) <= 2099:
            continue  # years belong to date_change
        if float(core.replace(",", "")) == 0:
            continue
        if re.match(r"\s*(?:to|-|\u2013|and|or)\s*\d", text[e:]) or re.search(r"\d\s*(?:to|-|\u2013|and|or)\s*$", text[:s]):
            continue  # part of a range such as "6 to 8"
        before = text[:s].rstrip(" \t")
        if "," not in core and "." not in core and int(core) <= 30 and re.match(r"[.)]\s", text[e:]) \
                and (before == "" or before[-1] in "\n.!?:;"):
            continue  # a list marker such as "3. Take another slide"
        if re.search(r"(?:passages?|steps?|sections?|chapters?|pages?|figures?|tables?|lines?|items?|no\.?|#)\s*$", text[:s], re.I):
            continue  # a reference label such as "Passage 3"
        yield s, e, m.group()


def _context_has_number(core: str, context: str) -> bool:
    return re.search(r"(?<![\d.,])" + re.escape(core) + r"(?!\d|,\d|\.\d)", context) is not None


def _format_like(core: str, new: float) -> str:
    decimals = len(core.split(".")[1]) if "." in core else 0
    if decimals:
        return f"{new:,.{decimals}f}" if "," in core else f"{new:.{decimals}f}"
    return f"{int(round(new)):,}" if "," in core else str(int(round(new)))


def _new_number(core: str, context: str, rng: random.Random) -> str | None:
    value = float(core.replace(",", ""))
    decimals = len(core.split(".")[1]) if "." in core else 0
    for _ in range(30):
        if decimals == 0 and value < 10:
            new = value + rng.choice([1, 2, 3, 5])
        else:
            new = value * rng.choice([0.5, 0.75, 1.25, 1.5, 2.0, 3.0])
        cand = _format_like(core, new)
        if cand != core and not _context_has_number(cand, context):
            return cand
    return None


def inject_number_swap(context: str, answer: str, rng: random.Random):
    blocked = _blocked_spans(answer)
    cands = [(s, e, tok) for s, e, tok in _number_candidates(answer)
             if _context_has_number(tok.rstrip("%"), context) and not _is_blocked(blocked, s, e)]
    rng.shuffle(cands)
    for s, e, tok in cands:
        new = _new_number(tok.rstrip("%"), context, rng)
        if new is None:
            continue
        new_tok = new + ("%" if tok.endswith("%") else "")
        edited = answer[:s] + new_tok + answer[e:]
        return edited, f"{tok} -> {new_tok}", _window(answer, s, e), _window(edited, s, s + len(new_tok))
    return None


# ------------------------------------------------------------------ entities
_ENT_RE = re.compile(r"\b[A-Z][a-z]+(?:[ ]+[A-Z][a-z]+)*\b")
_WORD_RE = re.compile(r"[A-Z][a-z]+")


def _at_sentence_start(text: str, start: int) -> bool:
    prefix = text[:start].rstrip(" \"'\u201c\u2018(*-")
    return prefix == "" or prefix[-1] in ".!?:\n"


def _entities(text: str):
    """Capitalised phrases as (start, end, phrase, n_words); a rough stand-in for named entities."""
    out = []
    for m in _ENT_RE.finditer(text):
        parts = [(w.start() + m.start(), w.end() + m.start(), w.group()) for w in _WORD_RE.finditer(m.group())]
        while parts and parts[0][2].lower() in _STOP:
            parts.pop(0)
        while parts and parts[-1][2].lower() in _STOP:
            parts.pop()
        if not parts:
            continue
        s, e = parts[0][0], parts[-1][1]
        phrase = text[s:e]
        if len(parts) == 1 and _at_sentence_start(text, s):
            continue  # a lone capitalised word at the start of a sentence is probably not a name
        if phrase in MONTHS or len(phrase) < 3:
            continue
        if len(parts) == 1 and len(phrase) < 4:
            continue  # short single words ("Ste", "Dr") are usually fragments of addresses or titles
        out.append((s, e, phrase, len(parts)))
    return out


_NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART"}
_NLP = None
_NLP_TRIED = False


def _get_nlp():
    """spaCy's small English model, or None when it is not installed."""
    global _NLP, _NLP_TRIED
    if not _NLP_TRIED:
        _NLP_TRIED = True
        try:
            import spacy

            _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            _NLP.max_length = 3_000_000
        except Exception:  # noqa: BLE001  (not installed, no model, version clash ...)
            _NLP = None
    return _NLP


def ner_available() -> bool:
    return _get_nlp() is not None


@lru_cache(maxsize=4096)
def _ner_spans(text: str):
    out = []
    for ent in _get_nlp()(text).ents:
        if ent.label_ not in _NER_LABELS:
            continue
        s, e = ent.start_char, ent.end_char
        if text[s:e].lower().startswith(("the ", "a ", "an ")):
            s += text[s:e].index(" ") + 1
        phrase = text[s:e]
        if len(phrase) < 3 or "\n" in phrase or phrase != phrase.strip():
            continue
        out.append((s, e, phrase, ent.label_))
    return tuple(out)


def _entity_spans(text: str):
    """(start, end, phrase, label). With spaCy the label is the entity type (PERSON, GPE, ...), so a swap replaces a
    city with a city. Without spaCy (tests only) the label is just the number of words."""
    if _get_nlp() is not None:
        return list(_ner_spans(text))
    return [(s, e, p, str(n)) for s, e, p, n in _entities(text)]


def inject_entity_swap(context: str, answer: str, rng: random.Random):
    blocked = _blocked_spans(answer)
    by_label: dict[str, set[str]] = defaultdict(set)
    for _, _, phrase, label in _entity_spans(context):
        by_label[label].add(phrase)
    cands = [(s, e, p, label) for s, e, p, label in _entity_spans(answer)
             if not _is_blocked(blocked, s, e) and re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", context)]
    rng.shuffle(cands)
    for s, e, phrase, label in cands:
        pool = sorted(x for x in by_label.get(label, ())
                      if x != phrase and x not in answer and phrase not in x and x not in phrase)
        if not pool:
            continue
        new = rng.choice(pool)
        edited = answer[:s] + new + answer[e:]
        return edited, f"{phrase} -> {new}", _window(answer, s, e), _window(edited, s, s + len(new))
    return None


# ------------------------------------------------------------------ negation
_NEG_ADD = re.compile(
    r"\b(is|are|was|were|can|will|would|should|could|does|did)\b(?![\u2019']t)(?!\s+not\b)"
    r"|\b(has|have|had)\b(?=\s+been\b)"
)
_NEG_REMOVE = re.compile(r"\b(is|are|was|were|has|have|had|does|did|will|would|should|could)\s+not\b")
_CANNOT = re.compile(r"\bcannot\b")
_CONTRACTION = re.compile(r"\b(isn|aren|wasn|weren|hasn|haven|hadn|doesn|didn|won|wouldn|shouldn|couldn|can)[\u2019']t\b")
_CONTRACTION_MAP = {"isn": "is", "aren": "are", "wasn": "was", "weren": "were", "hasn": "has", "haven": "have",
                    "hadn": "had", "doesn": "does", "didn": "did", "won": "will", "wouldn": "would",
                    "shouldn": "should", "couldn": "could", "can": "can"}


def inject_negation_flip(context: str, answer: str, rng: random.Random):
    cands = []
    for m in _NEG_ADD.finditer(answer):
        word = m.group(0)
        cands.append((m.start(), m.end(), "cannot" if word == "can" else f"{word} not"))
    for m in _NEG_REMOVE.finditer(answer):
        cands.append((m.start(), m.end(), m.group(1)))
    for m in _CANNOT.finditer(answer):
        cands.append((m.start(), m.end(), "can"))
    for m in _CONTRACTION.finditer(answer):
        cands.append((m.start(), m.end(), _CONTRACTION_MAP[m.group(1)]))
    blocked = _blocked_spans(answer)
    cands = [c for c in cands if not _is_blocked(blocked, c[0], c[1])]
    if not cands:
        return None
    s, e, repl = rng.choice(sorted(cands))
    edited = answer[:s] + repl + answer[e:]
    return edited, f"{answer[s:e]} -> {repl}", _window(answer, s, e), _window(edited, s, s + len(repl))


# ------------------------------------------------------------------ dates
_YEAR_RE = re.compile(r"(?<![\d-])(?:19|20)\d{2}(?![\d-])")
_MONTH_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\b")


def inject_date_change(context: str, answer: str, rng: random.Random):
    cands = []
    blocked = _blocked_spans(answer)
    for m in _YEAR_RE.finditer(answer):
        following = answer[m.end():].lstrip(" ")[:1]
        if following.isupper():
            continue  # "1940 Cliff Dr" is a street number, not a year
        if re.search(rf"(?<!\d){m.group()}(?!\d)", context):
            cands.append((m.start(), m.end(), "year"))
    for m in _MONTH_RE.finditer(answer):
        if re.search(rf"\b{m.group()}\b", context):
            cands.append((m.start(), m.end(), "month"))
    cands = [c for c in cands if not _is_blocked(blocked, c[0], c[1])]
    rng.shuffle(cands)
    for s, e, kind in cands:
        old = answer[s:e]
        if kind == "year":
            options = [str(int(old) + d) for d in (-10, -5, -3, -2, -1, 1, 2, 3, 5, 10)]
            options = [o for o in options if 1900 <= int(o) <= 2099 and o not in context]
        else:
            options = [m for m in MONTHS if m != old and not re.search(rf"\b{m}\b", context)]
        if not options:
            continue
        new = rng.choice(options)
        edited = answer[:s] + new + answer[e:]
        return edited, f"{old} -> {new}", _window(answer, s, e), _window(edited, s, s + len(new))
    return None


# ------------------------------------------------------------------ unsupported addition
def inject_unsupported_addition(context: str, answer: str, rng: random.Random):
    if not answer.strip():
        return None
    pool = [s for s in UNSUPPORTED_SENTENCES if s not in answer]
    if not pool:
        return None
    sentence = rng.choice(pool)
    base = answer.rstrip()
    edited = base + " " + sentence
    return edited, sentence, _window(base, len(base) - 1, len(base), pad=60), _window(edited, len(base), len(edited), pad=60)


INJECTORS = {
    "number_swap": inject_number_swap,
    "entity_swap": inject_entity_swap,
    "negation_flip": inject_negation_flip,
    "date_change": inject_date_change,
    "unsupported_addition": inject_unsupported_addition,
}


# ------------------------------------------------------------------ building the test set
def build_injections(examples, max_per_type: int = 200, seed: int = 42, error_types=ERROR_TYPES):
    """examples must already be faithful. Returns (injections, stats) where stats rows say how often each
    injector could be applied. The result does not depend on the order of `examples`."""
    by_task: dict[str, list] = defaultdict(list)
    for ex in examples:
        by_task[ex.task_type].append(ex)
    injections: list[Injection] = []
    stats: list[dict] = []
    for task in sorted(by_task):
        exs = sorted(by_task[task], key=lambda e: e.id)
        for et in error_types:
            rng = random.Random(f"{seed}-{task}-{et}")
            order = exs[:]
            rng.shuffle(order)
            made = tried = 0
            for ex in order:
                tried += 1
                out = INJECTORS[et](ex.context, ex.answer, rng)
                if out is None:
                    continue
                edited, desc, before, after = out
                if edited == ex.answer:
                    continue
                injections.append(Injection(ex.id, task, et, ex.context, ex.answer, edited, desc, before, after))
                made += 1
                if made >= max_per_type:
                    break
            stats.append({"task_type": task, "error_type": et, "n_faithful": len(exs), "n_tried": tried,
                          "n_made": made, "applicable_rate": round(made / tried, 3) if tried else 0.0})
    return injections, stats


def write_injections(path: str | Path, injections) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for inj in injections:
            f.write(json.dumps(inj.to_dict(), ensure_ascii=False) + "\n")


def read_injections(path: str | Path) -> list[Injection]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m halludetect inject` first.")
    with open(path, "r", encoding="utf-8") as f:
        return [Injection(**json.loads(line)) for line in f if line.strip()]


def write_samples_csv(path: str | Path, injections, per_group: int = 3) -> None:
    """A few edits per (task, error type), so a person can check by eye that the edits look right."""
    seen: dict[tuple[str, str], int] = defaultdict(int)
    rows = []
    for inj in injections:
        key = (inj.task_type, inj.error_type)
        if seen[key] < per_group:
            seen[key] += 1
            rows.append({"task_type": inj.task_type, "error_type": inj.error_type, "example_id": inj.example_id,
                         "description": inj.description, "before": inj.before, "after": inj.after})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "error_type", "example_id", "description", "before", "after"])
        writer.writeheader()
        writer.writerows(rows)


def run_injection(cfg, max_per_type: int = 200, out_path=None, split: str = "test", examples=None,
                  allow_regex_entities: bool = False) -> dict:
    """Build and save the injected test set from faithful RAGTruth responses of the given split."""
    import pandas as pd

    from halludetect.data.ragtruth import load_ragtruth

    if not ner_available():
        if not allow_regex_entities:
            raise RuntimeError(
                "spaCy's English model is not installed, and entity_swap needs it to swap a name for a name of "
                "the same kind. On Kaggle run:  !python -m spacy download en_core_web_sm  "
                "(if `import spacy` fails, run  !pip install -q spacy  first). "
                "Or pass --allow-regex-entities to use a rough fallback (not recommended for results)."
            )
        print("[warn] spaCy model not found: entity_swap uses a rough regex fallback")

    out_path = Path(out_path or Path(cfg.eval.results_dir) / "phase2_injected.jsonl")
    if examples is None:
        examples = load_ragtruth(cfg.data.ragtruth_dir, split, None, cfg.seed)
    faithful = [e for e in examples
                if not e.label_hallucinated and e.meta.get("quality") in (None, "good")]
    injections, stats = build_injections(faithful, max_per_type=max_per_type, seed=cfg.seed)
    write_injections(out_path, injections)
    write_samples_csv(out_path.with_name("phase2_injection_samples.csv"), injections)
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out_path.with_name("phase2_injection_stats.csv"), index=False)
    print(f"{len(faithful)} faithful '{split}' responses -> {len(injections)} injected examples")
    print(stats_df.to_string(index=False))
    print(f"Saved {out_path.name}, phase2_injection_samples.csv and phase2_injection_stats.csv in {out_path.parent}")
    return {"injections": injections, "stats": stats_df}
