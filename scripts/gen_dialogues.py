"""Generate Latin classroom dialogues with the local `claude -p` and `codex exec` CLIs.

Uses the user's existing subscriptions, not API billing. Each call asks for a few
short dialogues on a given topic and level; output is parsed, normalised, and
filtered (Latin-only, well-formed turns) before being written to JSONL.

Corrections are NOT generated here -- they come from our deterministic
corrupt+template pipeline so they are guaranteed correct.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path

LEVELS = {
    "tiro": "absolute beginner: very short sentences (3-7 words), present tense, common everyday vocabulary",
    "medius": "intermediate: moderate sentences, several tenses, simple subordinate clauses",
    "peritus": "advanced: natural classical prose style, varied syntax, as between educated equals",
}
TOPICS = [
    "family and home", "the school day", "food and meals", "weather and seasons", "the market",
    "a journey to Rome", "animals on the farm", "Roman gods and myths", "a visit to the baths",
    "games and sport", "friends", "the city and its buildings", "the army", "a letter from a friend",
    "the sea and ships", "books and reading", "a feast", "health and the doctor", "the countryside",
    "clothes", "time and the calendar", "a famous Roman", "work and trades", "a festival",
    "the garden", "a storm", "a lost object", "music and poetry", "history of Rome", "the forum",
]
#: Everyday openings. A 400M tutor trained on topic chats alone could not answer
#: QUID AGIS? or QUIS ES? ("EGO QUOQUE SUM TUSALUS. AMASNE MORTUOS?"), so these
#: are listed several times to weight them up in the topic cycle.
SMALL_TALK = [
    "greetings and how are you (QUID AGIS? BENE, GRATIAS / VALEO)",
    "the student asks who the teacher is; he answers that he is Claudius, their Latin teacher",
    "how the student's day was", "the weather today", "what the student did yesterday",
    "the student's hobbies", "the student's family", "the student's school and lessons",
    "the student's friends", "food the student likes", "plans for tomorrow", "saying goodbye",
]
TOPICS += SMALL_TALK * 3

PROMPT = """Write {n} short, separate classroom conversations entirely in classical Latin between a DISCIPULUS (student) and a MAGISTER (friendly Latin teacher who chats with the student). The MAGISTER is named Claudius; if asked who he is, he says so (e.g. SUM CLAUDIUS, MAGISTER TUUS).

Level: {level} -- {level_desc}.
Topic: {topic}.

Rules:
- Latin ONLY. No English, no translations, no notes, no titles, no commentary.
- Both speakers write grammatically correct Latin. No grammar corrections.
- Every MAGISTER turn is short (1-2 sentences): it responds directly to what the student just said, then ends with ONE short follow-up question about it. It never changes the subject on its own.
- Vary the openings: many conversations start with a greeting (SALVE, MAGISTER! / QUID AGIS?) and small talk before the topic.
- Each conversation has 4 to 8 turns, alternating, starting with DISCIPULUS.
- Use classical spelling; no macrons.

Output format, exactly -- one line per turn, conversations separated by a line containing only ###:
DISCIPULUS: ...
MAGISTER: ...
DISCIPULUS: ...
MAGISTER: ...
###
DISCIPULUS: ...
"""

DISALLOWED = "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Agent,NotebookEdit"


def call_claude(prompt: str, timeout: int = 300) -> str:
    r = subprocess.run(["claude", "-p", prompt, "--output-format", "text",
                        "--disallowedTools", DISALLOWED, "--effort", "low",
                        "--strict-mcp-config", "--settings", '{"disableAllHooks": true}'],
                       capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-300:])
    return r.stdout


def call_codex(prompt: str, timeout: int = 300) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        out = f.name
    r = subprocess.run(["codex", "exec", "--skip-git-repo-check", "--ephemeral",
                        "-s", "read-only", "-c", "model_reasoning_effort=low", "-o", out, prompt],
                       capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    text = Path(out).read_text() if Path(out).exists() else ""
    Path(out).unlink(missing_ok=True)
    if r.returncode != 0 and not text:
        raise RuntimeError(r.stderr[-300:])
    return text


def parse(raw: str) -> list[list[tuple[str, str]]]:
    """Split raw output into dialogues of (role, text) turns."""
    dialogues = []
    for block in raw.split("###"):
        turns = []
        for line in block.splitlines():
            line = line.strip().strip("*").strip()
            for role in ("DISCIPULUS", "MAGISTER"):
                for pref in (role + ":", role.title() + ":", role.lower() + ":"):
                    if line.startswith(pref):
                        text = line[len(pref):].strip()
                        if text:
                            turns.append((role, text))
        if turns:
            dialogues.append(turns)
    return dialogues


def valid(turns, rival_score, latin_score, normalize) -> tuple[bool, list]:
    if not (4 <= len(turns) <= 10) or turns[0][0] != "DISCIPULUS":
        return False, []
    if any(a[0] == b[0] for a, b in zip(turns, turns[1:])):
        return False, []           # must alternate
    norm = [(r, normalize(t)) for r, t in turns]
    joined = " ".join(t for _, t in turns)
    # Latin-only: no English/modern-language function words leaking in.
    words = joined.split()
    if len(words) >= 20 and (rival_score(joined) > 0.02 or latin_score(joined) < 0.04):
        return False, []
    return True, norm


#: Personal and perspective dialogues (--focus persona). The chat swapped
#: ego/tu and meus/tuus, answered "ET TU?" with its name, and used the singular
#: after the student named two brothers. These topics make the student ask about
#: Claudius's own life and talk about several people at once.
PERSONA_TOPICS = [
    "the student tells about his/her siblings (several, by name) and then asks ET TU? -- Claudius answers about his own family",
    "the student asks Claudius personal questions: where he lives, his family, his dog, what he likes",
    "the student names two or three friends; Claudius asks about them by name (e.g. DE MARCO AN DE LUCIO?)",
    "the student describes his/her parents; Claudius asks about each, then the student asks about Claudius's parents",
    "Claudius and the student compare their days (EGO ... ET TU?)",
    "the student asks what Claudius reads and likes; Claudius asks the same back",
    "the student talks about his/her pets; the student asks whether Claudius has an animal",
    "the student tells about a family trip with brothers/sisters; Claudius asks what they did",
]

PERSONA_RULES = """
The MAGISTER is always the same person. Facts about him (never contradict them, use only the ones that fit):
{sheet}

Extra rules for these conversations:
- Keep the perspective straight: the MAGISTER says EGO/MEUS/MIHI about himself and TU/TUUS/TIBI about the student; when the student asks "ET TU?" or asks about the MAGISTER, he answers about HIMSELF from the facts above.
- When the student names several people, the MAGISTER refers to them correctly: plural when there are several (FRATRES TUI, DE FRATRIBUS TUIS), the right name when asking about one (DE STIPE AN DE DARIO?).
- Use the student's names and details exactly as given.
"""
#: Messy beginner input (--focus messy). Every student turn in the other data is
#: clean Latin, but real beginners write "TU NOMUS?", "SIC FRANE, ET TU?",
#: "TUUS NEMEN EST?" -- and the model answered those with nonsense.
MESSY_TOPICS = [
    "the student introduces himself/herself in broken Latin and asks the teacher's name",
    "the student asks simple personal questions with misspellings and missing verbs",
    "the student answers the teacher's questions with single words or word salad",
    "the student mixes up pronouns and endings while talking about family",
    "the student tries to say what he/she likes or does today, with many errors",
    "the student greets the teacher and asks how he is, with typos",
]
MESSY_RULES = """
In these conversations the DISCIPULUS is an absolute beginner who writes BROKEN Latin, the way real beginners do:
misspelled words (NEMEN for NOMEN, NOMUS, CLAUDUS), wrong endings (TUUS NOMEN, EGO HABET), missing verbs (TU CLAUDIUS?),
word salad, one-word or pronoun-only questions, stray letters (IG?), and a modern name (e.g. FRANE, STIPE, DARIO, LUKA).
Use a different kind of mistake in each student turn. The MAGISTER:
- works out what the student most likely meant and answers THAT, in correct Latin;
- recasts the student's intended sentence correctly inside his reply, without explaining grammar
  (student: TU NOMUS? -> MAGISTER: NOMEN MEUM EST CLAUDIUS. QUOD EST NOMEN TUUM?);
- when the message is unintelligible, asks kindly what the student means (QUID DICERE UIS?) and offers a simple model sentence;
- treats the student's name as a name, never as a mistake.
The MAGISTER's own Latin must be perfect. The student's Latin must stay broken.
"""
FOCUS = {"prompt_extra": ""}


def job(spec):
    engine, level, topic, n, seed = spec
    prompt = PROMPT.format(n=n, level=level, level_desc=LEVELS[level], topic=topic)
    if FOCUS["prompt_extra"]:
        prompt = prompt.replace("\nOutput format", FOCUS["prompt_extra"] + "\nOutput format", 1)
    fn = call_claude if engine == "claude" else call_codex
    for attempt in range(3):
        try:
            return spec, fn(prompt)
        except Exception as e:  # rate limit, timeout -- back off and retry
            time.sleep(20 * (attempt + 1))
            err = str(e)
    return spec, "ERROR: " + err


def main():
    from claudius.filters import FilterConfig, latin_score, rival_score
    from claudius.normalize import normalize

    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--per-call", type=int, default=5)
    ap.add_argument("--engines", nargs="+", default=["claude", "codex"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="out/sft/dialogues.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--focus", choices=["general", "persona", "messy"], default="general")
    a = ap.parse_args()

    cfg = FilterConfig(min_words_for_score=20)
    ls = lambda t: latin_score(t, cfg)
    rs = lambda t: rival_score(t, cfg)
    rng = random.Random(a.seed)
    topics = TOPICS
    if a.focus == "persona":
        from claudius.sft.persona import sheet
        FOCUS["prompt_extra"] = PERSONA_RULES.format(sheet=sheet())
        topics = PERSONA_TOPICS
    elif a.focus == "messy":
        from claudius.sft.persona import sheet
        FOCUS["prompt_extra"] = PERSONA_RULES.format(sheet=sheet()) + MESSY_RULES
        topics = MESSY_TOPICS
    combos = list(itertools.product(a.engines, LEVELS, topics))
    rng.shuffle(combos)
    specs = [(e, l, t, a.per_call, i) for i, (e, l, t) in zip(range(a.calls), itertools.cycle(combos))]

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    kept = rejected = errors = 0
    with open(out, "a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(a.workers) as ex:
        for (engine, level, topic, _, _), raw in ex.map(job, specs):
            if raw.startswith("ERROR"):
                errors += 1; print(f"[{engine}] error: {raw[:120]}", flush=True); continue
            for turns in parse(raw):
                ok, norm = valid(turns, rs, ls, normalize)
                if not ok:
                    rejected += 1; continue
                kept += 1
                fh.write(json.dumps({"engine": engine, "level": level, "topic": topic,
                                     "turns": norm}, ensure_ascii=False) + "\n")
            print(f"[{engine}] {level:<7} {topic:<24} kept={kept} rejected={rejected} errors={errors}", flush=True)
    print(f"DONE kept={kept} rejected={rejected} errors={errors} -> {out}")


if __name__ == "__main__":
    main()
