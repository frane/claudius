<h1 align="center">CLAVDIVS</h1>

<p align="center"><b>A Latin-only language model, trained from scratch — and a Latin tutor you can talk to.</b><br>
It has never seen a word of English.</p>

<p align="center">
<a href="https://huggingface.co/fbandov/claudius-latin-400m-chat">Chat model</a> ·
<a href="https://huggingface.co/fbandov/claudius-latin-400m">Base model</a> ·
<a href="#run-the-tutor">Run the tutor</a> ·
<a href="#how-it-was-built">How it was built</a> ·
<a href="docs/decisions.md">Every decision, measured</a>
</p>

```
DISCIPVLVS: Salvē, magister! Quis es?
MAGISTER:   SVM CLAVDIVS, MAGISTER TVVS. QVID HODIE DISCERE VIS?

DISCIPVLVS: Tu nomus?
MAGISTER:   NOMEN MEVM EST CLAVDIVS. QVOD EST NOMEN TVVM?

DISCIPVLVS: Duos fratres habeo, Stipe et Dario. Et tu?
MAGISTER:   EGO FRATRES NON HABEO, SED SOROREM CLAVDIAM. SVNTNE FRATRES TVI MAIORES AN MINORES?

DISCIPVLVS: Puella rosam amant.
MAGISTER:   PAENE RECTE! SED DICENDVM EST "AMAT", NON "AMANT".
            PVELLA NVMERI SINGVLARIS EST, ERGO ET VERBVM: AMAT. VBI SVNT ROSAE?
```

Claudius is a 342M-parameter Llama-style model pretrained compute-optimally on
6.8 billion tokens of Latin — classical, medieval and neo-Latin — and nothing
else. On top of it sits a conversation tutor that chats with beginners in
Latin, copes with their broken Latin, and corrects their mistakes only when it
can back the correction up.

It writes in the Roman way: capitals, `V` for both U and V, no lowercase, no
diacritics.

## Models

All on Hugging Face, CC BY-SA 4.0.

| model | what it is | held-out bits/byte |
|---|---|---|
| [`claudius-latin-400m-chat`](https://huggingface.co/fbandov/claudius-latin-400m-chat) | **the tutor**: chat template, levels, correction mode, browser (ONNX) weights | — |
| [`claudius-latin-400m`](https://huggingface.co/fbandov/claudius-latin-400m) | base model, annealed on classical Latin and Vicipaedia | 1.097 |
| [`claudius-latin-400m-base`](https://huggingface.co/fbandov/claudius-latin-400m-base) | pretrained checkpoint before the anneal | 1.091 |

Bits per byte are measured on a **whole-author holdout** (no author in the test
set appears in training), against a character-bigram floor of 3.518 that was
fixed before any model was trained. No 13-word run of generated text occurs
verbatim in the training text.

## Use the models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("fbandov/claudius-latin-400m-chat")
model = AutoModelForCausalLM.from_pretrained("fbandov/claudius-latin-400m-chat")

messages = [{"role": "system", "content": "tiro"},            # tiro | medius | peritus
            {"role": "user", "content": "Salvē, magister! Quis es?"}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                              return_tensors="pt", return_dict=True)["input_ids"]
print(tok.decode(model.generate(ids)[0][ids.shape[1]:], skip_special_tokens=True))
# SUM CLAUDIUS, MAGISTER TUUS. QUID HODIE DISCERE UIS?
```

The tokenizer normalises any Latin you give it (macrons, lowercase, J and V),
and the generation defaults are already set. The [model card](https://huggingface.co/fbandov/claudius-latin-400m-chat)
covers the correction mode and its measured error rates.

## Run the tutor

```bash
git clone https://github.com/frane/claudius && cd claudius
uv venv && uv pip install -e ".[web]"
python -m claudius.cli serve          # → http://127.0.0.1:8000
```

- **`/`** — the full tutor. The model converses; corrections come from three
  detectors (the model's own correction mode, a scorer over word endings, a
  misspelling check against a clean Latin dictionary) and the
  [LatinCy](https://huggingface.co/latincy) parser. A **correction** (red, with
  the grammatical rule) needs the detectors and the parser to agree; anything
  less is asked as a **question** ("NVM 'NOMEN' DICERE VOLVISTI?").
- **`/browser`** — the same tutor running **entirely in your browser** with
  [transformers.js](https://github.com/huggingface/transformers.js): about 350 MB,
  downloaded once, WebGPU with a self-test and a CPU fallback. Nothing leaves
  the machine.
- **`python -m claudius.cli chat`** — the tutor in the terminal.

## How it was built

1. **Corpus.** Latin-PD, the latin-literature corpus (Corpus Corporum),
   the Latin Library, Perseus and Vicipaedia. Two-sided language ID (Latin score
   and rival-language rejection), OCR line cleaning, foreign-paragraph cutting,
   removal of edition line and section numbers, then global deduplication:
   exact paragraph hashing plus MinHash-LSH at 94.7% recall (J ≥ 0.8), as a
   parallel map/reduce on HF Jobs.
2. **One orthography.** Everything is normalised to the classical alphabet:
   uppercase, U for V, I for J, no diacritics, no Greek. The same normaliser is
   built into the tokenizer.
3. **Tokenizer.** 32k byte-level BPE. Fertility has no elbow for Latin, so
   the size was chosen on embedding cost and settled by held-out bits per byte.
4. **Pretraining.** A 342M Llama, 6.84B tokens (about 20 per parameter),
   AdamW, cosine schedule to 10%. Then an anneal on classical Latin and
   Vicipaedia, which moved the register (church vocabulary 17.0 → 11.6 per
   1,000 words, edition numbers 10.9 → 0) at the same bits per byte.
5. **Tutor.** Supervised fine-tuning on about 24,000 synthetic Latin
   dialogues: a fixed persona, three levels, beginners' broken Latin. Plus
   37,770 learner errors whose fix is always the original, correct sentence. The
   loss is on the teacher's words only, and correction text is context, never a
   target, so the model learns to continue after a fix but never to invent one.
6. **Measured, then kept or reverted.** Every change was tested against
   held-out data before it stayed: the regression sets are in `eval/`, and
   `docs/decisions.md` records each result, including the fixes that were
   rolled back.

Roughly $150 of GPU time on HF Jobs for everything.

## What it can and cannot do

| | |
|---|---|
| identity questions answered as Claudius | 7 / 7 |
| persona kept consistent (23 multi-turn cases) | 99% |
| off-topic replies | 8% |
| correct learner sentences wrongly corrected (tutor app) | 0 / 200 |
| misspellings caught (tutor app, fresh test) | 141 / 200 |

It writes fluent, grammatical Latin, but at 342M parameters it **states things
that are not true** — treat its facts as conversation, not information. It is
at its best on everyday topics at beginner and intermediate level.

## Repository

```
src/claudius/
  normalize.py  filters.py  quality.py   one orthography; language ID and cleaning; edition numbers
  dedup.py  corpora.py  prepare.py  parallel.py  mixing.py   corpus build, global dedup, weighted mix
  tokenizer.py  pipeline.py  model.py  train.py  evaluate.py  tokenizer, packing, pretraining, bits/byte
  sft/          dialogue template, persona, error generation, correction task, dataset packing
  tutor.py      the tutor: detectors, correction text, replies
  lexicon.py    misspelling check against a clean Latin dictionary
  web/          FastAPI server, the tutor page, the in-browser page
jobs/           HF Jobs entry points (corpus, tokens, training)
scripts/        data generation, dataset builds, every evaluation
eval/           the regression sets
docs/decisions.md   why each choice was made, with the numbers
```

## Prior art

[LATIN-VII / Varro](https://huggingface.co/vincent-b/latin-vii), LatinGPT,
`goldfish-models/lat_latn_100mb`, and the Latin encoders (Latin BERT, LaBerta,
PhilBERTa, PhilTa) came first. The evaluation borrows from the Varro model
card: a floor, a memorisation check, and standard incipits.

## License

Code: Apache-2.0. Models: CC BY-SA 4.0. The source corpora keep their own licences,
listed per source by `claudius sources`.
