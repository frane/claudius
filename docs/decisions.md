# Decisions

Why each choice was made, and what would change it. Decisions came out of the
handover in `#claudius`; the reasoning is recorded here so a later run does not
relitigate it from scratch.

## Normalisation: one classical orthography

Uppercase, `U` for `V`, `I` for `J`, no `W`, ligatures expanded, long-s folded,
diacritics stripped, hyphenation rejoined, Greek dropped, punctuation reduced to
a whitelist. Applied to the corpus before the tokenizer is trained and to every
prompt at inference.

*Why:* the corpus spans four centuries of typographic convention. Without
normalisation the tokenizer spends its vocabulary learning that `uir`, `vir`,
`Vir` and `VIR` are the same word. With it, the model sees the alphabet Rome
used, which is also the cleanest version of the claim being made.

*Cost:* vowel length is unrecoverable (it was not consistently marked anyway),
and Greek quotations are lost rather than transliterated. Transliteration was
the alternative; dropping was chosen for consistency, and consistency is what
matters — either is defensible, mixing them is not.

*What would change it:* if the chat phase needs to quote Greek, transliterate
instead and retrain the tokenizer. This is not a small change after pretraining.

## Language filter: stopword ratio, plus a rival check

The brief specified a stopword-ratio heuristic at ~0.06. On its own it does not
hold: French shares `et`, `est` and `qui` with Latin and scores ~0.12 — twice
the threshold. So the filter is two-sided. A document must clear 0.06 on Latin
function words *and* stay under 0.05 on the function words of French, German,
English, Italian and Spanish, *and* out-score the best rival by 1.5x.

Before either test, foreign-language *paragraphs* are cut from the document.
Many Latin-PD volumes are a Latin text inside a modern editorial frame; keeping
the whole volume would leak French into pretraining, and discarding it would
throw away the text. Cutting at paragraph level keeps the text and drops the
frame.

### Score the body, not the head

The scores are computed on five slices spread through the document, not on its
first 20,000 characters. This was found by running `filter-report` on real
Latin-PD rather than reasoning about it: a scanned book opens with a title page,
an imprimatur, a library plate and a column of OCR rubble, so the head is not
the text. Migne's *Patrologia Latina* -- one of the largest Latin corpora in
existence -- scored **0.0515** on its head and was rejected; on slices through
its body it scores **0.0871** and is kept. Sampling the head quietly discards
some of the most valuable books in the corpus.

### Stream shards separately

Latin-PD does not have one schema across its 118 shards. Of 24 sampled: 18 use
`identifier`, one uses `file_id`, two carry extra `language` and
`character_count` columns, three carry a stray `__index_level_0__`, and
`publication_date` is `int64` in some shards and `string` in others. Handing the
whole repo to `load_dataset` makes it unify those schemas and it raises
`CastError` at the first boundary -- about 1,000 rows in. That failure would
land hours into a 24-hour CPU job.

So each shard is streamed as its own dataset and column names are resolved per
row against a list of candidates. Verified against the exact pair that failed.

### Sample across shards

Latin-PD is 118 parquet shards and the rows are not shuffled across them, so the
first 5,000 rows of the stream are 5,000 rows of one shard -- one library, one
period, one press. `filter-report --shards N` opens N shards spread across the
repo and draws from them round-robin. Any accept rate measured on the head of
the stream describes a shard, not the corpus.

*Verify on the pilot:* `claudius filter-report` prints the documents nearest the
cutoff on both sides, plus the accept rate and how much text the paragraph cut
removes. That is where the mistakes are.

## Deduplication: fuzzy at document level, exact at paragraph level

MinHash + LSH over word 5-grams catches reprints and re-scans of the same
edition, which is the dominant form of duplication in Latin-PD. Paragraph-level
deduplication is exact hashing rather than MinHash: it exists to strip the
boilerplate (title pages, library plates, running apparatus) repeated verbatim
across volumes, and fuzzy matching at that scale costs far more than it returns.

*Measured on the real corpus:* the cross-shard duplicate rate is ~2% and stops
accumulating after the first few shards -- far below what "editions repeat"
implied. That looked like a threshold set too high, so it was checked directly:
group documents by title, and measure the Jaccard the deduper would compute for
same-title pairs.

    153 same-title pairs
      median Jaccard   0.000
      reaching 0.8      2/153
      in 0.3-0.8        0/153

The empty 0.3-0.8 band is the answer. A threshold set too high leaves real
reprints stranded just below it; there are none. Similarity is bimodal --
documents match at ~1.0 or share nothing.

The premise behind the worry was also wrong. Shared titles here are mostly
*multi-volume series*, not reprints: `patrologiaecur107mign` and
`patrologiaecur61mign` are Migne volumes 107 and 61, same title, entirely
different Latin. So 2% is real and the corpus is more diverse than expected.

*What would change it:* a paragraph-level near-duplicate rate that turns out
high on the pilot, which would mean sharding by MinHash band and running fuzzy
matching per shard.

## Tokenizer: byte-level BPE, size chosen by sweep

Vocabulary size is a hyperparameter, not something BPE discovers. One BPE is
trained to the ceiling and *truncated* to each candidate size — exact, not
approximate, because merges are learned in order and ids follow that order, so
the first N ids are what a run trained directly to N would have produced.
Fertility is measured on the held-out authors and the elbow is taken at 3%
relative gain per doubling.

Fertility alone does not settle it, so the shortlist is confirmed with held-out
**bits-per-byte** from pilot runs. Per-token loss is not comparable across
vocabularies; bits-per-byte is.

*Expectation from the brief:* 8k–32k. If the sweep lands outside that, believe
the sweep and say so in the tokenizer card.

## Model: three dense sizes, one recipe

150M / 400M / 1B, same tokenizer, same data, same hyperparameters, same tokens
per step. 1B is the flagship and the evaluation reference, 400M is the on-device
default (~250 MB at 4-bit), 150M is for drills and the browser. Three points on
one recipe is also a scaling curve worth publishing.

*MoE was rejected:* it cuts compute, not memory, and memory and download size
are the on-device constraints. Routing is also unstable at this scale.

## The floor, stated before training

**3.5181 bits per byte.** Interpolated character bigram, fit on 1,500 training
documents, scored on 150 documents from held-out authors of the clean corpus
(`fbandov/claudius-corpus-clean`), recorded 2026-09-19 — before any model
existed to compare against.

For scale: uniform choice over the 36-symbol normalised alphabet is 5.17
bits/byte, so letter-pair statistics alone already buy 1.65 bits. A language
model that does not beat 3.52 by a wide margin has learned nothing about Latin
beyond which letters follow which.

This number is fixed. It is easy to choose a floor after seeing a model's score
and declare victory; writing it down first removes that freedom. If the floor is
ever recomputed — on a different corpus, a different holdout, different
interpolation weights — it is a *new* number and must be reported as such
alongside this one, not quietly substituted for it.

## Evaluation: stated before training

* Whole-author holdout — no author in validation appears in training.
* Held-out bits-per-byte, against an interpolated character-bigram floor fit on
  training text. The floor is stated before the run so it cannot become a moving
  target.
* Memorisation check — word 13-gram windows of generated text searched verbatim
  against training text, with a genuinely unseen author as control. The control
  is what distinguishes regurgitation from ordinary Latin phrasing.
* Continuations of standard incipits, after normalisation.

The last three are borrowed from the Varro model card, which is the most honest
prior art on evaluating a small Latin model.

## The correctness guard is a veto, not a detector

The brief made a deterministic checker mandatory: a small model that mis-corrects
a student is worse than no tutor. Built on LatinCy (spaCy for Latin), checking
adjective-noun and subject-verb agreement. Measured on 300 correct Vulgate verses
and ~300 synthetic learner errors, with the same seeds throughout:

                                   false pos   verb detect   case detect
    first version                    11.7%         35%           14%
    drop gender                       6.7%         33%           11%
    distrust nsubj tagged Acc         4.7%         29%            8%
    same, medium model                4.7%         26%           12%

Three findings, each measured rather than assumed:

* **LatinCy cannot read our orthography raw.** On uppercase text it tags every
  word NUM -- PUELLA ROSAM PULCHRAM AMAT parses as Roman numerals. Lowercased,
  the same text parses *better* than ordinary Latin, and U-for-V is no problem.
* **Gender is unreliable** (16 of 41 false positives): Latin does not mark it on
  most third-declension endings, so the parser guesses. Dropped.
* **A bigger model does not help.** The limit is the approach, not parser
  accuracy. A dependency parser re-analyses an erroneous sentence to make it
  grammatical, so it often hides the error it should find -- faced with
  CAESAR GALLIAM DIUIDUNT it simply stops treating Caesar as the subject.

The remaining 4.7% is dominated by biblical proper names (EDOM, ELIPHAZ, LEUI),
which the Vulgate is dense with and learners rarely write, so it likely overstates
the rate on student text. That has not been measured.

**Consequence for the design.** At ~95% precision and ~10-30% recall, the checker
cannot find errors on its own. It can *confirm* them. So its role is a veto: the
buddy tells a student they are wrong only when the model proposes a correction
AND the checker independently finds the disagreement. Low recall then means the
buddy under-corrects, which is safe; the alternative, trusting either signal
alone, is how a tutor ends up confidently teaching mistakes.

**The trained model is the better detector -- measured.** Using the 110M pilot,
the weakest model the project will ever have, on the same ~300 synthetic errors:

                    model prefers correct    checker detects
    verb errors        91/112 = 81%               29%
    case errors       234/283 = 83%                8%
    median margin     3.4 nats (correct version ~30x more probable)

3x the checker on verbs, 10x on case, from a model at half an epoch. The parser
bends its analysis to make an erroneous sentence grammatical; the language model
does the opposite, because an improbable ending is exactly what it has learned.

So the design is two complementary signals: the model *detects* (~82% recall at
110M, should rise at 400M), the checker *confirms* (~95% precision, independent of
the model), and the buddy corrects only when both agree.

Caveat: this test hands the model the exact correct sentence to compare against.
In use, the buddy must generate candidates -- which the error generator already
does in reverse -- and pick the right one. That end-to-end rate is not yet
measured.

## Which model is actually usable: 400M

Measured, not argued. The pilot -- 110M at 15% of Chinchilla -- already produced
correct case endings, correct agreement and correct `EST`/`SUNT`/`QUOD` usage.
It had no content because it was barely trained, not because it was small.

The brief sets the bar as "simple GPT-2-class conversation". GPT-2 small is
124M; GPT-2 medium is 355M. The 400M size is 342M -- GPT-2 medium, trained
Chinchilla-optimally, on one language rather than all of English.

    size   tokens   A100 $   verdict
    150M     2.2B       10   grammatical Latin, completion only; too small to converse
    400M     6.8B       78   GPT-2-medium class; converses; ships at ~250MB in 4-bit
    1B      20.7B      904   ~30% better perplexity for 9x the money

Cost scales as N^2 under Chinchilla (20 tokens per parameter), so tripling
parameters costs nine times as much. That is arithmetic, not a hardware choice,
and no flavour closes a 9x gap.

**Measured 1B throughput** (job 6ab241fd, a100-large, 20 steps): 33 s/step
steady state at 524,288 tokens/step = ~15,900 tokens/s, 362 GPU-hours for
Chinchilla-optimal, $904.

What 400M will not do: know facts reliably, or self-correct grammar. That is
why the deterministic morphological checker is mandatory rather than optional --
the model converses, Whitaker's Words grades.

**Size is not what decides usability here; register is.** The pilot wrote
`[SCHOL. 50]` into a continuation of Virgil because 94% of the clean corpus is
medieval commentary. Without a final annealing phase on the classical subset,
the result is a fluent 14th-century scholiast rather than a conversation
partner.

## Results: the 400M and the anneal

Same held-out set (150 docs, whole-author holdout) and floor throughout.

                         params   tokens   bits/byte   beats 3.5181 floor by
    pilot                 110M    0.34B      1.918        1.65
    150M (Chinchilla)     110M    2.20B      1.351        2.17
    400M (Chinchilla)     342M    6.84B      1.091        2.43
    400M + anneal         342M   +0.07B      1.096        2.42

Held-out loss fell at every one of six in-run checks (5.666 -> 5.277). Zero
verbatim 13-grams against training text for every model.

**Annealing moved register, not general ability** -- which is what it is for.
Bits/byte is flat because the eval set is the clean corpus, mostly medieval.
The samples show it:

* *Gallia est omnis divisa* -> before: scholastic geometry (longitudo,
  latitudo, profunditas); after: geography (Rhenus, Germania, provincia).
* *Arma virumque cano* -> after: "Arma, duces, animosque tuos et regia
  signa. / Nec tu, si qua tuis olim via, si qua parentis / in terris pietas;
  non me, pater optime, coniunx / digna tua, Stygia non umquam ab sede recepi."
* *Quousque tandem* -> after: "Quid? Illud, Catilina, non est ferendum?"

**It is composing, not reciting** -- checked against the annealing set itself,
which it saw 3.5 times: 0/133 8-word runs, 1/141 6-word runs (0.7%), in the
continuations alone. An earlier reading of 7.1% at 6 words was an evaluation
bug: it scored prompt + continuation, and every prompt is a verbatim quotation
from the training data. Fixed; the automatic check now scores continuations
only. Remaining gap: the automatic check indexes the general clean corpus, not
the data a given run trained on.

Also found in the 400M's output: words fused at punctuation ("RABIE,IN"),
learned from OCR that drops the space after punctuation ~4,000 times per
million characters of Latin-PD. Normalisation now restores it for future
corpora and at inference. Pure-letter fusions ("DEDECOREATQUE") need a lexicon
and remain.

Spend: ~$42 before the 400M, ~$101 for it, ~$1.50 for the anneal.

## The buddy's error rates on the 400M

Same 80 correct and 80 corrupted Vulgate sentences as the pilot measurement;
the model generates its own candidate corrections and flags an error if any
beats the student's sentence by more than the margin.

                  FALSE ALARMS            CAUGHT
    margin     pilot     400M         pilot     400M
      0        55.0%     18.8%        88.8%     93.8%
      2        18.8%      7.5%        65.0%     87.5%
      4        11.2%      3.8%        45.0%     76.2%

Model + checker agreeing: 0.0% false alarms at every margin, 6-9% caught.

That makes the tiered design viable: both agree -> confident correction;
model alone at a strict margin -> soft suggestion; neither -> leave it.

Caveat: 3.8% is 3 of 80, a 95% interval of roughly 1-10%. A 300-sentence
run at margins 4/6/8 is measuring where to set the threshold.

## Code review after the $14 mistake

The 400M's first launch evaluated all 427M validation tokens every 500 steps --
45 minutes each, 27 times, 20 GPU-hours of waste on a 37-hour run. Caught only
because an ETA was asked for; the training and evaluation progress bars are
indistinguishable in the logs. It cost ~$14 to kill and relaunch. A review of
everything that bills or silently corrupts found six more:

1. **`--resume` did not work in a relaunched job.** The Trainer looks for
   checkpoint-* LOCALLY; a fresh container has none, so a timed-out run would
   have restarted from step 0. I had said twice that checkpoints made a timeout
   cheap. `restore_hub_checkpoint` now pulls `last-checkpoint/` first. Tested.
2. **Validation packed uncapped** -- the root cause of the $14, still in the
   code after the eval-side fix. Now capped at 20M tokens.
3. **Checkpoint pushes included optimizer state**: 4.1 GB x 17 = 70 GB per 400M
   run. Now every 2000 steps.
4. **LSH banding caught 20% of pairs at the stated 0.8 threshold**, not 95%:
   `_bands` targeted the S-curve's 0.5 crossing, the wrong criterion. Now the
   largest row count reaching 90% recall (16x8 at 128 permutations).
   *Measured impact on the built corpus: none* -- 404,550 real Latin-PD pairs
   compared directly, zero above 0.5. No rebuild.
5. **Candidates are verified with the MinHash estimate, not exact Jaccard.**
   Hidden by 4 until 4 was fixed: on a pathological synthetic corpus, pairs at
   true 0.69-0.77 were estimated 0.80-0.85 and accepted. Irrelevant on this
   corpus (see 4); documented, knob is num_perm=256.
6. **The CSV stream had no retry.** Connection retry added; a mid-stream
   failure still raises. Known gap.

Also found: `min_lr_ratio` was declared and never wired in (runs decayed to
zero, a standard choice, but the config lied); `tokenize.py` as a job-script
name shadowed the stdlib module and killed a job in dependency setup.

The pattern across all of these: each was invisible in a loss curve and in the
tests, and surfaced only from a measurement or a question about money. Four
could have cost real dollars; one was already paid.

## Compute utilisation is 17%, not 40%

Measured on the pilot: a 110M model at batch 8 x 2048, grad_accum 32, on
`a10g-small` runs at 16.4 s/step for 524,288 tokens -- about 32k tokens/s, or
2.1e13 FLOP/s against the A10G's 125 TFLOP/s peak. **17% MFU.**

The textbook 40% does not survive contact with a small model on a single
mid-range GPU. Budgeting the 1B at 40% understates it by 2.35x:

    1B on 20B tokens, H200
      at 40% MFU    87 GPU-hours    ~$436
      at 17% MFU   205 GPU-hours  ~$1,026

`claudius plan` now defaults to the measured figure and reports both. A
budgeting tool should be corrected upward by measurement, not flattered by a
constant.

*Caveat, and it matters:* 17% was measured on a 110M model on an A10G. A 1B
model on an H200 should do better -- larger models use the hardware more
efficiently and the H200 has far more memory bandwidth. The real number needs a
measured 1B step on the target hardware before anyone commits to it. Until then
the range is $436-$1,026 and the honest answer is the top of it.

## Compute

Pilot first: ~110M on ~2B tokens, a few A100-hours, roughly $10. It validates
the filter, the tokenizer and the loss curve before the 1B run is booked. The 1B
run is ~6e19 FLOPs, ~100–130 A100-hours, ~$300 at 40% MFU. `claudius plan`
computes this from the packed dataset rather than from a guess.

HF Jobs for both stages; checkpoints to the Hub so a timeout costs one relaunch
with `--resume`. A rented H100 is ~3x faster per hour if the job timeout becomes
the binding constraint.

## The claim

Not "first Latin LLM" — LATIN-VII / Varro, LatinGPT, `lat_latn_100mb` and the
Latin encoders came first. What is defensible: largest Latin-only generative
model, first at ~1B on ~10B tokens, first trained on the Neo-Latin bulk, first
chat-tuned. Phase 2 imports world knowledge from whatever model synthesises the
SFT data; the claim is about *pretraining*, and it should be worded that way.

## Tutor (fbandov/claudius-tutor, private)

- Data: 2,503 DISCIPULUS/MAGISTER dialogues written by `claude -p` (2,371) and
  `codex exec` (132, until its usage limit) via `scripts/gen_dialogues.py`, at
  low effort with hooks and MCP disabled (~0.5 -> ~40 dialogues/min). Deduped,
  split 95/5 by dialogue, 5 passes with freshly planted errors
  (`scripts/build_sft.py`): 1,957 x 1024-token sequences, 586k trained tokens.
- Fine-tune: annealed 400M, lr 5e-5, batch 24, 81 steps, 43 s on rtx-pro-6000.
  Val loss 3.43 -> 3.03 at step ~60, 3.04 at the end (flat, no more passes).
- The model must not write corrections. A first version trained on the woven
  corrections "corrected" FORUM to FORUM and missed PUELLA ROSAM AMANT. The
  correction prefix is now context (label -100); `claudius.tutor` decides.
- Correction explanations say only "the ending is wrong". The swap tables are
  string edits: -AM -> -A hits CURREBAM as well as ROSAM, so "the case is
  wrong" was sometimes false.
- Tiers (`claudius/tutor.py`), 300 correct + 200 corrupted Vulgate sentences:

  | tier | set | false alarms | caught | right fix |
  |---|---|---|---|---|
  | firm (checker + model > 0) | A | 2/300 | 24/200 | 22/24 |
  | hint (model >= 8 nats) | A | 3/300 | 91/200 | 90/91 |
  | firm (checker + model > 2.5) | B, fresh | 2/300 | 14/200 | 14/14 |
  | hint (model >= 8 nats) | B, fresh | 1/300 | 97/200 | 97/97 |

  2.5 was chosen on set A; set B is the honest number. Both B firm false alarms
  are Hebrew names in genealogy lists (ACCOS, ASEM). The model and the checker
  are complementary: the model barely prefers AMAT over AMANT (3.6 nats) or
  DEFENDUNT over DEFENDIT (1.6), which is what the parser checks.
- Known gaps: subject-verb errors with small model gain are missed (DEFENDIT);
  conversational content is fluent but often off-topic or factually wrong
  (e.g. 50,000 soldiers per legion).

## Tutor v2

- Data: 16,491 dialogues (13,988 new: Claudius persona, 61% greeting openings,
  every teacher turn ends in one question), 2 passes, 220 steps, 115 s. Val
  loss 2.86 -> 2.22. System prompt: "TU ES CLAUDIUS, MAGISTER LINGUAE LATINAE.
  DISCIPULUS EST {level}. ..." at training and inference.
- eval/tutor_messages.json (51), scripts/tune_tutor.py. v1 -> v2 at
  temperature 0.5: identity 0/7 -> 7/7, off-topic 36% -> 14%, loops 0%.
- Decoding grid (temperature 0.5-0.7 x penalty 1.1-1.2 x best-of 1/3): off-topic
  15/17/20% by temperature; best-of-3 worse in every pair and 3x slower.
  The claude -p judge is noisy (it rated one identical reply both ways), so
  differences under ~10 points are not meaningful at n=51. Defaults:
  temperature 0.5, top-p 0.9, penalty 1.15, best-of 1.
- Firm threshold re-measured on held-out Vulgate B with v2: firm false alarms
  9 / 5 / 4 of 300 at > 0 / 1 / 2.5 nats. Kept 2.5; MILITES URBEM DEFENDIT
  stays missed.
- Rejected patch: adjective gender swaps as fix candidates. They fixed one test
  case (ALBAM -> ALBUS instead of the wrong ALBA) but on held-out data added one
  firm and one hint false alarm for one extra catch. Rule: a fix for a quirk
  stays only if it improves held-out numbers, not the example that prompted it.

## Generated training data and provider terms (decided 2026-09-25, Frane)

The tutor data (16,359 dialogues and 37,770 error pairs from `claude -p`, plus 132
dialogues from codex) was generated on consumer subscriptions. Anthropic's
Consumer Terms bar using the Services "to develop any products or services that
compete with our Services, including to develop or train any artificial
intelligence or machine learning algorithms or models". OpenAI's terms bar using
Output "to develop models that compete with OpenAI". Decision: a free,
non-commercial Latin tutor shared with a few testers does not compete with either,
so it is fine. Revisit if Claudius goes commercial or the data or models are
released publicly. The base model (Latin corpus only) is unaffected either way.

## Error detection: learned + swaps combined (tutor v3)

- Learned detector: fbandov/claudius-tutor-v3, trained on dialogues plus a
  CORRIGE/RECTE task using 37,770 claude-written learner errors (8 types, the fix
  is the original sentence). v4 (more passes, 3x negatives, agreement-of-8-samples
  confidence) was more cautious and caught less, so it was dropped.
- Fair comparison on dev learner errors (1000 errors / 1000 correct, never
  trained on): learned 291 caught, swaps 89. On swap-generated Vulgate A/B the
  swaps win (101/109 vs 66/48). They are good at different things.
- Combination (Frane): firm only if both propose the same fix and the checker
  objects; either alone gives a hint question.

  | set | learned | swaps | combined |
  |---|---|---|---|
  | dev caught / firm FA | 291 / 3 | 89 / 5 | 321 / 0 (hint FA 11/1000) |
  | A caught / firm FA | 66 / 1 | 101 / 1 | 124 / 1 (hint FA 2/300) |
  | B caught / firm FA | 48 / 3 | 109 / 2 | 121 / 1 (hint FA 5/300) |
  | 51 messages caught / FA | 11 / 0 | 6 / 0 | 11 / 0 |

  Shipped. The cost is more hint questions on correct sentences (1-1.7%).
- Chat check, v3 at the tuned decoding: identity 7/7, off-topic 6% (v2: 14%),
  loops 0. Still missed: MILITES URBEM DEFENDIT, EGO PUELLA UIDEO, EGO LIBRUM
  LEGIT (the model praises them).
- `claudius chat` now defaults to fbandov/claudius-tutor-v3.

## Release model: anneal 2, and what a future full retrain must fix

Measured, per 1,000 words, on continuations of 40 fixed prompts
(`scripts/measure_output.py`; fusion judged by one fixed model, split only
if it prefers the split by more than 2 nats):

| | fused | edition numbers | church words | bits/byte |
|---|---|---|---|---|
| 400M | 0.4 | 10.9 | 17.0 | 1.091 |
| anneal 1 (Latin Library only) | 0.1 | 17.1 | 9.8 | 1.096 |
| anneal 2, first try (all numbers deleted) | 0.0 | 0.0 | 9.2 | 1.098 |

- **The old "21% of long words fused" was wrong.** It came from the frequency
  splitter, which "splits" rare but real words (CONSIGNIFICAT, PRAECOGNOSCERE,
  HILARISSIMUS). With the model-gated measure the 400M's output has almost no
  fusion (0.4 per 1,000). Fusion is real in the training text: latin-literature
  has about 29 per 1,000 in one sample and 1.1 in another; the classical core
  has 6.6.
- **Edition numbers.** Anneal 1 made them worse: it learned the Latin Library's
  verse numbering. `quality.strip_edition_numbers` now deletes edition
  numbering (line start or end, bracketed, glued to a word, "N. 4" marks) and
  drops whole sentences that carry a number as content. Deleting those numbers
  left holes ("INTER ANNOS ET A.C.", 22,561 of them), and the first anneal-2
  model learned the holes, so it was redone. This step is now part of
  `filters.prepare_document`, so every future corpus build gets it.
- **Church register.** latin-literature is 95% of the pretraining text and has
  2.6x the church vocabulary of the classical core (28.5 vs 10.8 per 1,000).
  Anneal 2 trains on the classical core plus Vicipaedia (9.6 per 1,000).
- **Tokenizer normalizer.** `tokenizer.classical_normalizer()` puts the
  orthography into the tokenizer (NFKD, drop accents, uppercase, V->U, J->I,
  W->UU). Plain input tokenizes exactly like normalised training text, and
  normalised text is unchanged (62/62 documents). New tokenizers get it
  automatically.

**Checklist for a future full retrain:**
1. Rebuild the corpus. Edition-number stripping is already in
   `prepare_document`.
2. Repair fusions in latin-literature with `quality.repair_fusions` before
   tokenizing. It runs at about 350 words/s unbatched on an M-series Mac, which
   is days for about 500M words, so batch it on a GPU and measure the cost
   first.
3. Cap latin-literature at about 60% of characters (now 95%). Add the
   graded-reader and colloquia sources (`readers-colloquia`, not yet built) and
   more Vicipaedia. Target church words per 1,000 at or below the classical
   core's (about 11).
4. Train the tokenizer with the classical normalizer (automatic now).
5. Re-measure with `scripts/measure_output.py` and whole-author bits per byte
   against the fixed 3.5181 floor.

**Released 2026-09-29 (public, CC BY-SA 4.0):**
- [fbandov/claudius-latin-400m](https://huggingface.co/fbandov/claudius-latin-400m)
  is anneal 2, redone with sentence-dropping instead of number deletion:
  1.097 bits/byte; per 1,000 words 0.0 numbers, 11.6 church words, 0.0 fused;
  0 date holes; 0 verbatim 13-grams.
- [fbandov/claudius-latin-400m-base](https://huggingface.co/fbandov/claudius-latin-400m-base)
  is the pretrained 400M.

Both ship only the weights, config, the tokenizer with the classical normaliser
built in, generation defaults with the anti-loop settings, and a model card.
Checkpoints, optimiser states, the training data and the tutor models stay
private. Verified by an anonymous download and generation.

**Terms revisit, 2026-09-29 (Frane):** publishing a chat model trained on the
claude-generated dialogues and error pairs is fine for a free, non-commercial
Latin model. This is the "released publicly" case noted above; decided to go.

## Chat model released: fbandov/claudius-latin-400m-chat (2026-09-29, public, CC BY-SA 4.0)

- The tutor recipe (v3: dialogues + 1 correction pass) on the release base
  (anneal 2), as fbandov/claudius-tutor-v5: 340 steps, 170 s, val loss 1.544.
  Chat: identity 7/7, off-topic 8%, loops 0, numbers/headings 0/51. The app
  (`claudius chat` / `serve`) now defaults to v5.
- The release ships a chat template (system message = level, the exact
  training format, verified token for token) and stops on the newline token
  (id 199) at the end of the teacher's turn.
- Bug caught before release: the chat defaults (repetition penalty 1.15, no
  repeated 4-grams) break correction mode, because correcting is mostly
  copying. The card shows correction with both switched off.
- Plain correction mode (`scripts/eval_correct_mode.py`): on the 51 learner
  messages 0/29 false alarms, 17/21 errors fixed exactly; on literary
  sentences it changes about 8% of correct ones. App pipeline on v5: firm
  false alarms 0/1000 (dev), 1/300 (A and B); caught 317/1000, 113 and 117/200.
- Open: on learner-style messages plain mode (17/21, 0 false alarms) beats the
  app's combined pipeline (11/21). Worth using it as a hint tier for short
  sentences, measured on held-out data first.

## Browser version, quantisation, persona, messy input (2026-09-29/30)

- **In-browser (transformers.js 4.3)** at `/browser`. ONNX export plus variants,
  measured with ONNX Runtime on the same inputs:

  | variant | size | bits/byte vs fp32 | same output as fp32 | corrections |
  |---|---|---|---|---|
  | fp16 | 752 MB | +0.03% | 100% | 17/21, 0 FA |
  | int8 (dynamic) | 377 MB | +4.3% | 85% | 17/21, 1 FA |
  | q4 (block 32) | 349 MB | +0.9% | 88% | 17/21, 0 FA |
  | q4f16 | 260 MB | +0.9% | 86% | 17/21, 0 FA |

  int8 dropped. BitNet rejected: it needs ternary pretraining, the quality gap
  is largest at this size, and it saves about 120 MB at best.
- **WebGPU bug.** On Frane's Mac, q4f16 on WebGPU produced digits and fusions
  (BONVS1, 1G?, HODIEHERIS, HABES NE) that the same logic on the CPU never does
  (replayed in Node on both q4 and q4f16). WebGPU now uses q4 (fp32
  arithmetic) and must pass a self-test (a known correction) on load, else it
  falls back to WASM.
- **Tokenizer class.** transformers 5 wrote `TokenizersBackend`, which
  transformers 4.x cannot load. Set to `PreTrainedTokenizerFast` in all
  released repos; verified on 4.57 and 5.17.
- **Persona (v6).** A fixed persona sheet (`sft/persona.py`), 4,732 persona
  dialogues (0 contradictions), and 25 old contradicting dialogues dropped.
  Regression (`eval/tutor_dialogues.json`, `scripts/eval_persona.py`): all
  three checks right 62.5% -> 79.2%; persona 75 -> 90, names 90 -> 94,
  perspective 90. Held-out correction mode unchanged (dev 634 vs 629 flagged,
  496 vs 483 exact, 87 vs 82 false alarms of 1,000).
- **Misspellings** (`lexicon.py`). The dictionary is clean Latin only (the
  dialogues, the classical core, Vicipaedia; 285k words). The pretraining
  corpus cannot serve: NEMEN occurs 136 times, ROSAM 70. Uncommon words get
  candidates within 2 edits, and the LM gates them. Name slots are never
  questioned. Margins: 12 nats for 1 edit, 20 for 2 (picked on dev with zero
  false alarms; test: 0/120 clean, 0/44 names, 83/120 typos caught). On short
  messages (<= 12 words) the learned fix is also a hint above 2 nats, or above
  0 when it respells a non-word. Final `tutor.detect`, fresh data (seed 11):
  0/200 clean and 0/66 name sentences flagged; typos caught 141/200, 128 exact.

## Chat model v7 released (2026-09-30)

- v7 = v6 data + 2,833 messy-input dialogues (broken beginner Latin; Claudius
  works out the intent and recasts), with no planted corrections in those.
- The 23-case regression (all four checks right): v5 62.5% (16 cases), v6
  62.3%, v7 65.2%, v7 + echo guard 71.0%. Persona 98.6%, perspective 92.8%,
  names 95.7%, sense 71.0%.
- Side effect found and handled: v7 echoes plain statements back as questions
  ("EGO CANEM HABEO." -> "TU CANEM HABES?"): 5/85 vs v6 1/85 over 5 samples
  each. It comes from the messy data's legitimate recasting (the training data
  has <= 1% such echoes). `tutor.echoes` + resampling (3 tries) brings it to
  2/85; the same guard is in the browser page. Off-topic back to 8%.
- Correction mode v7 (held-out, plain): dev 9.3% false alarms / 62% flagged /
  48% exact; A 8.7% / 62% / 48%; B 7.0% / 54% / 36%; messages 1/29, 19/21,
  17/21.
- Released to fbandov/claudius-latin-400m-chat with ONNX q4/q4f16/fp16 (q4:
  +1% bits/byte, same corrections). The local app defaults to tutor-v7.
- Upload note: `hf upload` stalled for an hour on 2.1 GB (hard-linked files);
  `HfApi.upload_folder` did it cleanly.
