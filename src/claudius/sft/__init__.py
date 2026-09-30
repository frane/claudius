"""Phase 2 -- the language buddy. Not started.

Scope, as decided: a conversation partner for a student practising Latin. Not a
general assistant, not a reasoner. Simple GPT-2-class conversation is the bar.

Design, so that phase 1 does not foreclose it:

* Register -- classical and textbook prose, not Neo-Latin theology. That is a
  data-mix decision, made in `corpora.py` by upweighting Perseus, the Latin
  Library, the Vulgate, Vicipaedia and 19th-century school readers, or by
  annealing on them at the end of pretraining.
* SFT data -- synthesised, since no Latin instruction set exists. Generate
  correct sentences, corrupt them with typical learner errors (case agreement,
  verb endings, word order), and pair each with a gentle correction and a
  follow-up question. Plus vocabulary drills, "say it more simply", short
  everyday dialogues, and grammar explained in Latin.
* Level -- `tiro` / `medius` / `peritus` in the system prompt.
* Correctness guard -- the LM holds the conversation; a deterministic
  morphological checker (Whitaker's Words, LatinCy, CLTK) does the grading and
  correction. A small model that mis-corrects a student is worse than no tutor.
  This is not optional, and it is the reason the product can ship at 400M.
* Delivery -- GGUF via llama.cpp for iOS/Android/Windows/Mac, WebGPU
  (web-llm or transformers.js) in the browser. Offline, no API.

Note for the write-up: SFT imports world knowledge from whatever model
translates or synthesises the data. That is acceptable; the claim is about
*pretraining* having seen Latin only, and it should be stated that way.

Everything the chat phase needs from phase 1 already exists: the normalisation
is shared (prompts get the same treatment as the corpus), and the tokenizer has
room for chat special tokens below the vocabulary ceiling.
"""

#: Proficiency levels that go in the system prompt.
LEVELS = ("tiro", "medius", "peritus")
