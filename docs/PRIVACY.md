# Privacy & Consent Policy (Face Data)

A system that stores students' faces and names is a **biometric data system** — it needs clear
guardrails from day one, not as an afterthought.

## Core rules

1. **No storage without explicit consent**: the first time an unknown face is detected, the
   robot asks in plain terms ("Would you like me to remember you for next time?") before storing
   anything. A decline means no embedding is stored — that visit stays anonymous.
2. **Minimum necessary data**: only store (the name as given by the student, the face embedding,
   first/last visit date, visit count). No raw images are kept after the embedding is extracted.
   Conversation transcripts (rule 7) and a few short remembered facts (rule 8) are also stored —
   nothing else.
3. **Retention period**: any student's data is deleted automatically if they haven't visited the
   kiosk within a set period (e.g. one full semester) — an auto-expiry policy, not a manual one.
4. **Right to be forgotten**: there must be a way (even a simple voice request like "forget me")
   to delete a person's data on request.
5. **Local storage**: the face database stays local on the robot's device; it is never uploaded
   to an external cloud service without encryption and legal review by university
   administration.
6. **Transparency**: a clear sign/notice near the kiosk explaining that there's a camera doing
   face recognition, and what it's used for.
7. **Conversation transcripts**: what was said (both sides) and when is recorded, so the robot
   can refer back to a student's earlier visits and so wrong answers can actually be diagnosed
   afterwards. Two rules make this defensible rather than surveillance:
   - A transcript is only **linked to a named person** if that person consented to being
     remembered. A visitor who declined (or was never asked) still has their visit recorded, but
     with no identity attached to it — see the nullable `conversations.person_id` column.
   - Deleting a person deletes **their transcripts too**, automatically. This is enforced in the
     schema by `ON DELETE CASCADE`, not by remembering to run a second query, so both rule 3
     (retention) and rule 4 (right to be forgotten) cover conversations without any extra step.

   The kiosk notice (rule 6) must say that conversations are recorded, not only that there's a
   camera. This is the one part of this policy a student cannot infer from looking at the robot.
8. **Remembered facts**: at the end of a visit the robot extracts a handful of
   short notes about the student — what they were asking about, anything worth knowing next time
   — and stores them in `person_facts`, so a returning visitor can be greeted with "did you get
   anywhere with the Master of Computer Science?" rather than only being identified.

   These are **more sensitive than the face embeddings**, not less, and should be treated that
   way in any review: an embedding is a meaningless vector without the model that produced it,
   whereas these are readable English sentences about a named student. Four constraints, each
   enforced in code rather than left to a caller:
   - **Consent-gated by construction.** A fact can only be attached to a `person_id`, and a
     visitor who declined to be remembered has none — so there is nowhere to put one. It is not
     a check that can be forgotten.
   - **Bounded categories.** The extraction prompt explicitly refuses health or disability,
     religion, money problems, grades or exam results, immigration/visa status, anything about a
     person other than the student, and anything they asked to have forgotten. See
     `EXTRACTION_SYSTEM_PROMPT` in `stages/stage05_nlu_dialogue/visit_memory.py`.
   - **Auditable.** Every fact records the conversation it came from (`source_conversation_id`),
     so a student disputing something the robot "remembers" can be shown what was actually said.
   - **Deleted with the person**, via the same `ON DELETE CASCADE` as everything else, so rules
     3 and 4 cover them with no extra step. `delete_person_facts()` also exists for the narrower
     request — "forget what you know about me, but I still want to be recognised".

   A student asking what the robot remembers about them should be answered from `person_facts`,
   not by reading their transcripts aloud. That is the whole point of the table.

## Before actual deployment (outside the scope of the code)

- Legal review by university administration (local data protection laws that may resemble
  GDPR/FERPA depending on the country).
- Identify who is responsible for deletion requests/complaints.
- Formally document the retention policy and link it to the notice sign at the kiosk.

## Where the data actually lives

A MySQL database (default name `ai_reception_robot`), six tables — `robot_identity`, `people`,
`face_embeddings`, `conversations`, `utterances`, `person_facts`. Only the last five hold
anything about a visitor; `robot_identity` is the robot's own name and its creator's contact
details, which is configuration rather than personal data collected from anybody. Schema and
every statement that touches it are in `stages/stage02_face_recognition/robot_db.py`; connection
settings come from a gitignored `.env` at the project root (`.env.example` is the template).

**Without MySQL**: if no MySQL server answers, the robot now keeps the same
six tables in a local SQLite file, `data/robot_memory.db` (or `LUC_DB_SQLITE_PATH`), instead of
running with no memory. Everything in this policy applies to that file exactly as to MySQL - the
same consent rules, the same cascade deletes (SQLite enforces them; `PRAGMA foreign_keys = ON`
is set on every connection). It holds face embeddings and transcripts, so it must never be
copied off the robot, committed to git (`data/` is in `.gitignore`), or left on a shared
computer. Open it with "DB Browser for SQLite" to answer a request; the queries below work
unchanged.

Rule 5 (local storage) is a **deployment** requirement, and this is the file to check it
against: point `LUC_DB_HOST` at a machine inside the university, not at a managed cloud
database, unless the encryption/legal review below has actually happened.

Useful queries for handling a request in person:

```sql
-- everything the robot knows about one student
SELECT * FROM people WHERE name = 'Ahmad';

-- everything they have ever asked, newest first
SELECT c.started_at, u.speaker, u.text
FROM conversations c JOIN utterances u USING (conversation_id)
JOIN people p USING (person_id)
WHERE p.name = 'Ahmad'
ORDER BY c.started_at DESC, u.turn_index;

-- what the robot REMEMBERS about them, as opposed to what was said - answer a
-- "what do you know about me" request from this, not from the transcripts above
SELECT f.created_at, f.fact, f.source_conversation_id
FROM person_facts f JOIN people p USING (person_id)
WHERE p.name = 'Ahmad'
ORDER BY f.created_at DESC;

-- the narrower request: forget what you know about me, but keep recognising me
DELETE FROM person_facts WHERE person_id = '...';

-- right to be forgotten: takes their embeddings, transcripts and remembered facts with it
-- (ON DELETE CASCADE)
DELETE FROM people WHERE person_id = '...';
```

This file is reference-only — the technical implementation of the consent flag and the
auto-expiry policy lives in `stages/stage02_face_recognition/face_memory.py`, and the storage
and cascade-delete behaviour in `stages/stage02_face_recognition/robot_db.py`.

## What leaves the robot

Faces and the database never leave the computer, and neither do voices unless `STT_CLOUD=groq`
is set (below): face detection and recognition, speech recognition (Whisper or the streaming
Zipformer), the end-of-turn model, the wake word, echo cancellation and the voice (Kokoro or
Piper) all run locally. Microphone audio is not recorded
to disk; the only audio kept is a few hundred milliseconds in memory while it is being
recognised.

**The text of a visitor's question does leave it** when a cloud language model answers:
`ANSWER_BACKENDS` lists the order (default `groq,gemini,ollama`). With a Groq or Gemini key set,
the question, the matching knowledge-base passages and the last few lines of the current
conversation are sent to that provider to write the answer. No name, face or person id is sent.
To keep everything on the robot, set `ANSWER_BACKENDS=ollama` (the local model) and leave
`STT_CLOUD=off` - slower, but nothing leaves the building. The kiosk notice (rule 6) should say which one is in use.

**A recording of the visitor's question leaves it** when `STT_CLOUD=groq` is set (it is off by
default): each finished question, a few seconds of 16 kHz audio, is sent to Groq to be transcribed
by Whisper large-v3, together with a short hint listing the institution's and the robot's names.
The partial transcripts, the robot's own speech and any audio between questions are not sent.
Leave `STT_CLOUD=off` to keep all audio on the robot. The kiosk notice should mention it.

`LOG_TRANSCRIPTS=true` prints what visitors say into the console log. Leave it `false` on an
unattended kiosk whose logs are collected anywhere.
