# Privacy & Consent Policy (Face Data)

A system that stores students' faces and names is a **biometric data system** — it needs clear
guardrails from day one, not as an afterthought.

## Core rules

1. **No storage without explicit consent**: the first time an unknown face is detected, the
   robot asks in plain terms ("Would you like me to remember you for next time?") before storing
   anything. A decline means no embedding is stored — that visit stays anonymous.
2. **Minimum necessary data**: only store (the name as given by the student, the face embedding,
   first/last visit date, visit count). No raw images are kept after the embedding is extracted.
   Conversation transcripts are also stored (see rule 7) — nothing else.
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

## Before actual deployment (outside the scope of the code)

- Legal review by university administration (local data protection laws that may resemble
  GDPR/FERPA depending on the country).
- Identify who is responsible for deletion requests/complaints.
- Formally document the retention policy and link it to the notice sign at the kiosk.

## Where the data actually lives

A MySQL database (default name `luc_robot`), five tables — `robot_identity`, `people`,
`face_embeddings`, `conversations`, `utterances`. Only the last four hold anything about a
visitor; `robot_identity` is the robot's own name and its creator's contact details, which is
configuration rather than personal data collected from anybody. Schema and every statement that
touches it are in
`stages/stage02_face_recognition/robot_db.py`; connection settings come from a gitignored `.env`
at the project root (`.env.example` is the template).

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

-- right to be forgotten: takes their embeddings and transcripts with it (ON DELETE CASCADE)
DELETE FROM people WHERE person_id = '...';
```

This file is reference-only — the technical implementation of the consent flag and the
auto-expiry policy lives in `stages/stage02_face_recognition/face_memory.py`, and the storage
and cascade-delete behaviour in `stages/stage02_face_recognition/robot_db.py`.
