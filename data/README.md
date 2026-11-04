# Data

Everything the dispatcher says comes from these files. All of it is public
Wits information supplied by the project owner.

| File | What it is | Source |
| --- | --- | --- |
| `courses.json` | The 11 Honours Computer Science courses (COMS4xxx) and their November 2026 exams: date, time, type and venues. `here` names the building Pepper stands in. | Final November 2026 Assessment Timetable |
| `calendar.json` | 2026 academic periods (teaching blocks, breaks, exam periods) and key dates. | Wits 2026 academic calendar |
| `contacts.json` | School of Computer Science and Applied Mathematics contacts. `default` lists who is suggested when no one specific is asked for. | CSAM contact page |
| `locations.json` | Campus places with campus side, map grid square and map marker position. Rooms matching `MSL###` are in this building. | Braamfontein campus map 2025 |
| `intents.json` | The 15 canonical intent labels and their slots. | Project design |
| `branch_b_examples.json` | Labelled isiZulu requests and one label word per intent, shown to InkubaLM in the Branch B prompt. | Project recordings and the project owner's sentences |
| `dev_keywords.json` | Keyword lists for the stand-in classifiers. **Development only.** | Project design |
| `intent_dataset_template.csv` | Column layout for the labelled isiZulu command set. | Template |

## Not in the data yet

- **Weekly lecture times.** Add them per course under `sessions`, e.g.
  `{ "day": "monday", "start": "10:15", "end": "12:00", "venue": "MSL004" }`.
  They are only used during teaching blocks from `calendar.json`.
- **Lecturers.** Set `lecturer` per course. Until then, "who teaches X"
  points students to the Honours coordinator.

## Updating for a new period

When the 2027 timetables are published, replace the `exams` (and add
`sessions`) in `courses.json` and the periods in `calendar.json`, then run
`python -m pytest`. `tests/test_slots_and_dispatcher.py` checks that every
venue points to a known location and that the files are consistent.

Generated at runtime and ignored by git:

- `logs/research_log.csv`: one row per trial and branch, only when research mode has "Log trials" ticked.
- `recordings/<session>/<trial>.wav`: only when "Save audio" is also ticked. Participant audio stays off GitHub and must be handled according to the ethics approval.
