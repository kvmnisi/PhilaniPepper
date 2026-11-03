# Data

| File | What it is | Status |
| --- | --- | --- |
| `intents.json` | The 13 canonical intent labels and their slots. Shared by both branches, the dispatcher and the evaluation tools. | Matches the reduced ~12-intent design in the status report (pending supervisor sign-off). |
| `courses.json` | Timetable used by the dispatcher. | **Sample data.** Codes `COMS90xx`, lecturers, venues and semester dates are placeholders. Replace before any session. |
| `dev_keywords.json` | Keyword lists for the `dev-rules` stand-in classifiers. | **Development only, never a result.** The isiZulu entries are AI-drafted and unverified. |
| `intent_dataset_template.csv` | Column layout for the labelled isiZulu command set used by `backend.evaluation.batch`. | Template. Only rows with `verified=yes` are evaluated. |

Generated at runtime and ignored by git:

- `logs/research_log.csv`: one row per trial and branch, written only when research mode has "Log trials" ticked.
- `recordings/<session>/<trial>.wav`: written only when "Save audio" is also ticked. Participant audio must stay off GitHub and must be handled according to the ethics approval.
