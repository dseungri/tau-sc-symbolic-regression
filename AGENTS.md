# Instructions for Future Coding Agents

Before changing this repository, read `README.md`, `PROJECT_STATUS.md`,
`docs/scientific_scope.md`, and `docs/unresolved_decisions.md`.

Permanent safeguards:

- Never access or expose real subject data unless explicitly authorized.
- Never commit data, secrets, identifiers, model checkpoints, or generated subject outputs.
- Never change the scientific equation or network definitions without updating the decision register.
- Preserve subject-level split integrity.
- Never split by region or visit.
- Never use the held-out test set for preprocessing or model selection.
- Never infer ROI ordering.
- Never silently impute missing tau values.
- Never substitute functional connectivity for DWI structural connectivity.
- Keep symbolic regression restricted to the local reaction function unless the protocol is formally changed.
- Run tests before declaring work complete.
- Document every assumption.
- Do not claim biological causality from imaging-level reaction terms.
