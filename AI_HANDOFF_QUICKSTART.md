# FungiCirno — AI-first handoff

**Do not expect members to read every document manually.** Their coding AI should inspect the repository and the detailed docs. Humans only need this page, their assigned prompt, and the acceptance evidence.

## Human steps
1. Extract this release into a NEW folder; keep the previous project as backup.
2. Open the folder in your coding AI / IDE agent.
3. Create your branch: Minh Thinh → `member/minh-thinh-ml`; Thien Khang → `member/khang-growth-analytics`.
4. Paste your matching prompt from `docs/ai_handoff/`.
5. Let the AI inspect before editing. Review its short plan, then let it work.
6. Run the assigned acceptance checks and commit only your scoped changes.

## Rules
- Never work directly on `main`.
- Do not change shared DB schema/migrations, app entry point, shared inference contracts, or global dependencies without Hai Nam's approval.
- No broad rewrites. Fix demonstrated issues with the smallest change.
- Never claim a test passed unless it was run; include command and output.
- Keep weights, datasets, captures, credentials and local databases out of Git.
- If a fix crosses into the other member's scope, report the required interface change instead of editing their files.

## Return to Hai Nam
Send branch + commit hash, exact commands and results, changed files, screenshots/exports/logs, failures, and remaining risks. Avoid unsupported claims that everything works.

## AI usage expectation
Use the coding AI heavily for codebase exploration, implementation, debugging and tests. It must read relevant source/docs, preserve project conventions, explain root cause before broad changes, keep scope bounded, and label unverified assumptions. Detailed contracts and acceptance criteria are in `docs/INTEGRATION_CONTRACTS.md` and `docs/ACCEPTANCE_CHECKLIST.md`.
