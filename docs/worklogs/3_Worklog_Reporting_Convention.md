# 3 Worklog Reporting Convention

**Date:** 2026-09-11

## Decision

Every completed bounded task in this repository must end with a Markdown report in `docs/worklogs/` before its closing commit. Files use a monotonically increasing numeric prefix followed by a concise underscore-separated task name, for example `1_Experiment_Environment_Setup_Batch.md`.

## Required report content

Each worklog records the task objective and scope, completed work, verification, decisions and deviations, unresolved items or blockers, preserved boundaries, outcome, and relevant commit identifiers. A report must distinguish observed evidence from inference and must not represent environment preparation as an experimental result.

## Application

This convention is applied retroactively to the two completed prerequisite tasks in worklogs 1 and 2. Future task reports will be added and committed with the task they close; the corresponding local and LabServer63 worktrees must then be verified clean.
