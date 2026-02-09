# ADR 0033: Remove schema/validation modes; simplify agent outputs

Date: 2026-02-06

Context: The worker prioritizes clean control flow and robust execution. Most downstream consumers only require short human-readable summaries and readable debug artifacts.

Decision: Remove schema/validation mode configuration and treat planning/coding backend output as plain text (Markdown by convention). Replace heavy structured domain models with `PlanDocument` (plan Markdown + summary) and `ExecutionReport` (report Markdown + summary), derived on a best-effort basis.

Consequences: Cleaner control flow, fewer retries caused by output formatting, and more readable artifacts. The system no longer provides strict machine-validated output guarantees; downstream logic relies on summaries and raw Markdown evidence.

