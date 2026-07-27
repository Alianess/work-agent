# Product

## Register

product

## Users

The primary user is a new employee who attends work meetings mostly as a listener and needs a reliable local assistant for turning noisy meeting recordings, ASR transcripts, supplemental materials, and personal corrections into usable work artifacts.

The user works in a local workspace, switches between model endpoints, inspects generated files, and iterates on meeting notes without exposing API keys or private meeting content in the UI.

## Product Purpose

Work Agent is a local AI workbench for model-driven office workflows. Its first production skill is meeting-recording processing: take a transcript plus confirmed facts and supplemental files, then produce two Markdown outputs:

- a detailed internal archive for the user to keep locally;
- a concise work-submission version that is conservative, confirmed, and appropriate for a listener's role.

The broader purpose is to provide a clean base for future ReAct tools, model routing, file skills, and domain-specific office workflows.

## Brand Personality

Precise, calm, and operational. The interface should feel like an instrument panel for careful work: confident enough to guide decisions, quiet enough not to compete with the task, and explicit when information is uncertain.

## Anti-references

Avoid marketing landing pages, oversized hero sections, generic SaaS card grids, purple-blue AI gradients, decorative glassmorphism, mascot-like illustration, and copy that oversells the system.

Avoid meeting-note output patterns that make the user appear to be the meeting owner: responsibility assignments, aggressive follow-up wording, uncertain names or amounts written as facts, and ASR uncertainty leaking into the work-submission artifact.

## Design Principles

1. Make the workflow visible: show model, profile, skill, input, output, and status without forcing the user into hidden configuration.
2. Treat uncertainty as first-class: confirmed facts and uncertain ASR-derived content need separate surfaces.
3. Keep the workbench extensible: every skill should fit into the same run panel, artifact panel, and model profile system.
4. Prefer operational density over presentation: repeated use should be faster than first-use delight.
5. Protect private work by default: never display secrets, never persist API keys in frontend code, and keep generated files local.

## Accessibility & Inclusion

Target WCAG AA contrast, visible focus states, keyboard-accessible controls, reduced-motion support, explicit labels for form fields, semantic landmarks, and clear inline error messages with a next step.
