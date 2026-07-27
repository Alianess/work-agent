# Design

## Mood

Local operations desk: white task light, graphite controls, and a quiet teal signal line that reads as an instrument, not decoration.

## Color Strategy

Restrained product palette. The surface stays clear and neutral; teal marks the active route, confirmed model state, and primary action. Amber and red are reserved for warnings and errors.

## Tokens

```css
:root {
  color-scheme: light;
  --color-bg: oklch(1.000 0.000 0);
  --color-shell: oklch(0.973 0.003 250);
  --color-surface: oklch(0.985 0.002 250);
  --color-panel: oklch(0.955 0.006 250);
  --color-ink: oklch(0.205 0.018 250);
  --color-muted: oklch(0.455 0.020 250);
  --color-faint: oklch(0.635 0.017 250);
  --color-line: oklch(0.885 0.010 250);
  --color-primary: oklch(0.520 0.095 188);
  --color-primary-strong: oklch(0.420 0.105 188);
  --color-primary-soft: oklch(0.925 0.045 188);
  --color-accent: oklch(0.580 0.120 52);
  --color-accent-soft: oklch(0.940 0.050 52);
  --color-success: oklch(0.500 0.105 155);
  --color-warning: oklch(0.620 0.130 72);
  --color-danger: oklch(0.550 0.160 28);

  --radius-xs: 4px;
  --radius-sm: 6px;
  --radius-md: 8px;
  --shadow-focus: 0 0 0 3px oklch(0.720 0.100 188 / 0.24);
  --font-sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-mono: "SFMono-Regular", "SF Mono", Consolas, "Liberation Mono", monospace;
}
```

## Typography

Use one tuned sans family for product clarity. Headings are compact and direct; labels use the same family at smaller sizes with stronger weight. Data, paths, model names, and counts use the mono stack with tabular numerals.

Type scale:

- Page title: 24px / 32px, weight 720.
- Section title: 17px / 24px, weight 700.
- Body: 14px / 21px, weight 450.
- Label: 12px / 16px, weight 650.
- Code/data: 12px / 16px, tabular numerals.

## Layout

The default screen is an app shell:

```text
+----------------+-----------------------------------------------+
| sidebar        | top status bar                                |
|                +-----------------------------------------------+
| navigation     | active workflow surface                       |
| model summary  | split panes / forms / output previews         |
| tool status    |                                               |
+----------------+-----------------------------------------------+
```

Desktop keeps navigation persistent. Tablet and mobile collapse into a compact top navigation while preserving the same panel order: status, workflow, artifacts.

## Components

- Sidebar: persistent navigation, current model status, API key readiness, and tool count.
- Toolbar: current view title, primary action, refresh/status controls.
- Form controls: explicit labels, helper text when needed, inline errors, visible focus states.
- Segmented navigation: used for major workbench views.
- Tables/lists: dense rows with clear file paths, sizes, and modified times.
- Output preview: monospaced path, readable Markdown text, copy/open actions later.
- Toast/status region: `aria-live="polite"` and concise action result messages.

## Motion

Use short state transitions only: 160-220ms for hover, focus, panel reveal, and toast entry. Do not animate layout-heavy properties. Respect `prefers-reduced-motion`.

## Interaction Rules

- Primary action labels describe the result: "Run Agent", "Generate Minutes", "Add Profile".
- Missing environment keys are shown as configuration status, never as secret input fields.
- Errors must include a fix or next step.
- File paths are treated as first-class inputs because this is a local workspace tool.
- Meeting-skill controls separate confirmed facts from supplemental paths so the user can keep uncertain notes out of the work-submission output.
