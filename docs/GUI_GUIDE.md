# GUI guide

Launch from the repository root:

```bash
python scripts/gui_app.py
```

## Dashboard

Shows the main format and workflow boundaries. It is a summary page, not a live telemetry dashboard.

## Model Library

Supported file types:

- `.npz`: native controller checkpoint; numeric parameter vector is validated
- `.nrlmodel`: NeuroRaceLab bundle; manifest/schema and parameter count are validated
- `.gguf`: imported and checked for the GGUF magic header; inference is not integrated

Select a native `.npz` checkpoint and use **Export NPZ → NRLMODEL** to create a portable controller bundle. Matching training metadata is included when the history JSON can be found.

## Experiments

The current tab validates and displays a local experiment configuration. It does not launch the training pipeline.

## Live Run

The tab launches `scripts/population_workbench.py` as a subprocess, passes supported track/population/hidden-size arguments, streams process output to the GUI log, and offers a stop action. The simulation renders in a separate Pygame window.

## Reports

The report centre lists artifacts under `results/`. Select JSON/CSV experiment data and use **Export report…** to create PDF, DOCX, or PPTX. Numeric series may be plotted and included in the document. The generated PNG plot is also saved alongside the report.

Document export requires optional dependencies such as `matplotlib`, `reportlab`, `python-docx`, and `python-pptx`. Install them in the same environment used to run the GUI.

## Current boundaries

- GGUF inference is not implemented.
- Training is not launched from the Experiments tab.
- Pygame is not embedded in the Tkinter window.
- Structured per-car live telemetry charts are not yet integrated.
