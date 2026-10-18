from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from format_support import (
    SUPPORTED_MODEL_SUFFIXES,
    SUPPORTED_REPORT_SUFFIXES,
    export_nrlmodel,
    export_report,
    import_model_file,
    validate_nrlmodel,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models" / "library"
TRAINING_DIR = RESULTS_DIR / "training"


class NeuroRaceLabApp(tk.Tk):
    """Desktop shell with native racing-model bundles and report export."""

    def __init__(self):
        super().__init__()
        self.title("NeuroRaceLab Workbench")
        self.geometry("1180x760")
        self.minsize(900, 600)
        self.configure(bg="#101820")
        self.status = tk.StringVar(value="Ready")
        self.architecture = tk.StringVar(value="residual_rnn")
        self.algorithm = tk.StringVar(value="EA")
        self.track = tk.StringVar(value="oval")
        self.population = tk.IntVar(value=12)
        self.hidden_size = tk.IntVar(value=16)
        self.seed = tk.IntVar(value=42)
        self.budget = tk.IntVar(value=256)
        self.live_track = tk.StringVar(value="oval")
        self.live_population = tk.IntVar(value=12)
        self.live_hidden_size = tk.IntVar(value=16)
        self.live_seed = tk.IntVar(value=42)
        self.live_process = None
        self.live_queue = queue.Queue()
        self.live_log_after_id = None
        self._style()
        self._build()
        self.refresh_models()
        self.refresh_reports()

    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("TFrame", background="#101820")
        s.configure("Panel.TFrame", background="#1a2732")
        s.configure("TLabel", background="#101820", foreground="#e7edf2", font=("Sans", 10))
        s.configure("Panel.TLabel", background="#1a2732", foreground="#e7edf2")
        s.configure("Muted.TLabel", background="#101820", foreground="#a9bac7")
        s.configure("TButton", padding=(10, 7))
        s.configure("TNotebook", background="#101820", borderwidth=0)
        s.configure("TNotebook.Tab", padding=(16, 9))

    def _build(self):
        header = ttk.Frame(self, padding=(18, 14))
        header.pack(fill="x")
        ttk.Label(header, text="NEURORACELAB", font=("Sans", 17, "bold")).pack(side="left")
        ttk.Label(header, text="EXPERIMENT WORKBENCH", style="Muted.TLabel").pack(side="left", padx=16)
        ttk.Label(header, text="v0.2 · formats + reports", style="Muted.TLabel").pack(side="right")
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        self.dashboard = ttk.Frame(self.tabs, padding=18)
        self.models = ttk.Frame(self.tabs, padding=18)
        self.experiments = ttk.Frame(self.tabs, padding=18)
        self.live = ttk.Frame(self.tabs, padding=18)
        self.reports = ttk.Frame(self.tabs, padding=18)
        for tab, title in [
            (self.dashboard, "Dashboard"), (self.models, "Model Library"),
            (self.experiments, "Experiments"), (self.live, "Live Run"),
            (self.reports, "Reports"),
        ]:
            self.tabs.add(tab, text=title)
        self._dashboard()
        self._models()
        self._experiments()
        self._live()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(150, self._drain_live_output)
        self._reports()
        foot = ttk.Frame(self, padding=(18, 8))
        foot.pack(fill="x")
        ttk.Label(foot, textvariable=self.status, style="Muted.TLabel").pack(side="left")
        ttk.Label(foot, text="Controller models and GGUF language models are separate formats", style="Muted.TLabel").pack(side="right")

    def panel(self, parent, title, body):
        p = ttk.Frame(parent, style="Panel.TFrame", padding=16)
        ttk.Label(p, text=title, style="Panel.TLabel", font=("Sans", 12, "bold")).pack(anchor="w")
        ttk.Label(p, text=body, style="Panel.TLabel", wraplength=500, justify="left").pack(anchor="w", pady=(6, 0))
        return p

    def _dashboard(self):
        self.dashboard.columnconfigure((0, 1), weight=1)
        self.dashboard.rowconfigure((1, 2), weight=1)
        ttk.Label(self.dashboard, text="Overview", font=("Sans", 21, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 14)
        )
        cards = [
            ("Model formats", ".nrlmodel / .npz racing controllers; .gguf language models"),
            ("Experiment data", "JSON and CSV inputs can become PDF, DOCX, or PPTX reports"),
            ("Plots", "Numeric trends are rendered into PNG and embedded in document exports"),
            ("Runtime status", "Model import/export and report generation are connected; live training remains separate"),
        ]
        for i, (title, body) in enumerate(cards):
            self.panel(self.dashboard, title, body).grid(
                row=1 + i // 2, column=i % 2, sticky="nsew", padx=6, pady=6
            )
        self.panel(
            self.dashboard, "Format boundary",
            "GGUF files are registered and checked for their magic header, but inference needs a compatible GGUF runtime. "
            "NeuroRaceLab racing controllers use their own observation/action schema and are packaged as .nrlmodel.",
        ).grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=6)

    def _models(self):
        self.models.columnconfigure(0, weight=1)
        self.models.rowconfigure(3, weight=1)
        ttk.Label(self.models, text="Model Library", font=("Sans", 21, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.models, text="Import .gguf, .npz, and .nrlmodel. Native controller bundles are validated.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 14))
        bar = ttk.Frame(self.models)
        bar.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(bar, text="Import model…", command=self.import_model).pack(side="left")
        ttk.Button(bar, text="Export NPZ → NRLMODEL…", command=self.export_selected_npz).pack(side="left", padx=8)
        ttk.Button(bar, text="Inspect selected", command=self.inspect_selected_model).pack(side="left")
        ttk.Button(bar, text="Refresh", command=self.refresh_models).pack(side="left", padx=8)
        self.model_list = tk.Listbox(
            self.models, bg="#1a2732", fg="#e7edf2", selectbackground="#276678",
            borderwidth=0, highlightthickness=0, font=("Monospace", 10),
        )
        self.model_list.grid(row=3, column=0, sticky="nsew")
        ttk.Label(
            self.models,
            text=".nrlmodel is a ZIP-based NeuroRaceLab controller bundle. GGUF is a separate language-model format; importing it does not enable inference.",
            wraplength=850,
        ).grid(row=4, column=0, sticky="w", pady=10)

    def import_model(self):
        chosen = filedialog.askopenfilename(
            title="Import model",
            filetypes=[
                ("Supported model formats", "*.gguf *.npz *.nrlmodel"),
                ("GGUF language model", "*.gguf"),
                ("NeuroRaceLab checkpoint", "*.npz"),
                ("NeuroRaceLab model bundle", "*.nrlmodel"),
                ("All files", "*.*"),
            ],
        )
        if not chosen:
            return
        try:
            dst = import_model_file(Path(chosen), MODEL_DIR)
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))
            return
        self.refresh_models()
        self.status.set(f"Imported {dst.name}")
        messagebox.showinfo("Model imported", f"Imported to:\n{dst}")

    def refresh_models(self):
        if not hasattr(self, "model_list"):
            return
        self.model_list.delete(0, "end")
        files = sorted(
            (p for p in MODEL_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_MODEL_SUFFIXES),
            key=lambda p: p.name.lower(),
        ) if MODEL_DIR.exists() else []
        if not files:
            self.model_list.insert("end", "No model files registered")
        for p in files:
            size = p.stat().st_size
            self.model_list.insert("end", f"{p.name}  |  {size / (1024**2):.2f} MiB  |  {p.suffix.lower()}")

    def _selected_model_path(self) -> Path | None:
        selection = self.model_list.curselection()
        if not selection:
            return None
        value = self.model_list.get(selection[0])
        if value.startswith("No model"):
            return None
        name = value.split("  |  ", 1)[0]
        return MODEL_DIR / name

    def inspect_selected_model(self):
        path = self._selected_model_path()
        if not path or not path.exists():
            messagebox.showinfo("Select a model", "Select a model from the library first.")
            return
        try:
            if path.suffix.lower() == ".nrlmodel":
                details = validate_nrlmodel(path)
                text = json.dumps(details, indent=2)
            elif path.suffix.lower() == ".npz":
                from format_support import inspect_npz
                text = json.dumps(inspect_npz(path), indent=2)
            else:
                text = (
                    f"File: {path.name}\nSize: {path.stat().st_size:,} bytes\n"
                    "GGUF header is checked on import. Inference runtime is not integrated."
                )
        except Exception as exc:
            messagebox.showerror("Inspection failed", str(exc))
            return
        messagebox.showinfo("Model details", text)

    def export_selected_npz(self):
        source = self._selected_model_path()
        if not source or source.suffix.lower() != ".npz":
            messagebox.showinfo("Select checkpoint", "Select a .npz racing-controller checkpoint first.")
            return
        suggested = source.with_suffix(".nrlmodel").name
        destination = filedialog.asksaveasfilename(
            title="Export NeuroRaceLab model bundle",
            initialdir=str(MODEL_DIR),
            initialfile=suggested,
            defaultextension=".nrlmodel",
            filetypes=[("NeuroRaceLab model bundle", "*.nrlmodel")],
        )
        if not destination:
            return
        metadata_path = None
        # Training history usually lives beside the checkpoint with a matching stem.
        candidate = source.with_name(source.name.replace("_checkpoint.npz", "_history.json"))
        if candidate.exists():
            metadata_path = candidate
        else:
            training_candidate = TRAINING_DIR / source.name.replace("_checkpoint.npz", "_history.json")
            if training_candidate.exists():
                metadata_path = training_candidate
        try:
            result = export_nrlmodel(source, Path(destination), metadata_path)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status.set(f"Exported {result.name}")
        messagebox.showinfo("Export complete", f"Created bundle:\n{result}")

    def _experiments(self):
        self.experiments.columnconfigure(0, weight=1)
        self.experiments.rowconfigure(4, weight=1)
        ttk.Label(self.experiments, text="Experiment Configuration", font=("Sans", 21, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            self.experiments, text="Configuration validation is active; the GUI does not launch training yet.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 14))
        form = ttk.Frame(self.experiments, style="Panel.TFrame", padding=16)
        form.grid(row=2, column=0, sticky="nw")
        combos = [
            ("Architecture", self.architecture, ["residual_rnn", "continuous_time", "equilibrium"]),
            ("Algorithm", self.algorithm, ["EA", "ES"]),
            ("Track", self.track, ["oval", "circuit", "procedural"]),
        ]
        for r, (label, var, values) in enumerate(combos):
            ttk.Label(form, text=label, style="Panel.TLabel").grid(
                row=r, column=0, sticky="w", padx=(0, 24), pady=7
            )
            ttk.Combobox(form, textvariable=var, values=values, state="readonly", width=24).grid(
                row=r, column=1, sticky="ew", pady=7
            )
        for r, (label, var) in enumerate([
            ("Population size", self.population), ("Hidden size", self.hidden_size),
            ("Seed", self.seed), ("Evaluation budget", self.budget),
        ], start=3):
            ttk.Label(form, text=label, style="Panel.TLabel").grid(
                row=r, column=0, sticky="w", padx=(0, 24), pady=7
            )
            ttk.Spinbox(
                form, from_=0 if label == "Seed" else 1, to=1000000, textvariable=var, width=22
            ).grid(row=r, column=1, sticky="ew", pady=7)
        ttk.Button(self.experiments, text="Validate configuration", command=self.validate_config).grid(
            row=3, column=0, sticky="w", pady=12
        )
        self.preview = tk.Text(
            self.experiments, height=7, bg="#1a2732", fg="#e7edf2",
            insertbackground="#e7edf2", relief="flat", wrap="word",
        )
        self.preview.grid(row=4, column=0, sticky="nsew")

    def validate_config(self):
        try:
            config = {
                "architecture": self.architecture.get(), "algorithm": self.algorithm.get(),
                "track": self.track.get(), "population_size": int(self.population.get()),
                "hidden_size": int(self.hidden_size.get()), "seed": int(self.seed.get()),
                "evaluation_budget": int(self.budget.get()),
            }
            if config["population_size"] < 2 or config["hidden_size"] < 1 or config["evaluation_budget"] < 1:
                raise ValueError("Population must be at least 2; hidden size and budget must be positive.")
            self.preview.delete("1.0", "end")
            self.preview.insert("1.0", json.dumps(config, indent=2))
            self.status.set("Configuration validated locally; no training started")
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid configuration", str(exc))

    def _live(self):
        self.live.columnconfigure(0, weight=1)
        self.live.rowconfigure(4, weight=1)
        ttk.Label(self.live, text="Live Simulation", font=("Sans", 21, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            self.live,
            text="Launch and stop the existing multi-car Pygame workbench from this GUI. "
                 "The simulator renders in its own Pygame window; this tab manages its lifecycle and logs.",
            style="Muted.TLabel", wraplength=900,
        ).grid(row=1, column=0, sticky="w", pady=(4, 14))

        config = ttk.Frame(self.live, style="Panel.TFrame", padding=14)
        config.grid(row=2, column=0, sticky="ew")
        for col in range(8):
            config.columnconfigure(col, weight=1 if col % 2 else 0)
        ttk.Label(config, text="Track", style="Panel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=5)
        ttk.Combobox(
            config, textvariable=self.live_track, values=["oval", "circuit", "procedural"],
            state="readonly", width=14,
        ).grid(row=0, column=1, sticky="ew", padx=(0, 16), pady=5)
        ttk.Label(config, text="Cars", style="Panel.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 6), pady=5)
        ttk.Spinbox(config, from_=2, to=128, textvariable=self.live_population, width=7).grid(
            row=0, column=3, sticky="ew", padx=(0, 16), pady=5
        )
        ttk.Label(config, text="Hidden size", style="Panel.TLabel").grid(row=0, column=4, sticky="w", padx=(0, 6), pady=5)
        ttk.Spinbox(config, from_=1, to=512, textvariable=self.live_hidden_size, width=7).grid(
            row=0, column=5, sticky="ew", padx=(0, 16), pady=5
        )
        ttk.Label(config, text="Seed", style="Panel.TLabel").grid(row=0, column=6, sticky="w", padx=(0, 6), pady=5)
        ttk.Spinbox(config, from_=0, to=1000000, textvariable=self.live_seed, width=9).grid(
            row=0, column=7, sticky="ew", pady=5
        )

        controls = ttk.Frame(self.live)
        controls.grid(row=3, column=0, sticky="ew", pady=(10, 10))
        self.start_live_button = ttk.Button(controls, text="Start simulation", command=self.start_live)
        self.start_live_button.pack(side="left")
        self.stop_live_button = ttk.Button(controls, text="Stop simulation", command=self.stop_live, state="disabled")
        self.stop_live_button.pack(side="left", padx=8)
        ttk.Button(controls, text="Clear log", command=self.clear_live_log).pack(side="left", padx=8)
        self.live_status = tk.StringVar(value="Stopped")
        ttk.Label(controls, textvariable=self.live_status, style="Muted.TLabel").pack(side="right")

        self.live_log = tk.Text(
            self.live, height=16, bg="#0c1319", fg="#c8d8e4",
            insertbackground="#e7edf2", relief="flat", wrap="word",
            font=("Monospace", 9), state="disabled",
        )
        self.live_log.grid(row=4, column=0, sticky="nsew")
        ttk.Label(
            self.live,
            text="Controls and HUD remain in the Pygame window. Closing the GUI will stop the launched workbench.",
            style="Muted.TLabel", wraplength=900,
        ).grid(row=5, column=0, sticky="w", pady=(8, 0))

    def _append_live_log(self, message: str):
        self.live_log.configure(state="normal")
        self.live_log.insert("end", message.rstrip() + "\n")
        self.live_log.see("end")
        self.live_log.configure(state="disabled")

    def clear_live_log(self):
        self.live_log.configure(state="normal")
        self.live_log.delete("1.0", "end")
        self.live_log.configure(state="disabled")

    def start_live(self):
        if self.live_process and self.live_process.poll() is None:
            messagebox.showinfo("Simulation already running", "Stop the current simulation before starting another.")
            return
        try:
            population = int(self.live_population.get())
            hidden_size = int(self.live_hidden_size.get())
            seed = int(self.live_seed.get())
            track = self.live_track.get()
            if population < 2 or hidden_size < 1 or seed < 0:
                raise ValueError("Cars must be at least 2, hidden size positive, and seed non-negative.")
            script = ROOT / "scripts" / "population_workbench.py"
            if not script.is_file():
                raise FileNotFoundError(f"Could not find workbench script:\n{script}")
            command = [
                sys.executable, str(script),
                "--track", track,
                "--population", str(population),
                "--hidden-size", str(hidden_size),
            ]
            env = os.environ.copy()
            source_path = str(ROOT / "src")
            env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
            env.setdefault("PYTHONUNBUFFERED", "1")
            self._append_live_log("$ " + " ".join(command))
            self._append_live_log(f"Configured seed: {seed} (the current workbench CLI does not expose a verified seed option).")
            self._append_live_log("Launching Pygame workbench…")
            self.live_process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.start_live_button.configure(state="disabled")
            self.stop_live_button.configure(state="normal")
            self.live_status.set(f"Running · PID {self.live_process.pid}")
            self.status.set("Population workbench started")
            threading.Thread(
                target=self._read_live_stdout,
                args=(self.live_process,),
                daemon=True,
            ).start()
        except Exception as exc:
            self.live_status.set("Stopped")
            self._append_live_log(f"ERROR: {exc}")
            messagebox.showerror("Could not start simulation", str(exc))

    def _read_live_stdout(self, process):
        try:
            if process.stdout:
                for line in process.stdout:
                    self.live_queue.put(("line", line.rstrip("\n")))
        except Exception as exc:
            self.live_queue.put(("line", f"[output reader] {exc}"))
        return_code = process.wait()
        self.live_queue.put(("exit", (process.pid, return_code)))

    def _drain_live_output(self):
        try:
            while True:
                kind, payload = self.live_queue.get_nowait()
                if kind == "line":
                    self._append_live_log(payload)
                elif kind == "exit":
                    pid, return_code = payload
                    self._append_live_log(f"Workbench process {pid} exited with code {return_code}.")
                    if self.live_process and self.live_process.pid == pid:
                        self.live_process = None
                        self.start_live_button.configure(state="normal")
                        self.stop_live_button.configure(state="disabled")
                        self.live_status.set("Stopped" if return_code == 0 else f"Exited · code {return_code}")
                        self.status.set("Population workbench stopped")
        except queue.Empty:
            pass
        self.after(150, self._drain_live_output)

    def stop_live(self):
        process = self.live_process
        if not process or process.poll() is not None:
            self.live_process = None
            self.start_live_button.configure(state="normal")
            self.stop_live_button.configure(state="disabled")
            self.live_status.set("Stopped")
            return
        self._append_live_log("Stopping simulation…")
        try:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        except Exception as exc:
            self._append_live_log(f"ERROR while stopping: {exc}")
        self.live_process = None
        self.start_live_button.configure(state="normal")
        self.stop_live_button.configure(state="disabled")
        self.live_status.set("Stopped")
        self.status.set("Population workbench stopped")

    def _on_close(self):
        if self.live_process and self.live_process.poll() is None:
            self.stop_live()
        self.destroy()

    def _reports(self):
        self.reports.columnconfigure(0, weight=1)
        self.reports.rowconfigure(1, weight=1)
        ttk.Label(self.reports, text="Report Center", font=("Sans", 21, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.report_list = tk.Listbox(
            self.reports, bg="#1a2732", fg="#e7edf2", selectbackground="#276678",
            borderwidth=0, highlightthickness=0, font=("Monospace", 10),
        )
        self.report_list.grid(row=1, column=0, sticky="nsew", pady=(14, 10))
        bar = ttk.Frame(self.reports)
        bar.grid(row=2, column=0, sticky="ew")
        ttk.Button(bar, text="Refresh", command=self.refresh_reports).pack(side="left")
        ttk.Button(bar, text="Open selected", command=self.open_report).pack(side="left", padx=8)
        ttk.Button(bar, text="Export report…", command=self.export_report_dialog).pack(side="left")
        ttk.Label(
            self.reports,
            text="Select a JSON or CSV data file and export a PDF, DOCX, or PPTX report with an optional numeric plot.",
            style="Muted.TLabel", wraplength=850,
        ).grid(row=3, column=0, sticky="w", pady=(10, 0))

    def refresh_reports(self):
        if not hasattr(self, "report_list"):
            return
        self.report_list.delete(0, "end")
        files = [
            p for p in RESULTS_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_REPORT_SUFFIXES
        ] if RESULTS_DIR.exists() else []
        if not files:
            self.report_list.insert("end", "No report artifacts found")
            return
        for p in sorted(files):
            self.report_list.insert("end", str(p.relative_to(ROOT)))

    def _selected_report_path(self) -> Path | None:
        selection = self.report_list.curselection()
        if not selection:
            return None
        value = self.report_list.get(selection[0])
        if value.startswith("No report"):
            return None
        return ROOT / value

    def export_report_dialog(self):
        source_name = self._selected_report_path()
        if not source_name or source_name.suffix.lower() not in {".json", ".csv"}:
            chosen = filedialog.askopenfilename(
                title="Choose experiment data",
                initialdir=str(RESULTS_DIR if RESULTS_DIR.exists() else ROOT),
                filetypes=[("Experiment data", "*.json *.csv"), ("JSON", "*.json"), ("CSV", "*.csv")],
            )
            if not chosen:
                return
            source = Path(chosen)
        else:
            source = source_name
        output = filedialog.asksaveasfilename(
            title="Export report",
            initialdir=str(RESULTS_DIR if RESULTS_DIR.exists() else ROOT),
            initialfile=f"{source.stem}_report.pdf",
            defaultextension=".pdf",
            filetypes=[
                ("PDF document", "*.pdf"), ("Word document", "*.docx"),
                ("PowerPoint presentation", "*.pptx"),
            ],
        )
        if not output:
            return
        destination = Path(output)
        try:
            result = export_report(source, destination, destination.suffix)
        except Exception as exc:
            messagebox.showerror("Report export failed", str(exc))
            return
        self.refresh_reports()
        self.status.set(f"Exported {result.name}")
        messagebox.showinfo("Report exported", f"Created:\n{result}")

    def open_report(self):
        path = self._selected_report_path()
        if not path:
            return
        if not path.exists():
            messagebox.showerror("Missing file", str(path))
            return
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif sys.platform.startswith("win"):
                import os
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                messagebox.showinfo("Selected artifact", str(path))
        except OSError as exc:
            messagebox.showerror("Could not open file", str(exc))


if __name__ == "__main__":
    NeuroRaceLabApp().mainloop()
