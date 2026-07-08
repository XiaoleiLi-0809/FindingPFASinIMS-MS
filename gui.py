from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .engine import (
    ProcessingConfig,
    ProcessingError,
    default_database_path,
    list_excel_sheets,
    process_file,
    read_input_table,
    suggested_columns,
)


APP_TITLE = "PFAS CCS Screening"
NONE_VALUE = "(not selected)"

RIMP1_METHODS = {
    "Auto detect": "auto",
    "Use existing first isotopic peak ratio column": "existing",
    "Calculate from monoisotopic and first-isotope intensities": "abundance",
    "Pair M+1 to M+4 isotope peaks from m/z, RT, 3D_TC, intensity": "pairing",
}

MASS_CALIBRATION_METHODS = {
    "No mass calibration": "none",
    "Lock mass ppm offset": "lock",
    "Linear ppm correction from standards": "linear",
}

CALIBRATION_METHODS = {
    "None (recommended for experimental CCS)": "none",
    "Apply fixed POS/NEG transformation": "apply",
    "Auto (only predicted/uncalibrated CCS columns)": "auto",
}

FIRST_CALIBRATION_METHODS = {
    "cIMS calibration from standard tc, m/z and CCS": "cims_reduced_ccs",
    "Use an existing CCS column": "existing_ccs",
}


class ScreeningApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1060x1080")
        self.minsize(820, 700)
        self.option_add("*Font", ("Segoe UI", 10))

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.sheet_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="POS")
        self.rimp1_method_var = tk.StringVar(value=next(iter(RIMP1_METHODS)))
        self.mass_calibration_var = tk.StringVar(
            value=next(iter(MASS_CALIBRATION_METHODS))
        )
        self.calibration_var = tk.StringVar(value=next(iter(CALIBRATION_METHODS)))
        self.first_calibration_var = tk.StringVar(
            value=next(iter(FIRST_CALIBRATION_METHODS))
        )
        self.standards_path_var = tk.StringVar()
        self.standards_sheet_var = tk.StringVar()
        self.standards_tc_var = tk.StringVar(value=NONE_VALUE)
        self.standards_mz_var = tk.StringVar(value=NONE_VALUE)
        self.standards_ccs_var = tk.StringVar(value=NONE_VALUE)
        self.standards_charge_var = tk.StringVar(value=NONE_VALUE)
        self.mass_standards_path_var = tk.StringVar()
        self.mass_standards_sheet_var = tk.StringVar()
        self.mass_observed_mz_var = tk.StringVar(value=NONE_VALUE)
        self.mass_exact_mz_var = tk.StringVar(value=NONE_VALUE)
        self.default_charge_var = tk.IntVar(value=1)
        self.mc_var = tk.IntVar(value=200)
        self.minimum_intensity_var = tk.DoubleVar(value=0.0)
        self.priority_top_n_var = tk.IntVar(value=200)
        self.database_tolerance_var = tk.DoubleVar(value=5.0)
        self.database_path_var = tk.StringVar(value=str(default_database_path()))
        self.database_sheet_var = tk.StringVar(value="Worksheet1")
        self.status_var = tk.StringVar(value="Choose an Excel file to begin.")
        self.column_vars = {
            field: tk.StringVar(value=NONE_VALUE)
            for field in [
                "mz",
                "ccs",
                "tc",
                "charge",
                "rt",
                "intensity",
                "rimp1",
                "m_intensity",
                "mp1_intensity",
            ]
        }
        self.column_boxes: dict[str, ttk.Combobox] = {}
        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._build_ui()
        self.after(100, self._drain_events)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)

        title = ttk.Label(outer, text=APP_TITLE, font=("Segoe UI Semibold", 19))
        title.grid(row=0, column=0, columnspan=3, sticky="w")
        subtitle = ttk.Label(
            outer,
            text=(
                "Preprocess an Excel feature table, calculate experimental isotope "
                "ratios, calibrate CCS, and assign PFAS probability tiers."
            ),
            foreground="#4a5568",
        )
        subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 18))

        ttk.Label(outer, text="Input file").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(outer, textvariable=self.input_var).grid(
            row=2, column=1, sticky="ew", padx=8
        )
        ttk.Button(outer, text="Browse...", command=self._browse_input).grid(
            row=2, column=2, sticky="ew"
        )

        ttk.Label(outer, text="Worksheet").grid(row=3, column=0, sticky="w", pady=5)
        self.sheet_box = ttk.Combobox(
            outer, textvariable=self.sheet_var, state="readonly"
        )
        self.sheet_box.grid(row=3, column=1, sticky="ew", padx=8)
        self.sheet_box.bind("<<ComboboxSelected>>", lambda _event: self._load_columns())

        settings = ttk.LabelFrame(outer, text="Processing settings", padding=12)
        settings.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(14, 8))
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)

        ttk.Label(settings, text="Ion mode").grid(row=0, column=0, sticky="w", pady=5)
        mode_frame = ttk.Frame(settings)
        mode_frame.grid(row=0, column=1, sticky="w")
        for value, text in [("POS", "Positive [M+H]+"), ("NEG", "Negative [M-H]-")]:
            ttk.Radiobutton(
                mode_frame,
                text=text,
                value=value,
                variable=self.mode_var,
                command=self._load_columns,
            ).pack(side="left", padx=(0, 18))

        ttk.Label(settings, text="rMp1 source").grid(
            row=1, column=0, sticky="w", pady=5
        )
        ttk.Combobox(
            settings,
            textvariable=self.rimp1_method_var,
            values=list(RIMP1_METHODS),
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(10, 0))

        ttk.Label(settings, text="First calibration").grid(
            row=2, column=0, sticky="w", pady=5
        )
        ttk.Combobox(
            settings,
            textvariable=self.first_calibration_var,
            values=list(FIRST_CALIBRATION_METHODS),
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=(10, 0))

        ttk.Label(settings, text="Mass calibration").grid(
            row=3, column=0, sticky="w", pady=5
        )
        ttk.Combobox(
            settings,
            textvariable=self.mass_calibration_var,
            values=list(MASS_CALIBRATION_METHODS),
            state="readonly",
        ).grid(row=3, column=1, sticky="ew", padx=(10, 0))

        ttk.Label(settings, text="Second calibration").grid(
            row=4, column=0, sticky="w", pady=5
        )
        ttk.Combobox(
            settings,
            textvariable=self.calibration_var,
            values=list(CALIBRATION_METHODS),
            state="readonly",
        ).grid(row=4, column=1, sticky="ew", padx=(10, 0))

        ttk.Label(settings, text="MC iterations").grid(
            row=5, column=0, sticky="w", pady=5
        )
        ttk.Spinbox(
            settings, from_=1, to=1000, textvariable=self.mc_var, width=10
        ).grid(row=5, column=1, sticky="w", padx=(10, 0))
        ttk.Label(settings, text="Minimum M intensity").grid(
            row=5, column=2, sticky="w", padx=(24, 0), pady=5
        )
        ttk.Entry(
            settings,
            textvariable=self.minimum_intensity_var,
            width=14,
        ).grid(row=5, column=3, sticky="ew", padx=(10, 0))

        ttk.Label(settings, text="Default |charge|").grid(
            row=6, column=0, sticky="w", pady=5
        )
        ttk.Spinbox(
            settings,
            from_=1,
            to=20,
            textvariable=self.default_charge_var,
            width=10,
        ).grid(row=6, column=1, sticky="w", padx=(10, 0))
        ttk.Label(settings, text="Top N intense PFAS peaks").grid(
            row=6, column=2, sticky="w", padx=(24, 0), pady=5
        )
        ttk.Spinbox(
            settings,
            from_=1,
            to=100000,
            textvariable=self.priority_top_n_var,
            width=14,
        ).grid(row=6, column=3, sticky="ew", padx=(10, 0))

        ttk.Label(settings, text="Database tolerance (ppm)").grid(
            row=7, column=0, sticky="w", pady=5
        )
        ttk.Entry(
            settings,
            textvariable=self.database_tolerance_var,
            width=10,
        ).grid(row=7, column=1, sticky="w", padx=(10, 0))

        ttk.Label(settings, text="PFAS database").grid(
            row=8, column=0, sticky="w", pady=5
        )
        ttk.Entry(settings, textvariable=self.database_path_var).grid(
            row=8, column=1, columnspan=2, sticky="ew", padx=(10, 8)
        )
        ttk.Button(
            settings,
            text="Browse...",
            command=self._browse_database,
        ).grid(row=8, column=3, sticky="ew")

        ttk.Label(settings, text="Database worksheet").grid(
            row=9, column=0, sticky="w", pady=5
        )
        self.database_sheet_box = ttk.Combobox(
            settings,
            textvariable=self.database_sheet_var,
            state="readonly",
            values=["Worksheet1"],
        )
        self.database_sheet_box.grid(
            row=9, column=1, columnspan=3, sticky="ew", padx=(10, 0)
        )

        mapping = ttk.LabelFrame(outer, text="Advanced column mapping", padding=12)
        mapping.grid(row=5, column=0, columnspan=3, sticky="ew", pady=8)
        mapping.columnconfigure(1, weight=1)
        labels = [
            ("mz", "m/z"),
            ("ccs", "Existing CCS"),
            ("tc", "3D_TC / corrected arrival time"),
            ("charge", "Charge (optional; default = 1)"),
            ("rt", "Retention time"),
            ("intensity", "Base peak intensity"),
            ("rimp1", "Existing rMp1 / first isotopic peak ratio"),
            ("m_intensity", "Monoisotopic peak intensity"),
            ("mp1_intensity", "First isotopic peak intensity"),
        ]
        for row, (field, label) in enumerate(labels):
            ttk.Label(mapping, text=label).grid(row=row, column=0, sticky="w", pady=3)
            box = ttk.Combobox(
                mapping, textvariable=self.column_vars[field], state="readonly"
            )
            box.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
            self.column_boxes[field] = box

        mass_standards = ttk.LabelFrame(
            outer, text="Optional mass calibration standards", padding=12
        )
        mass_standards.grid(row=6, column=0, columnspan=3, sticky="ew", pady=8)
        mass_standards.columnconfigure(1, weight=1)
        ttk.Label(mass_standards, text="Mass standards file").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Entry(mass_standards, textvariable=self.mass_standards_path_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(
            mass_standards, text="Browse...", command=self._browse_mass_standards
        ).grid(row=0, column=2)
        ttk.Label(mass_standards, text="Worksheet").grid(
            row=1, column=0, sticky="w", pady=3
        )
        self.mass_standards_sheet_box = ttk.Combobox(
            mass_standards,
            textvariable=self.mass_standards_sheet_var,
            state="readonly",
        )
        self.mass_standards_sheet_box.grid(row=1, column=1, sticky="ew", padx=8)
        self.mass_standards_sheet_box.bind(
            "<<ComboboxSelected>>", lambda _event: self._load_mass_standard_columns()
        )
        ttk.Label(mass_standards, text="Observed m/z column").grid(
            row=2, column=0, sticky="w", pady=3
        )
        self.mass_observed_mz_box = ttk.Combobox(
            mass_standards, textvariable=self.mass_observed_mz_var, state="readonly"
        )
        self.mass_observed_mz_box.grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Label(mass_standards, text="Exact m/z column").grid(
            row=3, column=0, sticky="w", pady=3
        )
        self.mass_exact_mz_box = ttk.Combobox(
            mass_standards, textvariable=self.mass_exact_mz_var, state="readonly"
        )
        self.mass_exact_mz_box.grid(row=3, column=1, sticky="ew", padx=8)

        standards = ttk.LabelFrame(
            outer, text="First-stage cIMS calibration standards", padding=12
        )
        standards.grid(row=7, column=0, columnspan=3, sticky="ew", pady=8)
        standards.columnconfigure(1, weight=1)
        ttk.Label(standards, text="Standards file").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Entry(standards, textvariable=self.standards_path_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(
            standards, text="Browse...", command=self._browse_standards
        ).grid(row=0, column=2)
        ttk.Label(standards, text="Worksheet").grid(
            row=1, column=0, sticky="w", pady=3
        )
        self.standards_sheet_box = ttk.Combobox(
            standards, textvariable=self.standards_sheet_var, state="readonly"
        )
        self.standards_sheet_box.grid(row=1, column=1, sticky="ew", padx=8)
        self.standards_sheet_box.bind(
            "<<ComboboxSelected>>", lambda _event: self._load_standard_columns()
        )
        ttk.Label(standards, text="Standard tc column").grid(
            row=2, column=0, sticky="w", pady=3
        )
        self.standards_tc_box = ttk.Combobox(
            standards, textvariable=self.standards_tc_var, state="readonly"
        )
        self.standards_tc_box.grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Label(standards, text="Standard m/z column").grid(
            row=3, column=0, sticky="w", pady=3
        )
        self.standards_mz_box = ttk.Combobox(
            standards, textvariable=self.standards_mz_var, state="readonly"
        )
        self.standards_mz_box.grid(row=3, column=1, sticky="ew", padx=8)
        ttk.Label(standards, text="Reference CCS column").grid(
            row=4, column=0, sticky="w", pady=3
        )
        self.standards_ccs_box = ttk.Combobox(
            standards, textvariable=self.standards_ccs_var, state="readonly"
        )
        self.standards_ccs_box.grid(row=4, column=1, sticky="ew", padx=8)
        ttk.Label(standards, text="Standard charge (optional)").grid(
            row=5, column=0, sticky="w", pady=3
        )
        self.standards_charge_box = ttk.Combobox(
            standards, textvariable=self.standards_charge_var, state="readonly"
        )
        self.standards_charge_box.grid(row=5, column=1, sticky="ew", padx=8)

        ttk.Label(outer, text="Output file").grid(row=8, column=0, sticky="w", pady=8)
        ttk.Entry(outer, textvariable=self.output_var).grid(
            row=8, column=1, sticky="ew", padx=8
        )
        ttk.Button(outer, text="Browse...", command=self._browse_output).grid(
            row=8, column=2, sticky="ew"
        )

        action_frame = ttk.Frame(outer)
        action_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(14, 6))
        action_frame.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(action_frame, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        self.run_button = ttk.Button(
            action_frame, text="Run screening", command=self._start_processing
        )
        self.run_button.grid(row=0, column=1)

        ttk.Label(outer, textvariable=self.status_var, foreground="#2d5f8b").grid(
            row=10, column=0, columnspan=3, sticky="w", pady=(4, 6)
        )
        self.log = tk.Text(
            outer,
            height=8,
            wrap="word",
            state="disabled",
            background="#f7f9fb",
            relief="solid",
            borderwidth=1,
        )
        self.log.grid(row=11, column=0, columnspan=3, sticky="nsew")
        outer.rowconfigure(11, weight=1)

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose feature table",
            filetypes=[
                ("Excel workbooks", "*.xlsx *.xlsm *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.input_var.set(path)
        source = Path(path)
        self.output_var.set(str(source.with_name(f"{source.stem}_PFAS_labeled.xlsx")))
        try:
            if source.suffix.lower() == ".csv":
                sheets = ["CSV"]
            else:
                sheets = list_excel_sheets(source)
            self.sheet_box["values"] = sheets
            self.sheet_var.set(sheets[0])
            self._load_columns()
            self.status_var.set(f"Loaded {source.name}")
        except ProcessingError as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _browse_output(self) -> None:
        initial = Path(self.output_var.get()) if self.output_var.get() else None
        path = filedialog.asksaveasfilename(
            title="Save labeled output",
            defaultextension=".xlsx",
            initialdir=str(initial.parent) if initial else None,
            initialfile=initial.name if initial else "PFAS_labeled.xlsx",
            filetypes=[("Excel workbook", "*.xlsx")],
        )
        if path:
            self.output_var.set(path)

    def _browse_standards(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose cIMS calibration standards table",
            filetypes=[
                ("Excel workbooks", "*.xlsx *.xlsm *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.standards_path_var.set(path)
        source = Path(path)
        try:
            sheets = (
                ["CSV"]
                if source.suffix.lower() == ".csv"
                else list_excel_sheets(source)
            )
            self.standards_sheet_box["values"] = sheets
            self.standards_sheet_var.set(sheets[0])
            self._load_standard_columns()
        except ProcessingError as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _browse_database(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose PFAS chemical database",
            filetypes=[
                ("Excel workbooks", "*.xlsx *.xlsm *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.database_path_var.set(path)
        source = Path(path)
        try:
            sheets = (
                ["CSV"]
                if source.suffix.lower() == ".csv"
                else list_excel_sheets(source)
            )
            self.database_sheet_box["values"] = sheets
            self.database_sheet_var.set(sheets[0])
        except ProcessingError as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _browse_mass_standards(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose mass calibration standards table",
            filetypes=[
                ("Excel workbooks", "*.xlsx *.xlsm *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.mass_standards_path_var.set(path)
        source = Path(path)
        try:
            sheets = (
                ["CSV"]
                if source.suffix.lower() == ".csv"
                else list_excel_sheets(source)
            )
            self.mass_standards_sheet_box["values"] = sheets
            self.mass_standards_sheet_var.set(sheets[0])
            self._load_mass_standard_columns()
        except ProcessingError as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _load_mass_standard_columns(self) -> None:
        path_text = self.mass_standards_path_var.get().strip()
        if not path_text or not Path(path_text).exists():
            return
        try:
            path = Path(path_text)
            sheet: str | int = (
                0
                if path.suffix.lower() == ".csv"
                else self.mass_standards_sheet_var.get()
            )
            frame = read_input_table(path, sheet)
            values = [NONE_VALUE] + [str(column) for column in frame.columns]
            suggestions = suggested_columns(frame, self.mode_var.get())
            self.mass_observed_mz_box["values"] = values
            self.mass_exact_mz_box["values"] = values
            self.mass_observed_mz_var.set(
                suggestions.get("observed_mz") or NONE_VALUE
            )
            self.mass_exact_mz_var.set(suggestions.get("exact_mz") or NONE_VALUE)
        except ProcessingError as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _load_standard_columns(self) -> None:
        path_text = self.standards_path_var.get().strip()
        if not path_text or not Path(path_text).exists():
            return
        try:
            path = Path(path_text)
            sheet: str | int = (
                0 if path.suffix.lower() == ".csv" else self.standards_sheet_var.get()
            )
            frame = read_input_table(path, sheet)
            values = [NONE_VALUE] + [str(column) for column in frame.columns]
            suggestions = suggested_columns(frame, self.mode_var.get())
            self.standards_tc_box["values"] = values
            self.standards_mz_box["values"] = values
            self.standards_ccs_box["values"] = values
            self.standards_charge_box["values"] = values
            self.standards_tc_var.set(suggestions.get("tc") or NONE_VALUE)
            self.standards_mz_var.set(suggestions.get("mz") or NONE_VALUE)
            self.standards_ccs_var.set(suggestions.get("ccs") or NONE_VALUE)
            self.standards_charge_var.set(
                suggestions.get("charge") or NONE_VALUE
            )
        except ProcessingError as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _load_columns(self) -> None:
        path_text = self.input_var.get().strip()
        if not path_text or not Path(path_text).exists():
            return
        try:
            path = Path(path_text)
            sheet: str | int = 0 if path.suffix.lower() == ".csv" else self.sheet_var.get()
            frame = read_input_table(path, sheet)
            values = [NONE_VALUE] + [str(column) for column in frame.columns]
            suggestions = suggested_columns(frame, self.mode_var.get())
            for field, box in self.column_boxes.items():
                box["values"] = values
                self.column_vars[field].set(suggestions.get(field) or NONE_VALUE)
            self._append_log(
                f"Detected {len(frame):,} rows and {len(frame.columns)} columns."
            )
        except ProcessingError as exc:
            self.status_var.set(str(exc))

    def _selected_column(self, field: str) -> str | None:
        value = self.column_vars[field].get()
        return None if value == NONE_VALUE or not value else value

    def _start_processing(self) -> None:
        input_path = Path(self.input_var.get().strip())
        output_path = Path(self.output_var.get().strip())
        if not input_path.exists():
            messagebox.showerror(APP_TITLE, "Choose a valid input file.")
            return
        if not self.output_var.get().strip():
            messagebox.showerror(APP_TITLE, "Choose an output file.")
            return
        try:
            mc_iterations = int(self.mc_var.get())
            default_charge = int(self.default_charge_var.get())
            priority_top_n = int(self.priority_top_n_var.get())
            minimum_intensity = float(self.minimum_intensity_var.get())
            database_tolerance = float(self.database_tolerance_var.get())
        except (TypeError, ValueError):
            messagebox.showerror(
                APP_TITLE,
                "Check MC iterations, charge, intensity threshold, Top N, and ppm tolerance.",
            )
            return
        if default_charge < 1:
            messagebox.showerror(APP_TITLE, "Default charge must be at least 1.")
            return
        if minimum_intensity < 0:
            messagebox.showerror(APP_TITLE, "Minimum M intensity cannot be negative.")
            return
        if priority_top_n < 1:
            messagebox.showerror(APP_TITLE, "Top N must be at least 1.")
            return
        if database_tolerance <= 0:
            messagebox.showerror(APP_TITLE, "Database ppm tolerance must be positive.")
            return
        database_path = Path(self.database_path_var.get().strip())
        if not database_path.exists():
            messagebox.showerror(APP_TITLE, "Choose a valid PFAS database file.")
            return

        sheet: str | int = (
            0 if input_path.suffix.lower() == ".csv" else self.sheet_var.get()
        )
        config = ProcessingConfig(
            input_path=input_path,
            output_path=output_path,
            sheet_name=sheet,
            ion_mode=self.mode_var.get(),
            rimp1_method=RIMP1_METHODS[self.rimp1_method_var.get()],
            mass_calibration_method=MASS_CALIBRATION_METHODS[
                self.mass_calibration_var.get()
            ],
            first_calibration_method=FIRST_CALIBRATION_METHODS[
                self.first_calibration_var.get()
            ],
            second_calibration_method=CALIBRATION_METHODS[
                self.calibration_var.get()
            ],
            mz_column=self._selected_column("mz"),
            ccs_column=self._selected_column("ccs"),
            tc_column=self._selected_column("tc"),
            charge_column=self._selected_column("charge"),
            default_charge=default_charge,
            rt_column=self._selected_column("rt"),
            intensity_column=self._selected_column("intensity"),
            rimp1_column=self._selected_column("rimp1"),
            m_intensity_column=self._selected_column("m_intensity"),
            mp1_intensity_column=self._selected_column("mp1_intensity"),
            standards_path=(
                Path(self.standards_path_var.get().strip())
                if self.standards_path_var.get().strip()
                else None
            ),
            standards_sheet_name=(
                0
                if self.standards_path_var.get().lower().endswith(".csv")
                else self.standards_sheet_var.get()
            ),
            standards_tc_column=(
                None
                if self.standards_tc_var.get() == NONE_VALUE
                else self.standards_tc_var.get()
            ),
            standards_mz_column=(
                None
                if self.standards_mz_var.get() == NONE_VALUE
                else self.standards_mz_var.get()
            ),
            standards_ccs_column=(
                None
                if self.standards_ccs_var.get() == NONE_VALUE
                else self.standards_ccs_var.get()
            ),
            standards_charge_column=(
                None
                if self.standards_charge_var.get() == NONE_VALUE
                else self.standards_charge_var.get()
            ),
            mass_standards_path=(
                Path(self.mass_standards_path_var.get().strip())
                if self.mass_standards_path_var.get().strip()
                else None
            ),
            mass_standards_sheet_name=(
                0
                if self.mass_standards_path_var.get().lower().endswith(".csv")
                else self.mass_standards_sheet_var.get()
            ),
            mass_observed_mz_column=(
                None
                if self.mass_observed_mz_var.get() == NONE_VALUE
                else self.mass_observed_mz_var.get()
            ),
            mass_exact_mz_column=(
                None
                if self.mass_exact_mz_var.get() == NONE_VALUE
                else self.mass_exact_mz_var.get()
            ),
            minimum_m_intensity=minimum_intensity,
            priority_top_n=priority_top_n,
            database_path=database_path,
            database_sheet_name=(
                0
                if database_path.suffix.lower() == ".csv"
                else self.database_sheet_var.get()
            ),
            database_mass_tolerance_ppm=database_tolerance,
            mc_iterations=mc_iterations,
        )
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Processing...")
        self._append_log("Starting screening run.")
        threading.Thread(
            target=self._worker, args=(config,), daemon=True
        ).start()

    def _worker(self, config: ProcessingConfig) -> None:
        try:
            summary = process_file(
                config,
                progress=lambda message: self.event_queue.put(("progress", message)),
            )
            self.event_queue.put(("complete", summary))
        except Exception as exc:
            self.event_queue.put(("error", exc))

    def _drain_events(self) -> None:
        try:
            while True:
                event, payload = self.event_queue.get_nowait()
                if event == "progress":
                    self.status_var.set(str(payload))
                    self._append_log(str(payload))
                elif event == "complete":
                    self.progress.stop()
                    self.run_button.configure(state="normal")
                    summary = payload
                    self.status_var.set(
                        f"Complete: {summary.predictable_rows:,}/{summary.input_rows:,} "
                        "rows predicted."
                    )
                    self._append_log(
                        f"Saved labeled workbook to {summary.output_path}"
                    )
                    messagebox.showinfo(
                        APP_TITLE,
                        (
                            f"Screening complete.\n\n"
                            f"Predictable rows: {summary.predictable_rows:,}/"
                            f"{summary.input_rows:,}\n"
                            f"Level 3: "
                            f"{summary.level_counts['Level 3 High-confidence PFAS']:,}\n\n"
                            f"Output:\n{summary.output_path}"
                        ),
                    )
                elif event == "error":
                    self.progress.stop()
                    self.run_button.configure(state="normal")
                    message = (
                        str(payload)
                        if isinstance(payload, ProcessingError)
                        else f"Unexpected error: {payload}"
                    )
                    self.status_var.set(message)
                    self._append_log(message)
                    messagebox.showerror(APP_TITLE, message)
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def main() -> None:
    app = ScreeningApp()
    app.mainloop()
