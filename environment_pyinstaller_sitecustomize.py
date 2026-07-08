from __future__ import annotations

import os
import traceback
from pathlib import Path


def _show_startup_error(message: str) -> None:
    error_path = Path.cwd() / "PFAS_CCS_Screening_startup_error.txt"
    error_path.write_text(message, encoding="utf-8")
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "PFAS CCS Screening",
            f"The application could not start.\n\nDetails: {error_path}",
        )
        root.destroy()
    except Exception:
        pass


if not os.environ.get("PFAS_APP_SKIP_LAUNCH"):
    try:
        from pfas_screening_app.gui import main

        main()
    except Exception:
        _show_startup_error(traceback.format_exc())
