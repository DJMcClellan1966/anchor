#!/usr/bin/env python3
"""
Tkinter GUI for Anchor: question input and response display.
Uses the same wiring as run_anchor.py; runs query in a thread to keep UI responsive.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

_ANCHOR_ROOT = Path(__file__).resolve().parent
if str(_ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANCHOR_ROOT))

import tkinter as tk
from tkinter import messagebox, scrolledtext

from anchor.engine import AnchorEngine
from anchor.wire import get_config, get_engine, get_generator_kind


def _format_status(generator_kind: str, generator_meta: dict | None, critic_info: dict) -> str:
    """Build status line: generator and critic score/decision."""
    parts = []
    gen_meta = generator_meta or {}
    if gen_meta.get("fallback_reason"):
        parts.append(gen_meta["fallback_reason"])
    g_sentences = gen_meta.get("graph_sentences")
    g_vocab = gen_meta.get("vocab_size")
    if g_sentences is not None and g_vocab is not None:
        parts.append(f"Generator: graph_attention (graph: {g_sentences} sentences, vocab: {g_vocab})")
    else:
        parts.append(f"Generator: {generator_kind}")
    score = critic_info.get("score")
    decision = critic_info.get("decision", "")
    if score is not None and decision:
        parts.append(f"score={score} decision={decision}")
    return "  |  ".join(parts)


def main() -> None:
    config = get_config()
    engine = get_engine()
    if engine is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Anchor",
            "Dictionary not configured.\n\n"
            "Set dictionary_path in config/paths.json or set ANCHOR_DICTIONARY_PATH.",
        )
        root.destroy()
        sys.exit(1)

    generator_kind = get_generator_kind()
    anchor_engine = AnchorEngine(engine, config, generator_kind=generator_kind)

    root = tk.Tk()
    root.title("Anchor")
    root.minsize(500, 400)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(2, weight=1)

    # Question
    tk.Label(root, text="Question:", anchor="w").grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 2))
    question_text = tk.Text(root, height=3, wrap=tk.WORD, font=("Segoe UI", 10))
    question_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
    question_text.focus_set()

    # Response
    tk.Label(root, text="Response:", anchor="w").grid(row=2, column=0, sticky="ew", padx=8, pady=(8, 2))
    response_text = scrolledtext.ScrolledText(
        root, wrap=tk.WORD, font=("Segoe UI", 10), state=tk.DISABLED
    )
    response_text.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 4))
    root.rowconfigure(3, weight=1)

    # Status line
    status_var = tk.StringVar(value="Ready. Enter a question and click Ask.")
    status_label = tk.Label(root, textvariable=status_var, anchor="w", font=("Segoe UI", 9))
    status_label.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))

    def _update_ui(
        response: str,
        critic_info: dict,
        extras: dict | None,
        status: str,
        error: bool = False,
    ) -> None:
        response_text.config(state=tk.NORMAL)
        response_text.delete("1.0", tk.END)
        response_text.insert(tk.END, response.strip() or "(no response)")
        response_text.config(state=tk.DISABLED)
        status_var.set(status)
        if error:
            messagebox.showerror("Anchor", status)

    def submit_worker() -> None:
        """Run one query in background and update UI when done."""
        q = question_text.get("1.0", tk.END).strip()
        if not q:
            status_var.set("Enter a question.")
            return
        status_var.set("Thinking...")
        response_text.config(state=tk.NORMAL)
        response_text.delete("1.0", tk.END)
        response_text.insert(tk.END, "Thinking...")
        response_text.config(state=tk.DISABLED)

        def do_query() -> None:
            try:
                response, critic_info, extras = anchor_engine.query(q, return_extras=True)
                extras = extras or {}
                gen_meta = extras.get("generator_meta") or {}
                status = _format_status(generator_kind, gen_meta, critic_info)
                root.after(0, lambda: _update_ui(response, critic_info, extras, status))
            except Exception as exc:
                root.after(0, lambda: _update_ui("", {}, None, f"Error: {exc}", error=True))

        t = threading.Thread(target=do_query, daemon=True)
        t.start()

    ask_btn = tk.Button(root, text="Ask", command=submit_worker, font=("Segoe UI", 10))
    ask_btn.grid(row=1, column=1, padx=(0, 8), pady=(0, 4), sticky="n")

    def on_key(event) -> None:
        if event.state & 0x4 and event.keysym == "Return":
            submit_worker()

    question_text.bind("<Control-Return>", on_key)

    root.mainloop()


if __name__ == "__main__":
    main()
