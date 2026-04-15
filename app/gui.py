from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .core import FingerprintEngine
from .receiver import UdpReceiverThread


class FingerprintAppWindow:
    def __init__(self, engine: FingerprintEngine, receiver: UdpReceiverThread) -> None:
        self.engine = engine
        self.receiver = receiver
        self.root = tk.Tk()
        self.root.title("ESP32 CSI Fingerprinting App")
        self.root.geometry("1420x920")
        self.root.minsize(1100, 700)
        self.root.configure(bg="#eef2f6")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cols_var = tk.IntVar(value=self.engine.grid_cols)
        self.rows_var = tk.IntVar(value=self.engine.grid_rows)
        self.capture_var = tk.DoubleVar(value=self.engine.capture_seconds)
        self.window_var = tk.DoubleVar(value=self.engine.window_seconds)
        self.window_step_var = tk.DoubleVar(value=self.engine.window_step_seconds)
        self.keepalive_var = tk.DoubleVar(value=self.engine.keepalive_pings_per_second)
        self.model_var = tk.StringVar(value=self.engine.active_model_name or "")
        self.summary_var = tk.StringVar()
        self.capture_status_var = tk.StringVar()
        self.prediction_var = tk.StringVar()
        self.udp_var = tk.StringVar()
        self.message_var = tk.StringVar()
        self.cell_widgets: dict[str, dict[str, tk.Widget]] = {}
        self.cell_render_cache: dict[str, dict[str, object]] = {}
        self.model_selector: ttk.Combobox | None = None
        self.train_button: ttk.Button | None = None
        self.comm_log_listbox: tk.Listbox | None = None
        self.refresh_job: str | None = None

        self._build_layout()
        self._rebuild_grid(self.engine.grid_cols, self.engine.grid_rows)

    def run(self) -> None:
        self._refresh()
        self.root.mainloop()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=4)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(outer, text="Shared Config Driven Controls", padding=12)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for column in range(18):
            controls.columnconfigure(column, weight=1)

        ttk.Label(controls, text="Grid X").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(controls, from_=1, to=20, textvariable=self.cols_var, width=6).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(controls, text="Grid Y").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(controls, from_=1, to=20, textvariable=self.rows_var, width=6).grid(
            row=0, column=3, sticky="w"
        )
        ttk.Label(controls, text="Capture (sec)").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(
            controls,
            from_=1,
            to=20,
            increment=0.5,
            textvariable=self.capture_var,
            width=8,
        ).grid(row=0, column=5, sticky="w")
        ttk.Label(controls, text="Window (sec)").grid(row=0, column=6, sticky="w")
        ttk.Spinbox(
            controls,
            from_=0.25,
            to=20,
            increment=0.25,
            textvariable=self.window_var,
            width=8,
        ).grid(row=0, column=7, sticky="w")
        ttk.Label(controls, text="Window Step (sec)").grid(row=0, column=8, sticky="w")
        ttk.Spinbox(
            controls,
            from_=0.05,
            to=20,
            increment=0.05,
            textvariable=self.window_step_var,
            width=8,
        ).grid(row=0, column=9, sticky="w")
        ttk.Label(controls, text="Ping/s").grid(row=0, column=10, sticky="w")
        ttk.Spinbox(
            controls,
            from_=0,
            to=20,
            increment=0.5,
            textvariable=self.keepalive_var,
            width=8,
        ).grid(row=0, column=11, sticky="w")
        ttk.Button(controls, text="Apply To Config", command=self.on_apply_grid).grid(
            row=0, column=12, padx=(8, 0), sticky="ew"
        )
        ttk.Button(controls, text="Clear All", command=self.on_clear_all).grid(
            row=0, column=13, padx=(8, 0), sticky="ew"
        )
        self.train_button = ttk.Button(
            controls, text="Train Models", command=self.on_train_models
        )
        self.train_button.grid(row=0, column=14, padx=(8, 0), sticky="ew")
        ttk.Label(controls, text="Inference Model").grid(row=0, column=15, sticky="w")
        self.model_selector = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            state="readonly",
            values=(),
            width=18,
        )
        self.model_selector.grid(row=0, column=16, columnspan=2, sticky="ew")
        self.model_selector.bind("<<ComboboxSelected>>", self.on_select_model)
        ttk.Label(controls, textvariable=self.summary_var).grid(
            row=1, column=0, columnspan=8, sticky="w", pady=(10, 0)
        )
        ttk.Label(controls, textvariable=self.capture_status_var).grid(
            row=1, column=8, columnspan=4, sticky="w", pady=(10, 0)
        )
        ttk.Label(controls, textvariable=self.prediction_var).grid(
            row=1, column=12, columnspan=6, sticky="w", pady=(10, 0)
        )
        ttk.Label(controls, textvariable=self.udp_var).grid(
            row=2, column=0, columnspan=18, sticky="w", pady=(8, 0)
        )
        ttk.Label(
            controls,
            textvariable=self.message_var,
            foreground="#23435f",
        ).grid(row=3, column=0, columnspan=18, sticky="w", pady=(8, 0))

        grid_panel = ttk.LabelFrame(outer, text="Grid Heatmap", padding=8)
        grid_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        grid_panel.columnconfigure(0, weight=1)
        grid_panel.rowconfigure(0, weight=1)

        self.grid_canvas = tk.Canvas(grid_panel, bg="#eef2f6", highlightthickness=0)
        self.grid_canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(
            grid_panel, orient="vertical", command=self.grid_canvas.yview
        )
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(
            grid_panel, orient="horizontal", command=self.grid_canvas.xview
        )
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.grid_canvas.configure(
            yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set
        )

        self.grid_inner = tk.Frame(self.grid_canvas, bg="#eef2f6")
        self.grid_window = self.grid_canvas.create_window(
            (0, 0), window=self.grid_inner, anchor="nw"
        )
        self.grid_inner.bind(
            "<Configure>",
            lambda _event: self.grid_canvas.configure(
                scrollregion=self.grid_canvas.bbox("all")
            ),
        )
        self.grid_canvas.bind(
            "<Configure>",
            lambda event: self.grid_canvas.itemconfigure(
                self.grid_window, width=max(400, event.width)
            ),
        )

        side_panel = ttk.LabelFrame(outer, text="Node Telemetry", padding=10)
        side_panel.grid(row=1, column=1, sticky="nsew")
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(0, weight=1)
        side_panel.rowconfigure(1, weight=1)

        self.node_tree = ttk.Treeview(
            side_panel,
            columns=("node", "age", "rssi", "window", "source"),
            show="headings",
            height=18,
        )
        self.node_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(
            side_panel, orient="vertical", command=self.node_tree.yview
        )
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.node_tree.configure(yscrollcommand=tree_scroll.set)
        for key, title, width in (
            ("node", "Node", 60),
            ("age", "Age ms", 80),
            ("rssi", "RSSI", 80),
            ("window", "Window", 80),
            ("source", "Source", 180),
        ):
            self.node_tree.heading(key, text=title)
            self.node_tree.column(key, width=width, anchor="center")

        log_frame = ttk.LabelFrame(side_panel, text="Communication Logs", padding=8)
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.comm_log_listbox = tk.Listbox(log_frame, height=10, font=("Consolas", 9))
        self.comm_log_listbox.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.comm_log_listbox.yview
        )
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.comm_log_listbox.configure(yscrollcommand=log_scroll.set)

    def _rebuild_grid(self, cols: int, rows: int) -> None:
        for child in self.grid_inner.winfo_children():
            child.destroy()
        self.cell_widgets.clear()
        self.cell_render_cache.clear()
        for grid_y in range(rows):
            self.grid_inner.rowconfigure(grid_y, weight=1)
            for grid_x in range(cols):
                self.grid_inner.columnconfigure(grid_x, weight=1)
                cell_key = self.engine.cell_key(grid_x, grid_y)
                card = tk.Frame(
                    self.grid_inner,
                    bg="#dce3ea",
                    bd=1,
                    relief="solid",
                    padx=10,
                    pady=10,
                    width=180,
                    height=140,
                )
                card.grid(row=grid_y, column=grid_x, padx=6, pady=6, sticky="nsew")
                card.grid_propagate(False)
                title = tk.Label(
                    card,
                    text=f"Cell ({grid_x + 1}, {grid_y + 1})",
                    font=("Segoe UI", 11, "bold"),
                    bg=card["bg"],
                    fg="#16283a",
                )
                title.pack(anchor="w")
                probability = tk.Label(
                    card,
                    text="Not trained",
                    font=("Segoe UI", 11),
                    bg=card["bg"],
                    fg="#16283a",
                )
                probability.pack(anchor="w", pady=(8, 0))
                meta = tk.Label(
                    card,
                    text="Press Learn",
                    justify="left",
                    bg=card["bg"],
                    fg="#334b61",
                )
                meta.pack(anchor="w", pady=(8, 0))
                button_row = tk.Frame(card, bg=card["bg"])
                button_row.pack(fill="x", side="bottom", pady=(10, 0))
                learn = tk.Button(
                    button_row,
                    text="Learn",
                    command=lambda x=grid_x, y=grid_y: self.on_learn(x, y),
                )
                learn.pack(side="left", fill="x", expand=True)
                clear = tk.Button(
                    button_row,
                    text="Clear",
                    command=lambda x=grid_x, y=grid_y: self.on_clear_cell(x, y),
                )
                clear.pack(side="left", fill="x", expand=True, padx=(8, 0))
                self.cell_widgets[cell_key] = {
                    "card": card,
                    "title": title,
                    "probability": probability,
                    "meta": meta,
                    "button_row": button_row,
                    "learn": learn,
                    "clear": clear,
                }

    def on_apply_grid(self) -> None:
        snapshot = self.engine.snapshot()
        grid_changed = (
            self.cols_var.get() != snapshot["grid"]["cols"]
            or self.rows_var.get() != snapshot["grid"]["rows"]
        )
        window_changed = abs(
            self.window_var.get() - snapshot["grid"]["window_seconds"]
        ) > 1e-9
        window_step_changed = abs(
            self.window_step_var.get() - snapshot["grid"]["window_step_seconds"]
        ) > 1e-9
        if snapshot["training"]["trained_cells"] > 0 and (
            grid_changed or window_changed or window_step_changed
        ):
            okay = messagebox.askyesno(
                "Reset training data?",
                "Changing the grid, window size, or window step clears every learned cell and every trained model. Continue?",
            )
            if not okay:
                return
        self.engine.apply_grid_settings(
            self.cols_var.get(),
            self.rows_var.get(),
            self.capture_var.get(),
            self.window_var.get(),
            self.window_step_var.get(),
            self.keepalive_var.get(),
        )
        self._rebuild_grid(self.engine.grid_cols, self.engine.grid_rows)

    def on_clear_all(self) -> None:
        if messagebox.askyesno(
            "Clear all?",
            "Clear every learned cell dataset and all trained models?",
        ):
            self.engine.clear_all()

    def on_train_models(self) -> None:
        try:
            self.engine.train_models()
        except RuntimeError as exc:
            messagebox.showwarning("Training unavailable", str(exc))

    def on_learn(self, grid_x: int, grid_y: int) -> None:
        try:
            self.engine.start_capture(grid_x, grid_y)
        except RuntimeError as exc:
            messagebox.showwarning("Capture active", str(exc))

    def on_clear_cell(self, grid_x: int, grid_y: int) -> None:
        if messagebox.askyesno(
            "Clear cell?",
            f"Clear the dataset for cell ({grid_x + 1}, {grid_y + 1}) and retrain later?",
        ):
            self.engine.clear_cell(grid_x, grid_y)

    def on_select_model(self, _event: object | None = None) -> None:
        model_name = self.model_var.get().strip()
        if not model_name:
            return
        try:
            self.engine.set_active_model(model_name)
        except RuntimeError as exc:
            messagebox.showwarning("Model unavailable", str(exc))

    def _refresh(self) -> None:
        snapshot = self.engine.snapshot()
        self.cols_var.set(snapshot["grid"]["cols"])
        self.rows_var.set(snapshot["grid"]["rows"])
        self.capture_var.set(snapshot["grid"]["capture_seconds"])
        self.window_var.set(snapshot["grid"]["window_seconds"])
        self.window_step_var.set(snapshot["grid"]["window_step_seconds"])
        self.keepalive_var.set(snapshot["host"]["keepalive_pings_per_second"])
        self.model_var.set(snapshot["training"]["active_model"] or "")

        self.summary_var.set(
            f"Cells: {snapshot['training']['trained_cells']} / {snapshot['training']['total_cells']} | "
            f"Samples: {snapshot['training']['dataset_samples']} | "
            f"Models: {snapshot['training']['trained_model_count']} / 3 | "
            f"Nodes: {snapshot['training']['required_nodes']} | "
            f"Packets: {snapshot['metrics']['packet_count']}"
        )
        capture = snapshot["capture"]
        if capture["active"]:
            self.capture_status_var.set(
                f"Capturing ({capture['grid_x'] + 1}, {capture['grid_y'] + 1}) - "
                f"{capture['remaining_seconds']:.1f}s left"
            )
        else:
            self.capture_status_var.set("Capture idle")

        prediction = snapshot["prediction"]
        if prediction["ready"] and prediction["best_cell_key"]:
            best_x, best_y = [
                int(value) for value in prediction["best_cell_key"].split(",")
            ]
            self.prediction_var.set(
                f"{prediction['active_model']} | Best cell: ({best_x + 1}, {best_y + 1}) - "
                f"{prediction['best_probability'] * 100.0:.1f}%"
            )
        elif prediction["model_ready"]:
            self.prediction_var.set(
                f"{prediction['active_model']} ready. Waiting for a full live inference window."
            )
        elif snapshot["training"]["can_train"]:
            self.prediction_var.set("Learn data is ready. Click Train Models.")
        else:
            self.prediction_var.set(
                "Capture each cell with Learn, then click Train Models."
            )

        self.udp_var.set(
            f"{snapshot['udp_status']} | target IP for firmware/simulator: "
            f"{snapshot['host']['target_ip']}:{snapshot['host']['udp_port']} | "
            f"keepalive ping/s: {snapshot['host']['keepalive_pings_per_second']:.1f} | "
            f"config: {snapshot['host']['config_path']} | "
            f"log: {snapshot['comm_log_path']}"
        )
        self.message_var.set(snapshot["status_message"])
        self._refresh_model_controls(snapshot)

        if len(self.cell_widgets) != snapshot["grid"]["cols"] * snapshot["grid"]["rows"]:
            self._rebuild_grid(snapshot["grid"]["cols"], snapshot["grid"]["rows"])

        ready = snapshot["training"]["ready_for_inference"]
        capture_active = capture["active"]
        for cell in snapshot["cells"]:
            widgets = self.cell_widgets[cell["cell_key"]]
            background, foreground = self._cell_colors(cell, ready)
            probability_text = self._probability_text(cell, ready)
            meta_text = self._meta_text(cell)
            learn_state = "disabled" if capture_active else "normal"
            clear_state = (
                "normal" if cell["trained"] and not capture_active else "disabled"
            )
            render_state = {
                "background": background,
                "foreground": foreground,
                "probability_text": probability_text,
                "meta_text": meta_text,
                "learn_state": learn_state,
                "clear_state": clear_state,
            }
            cached_state = self.cell_render_cache.get(cell["cell_key"])

            if cached_state is None or cached_state["background"] != background:
                for key in ("card", "title", "probability", "meta", "button_row"):
                    widgets[key].configure(bg=background)
            if cached_state is None or cached_state["foreground"] != foreground:
                for key in ("title", "probability", "meta"):
                    widgets[key].configure(fg=foreground)
            if (
                cached_state is None
                or cached_state["probability_text"] != probability_text
            ):
                widgets["probability"].configure(text=probability_text)
            if cached_state is None or cached_state["meta_text"] != meta_text:
                widgets["meta"].configure(text=meta_text)
            if cached_state is None or cached_state["learn_state"] != learn_state:
                widgets["learn"].configure(state=learn_state)
            if cached_state is None or cached_state["clear_state"] != clear_state:
                widgets["clear"].configure(state=clear_state)

            self.cell_render_cache[cell["cell_key"]] = render_state

        self._refresh_nodes(snapshot["nodes"])
        self._refresh_logs(snapshot["comm_logs"])
        self.refresh_job = self.root.after(200, self._refresh)

    def _refresh_model_controls(self, snapshot: dict[str, object]) -> None:
        training = snapshot["training"]
        available_models = list(training["available_models"])
        selected_model = training["active_model"] or ""
        capture_active = bool(snapshot["capture"]["active"])

        if self.model_selector is not None:
            if tuple(self.model_selector.cget("values")) != tuple(available_models):
                self.model_selector.configure(values=available_models)
            if self.model_var.get() != selected_model:
                self.model_var.set(selected_model)
            selector_state = "readonly" if available_models and not capture_active else "disabled"
            if str(self.model_selector.cget("state")) != selector_state:
                self.model_selector.configure(state=selector_state)

        if self.train_button is not None:
            train_state = (
                "normal" if training["can_train"] and not capture_active else "disabled"
            )
            if str(self.train_button.cget("state")) != train_state:
                self.train_button.configure(state=train_state)

    def _refresh_nodes(self, nodes: list[dict[str, object]]) -> None:
        current = set(self.node_tree.get_children())
        needed = set()
        for node in nodes:
            item_id = str(node["node_id"])
            needed.add(item_id)
            values = (
                node["label"],
                "-" if node["age_ms"] is None else f"{node['age_ms']:.0f}",
                f"{node['rssi_dbm']:.1f}",
                node["window_samples"],
                node["source"],
            )
            if item_id in current:
                self.node_tree.item(item_id, values=values)
            else:
                self.node_tree.insert("", "end", iid=item_id, values=values)
        for item_id in current - needed:
            self.node_tree.delete(item_id)

    def _refresh_logs(self, logs: list[str]) -> None:
        if self.comm_log_listbox is None:
            return
        current_size = self.comm_log_listbox.size()
        if current_size == len(logs):
            return
        self.comm_log_listbox.delete(0, tk.END)
        for line in logs:
            self.comm_log_listbox.insert(tk.END, line)
        if logs:
            self.comm_log_listbox.yview_moveto(1.0)

    @staticmethod
    def _probability_text(cell: dict[str, object], ready: bool) -> str:
        if cell["is_capturing"]:
            return "Capturing..."
        if not cell["trained"]:
            return "Not trained"
        if ready:
            return f"Probability: {cell['probability'] * 100.0:.1f}%"
        return f"Samples: {cell['window_sample_count']}"

    @staticmethod
    def _meta_text(cell: dict[str, object]) -> str:
        if not cell["trained"]:
            return "Press Learn to collect windowed training samples."
        return (
            f"Nodes: {cell['node_count']}\n"
            f"Frames: {cell['total_frames']}\n"
            f"Windows: {cell['window_sample_count']}\n"
            f"Captures: {cell['capture_count']}"
        )

    @staticmethod
    def _cell_colors(cell: dict[str, object], ready: bool) -> tuple[str, str]:
        if cell["is_capturing"]:
            return "#ffe8a6", "#402800"
        if not cell["trained"]:
            return "#dce3ea", "#16283a"
        if ready:
            probability = max(0.0, min(1.0, float(cell["probability"])))
            probability = round(probability * 50.0) / 50.0
            if cell["is_best"]:
                red = 255
                green = int(230 - 90 * probability)
                blue = int(145 - 85 * probability)
            else:
                red = int(234 - 60 * probability)
                green = int(244 - 72 * probability)
                blue = int(240 - 145 * probability)
            foreground = "#201400" if probability >= 0.35 else "#16283a"
            return f"#{red:02x}{green:02x}{blue:02x}", foreground
        return "#cae7cf", "#16311d"

    def on_close(self) -> None:
        if self.refresh_job is not None:
            self.root.after_cancel(self.refresh_job)
            self.refresh_job = None
        self.receiver.stop()
        self.root.destroy()
