from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .raw_capture import RawCaptureEngine
from .receiver import UdpReceiverGroup


class RawCaptureAppWindow:
    def __init__(self, engine: RawCaptureEngine, receiver: UdpReceiverGroup) -> None:
        self.engine = engine
        self.receiver = receiver
        self.root = tk.Tk()
        self.root.title("ESP32 CSI Raw Capture App")
        self.root.geometry("1320x860")
        self.root.minsize(1040, 680)
        self.root.configure(bg="#edf1f5")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cols_var = tk.IntVar(value=self.engine.grid_cols)
        self.rows_var = tk.IntVar(value=self.engine.grid_rows)
        self.capture_var = tk.DoubleVar(value=self.engine.capture_seconds)
        self.empty_room_delay_var = tk.DoubleVar(value=self.engine.empty_room_delay_seconds)
        self.summary_var = tk.StringVar()
        self.capture_status_var = tk.StringVar()
        self.udp_var = tk.StringVar()
        self.message_var = tk.StringVar()
        self.path_var = tk.StringVar()
        self.cell_widgets: dict[str, dict[str, tk.Widget]] = {}
        self.cell_render_cache: dict[str, dict[str, object]] = {}
        self.apply_button: ttk.Button | None = None
        self.empty_room_button: ttk.Button | None = None
        self.clear_raw_button: ttk.Button | None = None
        self.stop_button: ttk.Button | None = None
        self.comm_log_listbox: tk.Listbox | None = None
        self.refresh_job: str | None = None
        self.settings_inputs: tuple[tk.Widget, ...] = ()

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

        controls = ttk.LabelFrame(outer, text="Raw Capture Controls", padding=12)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for column in range(12):
            controls.columnconfigure(column, weight=1)

        ttk.Label(controls, text="Grid X").grid(row=0, column=0, sticky="w")
        cols_spinbox = ttk.Spinbox(controls, from_=1, to=20, textvariable=self.cols_var, width=6)
        cols_spinbox.grid(row=0, column=1, sticky="w")
        ttk.Label(controls, text="Grid Y").grid(row=0, column=2, sticky="w")
        rows_spinbox = ttk.Spinbox(controls, from_=1, to=20, textvariable=self.rows_var, width=6)
        rows_spinbox.grid(row=0, column=3, sticky="w")
        ttk.Label(controls, text="Duration sec").grid(row=0, column=4, sticky="w")
        capture_spinbox = ttk.Spinbox(
            controls,
            from_=0,
            to=3600,
            increment=1,
            textvariable=self.capture_var,
            width=8,
        )
        capture_spinbox.grid(row=0, column=5, sticky="w")
        ttk.Label(controls, text="Empty Delay sec").grid(row=0, column=6, sticky="w")
        empty_room_delay_spinbox = ttk.Spinbox(
            controls,
            from_=0,
            to=3600,
            increment=1,
            textvariable=self.empty_room_delay_var,
            width=8,
        )
        empty_room_delay_spinbox.grid(row=0, column=7, sticky="w")
        self.settings_inputs = (
            cols_spinbox,
            rows_spinbox,
            capture_spinbox,
            empty_room_delay_spinbox,
        )

        self.apply_button = ttk.Button(
            controls,
            text="Apply To Config",
            command=self.on_apply_grid,
        )
        self.apply_button.grid(row=0, column=8, padx=(8, 0), sticky="ew")
        self.empty_room_button = ttk.Button(
            controls,
            text="Empty Room",
            command=self.on_empty_room,
        )
        self.empty_room_button.grid(row=0, column=9, padx=(8, 0), sticky="ew")
        self.clear_raw_button = ttk.Button(
            controls,
            text="Clear Raw",
            command=self.on_clear_raw_data,
        )
        self.clear_raw_button.grid(row=0, column=10, padx=(8, 0), sticky="ew")
        self.stop_button = ttk.Button(
            controls,
            text="Stop",
            command=self.on_stop_capture,
        )
        self.stop_button.grid(row=0, column=11, padx=(8, 0), sticky="ew")

        ttk.Label(controls, textvariable=self.summary_var).grid(
            row=1, column=0, columnspan=5, sticky="w", pady=(10, 0)
        )
        ttk.Label(controls, textvariable=self.capture_status_var).grid(
            row=1, column=5, columnspan=7, sticky="w", pady=(10, 0)
        )
        ttk.Label(controls, textvariable=self.udp_var).grid(
            row=2, column=0, columnspan=12, sticky="w", pady=(8, 0)
        )
        ttk.Label(controls, textvariable=self.path_var).grid(
            row=3, column=0, columnspan=12, sticky="w", pady=(8, 0)
        )
        ttk.Label(
            controls,
            textvariable=self.message_var,
            foreground="#23435f",
        ).grid(row=4, column=0, columnspan=12, sticky="w", pady=(8, 0))

        grid_panel = ttk.LabelFrame(outer, text="Capture Grid", padding=8)
        grid_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        grid_panel.columnconfigure(0, weight=1)
        grid_panel.rowconfigure(0, weight=1)

        self.grid_canvas = tk.Canvas(grid_panel, bg="#edf1f5", highlightthickness=0)
        self.grid_canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(grid_panel, orient="vertical", command=self.grid_canvas.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(grid_panel, orient="horizontal", command=self.grid_canvas.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.grid_canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.grid_inner = tk.Frame(self.grid_canvas, bg="#edf1f5")
        self.grid_window = self.grid_canvas.create_window((0, 0), window=self.grid_inner, anchor="nw")
        self.grid_inner.bind(
            "<Configure>",
            lambda _event: self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox("all")),
        )
        self.grid_canvas.bind(
            "<Configure>",
            lambda event: self.grid_canvas.itemconfigure(self.grid_window, width=max(400, event.width)),
        )

        side_panel = ttk.LabelFrame(outer, text="ESP Telemetry", padding=10)
        side_panel.grid(row=1, column=1, sticky="nsew")
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(0, weight=1)
        side_panel.rowconfigure(1, weight=1)

        self.node_tree = ttk.Treeview(
            side_panel,
            columns=("node", "age", "packets", "rssi", "subcarriers", "source"),
            show="headings",
            height=18,
        )
        self.node_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(side_panel, orient="vertical", command=self.node_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.node_tree.configure(yscrollcommand=tree_scroll.set)
        for key, title, width in (
            ("node", "Node", 70),
            ("age", "Age ms", 80),
            ("packets", "Packets", 80),
            ("rssi", "RSSI", 70),
            ("subcarriers", "Subcarriers", 90),
            ("source", "Source", 160),
        ):
            self.node_tree.heading(key, text=title)
            self.node_tree.column(key, width=width, anchor="center")

        log_frame = ttk.LabelFrame(side_panel, text="Raw Capture Logs", padding=8)
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.comm_log_listbox = tk.Listbox(log_frame, height=10, font=("Consolas", 9))
        self.comm_log_listbox.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.comm_log_listbox.yview)
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
                    bg="#dfe7ee",
                    bd=1,
                    relief="solid",
                    padx=10,
                    pady=10,
                    width=180,
                    height=130,
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
                status = tk.Label(
                    card,
                    text="Ready",
                    font=("Segoe UI", 11),
                    bg=card["bg"],
                    fg="#16283a",
                )
                status.pack(anchor="w", pady=(8, 0))
                meta = tk.Label(
                    card,
                    text="Raw JSONL",
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
                self.cell_widgets[cell_key] = {
                    "card": card,
                    "title": title,
                    "status": status,
                    "meta": meta,
                    "button_row": button_row,
                    "learn": learn,
                }

    def on_apply_grid(self) -> None:
        try:
            cols = self.cols_var.get()
            rows = self.rows_var.get()
            capture = self.capture_var.get()
            empty_room_delay = self.empty_room_delay_var.get()
        except tk.TclError:
            messagebox.showwarning("Invalid input", "Please enter valid numeric values.")
            return
        try:
            self.engine.apply_grid_settings(cols, rows, capture, empty_room_delay)
        except RuntimeError as exc:
            messagebox.showwarning("Config unavailable", str(exc))
            return
        self._rebuild_grid(self.engine.grid_cols, self.engine.grid_rows)

    def on_empty_room(self) -> None:
        snapshot = self.engine.snapshot()
        if snapshot["capture"]["active"]:
            self.engine.stop_capture()
            return
        try:
            self.engine.start_empty_room_capture()
        except RuntimeError as exc:
            messagebox.showwarning("Capture active", str(exc))

    def on_learn(self, grid_x: int, grid_y: int) -> None:
        snapshot = self.engine.snapshot()
        capture = snapshot["capture"]
        if (
            capture["active"]
            and capture["kind"] == "cell"
            and capture["grid_x"] == grid_x
            and capture["grid_y"] == grid_y
        ):
            self.engine.stop_capture()
            return
        try:
            self.engine.start_cell_capture(grid_x, grid_y)
        except RuntimeError as exc:
            messagebox.showwarning("Capture active", str(exc))

    def on_clear_raw_data(self) -> None:
        snapshot = self.engine.snapshot()
        if snapshot["capture"]["active"]:
            messagebox.showwarning("Capture active", "Stop the active raw capture before clearing raw data.")
            return
        if not messagebox.askyesno(
            "Clear Raw Data",
            "Delete all files in the raw_data folder?",
        ):
            return
        try:
            deleted_count = self.engine.clear_raw_data()
        except RuntimeError as exc:
            messagebox.showwarning("Capture active", str(exc))
            return
        messagebox.showinfo("Clear Raw Data", f"Deleted {deleted_count} raw data files.")

    def on_stop_capture(self) -> None:
        self.engine.stop_capture()

    def _refresh(self) -> None:
        snapshot = self.engine.snapshot()
        if (
            not self._settings_edit_in_progress()
            and not self._settings_have_pending_edits(snapshot)
        ):
            self.cols_var.set(snapshot["grid"]["cols"])
            self.rows_var.set(snapshot["grid"]["rows"])
            self.capture_var.set(snapshot["grid"]["capture_seconds"])
            self.empty_room_delay_var.set(snapshot["grid"]["empty_room_delay_seconds"])

        duration = snapshot["grid"]["capture_seconds"]
        duration_text = "manual stop" if duration <= 0 else f"{duration:.1f}s"
        self.summary_var.set(
            f"Grid: {snapshot['grid']['cols']}x{snapshot['grid']['rows']} | "
            f"Duration: {duration_text} | "
            f"Empty delay: {snapshot['grid']['empty_room_delay_seconds']:.1f}s | "
            f"Packets seen: {snapshot['metrics']['packet_count']} | "
            f"Saved: {snapshot['metrics']['saved_packet_count']} | "
            f"Active nodes: {snapshot['metrics']['active_nodes']}"
        )
        capture = snapshot["capture"]
        if capture["active"]:
            if capture.get("pending"):
                time_text = f"starts in {capture['delay_remaining_seconds']:.1f}s"
            else:
                remaining = capture["remaining_seconds"]
                if remaining is None:
                    time_text = f"{capture['elapsed_seconds']:.1f}s elapsed"
                else:
                    time_text = f"{remaining:.1f}s left"
            self.capture_status_var.set(
                f"Recording {capture['label']} - {time_text} - "
                f"{capture['saved_packets']} packets ({capture['valid_packets']} valid)"
            )
            self.path_var.set(f"Current file: {capture['path']}")
        else:
            self.capture_status_var.set("Raw capture idle")
            self.path_var.set(
                f"Raw data: {snapshot['raw_data_dir']} | index: {snapshot['session_index_path']}"
            )

        self.udp_var.set(
            f"{snapshot['udp_status']} | firmware/simulator target: "
            f"{snapshot['host']['target_ip']} | ports: "
            f"{self._format_node_ports(snapshot['host']['node_ports'])} | "
            f"config: {snapshot['host']['config_path']} | "
            f"log: {snapshot['comm_log_path']}"
        )
        self.message_var.set(snapshot["status_message"])
        self._refresh_control_states(snapshot)

        if len(self.cell_widgets) != snapshot["grid"]["cols"] * snapshot["grid"]["rows"]:
            self._rebuild_grid(snapshot["grid"]["cols"], snapshot["grid"]["rows"])

        capture_active = capture["active"]
        for cell in snapshot["cells"]:
            widgets = self.cell_widgets[cell["cell_key"]]
            is_capturing = bool(cell["is_capturing"])
            background = "#d3ecdf" if is_capturing else "#dfe7ee"
            foreground = "#12301e" if is_capturing else "#16283a"
            status_text = "Recording" if is_capturing else "Ready"
            meta_text = (
                f"{capture['saved_packets']} packets"
                if is_capturing
                else "Press Learn"
            )
            button_text = "Stop" if is_capturing else "Learn"
            button_state = "normal" if (not capture_active or is_capturing) else "disabled"
            render_state = {
                "background": background,
                "foreground": foreground,
                "status_text": status_text,
                "meta_text": meta_text,
                "button_text": button_text,
                "button_state": button_state,
            }
            cached_state = self.cell_render_cache.get(cell["cell_key"])
            if cached_state is None or cached_state["background"] != background:
                for key in ("card", "title", "status", "meta", "button_row"):
                    widgets[key].configure(bg=background)
            if cached_state is None or cached_state["foreground"] != foreground:
                for key in ("title", "status", "meta"):
                    widgets[key].configure(fg=foreground)
            if cached_state is None or cached_state["status_text"] != status_text:
                widgets["status"].configure(text=status_text)
            if cached_state is None or cached_state["meta_text"] != meta_text:
                widgets["meta"].configure(text=meta_text)
            if cached_state is None or cached_state["button_text"] != button_text:
                widgets["learn"].configure(text=button_text)
            if cached_state is None or cached_state["button_state"] != button_state:
                widgets["learn"].configure(state=button_state)
            self.cell_render_cache[cell["cell_key"]] = render_state

        self._refresh_nodes(snapshot["nodes"])
        self._refresh_logs(snapshot["comm_logs"])
        self.refresh_job = self.root.after(250, self._refresh)

    def _refresh_control_states(self, snapshot: dict[str, object]) -> None:
        capture_active = bool(snapshot["capture"]["active"])
        if self.apply_button is not None:
            self.apply_button.configure(state="disabled" if capture_active else "normal")
        if self.empty_room_button is not None:
            if capture_active and snapshot["capture"]["kind"] == "empty_room":
                self.empty_room_button.configure(text="Stop Empty Room", state="normal")
            else:
                self.empty_room_button.configure(
                    text="Empty Room",
                    state="disabled" if capture_active else "normal",
                )
        if self.clear_raw_button is not None:
            self.clear_raw_button.configure(state="disabled" if capture_active else "normal")
        if self.stop_button is not None:
            self.stop_button.configure(state="normal" if capture_active else "disabled")

    def _refresh_nodes(self, nodes: list[dict[str, object]]) -> None:
        current = set(self.node_tree.get_children())
        needed = set()
        for node in nodes:
            item_id = str(node["node_id"])
            needed.add(item_id)
            values = (
                node["label"],
                "-" if node["age_ms"] is None else f"{node['age_ms']:.0f}",
                node["packets_received"],
                f"{node['rssi_dbm']:.1f}",
                node["subcarrier_count"],
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
        if current_size > len(logs):
            self.comm_log_listbox.delete(0, tk.END)
            for line in logs:
                self.comm_log_listbox.insert(tk.END, line)
        else:
            for line in logs[current_size:]:
                self.comm_log_listbox.insert(tk.END, line)
        self.comm_log_listbox.see(tk.END)

    @staticmethod
    def _format_node_ports(node_ports: list[dict[str, int]]) -> str:
        if not node_ports:
            return "-"
        return ", ".join(f"{item['node_id']}:{item['port']}" for item in node_ports)

    def _settings_edit_in_progress(self) -> bool:
        focused = self.root.focus_get()
        return focused in self.settings_inputs

    def _settings_have_pending_edits(self, snapshot: dict[str, object]) -> bool:
        try:
            return (
                self.cols_var.get() != snapshot["grid"]["cols"]
                or self.rows_var.get() != snapshot["grid"]["rows"]
                or abs(self.capture_var.get() - snapshot["grid"]["capture_seconds"]) > 1e-9
                or abs(
                    self.empty_room_delay_var.get()
                    - snapshot["grid"]["empty_room_delay_seconds"]
                )
                > 1e-9
            )
        except tk.TclError:
            return True

    def on_close(self) -> None:
        if self.refresh_job is not None:
            self.root.after_cancel(self.refresh_job)
            self.refresh_job = None
        self.engine.stop_capture()
        self.root.destroy()
