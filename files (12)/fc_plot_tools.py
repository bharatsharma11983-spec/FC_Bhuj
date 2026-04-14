#!/usr/bin/env python3
"""
fc_plot_tools.py — Reusable plot style editor, CSV exporter for Bhuj SFF GUI.
Provides:
  - PlotStyleEditor: popup to edit line colour/style/width/marker/label for any set of lines
  - add_style_toolbar: adds "Style…" and "Save CSV" buttons below any Matplotlib canvas
"""
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox
import numpy as np, csv, os

# ── Matplotlib line style / marker options ────────────────────────────────────
LINESTYLES = [('-','Solid'),('--','Dashed'),('-.',  'Dash-dot'),(':',' Dotted'),('none','None')]
MARKERS    = [('none','None'),('o','Circle'),('s','Square'),('D','Diamond'),
              ('^','Triangle up'),('v','Triangle dn'),('*','Star'),
              ('+','Plus'),('x','Cross'),('|','Vbar')]

class PlotStyleEditor(tk.Toplevel):
    """
    Modal dialog to edit line appearance (colour, style, width, marker,
    label visibility) for each line in a Matplotlib axes.
    Callback `on_apply(lines_cfg)` is called on Apply/OK.
    lines_cfg: list of dicts with keys: label, color, ls, lw, marker, visible
    """
    def __init__(self, parent, ax, canvas, on_apply=None):
        super().__init__(parent)
        self.title('Plot Style Editor')
        self.configure(bg='#1E2A3A')
        self.resizable(True, True)
        self.ax = ax; self.canvas = canvas; self.on_apply = on_apply
        self.lines = [l for l in ax.get_lines() if l.get_label() and not l.get_label().startswith('_')]
        self._rows = []
        self._build()

    def _build(self):
        tk.Label(self, text='Plot Style Editor — Edit Lines, Labels, Colours',
                 bg='#1A3A5C', fg='#64B5F6', font=('Helvetica',10,'bold'), pady=6).pack(fill=tk.X)

        hdr = tk.Frame(self, bg='#1A3A5C')
        hdr.pack(fill=tk.X, padx=6, pady=2)
        for i,(txt,w) in enumerate([('Label',18),('Colour',6),('Style',10),('Width',5),('Marker',10),('Visible',6)]):
            tk.Label(hdr, text=txt, bg='#1A3A5C', fg='white', font=('Helvetica',8,'bold'),
                     width=w).grid(row=0, column=i, padx=4)

        scr_f = tk.Frame(self, bg='#1E2A3A')
        scr_f.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        canvas_inner = tk.Canvas(scr_f, bg='#1E2A3A', highlightthickness=0)
        sb = tk.Scrollbar(scr_f, orient='vertical', command=canvas_inner.yview)
        canvas_inner.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_inner.pack(fill=tk.BOTH, expand=True)
        inner = tk.Frame(canvas_inner, bg='#1E2A3A')
        canvas_inner.create_window((0,0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas_inner.configure(scrollregion=canvas_inner.bbox('all')))

        for r, line in enumerate(self.lines):
            row = {}
            bg = '#0D1B2A' if r%2==0 else '#1E2A3A'
            f = tk.Frame(inner, bg=bg); f.pack(fill=tk.X, pady=1)

            # Label
            row['label'] = tk.StringVar(value=line.get_label())
            tk.Entry(f, textvariable=row['label'], bg='#0D2A3A', fg='#E3F2FD',
                     font=('Helvetica',8), width=20, relief=tk.FLAT).grid(row=0, column=0, padx=4, pady=2)

            # Colour button
            row['color'] = line.get_color()
            def make_color_btn(fr, row_dict, line_obj):
                btn = tk.Button(fr, bg=row_dict['color'], width=4,
                                relief=tk.RAISED, borderwidth=2,
                                command=lambda rd=row_dict, ln=line_obj: self._pick_color(rd, ln))
                btn.grid(row=0, column=1, padx=4)
                row_dict['_color_btn'] = btn
            make_color_btn(f, row, line)

            # Line style
            ls_opts = [ls for ls,_ in LINESTYLES]
            row['ls'] = tk.StringVar(value=line.get_linestyle())
            ls_cbo = ttk.Combobox(f, textvariable=row['ls'], values=ls_opts, width=8, state='readonly')
            ls_cbo.grid(row=0, column=2, padx=4)

            # Line width
            row['lw'] = tk.StringVar(value=str(round(line.get_linewidth(),1)))
            tk.Entry(f, textvariable=row['lw'], bg='#0D2A3A', fg='#E3F2FD',
                     font=('Helvetica',8), width=5, relief=tk.FLAT).grid(row=0, column=3, padx=4)

            # Marker
            mk_opts = [mk for mk,_ in MARKERS]
            cur_mk = line.get_marker() or 'none'
            row['marker'] = tk.StringVar(value=cur_mk if cur_mk else 'none')
            mk_cbo = ttk.Combobox(f, textvariable=row['marker'], values=mk_opts, width=9, state='readonly')
            mk_cbo.grid(row=0, column=4, padx=4)

            # Visible
            row['visible'] = tk.BooleanVar(value=line.get_visible())
            tk.Checkbutton(f, variable=row['visible'], bg=bg, selectcolor='#1E2A3A',
                           activebackground=bg).grid(row=0, column=5, padx=4)

            self._rows.append(row)

        # Buttons
        btn_f = tk.Frame(self, bg='#1E2A3A')
        btn_f.pack(fill=tk.X, padx=6, pady=6)
        tk.Button(btn_f, text='✓ Apply', command=self._apply,
                   bg='#27AE60', fg='white', font=('Helvetica',9,'bold'),
                   relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_f, text='✗ Close', command=self.destroy,
                   bg='#7F8C8D', fg='white', font=('Helvetica',9),
                   relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_f, text='Reset Labels', command=self._reset_labels,
                   bg='#1565C0', fg='white', font=('Helvetica',9),
                   relief=tk.FLAT, padx=8).pack(side=tk.RIGHT, padx=4)

        self.geometry('680x400')

    def _pick_color(self, row_dict, line_obj):
        c = colorchooser.askcolor(color=row_dict['color'], parent=self)[1]
        if c:
            row_dict['color'] = c
            row_dict['_color_btn'].configure(bg=c)

    def _reset_labels(self):
        for row, line in zip(self._rows, self.lines):
            row['label'].set(line.get_label())

    def _apply(self):
        for row, line in zip(self._rows, self.lines):
            try:
                line.set_label(row['label'].get())
                line.set_color(row['color'])
                ls = row['ls'].get(); line.set_linestyle(ls if ls!='none' else 'None')
                lw = float(row['lw'].get()); line.set_linewidth(lw)
                mk = row['marker'].get(); line.set_marker(None if mk=='none' else mk)
                line.set_visible(row['visible'].get())
            except: pass
        self.ax.legend(fontsize=8)
        self.canvas.draw()
        if self.on_apply: self.on_apply()


def save_csv_dialog(parent, ax, default_name='plot_data.csv'):
    """Export all visible line data from ax to CSV."""
    fp = filedialog.asksaveasfilename(
        title='Save Plot Data as CSV', defaultextension='.csv',
        filetypes=[('CSV','*.csv'),('All','*.*')],
        initialfile=default_name, parent=parent)
    if not fp: return
    lines = [l for l in ax.get_lines()
             if l.get_label() and not l.get_label().startswith('_') and l.get_visible()]
    if not lines:
        messagebox.showinfo('No data','No visible lines to export.', parent=parent); return
    # Build column-wise data
    headers = []; data_cols = []
    for ln in lines:
        xd, yd = ln.get_xdata(), ln.get_ydata()
        headers.append(f'X_{ln.get_label()[:20]}')
        headers.append(f'Y_{ln.get_label()[:20]}')
        data_cols.append(xd); data_cols.append(yd)
    max_rows = max(len(c) for c in data_cols)
    try:
        with open(fp,'w', newline='') as f:
            w = csv.writer(f); w.writerow(headers)
            for i in range(max_rows):
                row = [c[i] if i < len(c) else '' for c in data_cols]
                w.writerow(row)
        messagebox.showinfo('Saved', f'CSV saved:\n{fp}', parent=parent)
    except Exception as e:
        messagebox.showerror('Error', str(e), parent=parent)


def add_style_toolbar(parent, fig, canvas, ax_getter, csv_name='data.csv'):
    """
    Add a toolbar row below a canvas with:
      [Style Editor] [Save CSV] [Save Figure]
    ax_getter: callable returning the current primary axes.
    """
    tb = tk.Frame(parent, bg='#2C3E50'); tb.pack(fill=tk.X, pady=1)
    tk.Button(tb, text='🎨 Style', font=('Helvetica',8,'bold'),
               bg='#8E44AD', fg='white', relief=tk.FLAT, padx=6,
               command=lambda: PlotStyleEditor(parent, ax_getter(), canvas)).pack(side=tk.LEFT, padx=4)
    tk.Button(tb, text='💾 Save CSV', font=('Helvetica',8,'bold'),
               bg='#D35400', fg='white', relief=tk.FLAT, padx=6,
               command=lambda: save_csv_dialog(parent, ax_getter(), csv_name)).pack(side=tk.LEFT, padx=2)
    tk.Button(tb, text='🖼 Save Fig', font=('Helvetica',8,'bold'),
               bg='#1565C0', fg='white', relief=tk.FLAT, padx=6,
               command=lambda: _save_fig(parent, fig)).pack(side=tk.LEFT, padx=2)
    return tb

def _save_fig(parent, fig):
    fp = filedialog.asksaveasfilename(
        title='Save Figure', defaultextension='.png',
        filetypes=[('PNG','*.png'),('PDF','*.pdf'),('SVG','*.svg'),('All','*.*')],
        parent=parent)
    if fp:
        try: fig.savefig(fp, dpi=200, bbox_inches='tight'); messagebox.showinfo('Saved',fp,parent=parent)
        except Exception as e: messagebox.showerror('Error',str(e),parent=parent)
