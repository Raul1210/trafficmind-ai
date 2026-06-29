"""
PDF session report generator — ReportLab (Platypus) + matplotlib (Agg).

Key design choices:
  - Uses Figure + FigureCanvasAgg directly (no global pyplot state, thread-safe).
  - All data is fetched from TrafficSystem / SQLite — no HTTP calls.
  - _NumberedCanvas enables 'Page X of Y' footers via a two-pass render.
  - Handles empty DB, no alerts, freshly started system gracefully.
"""
from __future__ import annotations

import io
import json
import sqlite3
import time
from datetime import datetime
from typing import Optional

# matplotlib: use the non-interactive Agg backend via Figure directly.
# This avoids touching global pyplot state (ultralytics may have already
# imported matplotlib with a different backend).
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    Image, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.pdfgen import canvas as rl_canvas

from traffic_monitor.logger import get_logger

logger = get_logger(__name__)

# ── Design tokens ─────────────────────────────────────────────────────────────
_DARK    = HexColor('#0a0e1a')
_DARK2   = HexColor('#1e293b')
_CYAN    = HexColor('#00c8f0')
_AMBER   = HexColor('#ffb300')
_RED     = HexColor('#ff4444')
_GREEN   = HexColor('#00cc66')
_MUTED   = HexColor('#64748b')
_TEXT    = HexColor('#1e293b')
_ROW_A   = HexColor('#f0f4f8')
_ROW_B   = HexColor('#ffffff')
_BORDER  = HexColor('#cbd5e1')

PAGE_W, PAGE_H = A4
_ML = _MR = 2.0 * cm
_MT = 2.8 * cm
_MB = 1.8 * cm
_CW = PAGE_W - _ML - _MR   # usable content width ≈ 17 cm


# ── Two-pass canvas for 'Page X of Y' ────────────────────────────────────────

class _NumberedCanvas(rl_canvas.Canvas):
    """
    Defers actual page output until save() so every page can be stamped with
    the final total-page count.  Page 1 (cover) gets no header/footer.
    """

    def __init__(self, *args, **kwargs):
        rl_canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved: list[dict] = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._draw_chrome(total)
            rl_canvas.Canvas.showPage(self)
        rl_canvas.Canvas.save(self)

    def _draw_chrome(self, total: int) -> None:
        if self._pageNumber == 1:
            return
        self.saveState()

        # Header bar
        self.setFillColor(_DARK)
        self.rect(0, PAGE_H - 20 * mm, PAGE_W, 20 * mm, fill=1, stroke=0)
        self.setFillColor(_CYAN)
        self.rect(0, PAGE_H - 20 * mm - 1.5, PAGE_W, 1.5, fill=1, stroke=0)
        self.setFont('Helvetica-Bold', 8.5)
        self.setFillColor(white)
        self.drawString(_ML, PAGE_H - 12 * mm, 'TrafficMind AI  ·  Session Report')
        self.setFont('Helvetica', 8)
        self.setFillColor(_CYAN)
        self.drawRightString(PAGE_W - _MR, PAGE_H - 12 * mm,
                             datetime.now().strftime('%Y-%m-%d'))

        # Footer bar
        self.setFillColor(_DARK)
        self.rect(0, 0, PAGE_W, 13 * mm, fill=1, stroke=0)
        self.setFillColor(_CYAN)
        self.rect(0, 13 * mm, PAGE_W, 1, fill=1, stroke=0)
        self.setFont('Helvetica', 7.5)
        self.setFillColor(white)
        self.drawString(
            _ML, 4.5 * mm,
            'TrafficMind AI — Real-Time Intelligent Traffic Monitoring & Optimization',
        )
        self.setFillColor(_CYAN)
        self.drawRightString(PAGE_W - _MR, 4.5 * mm,
                             f'Page {self._pageNumber} of {total}')
        self.restoreState()


# ── Paragraph styles (created once at module level) ───────────────────────────

_S: dict[str, ParagraphStyle] = {
    'cover_title': ParagraphStyle(
        'tm_cvt', fontName='Helvetica-Bold', fontSize=30,
        textColor=_CYAN, alignment=TA_CENTER, leading=36,
    ),
    'cover_sub': ParagraphStyle(
        'tm_cvs', fontName='Helvetica', fontSize=14,
        textColor=white, alignment=TA_CENTER, leading=20,
    ),
    'section_hdr': ParagraphStyle(
        'tm_sh', fontName='Helvetica-Bold', fontSize=11,
        textColor=white, leading=14,
    ),
    'body': ParagraphStyle(
        'tm_b', fontName='Helvetica', fontSize=9.5,
        textColor=_TEXT, leading=14, spaceAfter=6,
    ),
    'caption': ParagraphStyle(
        'tm_cap', fontName='Helvetica', fontSize=8,
        textColor=_MUTED, alignment=TA_CENTER, leading=11,
    ),
    'big_num': ParagraphStyle(
        'tm_bn', fontName='Helvetica-Bold', fontSize=20,
        textColor=_CYAN, alignment=TA_CENTER, leading=24,
    ),
    'big_lbl': ParagraphStyle(
        'tm_bl', fontName='Helvetica', fontSize=7.5,
        textColor=_MUTED, alignment=TA_CENTER, leading=10,
    ),
    'tbl_hdr': ParagraphStyle(
        'tm_th', fontName='Helvetica-Bold', fontSize=9,
        textColor=white, leading=12,
    ),
    'meta_key': ParagraphStyle(
        'tm_mk', fontName='Helvetica-Bold', fontSize=10,
        textColor=_MUTED, leading=14,
    ),
    'meta_val': ParagraphStyle(
        'tm_mv', fontName='Helvetica', fontSize=10,
        textColor=_TEXT, leading=14,
    ),
}


# ── Main generator ────────────────────────────────────────────────────────────

class PDFReportGenerator:
    """
    Generates a comprehensive PDF session report for TrafficMind AI.

    Constructor args:
        system     — TrafficSystem instance
        db_path    — absolute path to traffic_logs.db
        username   — current admin username (shown on cover)
        cfg        — Settings instance
        start_time — time.time() recorded when the system started
    """

    def __init__(self, system, db_path: str, username: str, cfg, start_time: float):
        self._sys  = system
        self._db   = db_path
        self._user = username
        self._cfg  = cfg
        self._t0   = start_time

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> io.BytesIO:
        """Build the full PDF and return it as a seeked-to-0 BytesIO buffer."""
        buf = io.BytesIO()

        # --- 1. Collect all data (minimise lock hold time) ---
        rows      = self._fetch_rows()
        last50    = rows[-50:]
        stats     = self._compute_stats(rows)
        smoothed, avg_conf, _ = self._sys.capture.get_snapshot()
        video_src = str(self._cfg.video_source)
        benchmark = self._load_benchmark()
        session_s = time.time() - self._t0

        with self._sys.rl_lock:
            rl_state  = dict(self._sys.rl_state)
            q_vals    = list(rl_state.get('q_values', [0.0, 0.0, 0.0, 0.0]))
            cur_phase = int(rl_state.get('phase', 0))

        # --- 2. Build charts (all thread-safe, no pyplot globals) ---
        img_cong   = self._chart_congestion(rows)
        img_mix    = self._chart_mix(smoothed)
        img_intens = self._chart_intensity(rows)
        img_qvals  = self._chart_qvals(q_vals, cur_phase)

        # --- 3. Assemble story ---
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=_ML, rightMargin=_MR,
            topMargin=_MT, bottomMargin=_MB,
        )
        story: list = []
        story += self._cover(stats, video_src, session_s)
        story += self._sec1_summary(stats, smoothed, avg_conf)
        story += self._sec2_analytics(img_cong, img_mix, img_intens)
        story += self._sec3_rl(rl_state, q_vals, cur_phase, img_qvals, benchmark)
        story += self._sec4_alerts(rows)
        story += self._sec5_snapshot(last50)

        doc.build(story, canvasmaker=_NumberedCanvas)
        buf.seek(0)
        return buf

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _fetch_rows(self) -> list[dict]:
        try:
            conn = sqlite3.connect(self._db, timeout=10)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM traffic_data ORDER BY id'
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f'PDF: DB read failed: {e}')
            return []

    def _load_benchmark(self) -> dict:
        try:
            with open(str(self._cfg.benchmark_results_path)) as f:
                return json.load(f)
        except Exception:
            return {}

    def _compute_stats(self, rows: list[dict]) -> dict:
        if not rows:
            return {
                'count': 0, 'avg_cong': 0.0, 'peak_cong': 0.0,
                'peak_ts': '—', 'total_veh': 0, 'avg_intens': 0.0,
            }
        congs  = [r.get('congestion_percentage', 0.0) for r in rows]
        vehs   = [r.get('vehicles_count', 0) for r in rows]
        intens = [r.get('traffic_intensity', 0.0) for r in rows]
        peak_i = int(np.argmax(congs))
        return {
            'count':      len(rows),
            'avg_cong':   round(float(np.mean(congs)), 1),
            'peak_cong':  round(float(max(congs)), 1),
            'peak_ts':    rows[peak_i].get('timestamp', '—'),
            'total_veh':  int(sum(vehs)),
            'avg_intens': round(float(np.mean(intens)), 2),
        }

    # ── Chart builders ────────────────────────────────────────────────────────

    @staticmethod
    def _new_fig(figsize=(9.5, 3.5)) -> tuple:
        """Create an isolated Figure + Axes using the Agg backend."""
        fig = Figure(figsize=figsize, facecolor='#f8fafc')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        return fig, ax

    @staticmethod
    def _style_ax(ax, ylabel: str = '') -> None:
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#475569', labelsize=7)
        if ylabel:
            ax.set_ylabel(ylabel, color='#475569', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e2e8f0')
        ax.spines['bottom'].set_color('#e2e8f0')
        ax.yaxis.grid(True, color='#f1f5f9', linewidth=0.8)
        ax.set_axisbelow(True)

    @staticmethod
    def _fig_to_image(fig: Figure, w_cm: float, h_cm: float) -> Optional[Image]:
        try:
            buf = io.BytesIO()
            fig.tight_layout(pad=0.6)
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
            buf.seek(0)
            return Image(buf, width=w_cm * cm, height=h_cm * cm)
        except Exception as e:
            logger.warning(f'Chart render error: {e}')
            return None

    def _chart_congestion(self, rows: list[dict]) -> Optional[Image]:
        try:
            vals = [r.get('congestion_percentage', 0.0) for r in rows]
            if not vals:
                return None
            fig, ax = self._new_fig((9.5, 3.5))
            x = range(len(vals))
            ax.fill_between(x, vals, alpha=0.12, color='#00c8f0')
            ax.plot(vals, color='#00c8f0', linewidth=1.6, zorder=3)
            ax.axhline(65, color='#ffb300', linewidth=0.9, linestyle='--',
                       label='Warning 65%', alpha=0.8)
            ax.axhline(85, color='#ff4444', linewidth=0.9, linestyle='--',
                       label='Critical 85%', alpha=0.8)
            ax.set_ylim(0, 105)
            ax.set_title('Congestion Over Time (%)', fontsize=9, color='#1e293b',
                         fontweight='bold', pad=6)
            ax.legend(fontsize=7, framealpha=0.6)
            self._style_ax(ax, 'Congestion %')
            if len(rows) > 1:
                step = max(1, len(rows) // 6)
                idxs = list(range(0, len(rows), step))
                ax.set_xticks(idxs)
                ax.set_xticklabels(
                    [rows[i].get('timestamp', '')[:16] for i in idxs],
                    rotation=20, ha='right', fontsize=6,
                )
            return self._fig_to_image(fig, 15.5, 5.5)
        except Exception as e:
            logger.warning(f'Congestion chart error: {e}')
            return None

    def _chart_mix(self, smoothed: dict) -> Optional[Image]:
        try:
            cars  = smoothed.get('car', 0)
            motos = smoothed.get('motorbike', 0)
            peds  = smoothed.get('person', 0)
            total = cars + motos + peds
            if total == 0:
                cars, motos, peds = 1, 1, 1
            fig, ax = self._new_fig((4.0, 3.5))
            wedges, texts, autotexts = ax.pie(
                [cars, motos, peds],
                labels=[f'Cars\n{cars}', f'Motorbikes\n{motos}', f'Pedestrians\n{peds}'],
                colors=['#00c8f0', '#ffb300', '#b464ff'],
                autopct='%1.0f%%', startangle=90,
                pctdistance=0.72, labeldistance=1.14,
                wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
            )
            for t in texts + autotexts:
                t.set_fontsize(7)
                t.set_color('#1e293b')
            ax.add_patch(__import__('matplotlib.patches', fromlist=['Circle'])
                         .Circle((0, 0), 0.44, fc='#f8fafc'))
            ax.set_title('Vehicle Mix', fontsize=9, color='#1e293b',
                         fontweight='bold', pad=4)
            return self._fig_to_image(fig, 6.5, 5.5)
        except Exception as e:
            logger.warning(f'Mix chart error: {e}')
            return None

    def _chart_intensity(self, rows: list[dict]) -> Optional[Image]:
        try:
            vals = [r.get('traffic_intensity', 0.0) for r in rows]
            if not vals:
                return None
            fig, ax = self._new_fig((8.0, 3.5))
            ax.bar(range(len(vals)), vals, color='#ffb300', width=0.8, alpha=0.85)
            ax.set_title('Traffic Intensity (v/s)', fontsize=9, color='#1e293b',
                         fontweight='bold', pad=6)
            self._style_ax(ax, 'v/s')
            if len(rows) > 1:
                step = max(1, len(rows) // 5)
                idxs = list(range(0, len(rows), step))
                ax.set_xticks(idxs)
                ax.set_xticklabels(
                    [rows[i].get('timestamp', '')[:16] for i in idxs],
                    rotation=20, ha='right', fontsize=6,
                )
            return self._fig_to_image(fig, 8.5, 5.5)
        except Exception as e:
            logger.warning(f'Intensity chart error: {e}')
            return None

    def _chart_qvals(self, q_vals: list, active: int) -> Optional[Image]:
        try:
            labels = ['Phase 0\nN-S Green', 'Phase 1\nE-W Green',
                      'Phase 2\nN-S Left',  'Phase 3\nE-W Left']
            bar_colors = ['#00c8f0' if i == active else '#cbd5e1'
                          for i in range(4)]
            fig, ax = self._new_fig((7.5, 3.5))
            bars = ax.bar(range(4), q_vals, color=bar_colors, width=0.55)
            ax.bar_label(bars, fmt='%.2f', fontsize=7, color='#475569', padding=3)
            ax.set_xticks(range(4))
            ax.set_xticklabels(labels, fontsize=7, color='#475569')
            ax.set_title('Q-Values per Signal Phase (active = cyan)',
                         fontsize=9, color='#1e293b', fontweight='bold', pad=6)
            self._style_ax(ax, 'Q-Value')
            return self._fig_to_image(fig, 8.5, 5.5)
        except Exception as e:
            logger.warning(f'Q-values chart error: {e}')
            return None

    # ── Layout helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _section_bar(title: str) -> Table:
        """Dark header bar for a section."""
        t = Table(
            [[Paragraph(title, _S['section_hdr'])]],
            colWidths=[_CW],
        )
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), _DARK2),
            ('TOPPADDING',    (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING',   (0, 0), (-1, -1), 12),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
            ('LINEABOVE',     (0, 0), (-1, 0), 3, _CYAN),
        ]))
        return t

    @staticmethod
    def _data_table(data: list[list], col_widths: list[float],
                    header_rows: int = 1,
                    extra_commands: list | None = None) -> Table:
        """Generic styled data table."""
        t = Table(data, colWidths=col_widths, repeatRows=header_rows)
        commands = [
            ('BACKGROUND',     (0, 0), (-1, header_rows - 1), _DARK2),
            ('TEXTCOLOR',      (0, 0), (-1, header_rows - 1), white),
            ('FONTNAME',       (0, 0), (-1, header_rows - 1), 'Helvetica-Bold'),
            ('FONTNAME',       (0, header_rows), (-1, -1), 'Helvetica'),
            ('FONTSIZE',       (0, 0), (-1, -1), 8.5),
            ('ROWBACKGROUNDS', (0, header_rows), (-1, -1), [_ROW_A, _ROW_B]),
            ('TOPPADDING',     (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING',  (0, 0), (-1, -1), 5),
            ('LEFTPADDING',    (0, 0), (-1, -1), 7),
            ('RIGHTPADDING',   (0, 0), (-1, -1), 7),
            ('BOX',            (0, 0), (-1, -1), 0.5, _BORDER),
            ('INNERGRID',      (0, 0), (-1, -1), 0.3, _BORDER),
        ]
        if extra_commands:
            commands.extend(extra_commands)
        t.setStyle(TableStyle(commands))
        return t

    # ── Story sections ────────────────────────────────────────────────────────

    def _cover(self, stats: dict, video_src: str, session_s: float) -> list:
        h, m = int(session_s // 3600), int((session_s % 3600) // 60)
        s_rem = int(session_s % 60)
        dur = f'{h}h {m}m {s_rem}s' if h > 0 else f'{m}m {s_rem}s'
        src_display = video_src if len(video_src) <= 55 else '…' + video_src[-52:]

        # Title banner
        banner = Table(
            [[Paragraph('TrafficMind AI', _S['cover_title'])],
             [Paragraph('Traffic Monitoring Session Report', _S['cover_sub'])]],
            colWidths=[_CW],
        )
        banner.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), _DARK),
            ('TOPPADDING',    (0, 0), (0, 0), 32),
            ('BOTTOMPADDING', (0, 0), (0, 0), 10),
            ('TOPPADDING',    (0, 1), (0, 1), 8),
            ('BOTTOMPADDING', (0, 1), (0, 1), 32),
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('LINEABOVE',     (0, 0), (-1, 0), 5, _CYAN),
        ]))

        # Cyan separator
        sep = Table([['']], colWidths=[_CW])
        sep.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), _CYAN),
            ('TOPPADDING',    (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))

        # Metadata
        meta = [
            [Paragraph('Generated at',      _S['meta_key']),
             Paragraph(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), _S['meta_val'])],
            [Paragraph('Generated by',      _S['meta_key']),
             Paragraph(self._user,           _S['meta_val'])],
            [Paragraph('Video source',       _S['meta_key']),
             Paragraph(src_display,          _S['meta_val'])],
            [Paragraph('Session duration',   _S['meta_key']),
             Paragraph(dur,                  _S['meta_val'])],
            [Paragraph('Database records',   _S['meta_key']),
             Paragraph(str(stats['count']),  _S['meta_val'])],
        ]
        meta_t = Table(meta, colWidths=[5 * cm, _CW - 5 * cm])
        meta_t.setStyle(TableStyle([
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING',   (0, 0), (-1, -1), 0),
            ('LINEBELOW',     (0, 0), (-1, -2), 0.5, _BORDER),
            ('LINEBELOW',     (0, -1), (-1, -1), 0.5, _BORDER),
        ]))

        return [
            Spacer(1, 28 * mm),
            banner,
            Spacer(1, 4 * mm),
            sep,
            Spacer(1, 8 * mm),
            meta_t,
            PageBreak(),
        ]

    def _sec1_summary(self, stats: dict, smoothed: dict,
                       avg_conf: float) -> list:
        story = [self._section_bar('1  —  Executive Summary'), Spacer(1, 5 * mm)]

        # 6 KPI cells in 2 rows of 3
        def kpi_cell(val: str, label: str, color: str = '#00c8f0') -> Table:
            inner = Table(
                [[Paragraph(f'<font face="Helvetica-Bold" size="18" '
                             f'color="{color}">{val}</font>', _S['big_num'])],
                 [Paragraph(label, _S['big_lbl'])]],
                colWidths=[_CW / 3 - 4 * mm],
            )
            inner.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, -1), _ROW_A),
                ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
                ('TOPPADDING',    (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('BOX',           (0, 0), (-1, -1), 0.5, _BORDER),
                ('LINEABOVE',     (0, 0), (-1, 0), 3, HexColor(color)),
            ]))
            return inner

        kw = _CW / 3
        row1 = Table(
            [[kpi_cell(str(stats['count']),    'Total Records',   '#00c8f0'),
              kpi_cell(f"{stats['avg_cong']}%", 'Avg Congestion', '#ffb300'),
              kpi_cell(f"{stats['peak_cong']}%",'Peak Congestion','#ff4444')]],
            colWidths=[kw, kw, kw],
        )
        row1.setStyle(TableStyle([
            ('LEFTPADDING',   (0, 0), (-1, -1), 2),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 2),
            ('TOPPADDING',    (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        row2 = Table(
            [[kpi_cell(str(stats['total_veh']),          'Total Vehicles',    '#00c8f0'),
              kpi_cell(f"{stats['avg_intens']} v/s",     'Avg Intensity',     '#ffb300'),
              kpi_cell(f"{round(avg_conf * 100, 1)}%",   'Avg Confidence',    '#00cc66')]],
            colWidths=[kw, kw, kw],
        )
        row2.setStyle(TableStyle([
            ('LEFTPADDING',   (0, 0), (-1, -1), 2),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 2),
            ('TOPPADDING',    (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story += [row1, row2, Spacer(1, 6 * mm)]

        # Auto-generated summary paragraph
        n, v = stats['count'], stats['total_veh']
        ac, pc, pt = stats['avg_cong'], stats['peak_cong'], stats['peak_ts']
        trend = ('above normal' if ac > 50 else ('moderate' if ac > 25 else 'light'))
        summary_style = ParagraphStyle(
            'tm_sum', fontName='Helvetica', fontSize=9.5, textColor=_TEXT,
            leading=15, spaceAfter=8, backColor=_ROW_A,
            borderPad=10, borderColor=_BORDER, borderWidth=0.5,
            leftIndent=8, rightIndent=8,
        )
        summary = (
            f'During this monitoring session the system recorded <b>{n} snapshots</b> '
            f'tracking a total of <b>{v} vehicles</b>. '
            f'Average congestion was <b>{ac}%</b> ({trend} traffic conditions). '
            f'Peak congestion of <b>{pc}%</b> was recorded at <b>{pt}</b>. '
            f'Average detection confidence: <b>{round(avg_conf * 100, 1)}%</b>.'
        )
        story += [Paragraph(summary, summary_style), Spacer(1, 4 * mm)]
        return story

    def _sec2_analytics(self, img_cong: Optional[Image],
                         img_mix: Optional[Image],
                         img_intens: Optional[Image]) -> list:
        story = [self._section_bar('2  —  Traffic Analytics'), Spacer(1, 5 * mm)]

        if img_cong:
            story += [
                img_cong,
                Paragraph(
                    'Figure 1 — Congestion percentage over session duration. '
                    'Dashed lines mark warning (65%) and critical (85%) thresholds.',
                    _S['caption'],
                ),
                Spacer(1, 5 * mm),
            ]
        else:
            story.append(Paragraph('No congestion data available yet.', _S['body']))

        if img_mix and img_intens:
            side = Table([[img_mix, img_intens]], colWidths=[7 * cm, 8.5 * cm])
            side.setStyle(TableStyle([
                ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING',   (0, 0), (-1, -1), 0),
                ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
                ('TOPPADDING',    (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ]))
            story += [
                side,
                Paragraph(
                    'Figure 2 — Current vehicle mix distribution (left)  '
                    '·  Figure 3 — Traffic intensity (v/s) over time (right)',
                    _S['caption'],
                ),
            ]
        elif img_mix:
            story += [img_mix, Paragraph('Figure 2 — Vehicle mix', _S['caption'])]
        elif img_intens:
            story += [img_intens, Paragraph('Figure 3 — Traffic intensity', _S['caption'])]

        story.append(Spacer(1, 4 * mm))
        return story

    def _sec3_rl(self, rl: dict, q_vals: list, cur_phase: int,
                  img_qvals: Optional[Image], benchmark: dict) -> list:
        from traffic_monitor.agent.q_learning import PHASE_NAMES
        story = [self._section_bar('3  —  RL Optimizer Performance'),
                 Spacer(1, 5 * mm)]

        phase_name = PHASE_NAMES.get(cur_phase, '—')
        state_rows = [
            ['Active Phase',         f'Phase {cur_phase}: {phase_name}'],
            ['Phase Duration',        f"{rl.get('phase_duration', 0)} steps"],
            ['Exploration Rate (ε)', str(rl.get('epsilon', '—'))],
            ['States Explored',       str(rl.get('states_explored', 0))],
            ['Last Reward',           str(rl.get('reward', 0.0))],
        ]
        st = Table(state_rows, colWidths=[6 * cm, _CW - 6 * cm])
        st.setStyle(TableStyle([
            ('FONTNAME',       (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME',       (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE',       (0, 0), (-1, -1), 9),
            ('TEXTCOLOR',      (0, 0), (0, -1), _MUTED),
            ('TEXTCOLOR',      (1, 0), (1, -1), _TEXT),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [_ROW_A, _ROW_B]),
            ('TOPPADDING',     (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING',  (0, 0), (-1, -1), 5),
            ('LEFTPADDING',    (0, 0), (-1, -1), 8),
            ('BOX',            (0, 0), (-1, -1), 0.5, _BORDER),
            ('INNERGRID',      (0, 0), (-1, -1), 0.3, _BORDER),
        ]))
        story += [st, Spacer(1, 5 * mm)]

        if img_qvals:
            story += [
                img_qvals,
                Paragraph(
                    'Figure 4 — Q-values per signal phase. '
                    'The active phase is highlighted in cyan.',
                    _S['caption'],
                ),
                Spacer(1, 5 * mm),
            ]

        if benchmark:
            bm_hdr = ['Scenario', 'Fixed-Time Avg Wait', 'Q-Learning Avg Wait',
                      'Improvement']
            bm_data = [bm_hdr]
            for scen in ('low', 'medium', 'peak'):
                r = benchmark.get(scen, {})
                if not r:
                    continue
                imp = r.get('wait_improvement_pct', 0)
                bm_data.append([
                    scen.capitalize(),
                    f"{r.get('fixed_avg_wait', '—')}s",
                    f"{r.get('rl_avg_wait', '—')}s",
                    f'+{imp}% faster' if imp > 0 else f'{imp}%',
                ])
            bm_extra = []
            for i in range(1, len(bm_data)):
                bm_extra += [
                    ('TEXTCOLOR', (3, i), (3, i), _GREEN),
                    ('FONTNAME',  (3, i), (3, i), 'Helvetica-Bold'),
                ]
            bm_t = self._data_table(bm_data,
                                    [3.5 * cm, 4.5 * cm, 4.5 * cm, 4.0 * cm],
                                    extra_commands=bm_extra)
            story += [
                Paragraph(
                    'Benchmark: Q-Learning vs Fixed-Time Controller',
                    ParagraphStyle('tm_bh', fontName='Helvetica-Bold', fontSize=9.5,
                                   textColor=_TEXT, spaceBefore=4, spaceAfter=4),
                ),
                bm_t,
            ]
        else:
            story.append(Paragraph(
                'Benchmark data not available. '
                'Run: python -m traffic_monitor.agent.q_learning --mode benchmark',
                _S['body'],
            ))

        story.append(Spacer(1, 4 * mm))
        return story

    def _sec4_alerts(self, rows: list[dict]) -> list:
        story = [self._section_bar('4  —  Alert History'), Spacer(1, 5 * mm)]

        # Derive alerts from traffic_data records (congestion thresholds)
        alerts = []
        for r in rows:
            c = r.get('congestion_percentage', 0)
            ts = r.get('timestamp', '')
            if c >= 85:
                alerts.append({
                    'ts':    ts,
                    'level': 'CRITICAL',
                    'msg':   f"High congestion: {c:.0f}% — "
                             f"{r.get('vehicles_count', 0)} vehicles detected",
                    'color': _RED,
                })
            elif c >= 65:
                alerts.append({
                    'ts':    ts,
                    'level': 'WARNING',
                    'msg':   f"Elevated traffic density: {c:.0f}% congestion",
                    'color': _AMBER,
                })

        alerts.sort(key=lambda x: x['ts'], reverse=True)
        alerts = alerts[:20]

        crit = sum(1 for a in alerts if a['level'] == 'CRITICAL')
        warn = len(alerts) - crit
        story.append(Paragraph(
            f'Total alerts in session: <b>{len(alerts)}</b>  '
            f'·  Critical: <b>{crit}</b>  '
            f'·  Warnings: <b>{warn}</b>',
            _S['body'],
        ))
        story.append(Spacer(1, 3 * mm))

        if not alerts:
            story.append(Paragraph(
                'No alert events recorded. Traffic conditions were normal throughout '
                'this session (congestion remained below 65%).',
                _S['body'],
            ))
        else:
            hdr = ['Timestamp', 'Level', 'Description']
            data = [hdr] + [[a['ts'], a['level'], a['msg']] for a in alerts]
            col_w = [4.5 * cm, 2.5 * cm, _CW - 7.0 * cm]
            al_extra = []
            for i, a in enumerate(alerts, start=1):
                al_extra += [
                    ('TEXTCOLOR', (1, i), (1, i), a['color']),
                    ('FONTNAME',  (1, i), (1, i), 'Helvetica-Bold'),
                    ('ALIGN',     (1, i), (1, i), 'CENTER'),
                ]
            al_t = self._data_table(data, col_w, extra_commands=al_extra)
            story.append(al_t)

        story.append(Spacer(1, 4 * mm))
        return story

    def _sec5_snapshot(self, last50: list[dict]) -> list:
        story = [self._section_bar('5  —  Database Snapshot (Last 50 Records)'),
                 Spacer(1, 5 * mm)]

        if not last50:
            story.append(Paragraph('No database records available.', _S['body']))
            return story

        hdr = ['ID', 'Timestamp', 'Vehicles', 'Confidence', 'Intensity', 'Congestion']
        data = [hdr]
        for r in reversed(last50):
            cong = r.get('congestion_percentage', 0)
            conf = r.get('avg_confidence', 0)
            data.append([
                str(r.get('id', '')),
                str(r.get('timestamp', '')),
                str(r.get('vehicles_count', '')),
                f'{round(conf * 100, 1)}%',
                f"{r.get('traffic_intensity', 0):.2f}",
                f'{cong:.1f}%',
            ])

        # Column widths totalling _CW
        col_w = [1.4 * cm, 4.8 * cm, 2.0 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm]
        remaining = _CW - sum(col_w)
        col_w[1] += remaining   # extra space goes to the timestamp column

        snap_extra = [
            ('FONTSIZE',       (0, 0), (-1, -1), 8),
            ('TOPPADDING',     (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING',  (0, 0), (-1, -1), 4),
        ]
        for i, r in enumerate(reversed(last50), start=1):
            cong = r.get('congestion_percentage', 0)
            col = _RED if cong >= 80 else (_AMBER if cong >= 50 else _GREEN)
            snap_extra += [
                ('TEXTCOLOR', (5, i), (5, i), col),
                ('FONTNAME',  (5, i), (5, i), 'Helvetica-Bold'),
            ]
        snap_t = self._data_table(data, col_w, header_rows=1, extra_commands=snap_extra)
        story.append(snap_t)
        return story
