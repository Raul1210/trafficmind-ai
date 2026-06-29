"""
EmailAlertManager — monitors congestion readings and sends email alerts.

Lifecycle:
  - Created once by TrafficSystem.__init__()
  - check_and_alert() is called by DatabaseLogger after every successful DB write
    (~every 10 seconds, on the DatabaseLogger thread)
  - Actual email sending runs in a short-lived daemon thread so it never
    blocks the DatabaseLogger write loop
  - Settings are cached from SQLite with a 30-second TTL so API changes
    propagate within one read interval

Thresholds / state:
  - Counts consecutive readings where congestion >= threshold
  - Fires when count reaches consecutive_readings setting
  - Resets count and enforces cooldown_minutes after each send
"""
from __future__ import annotations

import smtplib
import sqlite3
import threading
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from traffic_monitor.logger import get_logger

logger = get_logger(__name__)

_SETTINGS_TTL = 30.0   # seconds between settings DB re-reads


class EmailAlertManager:
    """
    Non-blocking email alert manager hooked into the DatabaseLogger write cycle.

    Public methods:
        check_and_alert(congestion, vehicles, avg_conf, intensity, video_src)
        send_test_email() -> (ok: bool, error: Optional[str])
        invalidate_cache()
    """

    def __init__(self, db_path: str) -> None:
        self._db   = db_path
        self._lock = threading.Lock()

        self._consecutive: int   = 0     # readings above threshold in a row
        self._last_sent:   float = 0.0   # time.time() of last successful send

        self._cache:    Optional[dict] = None
        self._cache_ts: float          = 0.0

    # ── Public interface ──────────────────────────────────────────────────────

    def check_and_alert(
        self,
        congestion: float,
        vehicles:   int   = 0,
        avg_conf:   float = 0.0,
        intensity:  float = 0.0,
        video_src:  str   = '',
    ) -> None:
        """Called from DatabaseLogger thread after each write. Never blocks."""
        try:
            cfg = self._load_settings()
            if not cfg or not cfg.get('enabled'):
                with self._lock:
                    self._consecutive = 0
                return

            threshold = int(cfg.get('congestion_threshold', 85))
            needed    = int(cfg.get('consecutive_readings', 3))
            cooldown  = int(cfg.get('cooldown_minutes', 5)) * 60

            with self._lock:
                if congestion >= threshold:
                    self._consecutive += 1
                else:
                    self._consecutive = 0
                    return

                consec = self._consecutive
                since_last = time.time() - self._last_sent

            if consec < needed:
                return
            if since_last < cooldown:
                return

            # Conditions met — record the send time, reset counter, fire thread
            with self._lock:
                self._last_sent   = time.time()
                self._consecutive = 0

            stats = {
                'congestion': congestion,
                'vehicles':   vehicles,
                'avg_conf':   avg_conf,
                'intensity':  intensity,
                'duration':   needed * 10,   # ~10 s per DatabaseLogger cycle
                'video_src':  video_src or '—',
            }
            threading.Thread(
                target=self._do_send,
                args=(cfg, stats),
                daemon=True,
                name='EmailAlertSend',
            ).start()

        except Exception as e:
            logger.error(f'EmailAlertManager.check_and_alert: {e}')

    def send_test_email(self) -> tuple[bool, Optional[str]]:
        """
        Synchronous test send — called from the API route.
        Returns (success, error_message).
        """
        try:
            conn = sqlite3.connect(self._db, timeout=5)
            conn.row_factory = sqlite3.Row
            row  = conn.execute(
                'SELECT * FROM email_settings WHERE id = 1'
            ).fetchone()
            conn.close()
        except Exception as e:
            return False, f'DB read error: {e}'

        if not row:
            return False, 'No email settings found'

        cfg = dict(row)
        recipients = [
            r.strip()
            for r in (cfg.get('recipient_emails') or '').split(',')
            if r.strip()
        ]
        if not recipients:
            return False, 'No recipient email addresses configured'

        subject  = 'TrafficMind AI — Test Email (Configuration Verified)'
        html_body = self._build_test_html()
        ok, err  = self._smtp_send(cfg, recipients, subject, html_body)
        self._log_alert(
            recipients=', '.join(recipients),
            congestion=0.0,
            duration_s=0,
            subject='[TEST] ' + subject,
            status='sent' if ok else 'failed',
            error_msg=err,
        )
        return ok, err

    def invalidate_cache(self) -> None:
        """Force settings reload on next check (call after API update)."""
        with self._lock:
            self._cache    = None
            self._cache_ts = 0.0

    # ── Settings cache ────────────────────────────────────────────────────────

    def _load_settings(self) -> Optional[dict]:
        now = time.time()
        with self._lock:
            if self._cache and (now - self._cache_ts) < _SETTINGS_TTL:
                return self._cache
        try:
            conn = sqlite3.connect(self._db, timeout=5)
            conn.row_factory = sqlite3.Row
            row  = conn.execute(
                'SELECT * FROM email_settings WHERE id = 1'
            ).fetchone()
            conn.close()
            if row:
                cfg = dict(row)
                with self._lock:
                    self._cache    = cfg
                    self._cache_ts = now
                return cfg
        except Exception as e:
            logger.warning(f'EmailAlertManager settings load error: {e}')
        return None

    # ── Email building & sending ──────────────────────────────────────────────

    def _do_send(self, cfg: dict, stats: dict) -> None:
        """Build alert email and send; log result. Runs in daemon thread."""
        recipients = [
            r.strip()
            for r in (cfg.get('recipient_emails') or '').split(',')
            if r.strip()
        ]
        if not recipients:
            return

        threshold = int(cfg.get('congestion_threshold', 85))
        subject   = (
            f'⚠️ TrafficMind AI — '
            f"High Congestion Alert ({stats['congestion']:.0f}%)"
        )
        html_body = self._build_alert_html(threshold, stats)
        ok, err   = self._smtp_send(cfg, recipients, subject, html_body)

        self._log_alert(
            recipients=', '.join(recipients),
            congestion=stats['congestion'],
            duration_s=stats['duration'],
            subject=subject,
            status='sent' if ok else 'failed',
            error_msg=err,
        )
        if ok:
            logger.info(
                f"Email alert sent to {recipients} "
                f"— congestion {stats['congestion']:.1f}%"
            )
        else:
            logger.error(f'Email alert failed: {err}')

    def _smtp_send(
        self,
        cfg:        dict,
        recipients: list[str],
        subject:    str,
        html_body:  str,
    ) -> tuple[bool, Optional[str]]:
        """Low-level SMTP send. Returns (success, error_msg)."""
        try:
            msg             = MIMEMultipart('alternative')
            msg['Subject']  = subject
            msg['From']     = cfg.get('sender_email', '')
            msg['To']       = ', '.join(recipients)
            msg.attach(MIMEText(html_body, 'html'))

            smtp_host = cfg.get('smtp_server', '')
            smtp_port = int(cfg.get('smtp_port', 587))
            username  = cfg.get('smtp_username', '')
            password  = cfg.get('smtp_password', '')
            sender    = cfg.get('sender_email', '')

            if smtp_port == 465:
                server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30)
            else:
                server = smtplib.SMTP(smtp_host, smtp_port, timeout=30)
                server.ehlo()
                server.starttls()
                server.ehlo()

            if username and password:
                server.login(username, password)

            server.sendmail(sender, recipients, msg.as_string())
            server.quit()
            return True, None

        except Exception as e:
            return False, str(e)

    # ── HTML builders ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_alert_html(threshold: int, stats: dict) -> str:
        ts        = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cong      = stats['congestion']
        duration  = stats['duration']
        vehicles  = stats['vehicles']
        intensity = stats['intensity']
        src       = stats.get('video_src', '—')

        return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>TrafficMind AI Alert</title></head>
<body style="margin:0;padding:20px;background:#f0f4f8;font-family:Arial,Helvetica,sans-serif">
<div style="max-width:580px;margin:0 auto;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.1)">

  <div style="background:#0a0e1a;padding:22px 28px;border-top:4px solid #00c8f0">
    <p style="color:#00c8f0;margin:0;font-size:20px;font-weight:700;letter-spacing:2px">TrafficMind AI</p>
    <p style="color:#94a3b8;margin:4px 0 0;font-size:11px;letter-spacing:1px;text-transform:uppercase">
      Real-Time Intelligent Traffic Monitoring &amp; Optimization
    </p>
  </div>

  <div style="padding:24px 28px">
    <div style="background:#fff3cd;border:1px solid #ffc107;border-radius:6px;padding:14px 16px;margin-bottom:20px">
      <p style="color:#856404;margin:0;font-size:15px;font-weight:700">&#9888; High Congestion Alert</p>
      <p style="color:#856404;margin:8px 0 0;font-size:13px">
        Congestion has exceeded <strong>{threshold}%</strong> for the last
        <strong>{duration}&nbsp;seconds</strong>.
      </p>
    </div>

    <p style="color:#1e293b;font-size:13px;font-weight:700;margin:0 0 10px">Current Traffic Statistics</p>
    <table style="width:100%;border-collapse:collapse;font-size:12px;border:1px solid #e2e8f0">
      <tr style="background:#f8fafc">
        <td style="padding:9px 14px;color:#64748b;font-weight:600;border-bottom:1px solid #e2e8f0;width:45%">Congestion Level</td>
        <td style="padding:9px 14px;color:#ff4444;font-weight:700;border-bottom:1px solid #e2e8f0">{cong:.1f}%</td>
      </tr>
      <tr>
        <td style="padding:9px 14px;color:#64748b;font-weight:600;border-bottom:1px solid #e2e8f0">Vehicles Detected</td>
        <td style="padding:9px 14px;color:#1e293b;border-bottom:1px solid #e2e8f0">{vehicles}</td>
      </tr>
      <tr style="background:#f8fafc">
        <td style="padding:9px 14px;color:#64748b;font-weight:600;border-bottom:1px solid #e2e8f0">Traffic Intensity</td>
        <td style="padding:9px 14px;color:#1e293b;border-bottom:1px solid #e2e8f0">{intensity:.2f}&nbsp;v/s</td>
      </tr>
      <tr>
        <td style="padding:9px 14px;color:#64748b;font-weight:600;border-bottom:1px solid #e2e8f0">Alert Generated</td>
        <td style="padding:9px 14px;color:#1e293b;border-bottom:1px solid #e2e8f0">{ts}</td>
      </tr>
      <tr style="background:#f8fafc">
        <td style="padding:9px 14px;color:#64748b;font-weight:600">Video Source</td>
        <td style="padding:9px 14px;color:#1e293b">{src}</td>
      </tr>
    </table>
  </div>

  <div style="background:#f8fafc;border-top:1px solid #e2e8f0;padding:14px 28px">
    <p style="color:#94a3b8;font-size:10px;margin:0;line-height:1.5">
      This is an automated alert from TrafficMind AI.<br>
      To modify alert settings, visit the admin dashboard and open the AI&nbsp;Alerts tab.
    </p>
  </div>

</div>
</body>
</html>"""

    @staticmethod
    def _build_test_html() -> str:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>TrafficMind AI Test</title></head>
<body style="margin:0;padding:20px;background:#f0f4f8;font-family:Arial,Helvetica,sans-serif">
<div style="max-width:580px;margin:0 auto;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.1)">

  <div style="background:#0a0e1a;padding:22px 28px;border-top:4px solid #00c8f0">
    <p style="color:#00c8f0;margin:0;font-size:20px;font-weight:700;letter-spacing:2px">TrafficMind AI</p>
    <p style="color:#94a3b8;margin:4px 0 0;font-size:11px;letter-spacing:1px;text-transform:uppercase">
      Real-Time Intelligent Traffic Monitoring &amp; Optimization
    </p>
  </div>

  <div style="padding:24px 28px">
    <div style="background:#d1fae5;border:1px solid #34d399;border-radius:6px;padding:14px 16px;margin-bottom:20px">
      <p style="color:#065f46;margin:0;font-size:15px;font-weight:700">&#10003; Configuration Test Successful</p>
      <p style="color:#065f46;margin:8px 0 0;font-size:13px">
        Your SMTP configuration is working correctly. Email alerts will be delivered to this address.
      </p>
    </div>
    <p style="color:#64748b;font-size:12px;margin:0">Test sent at: <strong>{ts}</strong></p>
  </div>

  <div style="background:#f8fafc;border-top:1px solid #e2e8f0;padding:14px 28px">
    <p style="color:#94a3b8;font-size:10px;margin:0;line-height:1.5">
      This is an automated test from TrafficMind AI.<br>
      To modify alert settings, visit the admin dashboard and open the AI&nbsp;Alerts tab.
    </p>
  </div>

</div>
</body>
</html>"""

    # ── DB logging ────────────────────────────────────────────────────────────

    def _log_alert(
        self,
        recipients:  str,
        congestion:  float,
        duration_s:  int,
        subject:     str,
        status:      str,
        error_msg:   Optional[str],
    ) -> None:
        try:
            conn = sqlite3.connect(self._db, timeout=5)
            conn.execute(
                'INSERT INTO email_alert_history '
                '(recipient_emails, congestion_value, duration_seconds, '
                ' subject, status, error_message) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (recipients, round(congestion, 2), duration_s,
                 subject, status, error_msg),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f'EmailAlertManager._log_alert error: {e}')
