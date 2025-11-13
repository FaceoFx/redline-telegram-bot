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
import sys
import traceback
import signal
from http.server import BaseHTTPRequestHandler, HTTPServer
try:
    from aiohttp import web
    HAS_AIOHTTP = True
except Exception:
    HAS_AIOHTTP = False
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

# ============================================
# HELPER UTILITIES (Progress Tracking & Caching)
# ============================================

class ProgressTracker:
    """Real-time progress updates for long operations"""
    
    def __init__(self, message, total: int, operation: str = "Processing"):
        self.message = message
        self.total = total
        self.current = 0
        self.operation = operation
        self.last_update = 0
        self.update_interval = 2.0  # Update every 2 seconds
        self.start_time = time.time()
        
    async def update(self, current: int, force: bool = False):
        """Update progress (throttled to avoid API rate limits)"""
        self.current = current
        now = time.time()
        
        # Only update if enough time passed or forced
        if not force and (now - self.last_update) < self.update_interval:
            return
            
        self.last_update = now
        percent = (current / self.total) * 100 if self.total > 0 else 0
        
        # Progress bar (20 chars)
        filled = int(percent / 5)
        bar = '█' * filled + '░' * (20 - filled)
        
        # ETA calculation
        elapsed = now - self.start_time
        if current > 0:
            rate = current / elapsed
            remaining = (self.total - current) / rate if rate > 0 else 0
            eta = f"{int(remaining)}s" if remaining < 60 else f"{int(remaining/60)}m"
        else:
            eta = "calculating..."
        
        try:
            await self.message.edit_text(
                f"⏳ <b>{self.operation}</b>\n\n"
                f"[{bar}] {percent:.1f}%\n\n"
                f"📊 {current:,} / {self.total:,} items\n"
                f"⏱️ ETA: {eta}",
                parse_mode='HTML'
            )
        except Exception:
            # Ignore rate limit errors on progress updates
            pass
    
    async def complete(self, result_summary: str = ""):
        """Mark operation as complete"""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed / 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        
        try:
            await self.message.edit_text(
                f"✅ <b>{self.operation} Complete!</b>\n\n"
                f"📊 Processed: {self.total:,} items\n"
                f"⏱️ Time: {time_str}\n"
                f"{result_summary}",
                parse_mode='HTML'
            )
        except Exception:
            pass


class SimpleCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key: str):
        """Get cached value if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value):
        """Cache value with current timestamp"""
        self.cache[key] = (value, time.time())
    
    def clear_expired(self):
        """Remove expired entries"""
        now = time.time()
        expired = [k for k, (_, ts) in self.cache.items() if now - ts > self.ttl]
        for k in expired:
            del self.cache[k]


def format_number(num: int) -> str:
    """Format large numbers with K/M suffixes"""
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num/1000:.1f}K"
    else:
        return f"{num/1000000:.1f}M"


def format_file_size(size_bytes: int) -> str:
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def count_lines_efficient(file_path: str, max_lines: int = None) -> int:
    """Count lines in file efficiently without loading into memory"""
    count = 0
    with open(file_path, 'rb') as f:
        for _ in f:
            count += 1
            if max_lines and count > max_lines:
                return count
    return count


def read_file_chunked(file_path: str, max_lines: int = None):
    """Generator to read large files line by line (memory efficient)"""
    count = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if max_lines and count >= max_lines:
                break
            yield line
            count += 1

# ============================================
# BOT CONFIGURATION
# ============================================

# Read sensitive values from environment variables
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN env var is required")

# Koyeb public URL for keep-alive (prevents auto-sleep)
# Set this to your Koyeb app URL (e.g., https://your-app-name.koyeb.app)
# If not set, will fall back to internal pings (less effective)
KOYEB_PUBLIC_URL = os.environ.get("KOYEB_PUBLIC_URL", "").strip()

# Webhook mode (more stable than polling on Koyeb)
# Set to 'true' to enable webhook mode (recommended for production)
USE_WEBHOOK = os.environ.get("USE_WEBHOOK", "true").lower() == "true"  # Default to true for better stability
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "").strip()  # Optional security token
# Strict webhook behavior: if true and webhook fails, exit so platform restarts the instance
# Values: 'true' | 'false' | 'auto' (auto => true when KOYEB_PUBLIC_URL is set)
_strict_env = os.environ.get("KOYEB_STRICT_WEBHOOK", "auto").strip().lower()
if _strict_env == "true":
    KOYEB_STRICT_WEBHOOK = True
elif _strict_env == "false":
    KOYEB_STRICT_WEBHOOK = False
else:
    KOYEB_STRICT_WEBHOOK = bool(os.environ.get("KOYEB_PUBLIC_URL", "").strip())
LOG_JSON = os.environ.get("LOG_JSON", "false").strip().lower() == "true"

# Logging configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Suppress verbose third-party logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)
logging.getLogger('telegram.ext').setLevel(logging.INFO)  # Keep Application logs
logging.getLogger('apscheduler').setLevel(logging.WARNING)  # Reduce scheduler noise

# Optional JSON logging for easier parsing in platforms
if LOG_JSON:
    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            try:
                import json as _json
                return _json.dumps({
                    "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "name": record.name,
                    "message": record.getMessage(),
                }, ensure_ascii=False)
            except Exception:
                return super().format(record)
    _fmt = _JsonFormatter()
    for _h in logging.getLogger().handlers:
        _h.setFormatter(_fmt)

# ============================================
# AIOHTTP helpers for unified webhook + ping
# ============================================

async def _ping_handler(request: web.Request) -> web.Response:
    try:
        return web.json_response({"ok": True, "time": int(time.time())})
    except Exception:
        return web.Response(status=200, text="ok")

async def _metrics_handler(request: web.Request) -> web.Response:
    try:
        data = {
            "uptime": bot_stats.get_uptime(),
            "commands": bot_stats.commands_executed,
            "files": bot_stats.files_processed,
            "errors": bot_stats.errors_caught,
            "extractions": bot_stats.total_extractions,
            "lines": bot_stats.total_lines_processed,
            "users": len(bot_stats.users_served),
            "time": int(time.time()),
        }
        return web.json_response(data)
    except Exception:
        return web.Response(status=200, text="{}")

async def _maintenance_job(context):
    try:
        m3u_cache.clear_expired()
    except Exception:
        pass
    try:
        result_cache.clear_expired()
    except Exception:
        pass

async def _start_aiohttp_webhook(application: Application, webhook_url: str, port: int, secret: str | None):
    await application.initialize()
    await application.bot.set_webhook(
        url=webhook_url,
        secret_token=secret if secret else None,
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
    )

    app = web.Application()
    app.router.add_post('/webhook', application.webhook_handler())
    app.router.add_get('/ping', _ping_handler)
    app.router.add_get('/metrics', _metrics_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()

    await application.start()
    logger.info(f"✅ AIOHTTP webhook server started on :{port} with /webhook and /ping")
    # Schedule periodic maintenance if JobQueue is available
    try:
        if application.job_queue:
            application.job_queue.run_repeating(_maintenance_job, interval=300, first=60)
    except Exception:
        pass
    # Sleep forever until canceled
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await application.stop()
        await runner.cleanup()

# Global exception handler for uncaught exceptions
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Log all uncaught exceptions with full traceback"""
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupts
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.error("="*70)
    logger.error("❌ UNCAUGHT EXCEPTION")
    logger.error("="*70)
    logger.error(f"Exception Type: {exc_type.__name__}")
    logger.error(f"Exception Value: {exc_value}")
    logger.error("📍 FULL TRACEBACK:")
    logger.error(''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    logger.error("="*70)

# Set the global exception handler
sys.excepthook = global_exception_handler

# Temp directory for file processing
TEMP_DIR = 'bot_temp'
os.makedirs(TEMP_DIR, exist_ok=True)

# File size limits (Koyeb-optimized)
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB default
MAX_LINES = int(os.environ.get('MAX_LINES', 500000))  # 500K lines max
MAX_MEMORY_MB = int(os.environ.get('MAX_MEMORY_MB', 300))  # 300MB RAM limit

# Performance settings
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', '8'))  # Reduced for Koyeb
PROCESSING_TIMEOUT = 300  # 5 minutes max per operation

# Initialize global cache (1 hour TTL)
m3u_cache = SimpleCache(ttl=3600)
result_cache = SimpleCache(ttl=1800)  # 30 min for results

# Proxy configuration (optional) - Read from environment variables
# Supports HTTP, HTTPS, and SOCKS5 proxies for M3U checking and validation
PROXY_ENABLED = os.environ.get("PROXY_ENABLED", "false").lower() == "true"
PROXY_VALUE = os.environ.get("PROXY_VALUE", "").strip()

# Disable proxy if it appears to be a placeholder
if PROXY_VALUE in ("123.45.67.89:8080", "127.0.0.1:8080", "localhost:8080"):
    PROXY_ENABLED = False
    PROXY_CONFIG = None
    logger.warning("⚠️ Placeholder proxy detected - proxy disabled")
elif PROXY_ENABLED and PROXY_VALUE:
    # Parse proxy URL and create config dict
    if PROXY_VALUE.startswith("socks"):
        # SOCKS5 proxy: socks5://host:port
        PROXY_CONFIG = {
            'http': PROXY_VALUE,
            'https': PROXY_VALUE
        }
        logger.info(f"🔐 Proxy enabled: SOCKS5 (requires requests[socks])")
    elif PROXY_VALUE.startswith("http"):
        # HTTP/HTTPS proxy: http://host:port
        PROXY_CONFIG = {
            'http': PROXY_VALUE,
            'https': PROXY_VALUE
        }
        logger.info(f"🔐 Proxy enabled: HTTP/HTTPS")
    else:
        # Assume http:// if no scheme provided
        proxy_url = f"http://{PROXY_VALUE}"
        PROXY_CONFIG = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"🔐 Proxy enabled: {proxy_url}")
else:
    PROXY_CONFIG = None

# ============================================
# BOT STATISTICS & MONITORING SYSTEM
# ============================================

class BotStatistics:
    """Track bot usage statistics and performance"""
    
    def __init__(self):
        self.start_time = time.time()
        self.commands_executed = 0
        self.files_processed = 0
        self.errors_caught = 0
        self.users_served = set()
        self.total_extractions = 0
        self.total_lines_processed = 0
        self.stats_file = os.path.join(TEMP_DIR, 'bot_stats.json')
        self.load_stats()
    
    def load_stats(self):
        """Load statistics from file"""
        try:
            if os.path.exists(self.stats_file):
                import json
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    self.commands_executed = data.get('commands', 0)
                    self.files_processed = data.get('files', 0)
                    self.errors_caught = data.get('errors', 0)
                    self.total_extractions = data.get('extractions', 0)
                    self.total_lines_processed = data.get('lines', 0)
                    self.users_served = set(data.get('users', []))
        except Exception as e:
            logger.warning(f"Could not load stats: {e}")
    
    def save_stats(self):
        """Save statistics to file"""
        try:
            import json
            data = {
                'commands': self.commands_executed,
                'files': self.files_processed,
                'errors': self.errors_caught,
                'extractions': self.total_extractions,
                'lines': self.total_lines_processed,
                'users': list(self.users_served),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.stats_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def record_command(self, user_id: int = None):
        """Record a command execution"""
        self.commands_executed += 1
        if user_id:
            self.users_served.add(user_id)
        if self.commands_executed % 10 == 0:  # Save every 10 commands
            self.save_stats()
    
    def record_file(self, lines: int = 0):
        """Record a file processed"""
        self.files_processed += 1
        self.total_lines_processed += lines
        self.save_stats()
    
    def record_extraction(self, results: int = 0):
        """Record an extraction"""
        self.total_extractions += results
        self.save_stats()
    
    def record_error(self):
        """Record an error"""
        self.errors_caught += 1
        self.save_stats()
    
    def get_uptime(self) -> str:
        """Get bot uptime"""
        uptime_seconds = int(time.time() - self.start_time)
        days = uptime_seconds // 86400
        hours = (uptime_seconds % 86400) // 3600
        minutes = (uptime_seconds % 3600) // 60
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def get_report(self) -> str:
        """Generate statistics report"""
        uptime = self.get_uptime()
        avg_per_file = self.total_lines_processed / self.files_processed if self.files_processed > 0 else 0
        
        return (
            f"📊 <b>BOT STATISTICS</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⏱️ <b>Uptime:</b> {uptime}\n"
            f"👥 <b>Users Served:</b> {len(self.users_served)}\n"
            f"⚡ <b>Commands:</b> {self.commands_executed:,}\n"
            f"📁 <b>Files Processed:</b> {self.files_processed:,}\n"
            f"📊 <b>Total Extractions:</b> {self.total_extractions:,}\n"
            f"📄 <b>Lines Processed:</b> {format_number(self.total_lines_processed)}\n"
            f"📈 <b>Avg Lines/File:</b> {format_number(int(avg_per_file))}\n"
            f"❌ <b>Errors Caught:</b> {self.errors_caught}\n\n"
            f"🤖 <b>REDLINE V15.0</b> | Production"
        )

# Initialize statistics
bot_stats = BotStatistics()

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
# ADMIN NOTIFICATION SYSTEM
# ============================================

# Get admin chat IDs for error notifications
ADMIN_CHAT_IDS = set()
_admin_env = os.environ.get('ADMIN_IDS', '').strip()
if _admin_env:
    for s in _admin_env.split(','):
        s = s.strip()
        if s:
            try:
                ADMIN_CHAT_IDS.add(int(s))
            except Exception:
                pass

# If no admin IDs, use owner IDs
if not ADMIN_CHAT_IDS and OWNER_IDS:
    ADMIN_CHAT_IDS = OWNER_IDS.copy()

async def notify_admins(application, message: str, parse_mode: str = 'HTML'):
    """Send notification to all admins"""
    if not ADMIN_CHAT_IDS:
        return
    
    for admin_id in ADMIN_CHAT_IDS:
        try:
            await application.bot.send_message(
                chat_id=admin_id,
                text=message,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.warning(f"Could not notify admin {admin_id}: {e}")

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
        # Allow ALL private chats (session management controls access)
        # Users from the 4 channels can interact with bot privately
        if chat.type == 'private':
            return True
        # Everything else denied
        return False
    except Exception:
        return False

# ============================================
# ERROR LOGGING DECORATOR
# ============================================

def log_errors(func):
    """Decorator to automatically log errors with full traceback in async handlers"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error("="*70)
            logger.error(f"❌ ERROR IN {func.__name__}")
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {e}")
            logger.error("📍 FULL TRACEBACK:")
            logger.error(traceback.format_exc())
            logger.error("="*70)
            raise  # Re-raise so error_handler can catch it
    return wrapper

# ============================================
# RATE LIMITER
# ============================================

_rate_state: Dict[int, Tuple[float, float]] = {}
_rate_warnings: Dict[int, float] = {}

def _allow_rate(chat_id: int, rate: float = 1.0, burst: int = 3) -> bool:
    """Token bucket rate limiter with user warning"""
    now = time.time()
    last, tokens = _rate_state.get(chat_id, (now, float(burst)))
    tokens = min(burst, tokens + (now - last) * rate)
    if tokens >= 1.0:
        _rate_state[chat_id] = (now, tokens - 1.0)
        return True
    _rate_state[chat_id] = (now, tokens)
    return False

async def send_rate_limit_warning(update: Update):
    """Send one-time rate limit warning per user"""
    chat_id = update.effective_chat.id
    now = time.time()
    last_warn = _rate_warnings.get(chat_id, 0)
    if now - last_warn > 60:  # Once per minute
        _rate_warnings[chat_id] = now
        try:
            await update.message.reply_text(
                "⏱️ <b>Slow down!</b>\n\n"
                "Please wait a moment before sending more requests.",
                parse_mode='HTML'
            )
        except Exception:
            pass

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
    
    @staticmethod
    async def check_links_batch_async(links: List[str], max_workers: int = 20, proxies: dict = None, progress_callback=None) -> Dict:
        """Check multiple links in parallel with async progress tracking"""
        results = {'alive': [], 'dead': []}
        processed = 0
        
        def check_with_progress(link):
            result = M3UChecker.check_single_link(link, 5, proxies)
            return result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(check_with_progress, link): link for link in links}
            
            for future in as_completed(futures):
                url, is_alive, status = future.result()
                if is_alive:
                    results['alive'].append((url, status))
                else:
                    results['dead'].append((url, status))
                
                processed += 1
                if progress_callback and processed % 5 == 0:  # Update every 5 links
                    await progress_callback(processed)
        
        # Final update
        if progress_callback:
            await progress_callback(len(links))
        
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
    
    @staticmethod
    def format_auto_block(info: Dict) -> str:
        """Format info for auto batch processing"""
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

class MACProbe:
    """MAC address IPTV portal probe (Stalker/MAG portals)"""
    
    @staticmethod
    def probe_mac(host: str, mac: str, timeout: int = 8, proxies: dict | None = None) -> Tuple[bool, Dict, str]:
        """Probe IPTV host with MAC address"""
        if not HAS_REQUESTS:
            return False, {}, "requests not installed"
        
        try:
            # Normalize host
            if not host.startswith(('http://', 'https://')):
                host = 'http://' + host
            base = host.rstrip('/')
            
            # Common MAC portal endpoints (Stalker/MAG portals)
            endpoints = [
                f"{base}/stalker_portal/server/load.php?type=stb&action=handshake&token=&mac={mac}",
                f"{base}/portal.php?type=stb&action=handshake&token=&mac={mac}",
                f"{base}/c/index.php?mac={mac}",
                f"{base}/server/load.php?type=stb&action=handshake&mac={mac}",
                f"{base}/api/v2/auth?mac={mac}",
            ]
            
            for endpoint in endpoints:
                try:
                    r = Net.get(
                        endpoint, 
                        timeout=timeout, 
                        proxies=proxies,
                        headers={"User-Agent": "Mozilla/5.0 (QtEmbedded; U; Linux; C)"}
                    )
                    
                    if r.status_code == 200:
                        content = r.text.lower()
                        # Check for valid portal response
                        if any(word in content for word in ['token', 'handshake', 'js', 'profile', 'cmd', 'expires']):
                            info = {
                                'mac': mac,
                                'host': host,
                                'endpoint': endpoint.split('?')[0],  # Clean endpoint
                                'status': 'Active',
                                'response_size': len(r.text)
                            }
                            return True, info, ''
                except Exception:
                    continue
            
            return False, {}, "MAC portal not found or inactive"
            
        except Exception as e:
            return False, {}, f"Error: {str(e)}"
    
    @staticmethod
    def format_mac_result(info: Dict) -> str:
        """Format MAC probe result"""
        lines = [
            "----------------------------",
            f"●► MAC: {info.get('mac', '-')}",
            f"●► Host: {info.get('host', '-')}",
            f"●► Endpoint: {info.get('endpoint', '-')}",
            f"●► Status: {info.get('status', '-')}",
            f"●► Response Size: {info.get('response_size', 0)} bytes",
            "----------------------------"
        ]
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
        # Try multiple APIs with fallback (avoid rate limits)
        apis = [
            # API 1: ip-api.com (free, no key needed, 45 req/min)
            {
                'url': f"http://ip-api.com/json/{ip}?fields=status,country,countryCode,region,regionName,city,zip,lat,lon,timezone,isp,org,as,asname",
                'map': lambda d: {
                    'country': d.get('countryCode', ''),
                    'country_name': d.get('country', ''),
                    'region': d.get('regionName', ''),
                    'region_code': d.get('region', ''),
                    'city': d.get('city', ''),
                    'postal': d.get('zip', ''),
                    'latitude': d.get('lat', ''),
                    'longitude': d.get('lon', ''),
                    'timezone': d.get('timezone', ''),
                    'org': d.get('isp', ''),
                    'asn': d.get('as', ''),
                    'as_name': d.get('asname', '')
                } if d.get('status') == 'success' else {}
            },
            # API 2: ipapi.co (backup, 1000/day limit)
            {
                'url': f"https://ipapi.co/{ip}/json/",
                'map': lambda d: d if d.get('country') else {}
            },
            # API 3: ipwhois.app (backup, free)
            {
                'url': f"https://ipwhois.app/json/{ip}",
                'map': lambda d: {
                    'country': d.get('country_code', ''),
                    'country_name': d.get('country', ''),
                    'region': d.get('region', ''),
                    'city': d.get('city', ''),
                    'postal': d.get('postal', ''),
                    'latitude': d.get('latitude', ''),
                    'longitude': d.get('longitude', ''),
                    'timezone': d.get('timezone', ''),
                    'org': d.get('isp', ''),
                    'asn': d.get('asn', '')
                } if d.get('success') else {}
            }
        ]
        
        for api in apis:
            try:
                r = Net.get(api['url'], timeout=timeout)
                if r.status_code == 200:
                    data = r.json()
                    result = api['map'](data)
                    if result:  # If we got valid data, return it
                        return result
            except Exception:
                continue  # Try next API
        
        return {}  # All APIs failed

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

# Global health server instance (singleton)
_health_server = None
_health_server_lock = threading.Lock()
_bot_health_status = {'last_update': time.time(), 'is_healthy': True, 'updates_received': 0}

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path in ('/', '/health', '/status'):
                # Advanced health check - verify bot is actually responsive
                idle_time = time.time() - _bot_health_status['last_update']
                is_healthy = _bot_health_status['is_healthy'] and idle_time < 600  # 10 min threshold
                
                if is_healthy:
                    body = b'OK'
                    status = 200
                else:
                    # Report unhealthy if bot seems stuck
                    body = f'DEGRADED: No activity for {int(idle_time)}s'.encode()
                    status = 503  # Service Unavailable
                    logger.warning(f"⚠️ Health check degraded: idle for {int(idle_time)}s")
                
                self.send_response(status)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                
            elif self.path == '/ping':
                # Simple ping endpoint for external monitors (always returns OK)
                body = b'PONG'
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                
            elif self.path == '/stats':
                # Stats endpoint for monitoring
                import json
                stats = {
                    'status': 'healthy' if _bot_health_status['is_healthy'] else 'degraded',
                    'uptime': int(time.time() - _bot_health_status.get('start_time', time.time())),
                    'last_activity': int(time.time() - _bot_health_status['last_update']),
                    'updates_received': _bot_health_status['updates_received']
                }
                body = json.dumps(stats).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
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
        except Exception as e:
            logger.debug(f"Health handler error: {e}")
            pass
    
    def log_message(self, format, *args):
        # Suppress health check logs to reduce noise
        pass

def start_health_server():
    """Start health server (singleton pattern to prevent 'Address already in use')"""
    global _health_server
    
    with _health_server_lock:
        # If server already running, don't create a new one
        if _health_server is not None:
            logger.info("🌐 Health server already running (reusing)")
            return
        
        try:
            # Use port 8000 for health server to avoid conflict with webhook on 8080
            port = int(os.environ.get('HEALTH_PORT', '8000'))  # Separate port for health server
            
            # Set SO_REUSEADDR to allow quick restarts
            class ReuseHTTPServer(HTTPServer):
                def server_bind(self):
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    super().server_bind()
            
            _health_server = ReuseHTTPServer(('', port), _HealthHandler)
            th = threading.Thread(target=_health_server.serve_forever, daemon=True)
            th.start()
            logger.info(f"🌐 Health server listening on port {port}")
            
            # Start keep-alive ping to prevent Koyeb sleep
            _start_keepalive_ping(port)
        except Exception as e:
            logger.warning(f"Health server failed to start: {e}")
            _health_server = None  # Reset on failure

def _start_keepalive_ping(port: int):
    """Generate external HTTP traffic every 2 minutes to prevent Koyeb sleep (AGGRESSIVE)"""
    def ping_loop():
        import time
        first_ping = True
        ping_failures = 0
        
        while True:
            try:
                time.sleep(120)  # 2 minutes (MORE AGGRESSIVE)
                
                # Try to ping our own external URL to generate "real" traffic
                koyeb_url = os.environ.get('KOYEB_PUBLIC_URL', '')
                
                if koyeb_url and HAS_REQUESTS:
                    # External HTTP request (counts as real traffic for Koyeb)
                    try:
                        import requests
                        # Use port 8000 for health checks if webhook is on 8080
                        health_port = os.environ.get('HEALTH_PORT', '8000')
                        if ':8080' in koyeb_url and health_port == '8000':
                            # For webhook on 8080, ping the main domain (assumes port 80/443)
                            ping_url = koyeb_url.rstrip('/') + '/ping'
                        else:
                            # Use the specified port
                            if ':' not in koyeb_url.split('//')[1]:
                                ping_url = koyeb_url.rstrip('/') + f':{health_port}/ping'
                            else:
                                ping_url = koyeb_url.rstrip('/') + '/ping'
                        
                        response = requests.get(ping_url, timeout=5)
                        if response.status_code == 200:
                            ping_failures = 0  # Reset failure counter
                            if first_ping:
                                logger.info("✅ Keep-alive system active (2min aggressive mode)")
                                first_ping = False
                            logger.debug("✅ Keep-alive: External ping OK")
                        else:
                            ping_failures += 1
                            logger.warning(f"⚠️ Keep-alive ping returned {response.status_code}")
                    except Exception as e:
                        ping_failures += 1
                        logger.debug(f"Keep-alive external failed: {e}")
                        # Fallback to internal ping
                        _internal_ping(port)
                        
                    # Alert if too many failures
                    if ping_failures >= 5:
                        logger.error(f"❌ Keep-alive failed {ping_failures} times - bot may sleep!")
                        ping_failures = 0  # Reset to avoid spam
                else:
                    # Fallback to internal socket ping
                    _internal_ping(port)
                    if first_ping:
                        logger.info("✅ Keep-alive active (internal mode - less effective)")
                        first_ping = False
                    
            except Exception as e:
                logger.debug(f"Keep-alive error: {e}")
    
    def _internal_ping(port: int):
        """Internal fallback ping"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(('127.0.0.1', port))
            sock.sendall(b'GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n')
            sock.recv(100)
            sock.close()
        except Exception:
            pass
    
    th = threading.Thread(target=ping_loop, daemon=True)
    th.start()
    logger.info("✅ Keep-alive mechanism started (2min aggressive interval)")

# ============================================
# GLOBAL ERROR HANDLER
# ============================================

# Conflict counter to reduce log spam
_conflict_count = 0

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    global _conflict_count
    err = context.error
    
    # Ignore duplicate polling conflicts during rolling deploys
    if isinstance(err, telegram.error.Conflict):
        _conflict_count += 1
        if _conflict_count <= 3:
            logger.warning("⚠️ Conflict detected: another bot instance polling. This is normal during Koyeb restarts.")
        elif _conflict_count == 4:
            logger.info("ℹ️ Conflict warnings suppressed (old instance shutting down...)")
        return
    
    # Ignore network timeout errors (will retry automatically)
    if isinstance(err, (telegram.error.NetworkError, telegram.error.TimedOut)):
        logger.warning(f"⚠️ Network error (will retry): {err}")
        return
    
    # Record error in statistics
    bot_stats.record_error()
    
    # Log full traceback for all other errors
    logger.error("="*70)
    logger.error("❌ UNHANDLED ERROR DETECTED")
    logger.error("="*70)
    logger.error(f"Error Type: {type(err).__name__}")
    logger.error(f"Error Message: {err}")
    if update:
        logger.error(f"Update: {update}")
    logger.error("\n📍 FULL TRACEBACK:")
    tb_list = traceback.format_exception(type(err), err, err.__traceback__)
    tb_string = ''.join(tb_list)
    logger.error(tb_string)
    logger.error("="*70)
    
    # Send notification to admins (like GitHub best practice example)
    if ADMIN_CHAT_IDS and context.application:
        try:
            import html
            import json
            
            # Build detailed error message
            update_str = update.to_dict() if isinstance(update, Update) else str(update)
            
            # Truncate if too long (Telegram limit 4096 chars)
            tb_short = tb_string[:1500] if len(tb_string) > 1500 else tb_string
            update_short = json.dumps(update_str, indent=2)[:500] if update_str else "No update"
            
            message = (
                "🚨 <b>BOT ERROR DETECTED</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"<b>Error:</b> <code>{html.escape(str(err)[:200])}</code>\n"
                f"<b>Type:</b> <code>{type(err).__name__}</code>\n\n"
                f"<b>Traceback:</b>\n<pre>{html.escape(tb_short)}</pre>\n\n"
                f"<b>Update:</b>\n<pre>{html.escape(update_short)}</pre>\n\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await notify_admins(context.application, message)
        except Exception as notify_error:
            logger.error(f"Could not notify admins: {notify_error}")

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
    """Create main menu keyboard"""
    keyboard = [
        # EXTRACTION
        [InlineKeyboardButton("📦 EXTRACTION", callback_data="noop")],
        [
            InlineKeyboardButton("📝 Mobile:Pass", callback_data="mode_np"),
            InlineKeyboardButton("👤 User:Pass", callback_data="mode_up"),
            InlineKeyboardButton("📧 Mail:Pass", callback_data="mode_mp")
        ],
        [
            InlineKeyboardButton("🔧 M3U", callback_data="mode_m3u"),
            InlineKeyboardButton("🔑 MAC:Key", callback_data="mode_mac"),
            InlineKeyboardButton("⭐ ALL", callback_data="mode_all")
        ],
        
        # M3U TOOLS
        [InlineKeyboardButton("🔧 M3U TOOLS", callback_data="noop")],
        [
            InlineKeyboardButton("📊 M3U→Combo", callback_data="m3u_to_combo"),
            InlineKeyboardButton("📊 Combo→M3U", callback_data="combo_to_m3u")
        ],
        [
            InlineKeyboardButton("🔀 M3U→MAC", callback_data="m3u_to_mac"),
            InlineKeyboardButton("🔀 MAC→M3U", callback_data="mac_to_m3u")
        ],
        
        # PANEL TOOLS
        [InlineKeyboardButton("⚡ PANEL TOOLS", callback_data="noop")],
        [
            InlineKeyboardButton("📦 Panel Searcher", callback_data="panel_searcher"),
            InlineKeyboardButton("🟢 Check Live Panels", callback_data="check_panels")
        ],
        
        # CHECKERS
        [InlineKeyboardButton("⚡ CHECKERS", callback_data="noop")],
        [
            InlineKeyboardButton("⚡ User:Pass Xtream (1)", callback_data="up_xtream_single"),
            InlineKeyboardButton("🔴 M3U Manual (1)", callback_data="m3u_manual")
        ],
        [
            InlineKeyboardButton("📱 MAC Host (1)", callback_data="mac_host_single")
        ],
        [
            InlineKeyboardButton("⚡ User:Pass Xtream Auto", callback_data="up_xtream_auto")
        ],
        [
            InlineKeyboardButton("🔴 M3U Scanner Auto", callback_data="check_m3u"),
            InlineKeyboardButton("📱 MAC Scanner Auto", callback_data="mac_scanner")
        ],
        
        # TOOLS
        [InlineKeyboardButton("🚩 TOOLS", callback_data="noop")],
        [
            InlineKeyboardButton("🔎 Keyword Searcher", callback_data="keyword_searcher"),
            InlineKeyboardButton("🔗 StreamCreed", callback_data="streamcreed_finder")
        ],
        [
            InlineKeyboardButton("🌐 WHOIS", callback_data="whois_lookup"),
            InlineKeyboardButton("🌐 Proxy Finder", callback_data="proxy_finder")
        ],
        [
            InlineKeyboardButton("🖊️ Combo Generator", callback_data="combo_generator")
        ],
        
        # SETTINGS
        [InlineKeyboardButton("⚙️ SETTINGS", callback_data="noop")],
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
    
    bot_stats.record_command(update.effective_user.id if update.effective_user else None)
    
    help_text = (
        "🔥 <b>REDLINE V15.0</b>\n"
        "Use the menu buttons to run tools.\n"
        "Upload files in the channel to start batch flows."
    )
    
    # Add admin commands for owners
    if update.effective_user and update.effective_user.id in OWNER_IDS:
        help_text += (
            "\n\n👑 <b>ADMIN COMMANDS:</b>\n"
            "/stats - View bot statistics\n"
            "/sysinfo - System information\n"
            "/broadcast - Broadcast message to users\n"
            "/restart - Restart bot (with confirmation)\n\n"
            "📦 <b>BACKUP & EXPORT:</b>\n"
            "/backup - Full bot backup (ZIP)\n"
            "/export_stats - Export statistics (JSON)\n"
            "/export_users - Export user list (TXT)"
        )
    
    await update.effective_message.reply_html(help_text)

# ============================================
# ADMIN COMMANDS
# ============================================

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show bot statistics (admin only)"""
    if not update.effective_user or update.effective_user.id not in OWNER_IDS:
        return
    
    bot_stats.record_command(update.effective_user.id)
    
    report = bot_stats.get_report()
    await update.message.reply_html(report)

async def sysinfo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show system information (admin only)"""
    if not update.effective_user or update.effective_user.id not in OWNER_IDS:
        return
    
    bot_stats.record_command(update.effective_user.id)
    
    try:
        import psutil
        import platform
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        sys_info = (
            f"💻 <b>SYSTEM INFORMATION</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🖥️ <b>Platform:</b> {platform.system()} {platform.release()}\n"
            f"🐍 <b>Python:</b> {platform.python_version()}\n\n"
            f"⚙️ <b>CPU Usage:</b> {cpu_percent}%\n"
            f"🧠 <b>Memory:</b> {memory.percent}% ({format_file_size(memory.used)} / {format_file_size(memory.total)})\n"
            f"💾 <b>Disk:</b> {disk.percent}% ({format_file_size(disk.used)} / {format_file_size(disk.total)})\n\n"
            f"📦 <b>Config:</b>\n"
            f"• Max File: {format_file_size(MAX_FILE_SIZE)}\n"
            f"• Max Lines: {format_number(MAX_LINES)}\n"
            f"• Workers: {MAX_WORKERS}\n\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except ImportError:
        sys_info = (
            f"💻 <b>SYSTEM INFORMATION</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⚠️ <code>psutil</code> not installed\n"
            f"Install with: <code>pip install psutil</code>\n\n"
            f"📦 <b>Config:</b>\n"
            f"• Max File: {format_file_size(MAX_FILE_SIZE)}\n"
            f"• Max Lines: {format_number(MAX_LINES)}\n"
            f"• Workers: {MAX_WORKERS}\n\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as e:
        sys_info = f"❌ Error getting system info: {e}"
    
    await update.message.reply_html(sys_info)

async def broadcast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Broadcast message to all users (admin only)"""
    if not update.effective_user or update.effective_user.id not in OWNER_IDS:
        return
    
    bot_stats.record_command(update.effective_user.id)
    
    if not context.args:
        await update.message.reply_html(
            "📢 <b>Broadcast Command</b>\n\n"
            "<b>Usage:</b> <code>/broadcast Your message here</code>\n\n"
            "<b>Warning:</b> This sends the message to all registered users!"
        )
        return
    
    message_text = ' '.join(context.args)
    users = list(bot_stats.users_served)
    
    if not users:
        await update.message.reply_html("📭 No users to broadcast to!")
        return
    
    # Confirm broadcast
    await update.message.reply_html(
        f"📢 <b>Broadcast Confirmation</b>\n\n"
        f"<b>Message:</b> <code>{message_text}</code>\n"
        f"<b>Recipients:</b> {len(users)} users\n\n"
        f"Send <code>/confirm_broadcast</code> to proceed"
    )
    
    # Store broadcast data temporarily
    context.user_data['pending_broadcast'] = message_text

async def confirm_broadcast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Confirm and execute broadcast (admin only)"""
    if not update.effective_user or update.effective_user.id not in OWNER_IDS:
        return
    
    message_text = context.user_data.get('pending_broadcast')
    if not message_text:
        await update.message.reply_html("❌ No pending broadcast!")
        return
    
    users = list(bot_stats.users_served)
    success = 0
    failed = 0
    
    status_msg = await update.message.reply_html("📤 Broadcasting...")
    
    for user_id in users:
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text=f"📢 <b>Announcement</b>\n\n{message_text}",
                parse_mode='HTML'
            )
            success += 1
            await asyncio.sleep(0.05)  # Rate limiting
        except Exception:
            failed += 1
    
    await status_msg.edit_text(
        f"✅ <b>Broadcast Complete</b>\n\n"
        f"✅ Success: {success}\n"
        f"❌ Failed: {failed}\n"
        f"📊 Total: {len(users)}"
    )
    
    context.user_data.pop('pending_broadcast', None)

async def restart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Restart bot (admin only)"""
    if not update.effective_user or update.effective_user.id not in OWNER_IDS:
        return
    
    bot_stats.record_command(update.effective_user.id)
    
    await update.message.reply_html(
        "🔄 <b>Restart Bot</b>\n\n"
        "⚠️ This will restart the bot process.\n"
        "The bot will be offline for ~10 seconds.\n\n"
        "Send <code>/confirm_restart</code> to proceed"
    )

async def confirm_restart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Confirm and execute restart (admin only)"""
    if not update.effective_user or update.effective_user.id not in OWNER_IDS:
        return
    
    await update.message.reply_html("🔄 Restarting bot...")
    
    # Log restart
    logger.info(f"Bot restart initiated by admin {update.effective_user.id}")
    
    # Force restart by raising exception
    raise SystemExit("Admin restart requested")

# ============================================
# BACKUP & EXPORT COMMANDS (ADMIN ONLY)
# ============================================

async def backup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Export all bot data - admin only (works in channels & private)"""
    if not update.effective_user or update.effective_user.id not in OWNER_IDS:
        return
    
    bot_stats.record_command(update.effective_user.id)
    
    status_msg = await update.message.reply_html("📦 Creating backup...")
    
    try:
        import json
        import zipfile
        from io import BytesIO
        
        # Create backup data
        backup_data = {
            'backup_date': datetime.now().isoformat(),
            'bot_version': 'REDLINE V15.0',
            'statistics': {
                'uptime': bot_stats.get_uptime(),
                'users_served': list(bot_stats.users_served),
                'commands_executed': bot_stats.commands_executed,
                'files_processed': bot_stats.files_processed,
                'total_extractions': bot_stats.total_extractions,
                'total_lines_processed': bot_stats.total_lines_processed,
                'errors_caught': bot_stats.errors_caught
            },
            'configuration': {
                'max_file_size': MAX_FILE_SIZE,
                'max_lines': MAX_LINES,
                'max_workers': MAX_WORKERS,
                'allowed_channels': list(ALLOWED_CHANNEL_IDS),
                'owner_ids': list(OWNER_IDS),
                'admin_ids': list(ADMIN_CHAT_IDS)
            }
        }
        
        # Create ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add backup JSON
            zip_file.writestr('backup.json', json.dumps(backup_data, indent=2))
            
            # Add stats file if exists
            if os.path.exists(bot_stats.stats_file):
                with open(bot_stats.stats_file, 'r') as f:
                    zip_file.writestr('bot_stats.json', f.read())
            
            # Add settings file if exists
            settings_path = os.path.join(TEMP_DIR, 'settings.json')
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    zip_file.writestr('settings.json', f.read())
        
        zip_buffer.seek(0)
        
        # Send backup file
        await status_msg.edit_text("✅ Backup ready! Sending...")
        
        filename = f"redline_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        await update.message.reply_document(
            document=zip_buffer,
            filename=filename,
            caption=(
                f"📦 <b>Bot Backup Complete</b>\n\n"
                f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"👥 Users: {len(bot_stats.users_served)}\n"
                f"📊 Commands: {bot_stats.commands_executed:,}\n"
                f"📁 Files: {bot_stats.files_processed:,}\n\n"
                f"🔒 Keep this file secure!"
            ),
            parse_mode='HTML'
        )
        
        await status_msg.delete()
        logger.info(f"Backup created by admin {update.effective_user.id}")
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Backup failed: {e}")
        logger.error(f"Backup error: {e}\n{traceback.format_exc()}")

async def export_stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Export statistics as JSON - admin only (works in channels & private)"""
    if not update.effective_user or update.effective_user.id not in OWNER_IDS:
        return
    
    bot_stats.record_command(update.effective_user.id)
    
    try:
        import json
        from io import BytesIO
        
        # Prepare stats data
        stats_data = {
            'export_date': datetime.now().isoformat(),
            'bot_version': 'REDLINE V15.0',
            'uptime': bot_stats.get_uptime(),
            'uptime_seconds': int(time.time() - bot_stats.start_time),
            'statistics': {
                'users_served': len(bot_stats.users_served),
                'user_ids': list(bot_stats.users_served),
                'commands_executed': bot_stats.commands_executed,
                'files_processed': bot_stats.files_processed,
                'total_extractions': bot_stats.total_extractions,
                'total_lines_processed': bot_stats.total_lines_processed,
                'errors_caught': bot_stats.errors_caught,
                'avg_lines_per_file': bot_stats.total_lines_processed / bot_stats.files_processed if bot_stats.files_processed > 0 else 0
            }
        }
        
        # Create JSON buffer
        json_buffer = BytesIO()
        json_str = json.dumps(stats_data, indent=2, ensure_ascii=False)
        json_buffer.write(json_str.encode('utf-8'))
        json_buffer.seek(0)
        
        # Send file
        filename = f"bot_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        await update.message.reply_document(
            document=json_buffer,
            filename=filename,
            caption=(
                f"📊 <b>Statistics Export</b>\n\n"
                f"⏱️ Uptime: {bot_stats.get_uptime()}\n"
                f"👥 Users: {len(bot_stats.users_served)}\n"
                f"⚡ Commands: {bot_stats.commands_executed:,}\n"
                f"📁 Files: {bot_stats.files_processed:,}\n"
                f"📄 Lines: {format_number(bot_stats.total_lines_processed)}"
            ),
            parse_mode='HTML'
        )
        
        logger.info(f"Stats exported by admin {update.effective_user.id}")
        
    except Exception as e:
        await update.message.reply_html(f"❌ Export failed: {e}")
        logger.error(f"Stats export error: {e}\n{traceback.format_exc()}")

async def export_users_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Export user list as text file - admin only (works in channels & private)"""
    if not update.effective_user or update.effective_user.id not in OWNER_IDS:
        return
    
    bot_stats.record_command(update.effective_user.id)
    
    try:
        from io import BytesIO
        
        if not bot_stats.users_served:
            await update.message.reply_html("📭 <b>No users to export!</b>")
            return
        
        # Create user list
        user_list = sorted(bot_stats.users_served)
        content = f"REDLINE V15.0 - User List\n"
        content += f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"Total Users: {len(user_list)}\n"
        content += f"=" * 50 + "\n\n"
        
        for idx, user_id in enumerate(user_list, 1):
            content += f"{idx}. {user_id}\n"
        
        # Create buffer
        buffer = BytesIO()
        buffer.write(content.encode('utf-8'))
        buffer.seek(0)
        
        # Send file
        filename = f"bot_users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        await update.message.reply_document(
            document=buffer,
            filename=filename,
            caption=(
                f"👥 <b>User List Export</b>\n\n"
                f"📊 Total Users: {len(user_list)}\n"
                f"📅 Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"🔒 Keep this file secure!"
            ),
            parse_mode='HTML'
        )
        
        logger.info(f"User list exported by admin {update.effective_user.id}")
        
    except Exception as e:
        await update.message.reply_html(f"❌ Export failed: {e}")
        logger.error(f"User export error: {e}\n{traceback.format_exc()}")

# Debug mode toggle (global variable)
DEBUG_MODE = False

async def debug_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle debug mode - admin only"""
    global DEBUG_MODE
    
    if not update.effective_user or update.effective_user.id not in OWNER_IDS:
        return
    
    bot_stats.record_command(update.effective_user.id)
    
    DEBUG_MODE = not DEBUG_MODE
    status = "🟢 ON" if DEBUG_MODE else "🔴 OFF"
    
    await update.message.reply_html(
        f"🐛 <b>Debug Mode</b>\n\n"
        f"Status: {status}\n\n"
        f"{'✅ Detailed error logs enabled' if DEBUG_MODE else '❌ Detailed error logs disabled'}\n\n"
        f"Use <code>/debug</code> again to toggle"
    )
    
    logger.info(f"Debug mode {'enabled' if DEBUG_MODE else 'disabled'} by admin {update.effective_user.id}")

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
        f"👋 <b>Welcome {user.first_name}!</b>\n\n"
        "🔥 <b>REDLINE V15.0 Enhanced</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
        user = query.from_user
        await query.edit_message_text(
            f"👋 <b>Welcome back, {user.first_name}!</b>\n\n"
            "🔥 <b>REDLINE V15.0 Enhanced</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
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
            "<b>⚡ User:Pass Xtream Auto</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📝 Send IPTV host (with port). Example: <code>http://example.com:8080</code>",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return
    
    # === MAC Scanner (batch) ===
    if data == "mac_scanner":
        context.user_data.clear()
        context.user_data['mode'] = 'mac_scanner'
        context.user_data['step'] = 'ask_host'
        await query.edit_message_text(
            "<b>📱 MAC Scanner (Auto)</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📝 Send IPTV host (with port)\n"
            "Example: <code>http://example.com:8080</code>\n\n"
            "⚡ <b>Features:</b>\n"
            "• Batch MAC testing\n"
            "• Auto portal detection\n"
            "• Progress tracking",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return

    # === Phase 2: WHOIS Lookup ===
    if data == "whois_lookup":
        context.user_data.clear()
        context.user_data['mode'] = 'whois_lookup'
        await query.edit_message_text(
            "🌐 <b>WHOIS Lookup</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📱 <b>Send IP or domain:</b>\n"
            "• <code>8.8.8.8</code> (IP)\n"
            "• <code>example.com</code> (Domain)\n\n"
            "💡 Get domain info, age, location & more!",
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

    # === Phase 2: Panel Searcher ===
    if data == "panel_searcher":
        context.user_data.clear()
        context.user_data['mode'] = 'panel_searcher'
        await query.edit_message_text(
            "<b>📦 Panel Searcher</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📤 Send log/text file to search for panel URLs.",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return

    # === Phase 2: Check Live Panels ===
    if data == "check_panels":
        context.user_data.clear()
        context.user_data['mode'] = 'check_panels'
        await query.edit_message_text(
            "<b>🟢 Check Live Panels</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📤 Send a file with panel URLs (one per line).",
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
    
    # MAC Scanner (batch) - handle host input
    if mode == 'mac_scanner' and context.user_data.get('step') == 'ask_host':
        host = update.message.text.strip()
        
        # Validate and normalize host
        if not host:
            await update.message.reply_html("❌ <b>Invalid host</b>")
            return
        
        if not host.startswith(('http://', 'https://')):
            host = 'http://' + host
        
        context.user_data['host'] = host.rstrip('/')
        context.user_data['step'] = 'await_file'
        await update.message.reply_html(
            f"✅ <b>Host saved!</b>\n"
            f"<code>{host}</code>\n\n"
            f"📤 <b>Now send a file containing:</b>\n"
            f"• MAC addresses (one per line)\n"
            f"• Format: <code>00:1A:79:XX:XX:XX</code>\n"
            f"• Both colon and dash separators supported\n\n"
            f"⏳ Waiting for file..."
        )
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
            "<b>⚡ User:Pass Xtream Check (Single)</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
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
    
    # M3U to MAC
    elif data == "m3u_to_mac":
        context.user_data['mode'] = 'm3u_to_mac'
        context.user_data['action'] = 'convert'
        text = (
            "🔀 <b>M3U → MAC Converter</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🔀 Extract MAC addresses from M3U URLs\n\n"
            "📤 <b>Send M3U file:</b>\n"
            "• M3U playlist with MAC links\n"
            "• Bot extracts MAC addresses\n\n"
            "✅ <b>Output:</b>\n"
            "• Clean MAC addresses\n"
            "• Deduplicated\n\n"
            "⏳ Send your M3U file..."
        )
        await query.edit_message_text(text, parse_mode='HTML', reply_markup=get_back_button())
    
    # Back to menu
    elif data == "back":
        welcome_text = (
            "🔥 <b>REDLINE V15.0 Enhanced</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
    
    # Check if user has active session (used /redline and selected option)
    if not mode:
        await update.message.reply_text(
            "⚠️ Please select an option first\n"
            "Use /redline to show menu"
        )
        return
    
    # Check if user is already processing a file (one file at a time)
    if context.user_data.get('processing_file'):
        await update.message.reply_text(
            "⚠️ <b>Please wait!</b>\n\n"
            "You're already processing a file.\n"
            "Wait for it to finish before sending another.\n\n"
            "<i>Only one file at a time per user</i>",
            parse_mode='HTML'
        )
        return
    
    # Set processing lock (will be released in finally block)
    context.user_data['processing_file'] = True
    
    try:
        # Check file size
        file_size = update.message.document.file_size
        if file_size > MAX_FILE_SIZE:
            await update.message.reply_text(
                f"❌ <b>File too large!</b>\n\n"
                f"📊 <b>Limits:</b>\n"
                f"• Max file size: <code>{format_file_size(MAX_FILE_SIZE)}</code>\n"
                f"• Max lines: <code>{format_number(MAX_LINES)}</code>\n\n"
                f"📁 <b>Your file:</b> <code>{format_file_size(file_size)}</code>\n\n"
                f"💡 <b>Tips:</b>\n"
                f"• Split large files into smaller chunks\n"
                f"• Remove duplicate lines first\n"
                f"• Use compression (gzip) if possible\n\n"
                f"<i>Limits protect server resources on Koyeb</i>",
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
        # Download file
        file = await update.message.document.get_file()
        filename = update.message.document.file_name or "file.txt"
        file_path = os.path.join(TEMP_DIR, f"{update.effective_user.id}_{file.file_id}_{filename}")
        await file.download_to_drive(file_path)
        
        # Check line count (memory protection)
        line_count = count_lines_efficient(file_path, MAX_LINES + 1)
        if line_count > MAX_LINES:
            await status_msg.edit_text(
                f"❌ <b>Too many lines!</b>\n\n"
                f"📊 <b>Limits:</b>\n"
                f"• Max lines: <code>{format_number(MAX_LINES)}</code>\n"
                f"• Your file: <code>{format_number(line_count)}+</code>\n\n"
                f"💡 <b>Solution:</b>\n"
                f"• Split file into {(line_count // MAX_LINES) + 1} parts\n"
                f"• Remove empty lines\n"
                f"• Process in batches\n\n"
                f"<i>This protects against memory exhaustion</i>",
                parse_mode='HTML'
            )
            os.remove(file_path)
            context.user_data.clear()
            return
        
        # Update status
        await status_msg.edit_text(
            f"⏳ <b>Processing...</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📄 Reading {format_number(line_count)} lines...",
            parse_mode='HTML'
        )
        
        # Read file content (memory-safe)
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
            
            # Initialize ProgressTracker
            progress = ProgressTracker(status_msg, total, "Probing Xtream Accounts")
            
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
                    
                    # Update progress
                    if done % 10 == 0 or done == total:
                        await progress.update(done)
            
            # Complete progress
            await progress.complete(f"\n✅ Valid: {found}/{total}")
            
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
        
        # === HANDLE MAC SCANNER (batch) ===
        if mode == 'mac_scanner' and context.user_data.get('step') == 'await_file':
            await status_msg.edit_text(
                "⏳ <b>Scanning MAC Addresses...</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔍 Testing portal endpoints...",
                parse_mode='HTML'
            )
            
            host = context.user_data.get('host') or ''
            
            # Extract MAC addresses from file
            macs_raw = [ln.strip() for ln in text.split('\n') if ln.strip()]
            macs = []
            
            # Validate MAC format (support both colon and dash separators)
            mac_pattern = re.compile(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
            for mac in macs_raw:
                mac = mac.upper()
                if mac_pattern.match(mac):
                    # Normalize to colon separator
                    mac = mac.replace('-', ':')
                    macs.append(mac)
            
            if not macs:
                await status_msg.edit_text("❌ <b>No valid MAC addresses found!</b>", parse_mode='HTML')
                os.remove(file_path)
                context.user_data.clear()
                return
            
            # Apply limit
            limit = int(GLOBAL_SETTINGS.get('combo_limit', 500))
            macs = macs[:limit]
            workers = int(GLOBAL_SETTINGS.get('workers', 12))
            
            done = 0
            found = 0
            total = len(macs)
            blocks: List[str] = []
            
            # Initialize ProgressTracker
            progress = ProgressTracker(status_msg, total, "Scanning MAC Addresses")
            
            # Show typing
            try:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            except Exception:
                pass
            
            # Probe MACs in parallel
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(MACProbe.probe_mac, host, mac, 8, get_proxies()): mac for mac in macs}
                for fu in as_completed(futs):
                    ok, info, err = fu.result()
                    done += 1
                    if ok:
                        blocks.append(MACProbe.format_mac_result(info))
                        found += 1
                    
                    # Update progress
                    if done % 10 == 0 or done == total:
                        await progress.update(done)
            
            # Complete progress
            await progress.complete(f"\n✅ Valid: {found}/{total}")
            
            # Create result file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"MAC_SCANNER_{timestamp}.txt"
            result_path = os.path.join(TEMP_DIR, result_filename)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                if blocks:
                    f.write(f"# MAC Scanner Results\n")
                    f.write(f"# Host: {host}\n")
                    f.write(f"# Valid MACs: {found} / {total}\n")
                    f.write(f"# Scanned: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("\n".join(blocks) + "\n")
                else:
                    f.write("# No valid MAC addresses found\n")
            
            # Send result
            with open(result_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=result_filename,
                    caption=(
                        f"✅ <b>MAC Scanner Complete</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"🎯 Host: <code>{host}</code>\n"
                        f"📊 Valid: {found} / {total}"
                    ),
                    parse_mode='HTML',
                    reply_markup=get_main_menu()
                )
            
            await status_msg.delete()
            os.remove(result_path)
            os.remove(file_path)
            context.user_data.clear()
            return
        
        # === HANDLE M3U LINK CHECKER (AUTO, REDLINE info) ===
        if mode == 'check_m3u':
            await status_msg.edit_text(
                "⏳ <b>Starting M3U Probe...</b>",
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
            
            # Initialize ProgressTracker
            progress = ProgressTracker(status_msg, total, "Probing M3U Links")
            
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
                    
                    # Update progress
                    if done % 10 == 0 or done == total:
                        await progress.update(done)
            
            # Complete progress
            await progress.complete(f"\n✅ Valid: {found}/{total}")
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
            f"⏳ <b>Processing...</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🔍 Extracting data...",
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
                    f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
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
        logger.error("="*70)
        logger.error("❌ ERROR IN handle_document")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Message: {e}")
        logger.error("📍 FULL TRACEBACK:")
        logger.error(traceback.format_exc())
        logger.error("="*70)
        
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
    finally:
        # Always release the processing lock
        context.user_data.pop('processing_file', None)

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
        await send_rate_limit_warning(update)
        return
    mode = context.user_data.get('mode')
    # Support /redline in channels or allowed private
    if update.effective_message and getattr(update.effective_message, 'text', None):
        txt = update.effective_message.text.strip()
        if txt.startswith('/redline'):
            await start(update, context)
            return
    
    # WHOIS Lookup handler
    if mode == 'whois_lookup':
        target = update.message.text.strip()
        status_msg = await update.message.reply_html("⏳ <b>Running WHOIS...</b>")
        tgt, report = WHOISLookup.whois_report(target)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"WHOIS_{timestamp}.txt"
        result_path = os.path.join(TEMP_DIR, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(report + "\n")
        
        caption = f"🌐 <b>WHOIS:</b> <code>{tgt}</code>"
        
        with open(result_path, 'rb') as f:
            await update.message.reply_document(
                document=f,
                filename=result_filename,
                caption=caption,
                parse_mode='HTML',
                reply_markup=get_main_menu()
            )
        await status_msg.delete()
        os.remove(result_path)
        context.user_data.clear()
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
            logger.error("="*70)
            logger.error("❌ ERROR IN handle_text_message (combo_to_m3u)")
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {e}")
            logger.error("📍 FULL TRACEBACK:")
            logger.error(traceback.format_exc())
            logger.error("="*70)
            
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

    # No mode set - ignore message (don't spam users)
    # Bot only responds when user explicitly uses /redline or during active mode
    return

# ============================================
# AUTO-RESTART CONFIGURATION
# ============================================

# Global flags for restart control
_should_restart = True
_graceful_shutdown = False

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    global _should_restart, _graceful_shutdown
    logger.info(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
    _graceful_shutdown = True
    _should_restart = False

# Register signal handlers (with error handling for different platforms)
try:
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    logger.info("✅ Signal handlers registered successfully")
except Exception as e:
    logger.warning(f"⚠️ Could not register signal handlers: {e}")

# ============================================
# MAIN - START BOT
# ============================================

def main():
    """Start the bot with auto-restart capability"""
    logger.info("🤖 Starting REDLINE V15.0 Telegram Bot...")
    
    # Create application with proxy support (if Telegram is blocked)
    # Uncomment the proxy lines below if you need to use a proxy
    
    # Option 1: Without proxy (normal)
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .get_updates_read_timeout(30)
        .get_updates_write_timeout(30)
        .get_updates_connect_timeout(30)
        .get_updates_pool_timeout(30)
        .build()
    )
    
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
    application.add_handler(CommandHandler("redline", start))
    application.add_handler(CommandHandler("health", health_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    
    # Admin commands
    application.add_handler(CommandHandler("stats", stats_cmd))
    application.add_handler(CommandHandler("sysinfo", sysinfo_cmd))
    application.add_handler(CommandHandler("debug", debug_cmd))
    application.add_handler(CommandHandler("broadcast", broadcast_cmd))
    application.add_handler(CommandHandler("confirm_broadcast", confirm_broadcast_cmd))
    application.add_handler(CommandHandler("restart", restart_cmd))
    application.add_handler(CommandHandler("confirm_restart", confirm_restart_cmd))
    
    # Backup & Export commands
    application.add_handler(CommandHandler("backup", backup_cmd))
    application.add_handler(CommandHandler("export_stats", export_stats_cmd))
    application.add_handler(CommandHandler("export_users", export_users_cmd))
    
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
            
            # Clean expired cache entries
            m3u_cache.clear_expired()
            result_cache.clear_expired()
        except Exception:
            pass

    # Schedule cleanup job if JobQueue is available
    if application.job_queue:
        application.job_queue.run_repeating(_cleanup_temp_cb, interval=600, first=120)
    else:
        logger.warning("⚠️ JobQueue not available. Cleanup tasks disabled.")
    
    # Start bot
    if not ALLOWED_CHANNEL_IDS:
        logger.warning("No channels configured. Set CHANNEL_IDS or CHANNEL_ID env vars.")
    logger.info("✅ Bot is running! Press Ctrl+C to stop.")
    logger.info(f"📱 Channels: {sorted(list(ALLOWED_CHANNEL_IDS))}")
    logger.info("🚀 Progress tracking & caching enabled")
    logger.info(f"📊 Resource limits: {format_file_size(MAX_FILE_SIZE)} / {format_number(MAX_LINES)} lines / {MAX_WORKERS} workers")
    
    # Notify admins of successful startup (after bot initializes)
    async def send_startup_notification(app):
        """Send startup notification to admins after bot starts"""
        if not ADMIN_CHAT_IDS:
            return
        
        await asyncio.sleep(5)  # Wait for bot to fully initialize
        startup_msg = (
            "🚀 <b>BOT STARTED SUCCESSFULLY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"📱 <b>Channels:</b> {len(ALLOWED_CHANNEL_IDS)}\n"
            f"👥 <b>Total Users:</b> {len(bot_stats.users_served)}\n"
            f"🔄 <b>Auto-Restart:</b> ✅ Enabled\n"
            f"📊 <b>Stats Tracking:</b> ✅ Active\n\n"
            f"🤖 <b>REDLINE V15.0</b> | Production Ready"
        )
        await notify_admins(app, startup_msg)
    
    # Register post-init callback to send notification
    application.post_init = send_startup_notification
    
    # Run bot with retry on conflict
    logger.info("🚀 Starting polling loop...")
    
    # Initialize health status tracking
    _bot_health_status['start_time'] = time.time()
    _bot_health_status['last_update'] = time.time()
    _bot_health_status['is_healthy'] = True
    _bot_health_status['updates_received'] = 0
    
    # Activity monitor to detect Koyeb sleep
    def activity_monitor():
        """Monitor for signs of Koyeb sleep (no activity for 5+ minutes)"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                idle_time = time.time() - _bot_health_status['last_update']
                
                if idle_time > 360:  # 6 minutes of no activity
                    logger.warning(f"⚠️ No activity for {int(idle_time/60)} minutes - possible Koyeb sleep approaching")
                    _bot_health_status['is_healthy'] = False
                elif _bot_health_status['is_healthy'] == False and idle_time < 120:
                    # Recovered from unhealthy state
                    logger.info("✅ Bot recovered - receiving updates again")
                    _bot_health_status['is_healthy'] = True
            except Exception:
                pass
    
    # Start activity monitor
    monitor_thread = threading.Thread(target=activity_monitor, daemon=True)
    monitor_thread.start()
    
    # Update activity on each update
    async def track_activity(update: Update, context: ContextTypes.DEFAULT_TYPE):
        _bot_health_status['last_update'] = time.time()
        _bot_health_status['updates_received'] += 1
        _bot_health_status['is_healthy'] = True
    
    # Add activity tracker as first handler (runs before all others)
    from telegram.ext import TypeHandler
    application.add_handler(TypeHandler(Update, track_activity), group=-1)
    
    max_retries = 3
    
    # Determine if we should use webhook or polling
    use_webhook_mode = USE_WEBHOOK and KOYEB_PUBLIC_URL
    # Permanently disable webhook for this runtime if a previous failure occurred
    try:
        _webhook_disable_flag = os.path.join(TEMP_DIR, 'disable_webhook.flag')
        if os.path.exists(_webhook_disable_flag):
            logger.warning("⚠️ Webhook previously failed in this runtime. Disabling webhook mode until next deploy/restart.")
            use_webhook_mode = False
    except Exception:
        pass
    
    if use_webhook_mode:
        logger.info("🌐 Using WEBHOOK mode (more stable for Koyeb)")
        try:
            # Extract port from environment - use same port as health server
            port = int(os.environ.get('PORT', '8080'))  # Use 8080 to match Koyeb's default
            
            # Set webhook URL
            webhook_url = f"{KOYEB_PUBLIC_URL}/webhook"
            
            # Configure webhook with security token if provided
            if WEBHOOK_SECRET:
                webhook_url += f"?token={WEBHOOK_SECRET}"
            
            logger.info(f"🔗 Webhook URL: {webhook_url}")
            
            # Ensure aiohttp is available
            if not HAS_AIOHTTP:
                msg = "aiohttp is not installed. Add 'aiohttp' to requirements and redeploy."
                logger.error(msg)
                if KOYEB_STRICT_WEBHOOK:
                    sys.exit(1)
                logger.warning("⚠️ Falling back to polling mode due to missing aiohttp...")
                use_webhook_mode = False
            else:
                # Start aiohttp server hosting both /webhook and /ping on the public PORT
                asyncio.run(_start_aiohttp_webhook(
                    application,
                    webhook_url,
                    port,
                    WEBHOOK_SECRET if WEBHOOK_SECRET else None
                ))
            # If we reach here, webhook exited normally
            logger.warning("⚠️ Webhook loop exited - likely Koyeb container shutdown")
            return  # Exit main() normally
            
        except Exception as e:
            logger.error(f"❌ Webhook mode failed: {e}")
            if KOYEB_STRICT_WEBHOOK:
                logger.error("❌ Strict webhook mode enabled: exiting for platform restart")
                # Exit the process so Koyeb restarts a fresh instance
                sys.exit(1)
            logger.warning("⚠️ Falling back to polling mode...")
            use_webhook_mode = False
            # Drop a flag so subsequent restarts don't retry webhook endlessly
            try:
                with open(os.path.join(TEMP_DIR, 'disable_webhook.flag'), 'w', encoding='utf-8') as _f:
                    _f.write(f"disabled at {datetime.now().isoformat()} due to: {e}\n")
            except Exception:
                pass
            
            # Recreate application for polling (fresh event loop)
            logger.info("🔄 Recreating application for polling mode...")
            
            # Create a fresh event loop for polling
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            except Exception as loop_err:
                logger.warning(f"Could not create new event loop: {loop_err}")
            
            application = (
                Application.builder()
                .token(BOT_TOKEN)
                .get_updates_read_timeout(30)
                .get_updates_write_timeout(30)
                .get_updates_connect_timeout(30)
                .get_updates_pool_timeout(30)
                .build()
            )
            
            # Re-add all handlers
            application.add_handler(CommandHandler("redline", start))
            application.add_handler(CommandHandler("health", health_cmd))
            application.add_handler(CommandHandler("help", help_cmd))
            application.add_handler(CommandHandler("stats", stats_cmd))
            application.add_handler(CommandHandler("sysinfo", sysinfo_cmd))
            application.add_handler(CommandHandler("debug", debug_cmd))
            application.add_handler(CommandHandler("broadcast", broadcast_cmd))
            application.add_handler(CommandHandler("confirm_broadcast", confirm_broadcast_cmd))
            application.add_handler(CommandHandler("restart", restart_cmd))
            application.add_handler(CommandHandler("confirm_restart", confirm_restart_cmd))
            application.add_handler(CommandHandler("backup", backup_cmd))
            application.add_handler(CommandHandler("export_stats", export_stats_cmd))
            application.add_handler(CommandHandler("export_users", export_users_cmd))
            application.add_handler(CallbackQueryHandler(button_callback))
            application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
            application.add_error_handler(error_handler)
            
            # Re-add activity tracker
            from telegram.ext import TypeHandler
            application.add_handler(TypeHandler(Update, track_activity), group=-1)
    
    # Fallback to polling if webhook not enabled or failed
    if not use_webhook_mode:
        logger.info("🔄 Using POLLING mode")
        for attempt in range(max_retries):
            try:
                application.run_polling(
                    allowed_updates=Update.ALL_TYPES,
                    drop_pending_updates=True
                )
                # If polling exits normally, it's likely Koyeb stopped the container
                logger.warning("⚠️ Polling loop exited normally - likely Koyeb container shutdown")
                logger.warning("🔄 This is expected when Koyeb sleeps or redeploys the service")
                break
            except telegram.error.Conflict as e:
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"⚠️ Conflict on start (attempt {attempt+1}/{max_retries}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error("="*70)
                    logger.error(f"❌ FAILED TO START after {max_retries} attempts")
                    logger.error(f"Error: {e}")
                    logger.error("📍 TRACEBACK:")
                    logger.error(traceback.format_exc())
                    logger.error("="*70)
                    raise
            except Exception as e:
                logger.error("="*70)
                logger.error("❌ UNEXPECTED ERROR DURING POLLING")
                logger.error(f"Error Type: {type(e).__name__}")
                logger.error(f"Error Message: {e}")
                logger.error("📍 FULL TRACEBACK:")
                logger.error(traceback.format_exc())
                logger.error("="*70)
                raise

if __name__ == '__main__':
    # Startup validation
    try:
        logger.info("="*70)
        logger.info("🚀 REDLINE V15.0 BOT INITIALIZING")
        logger.info("="*70)
        
        # Validate environment
        if not BOT_TOKEN:
            logger.error("❌ BOT_TOKEN is not set!")
            sys.exit(1)
        
        logger.info(f"✅ Bot token configured: {BOT_TOKEN[:10]}...{BOT_TOKEN[-4:]}")
        logger.info(f"✅ Temp directory: {TEMP_DIR}")
        logger.info(f"✅ Allowed channels: {len(ALLOWED_CHANNEL_IDS)}")
        
        # Configuration summary
        logger.info("")
        logger.info("📋 Configuration:")
        logger.info(f"   Mode: {'WEBHOOK' if USE_WEBHOOK and KOYEB_PUBLIC_URL else 'POLLING'}")
        logger.info(f"   Keep-alive: {'2min aggressive' if KOYEB_PUBLIC_URL else 'Internal fallback'}")
        logger.info(f"   Health checks: Advanced (self-healing)")
        logger.info(f"   Proxy: {'Enabled' if PROXY_CONFIG else 'Disabled'}")
        
        # Warn if KOYEB_PUBLIC_URL is not set
        if not KOYEB_PUBLIC_URL:
            logger.warning("")
            logger.warning("⚠️ KOYEB_PUBLIC_URL not set - bot may auto-sleep!")
            logger.warning("💡 Set KOYEB_PUBLIC_URL to prevent auto-sleep")
            logger.warning("   Example: https://your-app-name.koyeb.app")
        else:
            logger.info(f"✅ Keep-alive target: {KOYEB_PUBLIC_URL}")
            if USE_WEBHOOK:
                logger.info("✅ Webhook mode enabled (maximum stability)")
        
    except Exception as e:
        logger.error(f"❌ Startup validation failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    restart_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 5
    base_restart_delay = 5  # Start with 5 seconds
    max_restart_delay = 300  # Max 5 minutes
    
    logger.info("")
    logger.info("="*70)
    logger.info("🔄 AUTO-RESTART MODE ENABLED")
    logger.info("Bot will automatically restart on crashes")
    logger.info("Press Ctrl+C to stop permanently")
    logger.info("="*70)
    logger.info("")
    
    while _should_restart:
        try:
            restart_count += 1
            if restart_count > 1:
                logger.info(f"\n🔄 RESTART #{restart_count} (Consecutive failures: {consecutive_failures})")
                # Create a fresh event loop for restart attempts
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                except Exception as loop_err:
                    logger.warning(f"Could not create new event loop: {loop_err}")
            
            main()
            
            # If main() exits normally (bot was stopped gracefully)
            if _graceful_shutdown:
                logger.info("✅ Bot shut down gracefully")
                break
            
            # Reset failure counter on successful run (normal exit = success)
            if consecutive_failures > 0:
                logger.info(f"✅ Successful restart after {consecutive_failures} failures")
            consecutive_failures = 0
            
            # Check if this is likely a Koyeb sleep/restart scenario
            logger.info("🔄 Bot stopped cleanly - normal for Koyeb sleep/restart")
            logger.info("⏳ Waiting 10 seconds before restarting...")
            time.sleep(10)  # Longer pause for Koyeb to fully restore
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user (Ctrl+C)")
            _should_restart = False
            break
            
        except Exception as e:
            consecutive_failures += 1
            
            logger.error("="*70)
            logger.error("❌ FATAL BOT ERROR - AUTO-RESTART TRIGGERED")
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {e}")
            logger.error("📍 FULL TRACEBACK:")
            logger.error(traceback.format_exc())
            logger.error("="*70)
            
            # Check if too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                logger.error(f"\n⚠️ TOO MANY CONSECUTIVE FAILURES ({consecutive_failures})")
                logger.error("Increasing restart delay to prevent crash loop...")
                # Exponential backoff with max cap
                restart_delay = min(
                    base_restart_delay * (2 ** (consecutive_failures - max_consecutive_failures)),
                    max_restart_delay
                )
            else:
                # Normal exponential backoff
                restart_delay = min(
                    base_restart_delay * (2 ** (consecutive_failures - 1)),
                    60  # Cap at 1 minute for normal failures
                )
            
            if _should_restart:
                logger.info(f"\n⏳ Waiting {restart_delay} seconds before restart...")
                time.sleep(restart_delay)
                logger.info("🔄 Attempting to restart bot...\n")
            else:
                break
    
    logger.info("\n" + "="*70)
    logger.info("🏁 BOT SHUTDOWN COMPLETE")
    logger.info(f"Total restarts: {restart_count - 1}")
    logger.info("="*70)
    sys.exit(0)
