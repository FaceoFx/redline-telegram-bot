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
from datetime import datetime
from typing import List, Set, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes
)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed - M3U checker disabled")

# ============================================
# BOT CONFIGURATION
# ============================================

BOT_TOKEN = "5687444692:AAHSzlDA9n8NlBjw7iqbpGEmBoQ9bGP4gdU"
CHANNEL_ID = "-880072951"  # Your channel

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
    def check_single_link(url: str, timeout: int = 5) -> Tuple[str, bool, str]:
        """Check if M3U link is alive"""
        if not HAS_REQUESTS:
            return (url, False, "❌ requests not installed")
        
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            
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
    def check_links_batch(links: List[str], max_workers: int = 20) -> Dict:
        """Check multiple links in parallel"""
        results = {'alive': [], 'dead': []}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(M3UChecker.check_single_link, link): link 
                      for link in links}
            
            for future in as_completed(futures):
                url, is_alive, status = future.result()
                if is_alive:
                    results['alive'].append((url, status))
                else:
                    results['dead'].append((url, status))
        
        return results

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
        [InlineKeyboardButton("═══ ℹ️ INFO ═══", callback_data="noop")],
        [
            InlineKeyboardButton("❓ Help", callback_data="help"),
            InlineKeyboardButton("📊 Stats", callback_data="stats")
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command - show welcome and main menu - ALL ENGLISH"""
    user = update.effective_user
    
    welcome_text = (
        f"👋 Welcome {user.mention_html()}!\n\n"
        "🔥 <b>REDLINE V15.0 - Enhanced Bot</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🚀 <b>All REDLINE Features + M3U Tools!</b>\n\n"
        "📊 <b>EXTRACTION:</b>\n"
        "• N:P (Phone:Password)\n"
        "• U:P (Username:Password)\n"
        "• M:P (Email:Password)\n"
        "• M3U Links\n"
        "• MAC:KEY\n"
        "• Extract ALL at once\n\n"
        "🔗 <b>M3U TOOLS:</b>\n"
        "• Check M3U Links (Live validation)\n"
        "• Convert M3U → Combo\n"
        "• Convert Combo → M3U\n\n"
        "⚡ <b>Features:</b>\n"
        "• Instant processing\n"
        "• Multi-threaded checking\n"
        "• Progress tracking\n"
        "• Professional results\n"
        "• Max file size: 50MB\n\n"
        "Select an option below:"
    )
    
    await update.message.reply_html(
        welcome_text,
        reply_markup=get_main_menu()
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all button callbacks - ALL ENGLISH"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    # No-op for section headers
    if data == "noop":
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
    elif data == "check_m3u":
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
            "Use /start to show menu"
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
        
        # === HANDLE M3U LINK CHECKER ===
        if mode == 'check_m3u':
            await status_msg.edit_text(
                "⏳ <b>Checking M3U Links...</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🔍 Validating links...",
                parse_mode='HTML'
            )
            
            # Extract links from file
            links = [line.strip() for line in text.split('\n') 
                    if line.strip() and line.strip().startswith('http')]
            
            if not links:
                await status_msg.edit_text(
                    "❌ <b>No M3U links found!</b>\n\n"
                    "Make sure file contains HTTP/HTTPS URLs",
                    parse_mode='HTML'
                )
                os.remove(file_path)
                context.user_data.clear()
                return
            
            # Check links (limit to 1000)
            links_to_check = links[:1000]
            results = M3UChecker.check_links_batch(links_to_check, max_workers=20)
            
            # Create result file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"M3U_CHECK_{timestamp}.txt"
            result_path = os.path.join(TEMP_DIR, result_filename)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"# M3U Link Checker Results\n")
                f.write(f"# Checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total: {len(links_to_check)} | Alive: {len(results['alive'])} | Dead: {len(results['dead'])}\n")
                f.write("#" + "="*50 + "\n\n")
                
                if results['alive']:
                    f.write("# ===== ALIVE LINKS =====\n")
                    for url, status in results['alive']:
                        f.write(f"{url}\n")
                    f.write("\n")
                
                if results['dead']:
                    f.write("# ===== DEAD LINKS =====\n")
                    for url, status in results['dead']:
                        f.write(f"{url}  # {status}\n")
            
            # Send result
            with open(result_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=result_filename,
                    caption=(
                        f"✅ <b>M3U Check Complete!</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"📊 Total Checked: {len(links_to_check)}\n"
                        f"✅ Alive: {len(results['alive'])}\n"
                        f"❌ Dead: {len(results['dead'])}\n\n"
                        f"⚡ Multi-threaded checking"
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
        
        # Post to channel (optional)
        try:
            await context.bot.send_message(
                chat_id=CHANNEL_ID,
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
    """Handle text messages - for Base URL input in combo_to_m3u"""
    mode = context.user_data.get('mode')
    
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
    else:
        # Not in a special mode - ignore text messages
        await update.message.reply_text(
            "ℹ️ <b>Unknown command</b>\n\n"
            "Use /start to show menu",
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
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    
    # Start bot
    logger.info("✅ Bot is running! Press Ctrl+C to stop.")
    logger.info(f"📱 Channel ID: {CHANNEL_ID}")
    
    # Run bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
