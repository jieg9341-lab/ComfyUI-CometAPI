import os
import io
import re
import json
import asyncio
import base64
import math
import mimetypes
import sqlite3
import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import requests
import torch
from aiohttp import web
from PIL import Image

import folder_paths
from server import PromptServer


_DEFERRED_CHAT_ROUTES = []
_CHAT_ROUTES_LOCK = threading.Lock()


class _DeferredChatRoutes:
    def _record(self, method: str, path: str):
        def decorator(handler):
            with _CHAT_ROUTES_LOCK:
                _DEFERRED_CHAT_ROUTES.append((method, path, handler))
            return handler

        return decorator

    def get(self, path: str):
        return self._record("get", path)

    def post(self, path: str):
        return self._record("post", path)


def _current_prompt_routes():
    inst = getattr(PromptServer, "instance", None)
    return getattr(inst, "routes", None) if inst is not None else None


CHAT_ROUTES = _current_prompt_routes() or _DeferredChatRoutes()


def _register_deferred_chat_routes() -> bool:
    routes = _current_prompt_routes()
    if routes is None:
        return False
    with _CHAT_ROUTES_LOCK:
        pending = list(_DEFERRED_CHAT_ROUTES)
        _DEFERRED_CHAT_ROUTES.clear()
    for method, path, handler in pending:
        getattr(routes, method)(path)(handler)
    return True


def _register_deferred_chat_routes_when_ready() -> None:
    if _register_deferred_chat_routes():
        return

    def _wait_for_server():
        import time

        deadline = time.time() + 120.0
        while time.time() < deadline:
            if _register_deferred_chat_routes():
                return
            time.sleep(0.05)
        print("[CometAPI Chat] route registration incomplete after 120s.")

    threading.Thread(target=_wait_for_server, daemon=True, name="CometAPIChatRouteRegister").start()


# ======================================================================
# 工具函数
# ======================================================================

def categorize_mime(mime_type: Optional[str]) -> str:
    mime_type = mime_type or "application/octet-stream"
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("video/"):
        return "video"
    if mime_type.startswith("audio/"):
        return "audio"
    if mime_type == "application/pdf":
        return "document"
    return "unknown"


def sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "upload.bin")
    base = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", base).strip("._")
    return base or "upload.bin"


def ensure_unique_path(folder: str, filename: str) -> str:
    stem, ext = os.path.splitext(filename)
    candidate = filename
    idx = 1
    while os.path.exists(os.path.join(folder, candidate)):
        candidate = f"{stem}_{idx}{ext}"
        idx += 1
    return os.path.join(folder, candidate)


def tensor_to_base64(tensor: torch.Tensor) -> Optional[str]:
    try:
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        import numpy as np

        img_np = (torch.clamp(tensor, 0, 1).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, "RGB" if img_np.shape[-1] == 3 else "RGBA")
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[CometChat] 图像张量转换 Base64 失败: {e}")
        return None


def _resolve_existing_file(filename: str, file_type: Optional[str] = None, subfolder: str = "") -> Optional[str]:
    filename = os.path.basename(filename or "")
    subfolder = (subfolder or "").strip().strip("/").strip("\\")

    search_roots: List[str] = []
    if file_type == "temp":
        search_roots.append(folder_paths.get_temp_directory())
    elif file_type == "input":
        search_roots.append(folder_paths.get_input_directory())
    else:
        search_roots.extend([folder_paths.get_temp_directory(), folder_paths.get_input_directory()])

    candidate_paths = []
    if os.path.isabs(filename) and os.path.isfile(filename):
        return filename
    for root in search_roots:
        if subfolder:
            candidate_paths.append(os.path.join(root, subfolder, filename))
        candidate_paths.append(os.path.join(root, filename))

    for path in candidate_paths:
        if os.path.isfile(path):
            return path
    return None


def resolve_file_to_base64(file_ref: Union[str, dict]) -> Optional[dict]:
    file_name = file_ref
    file_type = None
    subfolder = ""
    original_name = None
    file_id = None
    preview_text = ""
    size = 0
    remote_file_id = ""

    if isinstance(file_ref, dict):
        file_name = file_ref.get("name") or file_ref.get("filename") or file_ref.get("path") or ""
        file_type = file_ref.get("type")
        subfolder = file_ref.get("subfolder") or ""
        original_name = file_ref.get("original_name") or file_ref.get("name")
        file_id = file_ref.get("id") or file_ref.get("file_id") or None
        preview_text = str(file_ref.get("preview_text") or "").strip()
        remote_file_id = str(file_ref.get("remote_file_id") or "")
        try:
            size = int(file_ref.get("size") or 0)
        except Exception:
            size = 0

    full_path = _resolve_existing_file(str(file_name or ""), file_type=file_type, subfolder=subfolder)
    if not full_path:
        return None

    try:
        mime_type, _ = mimetypes.guess_type(full_path)
        if not mime_type:
            ext = full_path.lower().split(".")[-1]
            ext_map = {
                "pdf": "application/pdf",
                "mp4": "video/mp4",
                "mp3": "audio/mpeg",
                "wav": "audio/wav",
                "m4a": "audio/mp4",
                "webm": "video/webm",
                "ogg": "audio/ogg",
            }
            mime_type = ext_map.get(ext, "application/octet-stream")

        with open(full_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        category = categorize_mime(mime_type)
        source_root = folder_paths.get_temp_directory()
        source_type = "temp" if os.path.commonpath([source_root, full_path]) == source_root else "input"
        if not size:
            try:
                size = os.path.getsize(full_path)
            except OSError:
                size = 0

        resolved_name = original_name or os.path.basename(full_path)
        return {
            "mime_type": mime_type,
            "data": b64_data,
            "category": category,
            "name": os.path.basename(full_path),
            "original_name": resolved_name,
            "type": source_type,
            "subfolder": subfolder,
            "file_id": file_id or str(uuid.uuid4()),
            "preview_text": build_attachment_preview_text(resolved_name, category, preview_text),
            "size": size,
            "remote_file_id": remote_file_id,
        }
    except Exception as e:
        print(f"[CometChat] 读取文件失败 {full_path}: {e}")
        return None


def build_attachment_preview_text(name: str, category: str, preview_text: str = "") -> str:
    cleaned_preview = re.sub(r"\s+", " ", str(preview_text or "")).strip()
    if cleaned_preview:
        return cleaned_preview[:160]

    category_label_map = {
        "image": "图片",
        "video": "视频",
        "audio": "音频",
        "document": "PDF",
        "text": "文本",
    }
    category_label = category_label_map.get(category, "文件")
    return f"{category_label}附件：{name}"


def build_attachment_summary(attachments: List[dict]) -> str:
    if not attachments:
        return ""
    lines = []
    for att in attachments:
        label = att.get("original_name") or att.get("name") or "attachment"
        category = att.get("category") or "unknown"
        mime_type = att.get("mime_type") or "application/octet-stream"
        preview_text = build_attachment_preview_text(str(label), str(category), str(att.get("preview_text") or ""))
        lines.append(f"[Attachment: {label}, category={category}, mime={mime_type}] {preview_text}")
    return "\n".join(lines)


def normalize_api_format(api_format: Optional[str]) -> str:
    value = str(api_format or "").strip().lower()
    if "claude" in value:
        return "claude"
    if "gemini" in value:
        return "gemini"
    return "openai"


IMAGE_API_FORMATS = {"gemini_image", "gpt_image"}


def normalize_image_api_format(api_format: Optional[str], model: str = "") -> str:
    value = str(api_format or "").strip().lower().replace("-", "_")
    if value in IMAGE_API_FORMATS:
        return value
    if value in {"openai", "openai_image", "openai_images", "images"}:
        return "gpt_image"
    model_key = str(model or "").strip().lower()
    if "banana" in model_key or ("gemini" in model_key and "image" in model_key):
        return "gemini_image"
    if "gemini" in value:
        return "gemini_image"
    return "gpt_image"


def normalize_image_interface_mode(value: Optional[str], api_format: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    fmt = normalize_image_api_format(api_format)
    if fmt == "gemini_image":
        return text if text in {"native", "openai_compat"} else "native"
    return text if text in {"unified", "split"} else "unified"


def is_gemini_image_chat_model(api_format: Optional[str], model: str) -> bool:
    raw_format = str(api_format or "").strip().lower().replace("-", "_")
    if raw_format == "gemini_image":
        return True
    if normalize_api_format(api_format) != "gemini":
        return False
    model_key = str(model or "").strip().lower()
    if not model_key:
        return False
    if "banana" in model_key:
        return True
    return "gemini" in model_key and "image" in model_key


def is_gpt_image_chat_model(api_format: Optional[str], model: str, category: str = "") -> bool:
    raw_format = str(api_format or "").strip().lower().replace("-", "_")
    if raw_format == "gpt_image":
        return True
    if str(category or "").strip().lower() == "image" and normalize_image_api_format(raw_format, model) == "gpt_image":
        return True
    model_key = str(model or "").strip().lower().replace("_", "-")
    return bool(model_key and ("gpt-image" in model_key or "gptimage" in model_key))


def is_image_chat_model(api_format: Optional[str], model: str, category: str = "") -> bool:
    raw_format = str(api_format or "").strip().lower().replace("-", "_")
    if str(category or "").strip().lower() == "image" or raw_format in IMAGE_API_FORMATS:
        return True
    return is_gemini_image_chat_model(api_format, model) or is_gpt_image_chat_model(api_format, model, category)


def parse_image_prompt_options(prompt: str) -> dict:
    text = str(prompt or "")
    options: dict[str, str] = {}
    remove_spans: list[tuple[int, int]] = []

    pixel_match = re.search(r"(?<!\d)(\d{2,5})\s*(?:[xX×]|乘)\s*(\d{2,5})(?!\d)", text)
    if pixel_match:
        try:
            width = int(pixel_match.group(1))
            height = int(pixel_match.group(2))
            if 64 <= width <= 8192 and 64 <= height <= 8192:
                options["pixel_size"] = f"{width}x{height}"
                if width > 0 and height > 0:
                    divisor = math.gcd(width, height) or 1
                    options["aspect_ratio"] = f"{width // divisor}:{height // divisor}"
                remove_spans.append(pixel_match.span())
        except Exception:
            pass

    ratio_match = re.search(r"(?<!\d)(\d{1,4})\s*[:：比]\s*(\d{1,4})(?!\d)", text)
    if ratio_match:
        try:
            left = int(ratio_match.group(1))
            right = int(ratio_match.group(2))
            if left > 0 and right > 0:
                if not options.get("pixel_size"):
                    options["aspect_ratio"] = f"{left}:{right}"
                remove_spans.append(ratio_match.span())
        except Exception:
            pass

    size_match = re.search(r"(?<![A-Za-z0-9])([124])\s*[kK](?![A-Za-z0-9])", text)
    if size_match:
        options["image_size"] = f"{size_match.group(1)}K"
        remove_spans.append(size_match.span())

    if remove_spans:
        chars = list(text)
        for start, end in remove_spans:
            for index in range(start, end):
                chars[index] = " "
        cleaned = "".join(chars)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\s*([,，、;；])\s*", r"\1", cleaned)
        cleaned = re.sub(r"^[,，、;；\s]+|[,，、;；\s]+$", "", cleaned)
        cleaned = re.sub(r"([,，、;；]){2,}", r"\1", cleaned)
        options["prompt"] = cleaned.strip()
    else:
        options["prompt"] = text.strip()

    return options


def gemini_image_supports_size(model: str) -> bool:
    return "2.5" not in str(model or "").lower()


def validate_required_config(api_url: str, api_key: str, model: str) -> str:
    missing: List[str] = []
    if not str(api_url or "").strip():
        missing.append("API URL")
    if not str(api_key or "").strip():
        missing.append("API Key")
    if not str(model or "").strip():
        missing.append("模型名称")
    if not missing:
        return ""
    return f"请先在参数面板中配置：{'、'.join(missing)}。"


def attachment_category_label(category: str) -> str:
    if category == "image":
        return "图片"
    if category == "video":
        return "视频"
    if category == "audio":
        return "音频"
    if category == "document":
        return "PDF"
    return "文件"


def get_attachment_support_error(api_format: str, model: str, attachments: List[dict], preset: Optional[dict] = None) -> str:
    if not attachments:
        return ""

    categories = {str(att.get("category") or "unknown") for att in attachments}

    if preset:
        attach_map = preset.get("attachment_mapping", {})
        support_image = attach_map.get("support_image", False)
        support_video = attach_map.get("support_video", False)
        support_audio = attach_map.get("support_audio", False)
        support_document = attach_map.get("support_document", False)

        unsupported = []
        for category in categories:
            if category == "image" and not support_image:
                unsupported.append("图片")
            elif category == "video" and not support_video:
                unsupported.append("视频")
            elif category == "audio" and not support_audio:
                unsupported.append("音频")
            elif category == "document" and not support_document:
                unsupported.append("PDF")
            elif category not in ["image", "video", "audio", "document", "text"]:
                unsupported.append(attachment_category_label(category))
        
        if unsupported:
            return f"当前自定义预设不支持 {'、'.join(unsupported)} 附件。"
        return ""

    normalized_api_format = normalize_api_format(api_format)
    if normalized_api_format == "gemini":
        return ""

    if normalized_api_format == "claude":
        unsupported = [attachment_category_label(category) for category in categories if category not in {"image", "document"}]
        if unsupported:
            return f"Claude 原生格式当前不支持 {'、'.join(unsupported)} 附件。"
        return ""

    unsupported = [attachment_category_label(category) for category in categories if category != "image"]
    if unsupported:
        return f"OpenAI 兼容格式当前不支持 {'、'.join(unsupported)} 附件。"
    return ""


PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PLUGIN_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "cometapi_chat.db")
DB_LOCK = threading.RLock()
COMETAPI_CHAT_ROUTE_PREFIX = "/cometapi/virtual_wire_proto/chat"
COMETAPI_CHAT_BOOT_ID = str(uuid.uuid4())
COMETAPI_CHANNEL_ID_PREFIX = "cometapi_"
COMETAPI_GRS_DEFAULT_URL = "https://grsai.dakka.com.cn"


def db_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    ensure_data_dir()
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_database() -> None:
    with DB_LOCK, get_db_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv_store (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              auto_title INTEGER NOT NULL DEFAULT 1,
              pinned INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              draft TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS messages (
              id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              sort_order INTEGER NOT NULL,
              role TEXT NOT NULL,
              label TEXT NOT NULL DEFAULT '',
              text TEXT NOT NULL DEFAULT '',
              tone TEXT NOT NULL DEFAULT '',
              kind TEXT NOT NULL DEFAULT 'record',
              include_in_context INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              streaming INTEGER NOT NULL DEFAULT 0,
              FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attachments (
              id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              owner_type TEXT NOT NULL,
              owner_id TEXT NOT NULL,
              sort_order INTEGER NOT NULL DEFAULT 0,
              name TEXT NOT NULL DEFAULT '',
              original_name TEXT NOT NULL DEFAULT '',
              category TEXT NOT NULL DEFAULT 'unknown',
              type TEXT NOT NULL DEFAULT 'temp',
              subfolder TEXT NOT NULL DEFAULT '',
              mime_type TEXT NOT NULL DEFAULT '',
              preview_text TEXT NOT NULL DEFAULT '',
              size INTEGER NOT NULL DEFAULT 0,
              content TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              remote_file_id TEXT NOT NULL DEFAULT '',
              FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_order
              ON messages(session_id, sort_order);

            CREATE INDEX IF NOT EXISTS idx_attachments_owner
              ON attachments(session_id, owner_type, owner_id, sort_order);
            """
        )
        try:
            conn.execute("ALTER TABLE attachments ADD COLUMN remote_file_id TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass


def load_json_value(conn: sqlite3.Connection, key: str, fallback: Any) -> Any:
    row = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,)).fetchone()
    if not row:
        return fallback
    try:
        return json.loads(row["value"])
    except Exception:
        return fallback


def save_json_value(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        """
        INSERT INTO kv_store (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
          value = excluded.value,
          updated_at = excluded.updated_at
        """,
        (key, json.dumps(value, ensure_ascii=False), db_now_iso()),
    )


def normalize_attachment_record(raw: dict) -> dict:
    source = raw if isinstance(raw, dict) else {}
    category = str(source.get("category") or "unknown")
    name = str(source.get("name") or "")
    subfolder = str(source.get("subfolder") or "")
    file_type = normalize_chat_attachment_type(category, name, str(source.get("type") or "temp"), subfolder)
    return {
        "id": str(source.get("id") or source.get("file_id") or uuid.uuid4()),
        "name": name,
        "original_name": str(source.get("original_name") or source.get("name") or ""),
        "category": category,
        "type": file_type,
        "subfolder": subfolder,
        "mime_type": str(source.get("mime_type") or ""),
        "preview_text": build_attachment_preview_text(
            str(source.get("original_name") or name or "attachment"),
            category,
            str(source.get("preview_text") or ""),
        ),
        "size": int(source.get("size") or 0),
        "content": str(source.get("content") or ""),
        "remote_file_id": str(source.get("remote_file_id") or ""),
    }


def normalize_chat_attachment_type(category: str, name: str, file_type: str, subfolder: str) -> str:
    normalized_type = str(file_type or "temp")
    if category != "image" or subfolder != "cometapi_chat" or not str(name or "").startswith("generated_"):
        return normalized_type
    output_path = os.path.join(folder_paths.get_output_directory(), subfolder, name)
    temp_path = os.path.join(folder_paths.get_temp_directory(), subfolder, name)
    if os.path.exists(output_path) or not os.path.exists(temp_path):
        return "output"
    return normalized_type


def normalize_message_record(raw: dict) -> dict:
    source = raw if isinstance(raw, dict) else {}
    tone = "error" if str(source.get("tone") or "") == "error" else ""
    kind = str(source.get("kind") or "record")
    if kind not in {"record", "display-only", "synthetic"}:
        kind = "record"
    include_in_context = bool(source.get("includeInContext", True)) and tone != "error" and kind == "record"
    return {
        "id": str(source.get("id") or uuid.uuid4()),
        "role": "user" if str(source.get("role") or "") == "user" else "assistant",
        "label": str(source.get("label") or ""),
        "text": str(source.get("text") or ""),
        "tone": tone,
        "kind": kind,
        "includeInContext": include_in_context,
        "createdAt": str(source.get("createdAt") or db_now_iso()),
        "streaming": bool(source.get("streaming")),
        "files": [normalize_attachment_record(item) for item in (source.get("files") or []) if isinstance(item, dict)],
    }


def normalize_session_record(raw: dict) -> dict:
    source = raw if isinstance(raw, dict) else {}
    created_at = str(source.get("createdAt") or db_now_iso())
    return {
        "id": str(source.get("id") or uuid.uuid4()),
        "title": str(source.get("title") or "当前对话"),
        "autoTitle": source.get("autoTitle", True) is not False,
        "pinned": bool(source.get("pinned")),
        "createdAt": created_at,
        "updatedAt": str(source.get("updatedAt") or created_at),
        "draft": str(source.get("draft") or ""),
        "pendingFiles": [normalize_attachment_record(item) for item in (source.get("pendingFiles") or []) if isinstance(item, dict)],
        "messages": [normalize_message_record(item) for item in (source.get("messages") or []) if isinstance(item, dict)],
    }


def read_workspace_state_from_db(conn: sqlite3.Connection) -> Optional[dict]:
    workspace_meta = load_json_value(conn, "workspace_meta", None)
    config = load_json_value(conn, "plugin_config", None)

    session_rows = conn.execute(
        """
        SELECT id, title, auto_title, pinned, created_at, updated_at, draft
        FROM sessions
        ORDER BY pinned DESC, updated_at DESC, created_at DESC
        """
    ).fetchall()

    if not session_rows and config is None and workspace_meta is None:
        return None

    message_rows = conn.execute(
        """
        SELECT id, session_id, sort_order, role, label, text, tone, kind,
               include_in_context, created_at, streaming
        FROM messages
        ORDER BY session_id, sort_order ASC
        """
    ).fetchall()

    attachment_rows = conn.execute(
        """
        SELECT id, session_id, owner_type, owner_id, sort_order, name, original_name,
               category, type, subfolder, mime_type, preview_text, size, content, created_at, remote_file_id
        FROM attachments
        ORDER BY session_id, owner_type, owner_id, sort_order ASC
        """
    ).fetchall()

    attachments_by_owner: Dict[tuple, List[dict]] = {}
    for row in attachment_rows:
        key = (row["owner_type"], row["owner_id"])
        category = row["category"] or "unknown"
        name = row["name"] or ""
        subfolder = row["subfolder"] or ""
        attachments_by_owner.setdefault(key, []).append({
            "id": row["id"],
            "name": name,
            "original_name": row["original_name"],
            "category": category,
            "type": normalize_chat_attachment_type(category, name, row["type"] or "temp", subfolder),
            "subfolder": subfolder,
            "mime_type": row["mime_type"],
            "preview_text": row["preview_text"],
            "size": int(row["size"] or 0),
            "content": row["content"] or "",
            "remote_file_id": row["remote_file_id"] or "",
        })

    messages_by_session: Dict[str, List[dict]] = {}
    for row in message_rows:
        messages_by_session.setdefault(row["session_id"], []).append({
            "id": row["id"],
            "role": row["role"],
            "label": row["label"],
            "text": row["text"],
            "files": attachments_by_owner.get(("message", row["id"]), []),
            "tone": row["tone"],
            "kind": row["kind"],
            "includeInContext": bool(row["include_in_context"]),
            "createdAt": row["created_at"],
            "streaming": bool(row["streaming"]),
        })

    sessions = []
    for row in session_rows:
        session_id = row["id"]
        sessions.append({
            "id": session_id,
            "title": row["title"],
            "autoTitle": bool(row["auto_title"]),
            "pinned": bool(row["pinned"]),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "draft": row["draft"] or "",
            "pendingFiles": attachments_by_owner.get(("pending", session_id), []),
            "messages": messages_by_session.get(session_id, []),
        })

    current_session_id = ""
    sidebar_open = False
    if isinstance(workspace_meta, dict):
        current_session_id = str(workspace_meta.get("currentSessionId") or "")
        sidebar_open = workspace_meta.get("sidebarOpen") is True
    if not current_session_id and sessions:
        current_session_id = sessions[0]["id"]

    return {
        "config": config,
        "workspace_state": {
            "currentSessionId": current_session_id,
            "sidebarOpen": sidebar_open,
            "sessions": sessions,
        },
    }


def clear_stale_streaming_messages(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE messages
        SET streaming = 0
        WHERE streaming != 0
        """
    )


def replace_session_in_db(conn: sqlite3.Connection, session_payload: dict) -> None:
    session = normalize_session_record(session_payload)
    conn.execute(
        """
        INSERT INTO sessions (id, title, auto_title, pinned, created_at, updated_at, draft)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          title = excluded.title,
          auto_title = excluded.auto_title,
          pinned = excluded.pinned,
          created_at = excluded.created_at,
          updated_at = excluded.updated_at,
          draft = excluded.draft
        """,
        (
            session["id"],
            session["title"],
            1 if session["autoTitle"] else 0,
            1 if session["pinned"] else 0,
            session["createdAt"],
            session["updatedAt"],
            session["draft"],
        ),
    )

    conn.execute("DELETE FROM messages WHERE session_id = ?", (session["id"],))
    conn.execute("DELETE FROM attachments WHERE session_id = ?", (session["id"],))

    for index, attachment in enumerate(session["pendingFiles"]):
        conn.execute(
            """
            INSERT INTO attachments (
              id, session_id, owner_type, owner_id, sort_order, name, original_name,
              category, type, subfolder, mime_type, preview_text, size, content, created_at, remote_file_id
            ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attachment["id"],
                session["id"],
                session["id"],
                index,
                attachment["name"],
                attachment["original_name"],
                attachment["category"],
                attachment["type"],
                attachment["subfolder"],
                attachment["mime_type"],
                attachment["preview_text"],
                attachment["size"],
                attachment["content"],
                db_now_iso(),
                attachment.get("remote_file_id", ""),
            ),
        )

    for message_index, message in enumerate(session["messages"]):
        conn.execute(
            """
            INSERT INTO messages (
              id, session_id, sort_order, role, label, text, tone, kind,
              include_in_context, created_at, streaming
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message["id"],
                session["id"],
                message_index,
                message["role"],
                message["label"],
                message["text"],
                message["tone"],
                message["kind"],
                1 if message["includeInContext"] else 0,
                message["createdAt"],
                1 if message["streaming"] else 0,
            ),
        )
        for file_index, attachment in enumerate(message["files"]):
            conn.execute(
                """
                INSERT INTO attachments (
                  id, session_id, owner_type, owner_id, sort_order, name, original_name,
                  category, type, subfolder, mime_type, preview_text, size, content, created_at, remote_file_id
                ) VALUES (?, ?, 'message', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attachment["id"],
                    session["id"],
                    message["id"],
                    file_index,
                    attachment["name"],
                    attachment["original_name"],
                    attachment["category"],
                    attachment["type"],
                    attachment["subfolder"],
                    attachment["mime_type"],
                    attachment["preview_text"],
                    attachment["size"],
                    attachment["content"],
                    db_now_iso(),
                    attachment.get("remote_file_id", ""),
                ),
            )


def save_workspace_meta(conn: sqlite3.Connection, workspace_meta: dict) -> None:
    meta = workspace_meta if isinstance(workspace_meta, dict) else {}
    save_json_value(conn, "workspace_meta", {
        "currentSessionId": str(meta.get("currentSessionId") or ""),
        "sidebarOpen": meta.get("sidebarOpen") is True,
    })


def _stable_id(value: str, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    return text or fallback


def _get_cometapi_nodes_module():
    try:
        from . import nodes as nodes_module
    except Exception:
        import nodes as nodes_module
    return nodes_module


def _cometapi_channel_chat_id(channel_id: str) -> str:
    return f"{COMETAPI_CHANNEL_ID_PREFIX}{_stable_id(channel_id, 'channel')}"


def _cometapi_model_chat_id(channel_id: str, model_id: str) -> str:
    return f"{_cometapi_channel_chat_id(channel_id)}__{_stable_id(model_id, 'model')}"


def _cometapi_chat_api_url(nodes_module, channel_id: str) -> str:
    api_url = ""
    try:
        api_url = str(nodes_module.get_channel_api_url(channel_id) or "").strip()
    except Exception:
        api_url = ""
    if api_url:
        return api_url
    if str(channel_id or "").lower() == "grsai":
        return COMETAPI_GRS_DEFAULT_URL
    return "https://api.openai.com"


def _clean_api_key(value: Any) -> str:
    return str(value or "").strip()


def _settings_channel_api_key(channel_settings: dict) -> str:
    if not isinstance(channel_settings, dict):
        return ""
    return _clean_api_key(channel_settings.get("api_key"))


def _settings_model_api_key(channel_settings: dict, model_id: str) -> str:
    if not isinstance(channel_settings, dict):
        return ""
    model_settings = (channel_settings.get("models") or {}).get(model_id, {})
    if isinstance(model_settings, dict):
        return _clean_api_key(model_settings.get("api_key"))
    return ""


def _cometapi_chat_model_entry(
    nodes_module,
    channel_id: str,
    model_id: str,
    alias: str = "",
    api_format: str = "",
    override_api_key: str = "",
) -> dict:
    model = str(model_id or "").strip()
    if not model:
        return {}
    try:
        resolved_model = nodes_module.resolve_model_id(channel_id, model, "llm")
    except Exception:
        resolved_model = model
    try:
        final_api_format = nodes_module.get_model_api_format(channel_id, resolved_model)
    except Exception:
        final_api_format = api_format or "openai"
    return {
        "id": _cometapi_model_chat_id(channel_id, resolved_model),
        "name": resolved_model,
        "display_name": str(alias or resolved_model),
        "override_api_key": _clean_api_key(override_api_key),
        "api_format": final_api_format or api_format or "openai",
        "category": "llm",
    }


def build_cometapi_llm_chat_config() -> dict:
    """Convert CometAPI LLM channel settings into CometChat's channel config shape."""
    try:
        nodes_module = _get_cometapi_nodes_module()
        settings = nodes_module.load_comet_settings()
        catalog = nodes_module.get_model_catalog()
    except Exception as exc:
        print(f"[CometAPI Chat] failed to load CometAPI text model settings: {exc}")
        return {"channels": []}

    channels: List[dict] = []
    seen_channel_ids = set()

    for channel_id, categories in (catalog or {}).items():
        llm_items = categories.get("llm") if isinstance(categories, dict) else []
        channel_settings = (settings.get("channels") or {}).get(channel_id, {})
        channel_api_key = _settings_channel_api_key(channel_settings)
        models_by_id: Dict[str, dict] = {}

        if isinstance(llm_items, list):
            for item in llm_items:
                if not isinstance(item, dict) or not item.get("model"):
                    continue
                model_id = str(item.get("model") or "")
                model_settings = (channel_settings.get("models") or {}).get(model_id, {})
                alias = model_settings.get("alias") if isinstance(model_settings, dict) else ""
                model_api_key = _settings_model_api_key(channel_settings, model_id)
                if not channel_api_key and not model_api_key:
                    continue
                models_by_id[model_id] = _cometapi_chat_model_entry(
                    nodes_module,
                    channel_id,
                    model_id,
                    alias or item.get("alias") or model_id,
                    item.get("api_format") or "",
                    model_api_key,
                )

        for model_id, model_settings in (channel_settings.get("models") or {}).items():
            if not isinstance(model_settings, dict) or str(model_settings.get("category") or "").lower() != "llm":
                continue
            model_api_key = _clean_api_key(model_settings.get("api_key"))
            if not channel_api_key and not model_api_key:
                continue
            models_by_id.setdefault(
                str(model_id),
                _cometapi_chat_model_entry(
                    nodes_module,
                    channel_id,
                    str(model_id),
                    model_settings.get("alias") or str(model_id),
                    model_settings.get("api_format") or "",
                    model_api_key,
                ),
            )

        models = [model for model in models_by_id.values() if model]
        if not models:
            continue

        meta = categories.get("_meta") if isinstance(categories, dict) else {}
        label = str((meta or {}).get("name") or channel_id)
        chat_channel_id = _cometapi_channel_chat_id(channel_id)
        seen_channel_ids.add(chat_channel_id)
        channels.append(
            {
                "id": chat_channel_id,
                "name": label,
                "api_url": _cometapi_chat_api_url(nodes_module, channel_id),
                "api_key": channel_api_key,
                "enabled": True,
                "models": models,
                "source": "cometapi",
                "source_channel": channel_id,
            }
        )

    for custom_channel in settings.get("custom_channels") or []:
        if not isinstance(custom_channel, dict):
            continue
        channel_id = str(custom_channel.get("id") or "").strip().lower()
        if not channel_id or custom_channel.get("enabled") is False:
            continue
        channel_api_key = _clean_api_key(custom_channel.get("api_key"))
        models = []
        for item in custom_channel.get("models") or []:
            if not isinstance(item, dict) or str(item.get("category") or "").lower() != "llm":
                continue
            model_api_key = _clean_api_key(item.get("api_key"))
            if not channel_api_key and not model_api_key:
                continue
            entry = _cometapi_chat_model_entry(
                nodes_module,
                channel_id,
                str(item.get("model") or ""),
                item.get("alias") or item.get("model") or "",
                item.get("api_format") or custom_channel.get("api_format") or "openai",
                model_api_key,
            )
            if entry:
                models.append(entry)
        chat_channel_id = _cometapi_channel_chat_id(channel_id)
        if not models or chat_channel_id in seen_channel_ids:
            continue
        seen_channel_ids.add(chat_channel_id)
        channels.append(
            {
                "id": chat_channel_id,
                "name": str(custom_channel.get("name") or channel_id),
                "api_url": str(custom_channel.get("api_url") or "https://api.openai.com"),
                "api_key": channel_api_key,
                "enabled": True,
                "models": models,
                "source": "cometapi",
                "source_channel": channel_id,
            }
        )

    active_channel_id = channels[0]["id"] if channels else ""
    active_model_id = channels[0]["models"][0]["id"] if channels and channels[0].get("models") else ""
    return {
        "activeChannelId": active_channel_id,
        "activeModelId": active_model_id,
        "channels": channels,
        "presets": [],
        "system_prompt": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_tokens": 20000,
    }


init_database()


@CHAT_ROUTES.get("/cometapi/virtual_wire_proto/chat/bootstrap")
async def nkxx_cometapi_chat_bootstrap(_request):
    try:
        with DB_LOCK, get_db_connection() as conn:
            clear_stale_streaming_messages(conn)
            payload = read_workspace_state_from_db(conn)
            conn.commit()
        return web.json_response({
            "ok": True,
            "boot_id": COMETAPI_CHAT_BOOT_ID,
            "config": payload.get("config") if payload else None,
            "workspace_state": payload.get("workspace_state") if payload else None,
            "cometapi_config": build_cometapi_llm_chat_config(),
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@CHAT_ROUTES.get("/cometapi/virtual_wire_proto/chat/cometapi_llm_config")
async def nkxx_cometapi_chat_llm_config(_request):
    try:
        return web.json_response({"ok": True, "config": build_cometapi_llm_chat_config()})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@CHAT_ROUTES.post("/cometapi/virtual_wire_proto/chat/config")
async def nkxx_cometapi_chat_save_config(request):
    try:
        payload = await request.json()
        config = payload.get("config")
        with DB_LOCK, get_db_connection() as conn:
            save_json_value(conn, "plugin_config", config or {})
            conn.commit()
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@CHAT_ROUTES.post("/cometapi/virtual_wire_proto/chat/workspace_meta")
async def nkxx_cometapi_chat_save_workspace_meta(request):
    try:
        payload = await request.json()
        with DB_LOCK, get_db_connection() as conn:
            save_workspace_meta(conn, payload or {})
            conn.commit()
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@CHAT_ROUTES.post("/cometapi/virtual_wire_proto/chat/session/upsert")
async def nkxx_cometapi_chat_upsert_session(request):
    try:
        payload = await request.json()
        session = payload.get("session")
        if not isinstance(session, dict):
            return web.json_response({"error": "missing session"}, status=400)
        with DB_LOCK, get_db_connection() as conn:
            replace_session_in_db(conn, session)
            if "workspace_meta" in payload:
                save_workspace_meta(conn, payload.get("workspace_meta") or {})
            conn.commit()
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@CHAT_ROUTES.post("/cometapi/virtual_wire_proto/chat/session/delete")
async def nkxx_cometapi_chat_delete_session(request):
    try:
        payload = await request.json()
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            return web.json_response({"error": "missing session_id"}, status=400)
        with DB_LOCK, get_db_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            if "workspace_meta" in payload:
                save_workspace_meta(conn, payload.get("workspace_meta") or {})
            conn.commit()
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@CHAT_ROUTES.post("/cometapi/virtual_wire_proto/chat/upload")
async def nkxx_cometapi_chat_upload(request):
    try:
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "file":
            return web.json_response({"error": "缺少 file 字段"}, status=400)

        storage_dir = os.path.join(folder_paths.get_input_directory(), "cometapi_chat")
        os.makedirs(storage_dir, exist_ok=True)
        
        original_name = sanitize_filename(field.filename or "upload.bin")
        full_path = ensure_unique_path(storage_dir, original_name)

        with open(full_path, "wb") as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)

        mime_type = field.headers.get("Content-Type") or mimetypes.guess_type(full_path)[0] or "application/octet-stream"
        category = categorize_mime(mime_type)
        
        payload = {
            "ok": True,
            "file_id": str(uuid.uuid4()),
            "name": os.path.basename(full_path),
            "original_name": original_name,
            "mime_type": mime_type,
            "category": category,
            "type": "input",
            "subfolder": "cometapi_chat",
            "size": os.path.getsize(full_path),
            "preview_text": build_attachment_preview_text(original_name, category),
        }
        return web.json_response(payload)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ======================================================================
# 流控与僵尸线程管理核心
# ======================================================================

STREAM_CONTROL: Dict[str, Dict[str, Dict[str, Any]]] = {}
STREAM_CONTROL_LOCK = threading.Lock()

def begin_stream_control(plugin_id: str, task_kind: str = "text") -> str:
    if not plugin_id:
        return ""
    task_id = str(uuid.uuid4())
    with STREAM_CONTROL_LOCK:
        STREAM_CONTROL.setdefault(plugin_id, {})[task_id] = {
            "task_id": task_id,
            "kind": task_kind,
            "cancelled": False,
            "response": None,
        }
    return task_id

def attach_stream_response(plugin_id: str, task_id: str, response: requests.Response) -> None:
    if not plugin_id:
        return
    with STREAM_CONTROL_LOCK:
        control = STREAM_CONTROL.get(plugin_id, {}).get(task_id)
        if control:
            control["response"] = response

def is_stream_cancelled(plugin_id: str, task_id: str) -> bool:
    if not plugin_id:
        return False
    with STREAM_CONTROL_LOCK:
        control = STREAM_CONTROL.get(plugin_id, {}).get(task_id)
        if not control:
            return True
        return bool(control.get("cancelled"))

def cancel_stream_control(plugin_id: str, task_id: str = "") -> bool:
    if not plugin_id:
        return False
    responses = []
    cancelled_any = False
    with STREAM_CONTROL_LOCK:
        controls = STREAM_CONTROL.get(plugin_id)
        if not controls:
            return False
        target_ids = []
        for current_task_id, control in list(controls.items()):
            if task_id and current_task_id != task_id:
                continue
            if not task_id and control.get("kind") == "image":
                continue
            target_ids.append(current_task_id)
        for current_task_id in target_ids:
            control = controls.pop(current_task_id, None)
            if not control:
                continue
            cancelled_any = True
            control["cancelled"] = True
            response = control.get("response")
            if response is not None:
                responses.append(response)
        if not controls:
            STREAM_CONTROL.pop(plugin_id, None)
    for response in responses:
        try:
            response.close()
        except Exception:
            pass
    return cancelled_any

def clear_stream_control(plugin_id: str, task_id: str) -> None:
    if not plugin_id:
        return
    with STREAM_CONTROL_LOCK:
        controls = STREAM_CONTROL.get(plugin_id)
        if not controls:
            return
        controls.pop(task_id, None)
        if not controls:
            STREAM_CONTROL.pop(plugin_id, None)

def has_active_stream(plugin_id: str, task_kind: str = "") -> bool:
    if not plugin_id:
        return False
    with STREAM_CONTROL_LOCK:
        controls = STREAM_CONTROL.get(plugin_id) or {}
        if not task_kind:
            return bool(controls)
        return any(control.get("kind") == task_kind for control in controls.values())

def safe_send_sync(event: str, data: dict, plugin_id: str, task_id: str) -> None:
    if not plugin_id:
        return
    with STREAM_CONTROL_LOCK:
        control = STREAM_CONTROL.get(plugin_id, {}).get(task_id)
        if not control:
            return
    payload = dict(data or {})
    payload.setdefault("task_id", task_id)
    PromptServer.instance.send_sync(event, payload)


@CHAT_ROUTES.post("/cometapi/virtual_wire_proto/chat/cancel")
async def nkxx_cometapi_chat_cancel(request):
    try:
        payload = await request.json()
        plugin_id = str(payload.get("plugin_id") or "")
        task_id = str(payload.get("task_id") or "")
        if not plugin_id:
            return web.json_response({"error": "缺少 plugin_id"}, status=400)
        cancelled = cancel_stream_control(plugin_id, task_id)
        return web.json_response({"ok": True, "cancelled": cancelled})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@CHAT_ROUTES.post("/cometapi/virtual_wire_proto/chat/start")
async def nkxx_cometapi_chat_start(request):
    try:
        payload = await request.json()
        plugin_id = str(payload.get("plugin_id") or "")
        session_id = str(payload.get("session_id") or "")
        text_input = str(payload.get("text_input") or "")
        pending_files = payload.get("pending_files") or []
        config = payload.get("config") or {}
        history_data = payload.get("history") or []
        response_message_id = str(payload.get("message_id") or "")

        if not plugin_id:
            return web.json_response({"error": "缺少 plugin_id"}, status=400)
        if not text_input and not pending_files:
            return web.json_response({"error": "缺少消息内容"}, status=400)
        is_image_task = is_image_chat_model(
            (config or {}).get("api_format") if isinstance(config, dict) else "",
            str((config or {}).get("model") or "") if isinstance(config, dict) else "",
            str((config or {}).get("model_category") or (config or {}).get("category") or "") if isinstance(config, dict) else "",
        )
        if not is_image_task and has_active_stream(plugin_id, "text"):
            return web.json_response({"error": "当前插件正在生成，请先停止当前回复。"}, status=409)

        task_id = begin_stream_control(plugin_id, "image" if is_image_task else "text")

        def worker():
            try:
                run_chat_task(
                    plugin_id=plugin_id,
                    task_id=task_id,
                    session_id=session_id,
                    text_input=text_input,
                    pending_files=pending_files if isinstance(pending_files, list) else [],
                    config=config if isinstance(config, dict) else {},
                    history_data=history_data if isinstance(history_data, list) else [],
                    response_message_id=response_message_id,
                )
            finally:
                clear_stream_control(plugin_id, task_id)

        thread = threading.Thread(target=worker, name=f"cometapi-chat-{plugin_id}", daemon=True)
        thread.start()
        return web.json_response({"ok": True, "plugin_id": plugin_id, "session_id": session_id, "task_id": task_id, "mode": "image" if is_image_task else "text"})
    except Exception as e:
        try:
            plugin_id = str(payload.get("plugin_id") or "")
            with STREAM_CONTROL_LOCK:
                STREAM_CONTROL.pop(plugin_id, None)
        except Exception:
            pass
        return web.json_response({"error": str(e)}, status=500)


@CHAT_ROUTES.post("/cometapi/virtual_wire_proto/chat/selection/assist")
async def nkxx_cometapi_chat_selection_assist(request):
    try:
        payload = await request.json()
        action = str(payload.get("action") or "").strip().lower()
        text = str(payload.get("text") or "").strip()
        target_language = str(payload.get("target_language") or "").strip()
        config = payload.get("config") or {}
        if action not in {"translate", "explain", "optimize"}:
            return web.json_response({"error": "未知划词助手操作"}, status=400)
        if not text:
            return web.json_response({"error": "缺少选中文本"}, status=400)
        if len(text) > 12000:
            return web.json_response({"error": "选中文本过长，请缩短后再试"}, status=400)
        if not isinstance(config, dict):
            return web.json_response({"error": "缺少划词助手模型配置"}, status=400)
        result = await asyncio.to_thread(
            run_selection_assistant_sync,
            action,
            text,
            target_language,
            config,
        )
        return web.json_response({"ok": True, "result": result})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ======================================================================
# Provider Registry
# ======================================================================

class BaseChatProvider:
    def format_payload(self, model: str, system_prompt: str, history: list, new_text: str, attachments: list, kwargs: dict) -> dict:
        raise NotImplementedError

    def get_url(self, base_url: str, model: str) -> str:
        return base_url

    def get_headers(self, api_key: str) -> dict:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def build_user_history_message(self, text: str, attachments: list) -> dict:
        raise NotImplementedError

    def build_assistant_message(self, text: str) -> dict:
        raise NotImplementedError


class OpenAICompatibleProvider(BaseChatProvider):
    def get_url(self, base_url: str, model: str) -> str:
        base = base_url.strip().rstrip("/")
        if "/chat/completions" in base:
            return base
            
        parts = base.split("://")
        domain_path = parts[1] if len(parts) >= 2 else parts[0]
        
        if "/" not in domain_path:
            return f"{base}/v1/chat/completions"
        else:
            return f"{base}/chat/completions"

    def format_payload(self, model, system_prompt, history, new_text, attachments, kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(history)

        current_content = []
        if new_text:
            current_content.append({"type": "text", "text": new_text})

        for att in attachments:
            if att["category"] == "image":
                mime = att.get("mime_type", "image/jpeg")
                current_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{att['data']}"},
                })
            else:
                print(f"[CometChat] OpenAI compatible mode ignores unsupported attachment {att['category']} ({att.get('name')}).")

        if len(current_content) == 1 and current_content[0]["type"] == "text":
            messages.append({"role": "user", "content": new_text})
        else:
            messages.append({"role": "user", "content": current_content})

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }

        return payload

    def build_user_history_message(self, text: str, attachments: list) -> dict:
        content = text or ""
        summary = build_attachment_summary(attachments)
        if summary:
            content = f"{content}\n\n{summary}".strip()
        return {"role": "user", "content": content}

    def build_assistant_message(self, text: str) -> dict:
        return {"role": "assistant", "content": text}


class AnthropicProvider(BaseChatProvider):
    def get_url(self, base_url: str, model: str) -> str:
        if not base_url.endswith("/v1/messages"):
            return base_url.rstrip("/") + "/v1/messages"
        return base_url

    def get_headers(self, api_key: str) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    def format_payload(self, model, system_prompt, history, new_text, attachments, kwargs):
        messages = history.copy()
        current_content = []

        for att in attachments:
            if att["category"] == "image":
                current_content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": att["mime_type"], "data": att["data"]},
                })
            elif att["category"] == "document":
                current_content.append({
                    "type": "document",
                    "source": {"type": "base64", "media_type": att["mime_type"], "data": att["data"]},
                    "cache_control": {"type": "ephemeral"},
                })
            else:
                print(f"[CometChat] Claude native mode ignores unsupported attachment {att['category']} ({att.get('name')}).")

        if new_text:
            current_content.append({"type": "text", "text": new_text})

        messages.append({"role": "user", "content": current_content or [{"type": "text", "text": ""}]})

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def build_user_history_message(self, text: str, attachments: list) -> dict:
        content = []
        summary = build_attachment_summary(attachments)
        if text or summary:
            combined = text or ""
            if summary:
                combined = f"{combined}\n\n{summary}".strip()
            content.append({"type": "text", "text": combined})
        if not content:
            content = [{"type": "text", "text": ""}]
        return {"role": "user", "content": content}

    def build_assistant_message(self, text: str) -> dict:
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}


class GeminiNativeProvider(BaseChatProvider):
    # [修复2]: 修正 URL 生成丢失 `?alt=sse` 参数的问题
    def get_url(self, base_url: str, model: str) -> str:
        base = base_url.rstrip("/")
        url = ""
        
        if "/models/" in base and ":streamGenerateContent" in base:
            url = base
        elif base.endswith("/v1") or base.endswith("/v1beta"):
            url = f"{base}/models/{model}:streamGenerateContent"
        else:
            url = f"{base}/v1beta/models/{model}:streamGenerateContent"
            
        if "alt=sse" not in url:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}alt=sse"
            
        return url

    def get_headers(self, api_key: str) -> dict:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

    def format_payload(self, model, system_prompt, history, new_text, attachments, kwargs):
        contents = history.copy()
        parts = []

        if new_text:
            parts.append({"text": new_text})

        for att in attachments:
            # [修复1]: 修正为官方要求的驼峰命名 inlineData 和 mimeType
            parts.append({
                "inlineData": {
                    "mimeType": att["mime_type"],
                    "data": att["data"],
                }
            })

        # [修复4]: 空文本保护，防止 parts 为空字符串导致 Gemini 报错
        payload = {
            "contents": contents + [{"role": "user", "parts": parts or [{"text": " "}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 2048),
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        return payload

    def build_user_history_message(self, text: str, attachments: list) -> dict:
        combined = text or ""
        summary = build_attachment_summary(attachments)
        if summary:
            combined = f"{combined}\n\n{summary}".strip()
        return {"role": "user", "parts": [{"text": combined or " "}]}

    def build_assistant_message(self, text: str) -> dict:
        return {"role": "model", "parts": [{"text": text or " "}]}


# ======================================================================
# 全局 JSON Path 提取器
# ======================================================================
def get_json_path(obj: Any, path: str) -> Any:
    if not path: return None
    curr = obj
    try:
        for key in path.split('.'):
            if isinstance(curr, list):
                curr = curr[int(key)]
            else:
                curr = curr.get(key)
    except Exception:
        return None
    return curr

# ======================================================================
# 格式预设多模态引擎 (PresetProvider)
# ======================================================================

class PresetProvider(BaseChatProvider):
    def __init__(self, preset: dict):
        self.preset = preset
        self.msg_map = preset.get("message_mapping", {})
        self.attach_map = preset.get("attachment_mapping", {})

    def get_url(self, base_url: str, model: str) -> str:
        url_template = self.preset.get("url_template", "{api_url}")
        return url_template.replace("{api_url}", base_url.strip().rstrip("/"))

    def get_headers(self, api_key: str) -> dict:
        headers_template = self.preset.get("headers_template", {})
        headers = {}
        for k, v in headers_template.items():
            headers[k] = str(v).replace("{api_key}", api_key)
        return headers

    def process_uploads(self, attachments: list, api_url: str, api_key: str):
        upload_step = self.preset.get("upload_step")
        if not upload_step or not upload_step.get("enabled"):
            return

        upload_url_tpl = upload_step.get("url", "{api_url}/v1/files")
        upload_url = upload_url_tpl.replace("{api_url}", api_url.strip().rstrip("/"))
        
        headers_tpl = upload_step.get("headers", {})
        req_headers = {}
        for k, v in headers_tpl.items():
            if k.lower() == "content-type" and "multipart" in str(v).lower():
                continue
            req_headers[k] = str(v).replace("{api_key}", api_key)
            
        file_field = upload_step.get("file_field_name", "file")
        extra_fields = upload_step.get("extra_fields", {})
        extractor = upload_step.get("response_extractor", "id")

        for att in attachments:
            if att.get("remote_file_id"):
                continue

            try:
                file_bytes = base64.b64decode(att["data"])
                files = {file_field: (att.get("name", "file.bin"), file_bytes, att.get("mime_type", "application/octet-stream"))}
                data = dict(extra_fields)
                
                print(f"[CometChat Preset] 正在执行云端上传任务: {att.get('name')} ...")
                resp = requests.post(upload_url, headers=req_headers, files=files, data=data, timeout=120)
                resp.raise_for_status()
                
                remote_id = get_json_path(resp.json(), extractor)
                if remote_id:
                    att["remote_file_id"] = str(remote_id)
                    print(f"[CometChat Preset] 上传成功，缓存 Remote ID: {remote_id}")
                    
                    file_id = att.get("file_id")
                    if file_id:
                        with DB_LOCK, get_db_connection() as conn:
                            conn.execute("UPDATE attachments SET remote_file_id = ? WHERE id = ?", (str(remote_id), file_id))
                            conn.commit()
            except Exception as e:
                print(f"[CometChat Preset] 预设上传任务失败，回退等待直传 (如果配置了直传): {e}")

    def _build_content(self, text: str, attachments: list):
        support_image = self.attach_map.get("support_image", False)
        support_video = self.attach_map.get("support_video", False)
        support_audio = self.attach_map.get("support_audio", False)
        support_document = self.attach_map.get("support_document", False)

        text_tpl_str = json.dumps(self.attach_map.get("text_template", {"type": "text", "text": "{text}"}))
        img_tpl_str = json.dumps(self.attach_map.get("image_template", {"type": "image_url", "image_url": {"url": "data:{mime_type};base64,{data}"}}))
        video_tpl_str = json.dumps(self.attach_map.get("video_template", {"type": "video", "video_url": "data:{mime_type};base64,{data}"}))
        audio_tpl_str = json.dumps(self.attach_map.get("audio_template", {"type": "audio", "audio_url": "data:{mime_type};base64,{data}"}))
        doc_tpl_str = json.dumps(self.attach_map.get("document_template", {"type": "document", "document_url": "data:{mime_type};base64,{data}"}))

        if not attachments:
            return text

        content_list = []
        if text:
            txt_block = json.loads(text_tpl_str.replace('"{text}"', json.dumps(text)))
            content_list.append(txt_block)
            
        for att in attachments:
            cat = att.get("category", "unknown")
            mime_type = att.get("mime_type", "application/octet-stream")
            b64_data = att.get("data", "")
            remote_id = att.get("remote_file_id", "")
            
            if cat == "image" and support_image:
                content_list.append(json.loads(img_tpl_str.replace("{mime_type}", mime_type).replace("{data}", b64_data).replace("{remote_file_id}", remote_id)))
            elif cat == "video" and support_video:
                content_list.append(json.loads(video_tpl_str.replace("{mime_type}", mime_type).replace("{data}", b64_data).replace("{remote_file_id}", remote_id)))
            elif cat == "audio" and support_audio:
                content_list.append(json.loads(audio_tpl_str.replace("{mime_type}", mime_type).replace("{data}", b64_data).replace("{remote_file_id}", remote_id)))
            elif cat == "document" and support_document:
                content_list.append(json.loads(doc_tpl_str.replace("{mime_type}", mime_type).replace("{data}", b64_data).replace("{remote_file_id}", remote_id)))
            else:
                print(f"[CometChat Preset] 忽略预设不支持的附件类型: {cat}")
        
        if len(content_list) == 0:
            return text
        if len(content_list) == 1 and content_list[0].get("type") == "text" and not text:
            return text
            
        return content_list

    def build_user_history_message(self, text: str, attachments: list) -> dict:
        role_val = self.msg_map.get("user_role", "user")
        role_key = self.msg_map.get("role_key", "role")
        content_key = self.msg_map.get("content_key", "content")
        return {
            role_key: role_val,
            content_key: self._build_content(text, attachments)
        }

    def build_assistant_message(self, text: str) -> dict:
        role_val = self.msg_map.get("assistant_role", "assistant")
        role_key = self.msg_map.get("role_key", "role")
        content_key = self.msg_map.get("content_key", "content")
        return {
            role_key: role_val,
            content_key: text
        }

    def format_payload(self, model: str, system_prompt: str, history: list, new_text: str, attachments: list, kwargs: dict) -> dict:
        api_url = str(kwargs.get("api_url") or "")
        api_key = str(kwargs.get("api_key") or "")
        
        self.process_uploads(attachments, api_url, api_key)
        
        messages = []
        if system_prompt:
            sys_role = self.msg_map.get("system_role", "system")
            role_key = self.msg_map.get("role_key", "role")
            content_key = self.msg_map.get("content_key", "content")
            messages.append({role_key: sys_role, content_key: system_prompt})
            
        messages.extend(history)
        
        if new_text or attachments:
            usr_role = self.msg_map.get("user_role", "user")
            role_key = self.msg_map.get("role_key", "role")
            content_key = self.msg_map.get("content_key", "content")
            messages.append({
                role_key: usr_role,
                content_key: self._build_content(new_text, attachments)
            })
            
        body_template = self.preset.get("body_template", {})
        body_str = json.dumps(body_template)
        body_str = body_str.replace('"{model}"', json.dumps(model))
        body_str = body_str.replace('"{temperature}"', str(kwargs.get("temperature", 0.7)))
        body_str = body_str.replace('"{max_tokens}"', str(kwargs.get("max_tokens", 20000)))
        body_str = body_str.replace('"{messages}"', json.dumps(messages))
        
        return json.loads(body_str)


def build_selection_assistant_prompt(action: str, text: str, target_language: str = "") -> tuple[str, str]:
    selected = str(text or "").strip()
    system_prompt = (
        "你是 Comet 划词助手。回答要直接、清爽、可复制；不要编造上下文；"
        "除非用户文本本身要求，否则不要输出多余寒暄。"
    )
    if action == "translate":
        target = str(target_language or "中文（简体）").strip()
        user_prompt = (
            f"请将下面的划词内容翻译成{target}。\n"
            "要求：只输出译文；保留代码、URL、专有名词和必要格式；不要解释。\n\n"
            f"划词内容：\n{selected}"
        )
    elif action == "explain":
        user_prompt = (
            "请用中文解释下面的划词内容。\n"
            "要求：简洁准确；先给一句话结论，再列出必要要点；不要过度扩展。\n\n"
            f"划词内容：\n{selected}"
        )
    else:
        user_prompt = (
            "请在不编造信息的前提下，把下面的内容整理成更清晰、可直接复制的 AI 提示词。\n"
            "要求：保持原意和原语言；不要擅自添加具体事实、风格、背景或身份设定；"
            "如果信息不足，只做轻量润色，并在最后用一句话指出建议补充的关键点；"
            "如果明显是图像生成提示词，优先整理主体、场景、风格、构图、光照和细节；"
            "如果明显是任务提示词，优先整理目标、背景、要求和输出格式；"
            "结果要简洁可复制，不要输出分析过程。\n\n"
            f"原始内容：\n{selected}"
        )
    return system_prompt, user_prompt


def extract_non_stream_response_text(data: Any, api_format: str, preset: Optional[dict] = None) -> str:
    if not isinstance(data, dict):
        return ""
    candidates = [
        "choices.0.message.content",
        "choices.0.delta.content",
        "choices.0.message.reasoning_content",
        "content.0.text",
        "candidates.0.content.parts.0.text",
        "data.text",
        "text",
    ]
    if preset:
        parser_cfg = preset.get("stream_parser", {}) if isinstance(preset, dict) else {}
        content_path = str(parser_cfg.get("content_path") or "")
        if content_path:
            candidates.insert(0, content_path.replace(".delta.", ".message."))
            candidates.insert(1, content_path)
    normalized_format = normalize_api_format(api_format)
    if normalized_format == "gemini":
        parts = []
        for candidate in data.get("candidates") or []:
            for part in ((candidate.get("content") or {}).get("parts") or []):
                if isinstance(part, dict) and part.get("text"):
                    parts.append(str(part.get("text")))
        if parts:
            return "".join(parts).strip()
    if normalized_format == "claude":
        parts = []
        for item in data.get("content") or []:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item.get("text")))
        if parts:
            return "".join(parts).strip()
    for path in candidates:
        value = get_json_path(data, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def run_selection_assistant_sync(action: str, text: str, target_language: str, config: Dict[str, Any]) -> str:
    api_url = str(config.get("api_url") or "")
    api_key = str(config.get("api_key") or "")
    model = str(config.get("model") or "")
    api_format = config.get("api_format", "openai")
    model_category = str(config.get("model_category") or config.get("category") or "llm").strip().lower()
    if model_category == "image" or is_image_chat_model(api_format, model, model_category):
        raise ValueError("划词助手只能使用文本模型。")

    config_error = validate_required_config(api_url, api_key, model)
    if config_error:
        raise ValueError(config_error)

    preset = next((p for p in config.get("presets", []) if p.get("id") == api_format), None)
    if preset:
        provider: BaseChatProvider = PresetProvider(preset)
    else:
        normalized_api_format = normalize_api_format(api_format)
        if normalized_api_format == "claude":
            provider = AnthropicProvider()
        elif normalized_api_format == "gemini":
            provider = GeminiNativeProvider()
        else:
            provider = OpenAICompatibleProvider()

    system_prompt, user_prompt = build_selection_assistant_prompt(action, text, target_language)
    payload_kwargs = dict(config)
    payload_kwargs["api_url"] = api_url
    payload_kwargs["api_key"] = api_key
    payload_kwargs["temperature"] = config.get("temperature", 0.2)
    payload_kwargs["max_tokens"] = min(max(int(config.get("max_tokens") or 20000), 512), 20000)
    payload = provider.format_payload(model, system_prompt, [], user_prompt, [], payload_kwargs)

    url = provider.get_url(api_url, model)
    headers = provider.get_headers(api_key)
    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=(5, 180))
    response.raise_for_status()
    response.encoding = "utf-8"

    content_type = str(response.headers.get("content-type") or "").lower()
    if "text/event-stream" not in content_type and "stream" not in content_type:
        data = response.json()
        text_result = extract_non_stream_response_text(data, api_format, preset=preset)
        if not text_result:
            raise ValueError("划词助手没有收到有效回复。")
        return text_result

    full_response_text = ""
    stream_state: Dict[str, Any] = {"in_think": False, "gemini_last_full": ""}
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        chunk = extract_stream_chunk(api_format, data, stream_state, preset=preset)
        if chunk:
            full_response_text += chunk
    if stream_state.get("in_think"):
        full_response_text += "\n</think>\n"
    full_response_text = full_response_text.strip()
    if not full_response_text:
        raise ValueError("划词助手没有收到有效回复。")
    return full_response_text


# ======================================================================
# Streaming chunk 提取
# ======================================================================

def extract_stream_chunk(api_format: str, data: dict, state: dict, preset: Optional[dict] = None) -> str:
    parts: List[str] = []

    if preset:
        parser_cfg = preset.get("stream_parser", {})
        content_path = parser_cfg.get("content_path", "choices.0.delta.content")
        thinking_path = parser_cfg.get("thinking_path", "choices.0.delta.reasoning_content")

        think_chunk = get_json_path(data, thinking_path)
        if think_chunk:
            if not state.get("in_think"):
                parts.append("<think>\n")
                state["in_think"] = True
            parts.append(str(think_chunk))

        content_chunk = get_json_path(data, content_path)
        if content_chunk:
            if state.get("in_think"):
                parts.append("\n</think>\n")
                state["in_think"] = False
            parts.append(str(content_chunk))

        return "".join(parts)

    normalized_format = normalize_api_format(api_format)

    if normalized_format == "openai":
        choices = data.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}

        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            if not state.get("in_think"):
                parts.append("<think>\n")
                state["in_think"] = True
            parts.append(reasoning)

        normal_text = delta.get("content")
        if normal_text:
            if state.get("in_think"):
                parts.append("\n</think>\n")
                state["in_think"] = False
            parts.append(normal_text)

    elif normalized_format == "claude":
        event_type = data.get("type")
        if event_type == "content_block_delta":
            delta = data.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "thinking_delta":
                if not state.get("in_think"):
                    parts.append("<think>\n")
                    state["in_think"] = True
                parts.append(delta.get("thinking", ""))
            elif delta_type == "text_delta":
                if state.get("in_think"):
                    parts.append("\n</think>\n")
                    state["in_think"] = False
                parts.append(delta.get("text", ""))
        elif event_type == "message_delta":
            if state.get("in_think"):
                parts.append("\n</think>\n")
                state["in_think"] = False

    elif normalized_format == "gemini":
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts_data = candidates[0].get("content", {}).get("parts") or []
        text_joined = "".join(p.get("text", "") for p in parts_data if p.get("text"))
        
        # [修复3]: 移除了会导致截断吞字的 startswith(last_full) 累加判断。
        # Gemini 官方 SSE 流本来就是 Delta 增量返回，直接拼接即可。
        if text_joined:
            parts.append(text_joined)

    return "".join(parts)


def build_gemini_image_url(api_url: str, model: str) -> str:
    base = str(api_url or "").strip().rstrip("/")
    if "/models/" in base and (":generateContent" in base or ":streamGenerateContent" in base):
        url = base.replace(":streamGenerateContent", ":generateContent")
        return re.sub(r"([?&])alt=sse(&|$)", r"\1", url).rstrip("?&")
    if base.endswith("/v1") or base.endswith("/v1beta"):
        return f"{base}/models/{model}:generateContent"
    return f"{base}/v1beta/models/{model}:generateContent"


def save_generated_chat_image(image_data: str, mime_type: str, index: int = 1) -> dict:
    raw = str(image_data or "")
    if "," in raw and raw.lower().startswith("data:"):
        header, raw = raw.split(",", 1)
        match = re.match(r"data:([^;]+)", header, flags=re.I)
        if match:
            mime_type = match.group(1)
    image_bytes = base64.b64decode(raw)

    storage_dir = os.path.join(folder_paths.get_output_directory(), "cometapi_chat")
    os.makedirs(storage_dir, exist_ok=True)
    filename = ensure_unique_path(storage_dir, f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}.png")

    with Image.open(io.BytesIO(image_bytes)) as img:
        if img.mode not in {"RGB", "RGBA"}:
            img = img.convert("RGBA")
        img.save(filename, format="PNG")

    saved_name = os.path.basename(filename)
    return {
        "id": str(uuid.uuid4()),
        "name": saved_name,
        "original_name": saved_name,
        "category": "image",
        "type": "output",
        "subfolder": "cometapi_chat",
        "mime_type": "image/png",
        "size": os.path.getsize(filename),
        "preview_text": "生成图片",
    }


def save_generated_chat_pil_image(image: Image.Image, index: int = 1) -> dict:
    storage_dir = os.path.join(folder_paths.get_output_directory(), "cometapi_chat")
    os.makedirs(storage_dir, exist_ok=True)
    filename = ensure_unique_path(storage_dir, f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}.png")

    img = image.copy()
    if img.mode not in {"RGB", "RGBA"}:
        img = img.convert("RGBA")
    img.save(filename, format="PNG")

    saved_name = os.path.basename(filename)
    return {
        "id": str(uuid.uuid4()),
        "name": saved_name,
        "original_name": saved_name,
        "category": "image",
        "type": "output",
        "subfolder": "cometapi_chat",
        "mime_type": "image/png",
        "size": os.path.getsize(filename),
        "preview_text": "生成图片",
    }


def extract_gemini_image_response(data: dict) -> tuple[list[dict], str]:
    files: list[dict] = []
    texts: list[str] = []
    image_index = 1

    for candidate in (data.get("candidates") or []):
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if part.get("text"):
                texts.append(str(part.get("text") or ""))
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                mime_type = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                files.append(save_generated_chat_image(str(inline.get("data") or ""), str(mime_type), image_index))
                image_index += 1

    return files, "".join(texts).strip()


def image_attachments_to_pil(attachments: List[dict]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for att in attachments:
        if str(att.get("category") or "") != "image" or not att.get("data"):
            continue
        try:
            raw = base64.b64decode(str(att.get("data") or ""))
            image = Image.open(io.BytesIO(raw)).copy()
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA")
            images.append(image)
        except Exception as exc:
            print(f"[CometChat Image] failed to decode reference image {att.get('name')}: {exc}")
    return images


def nearest_chat_image_aspect_ratio(pil_images: list[Image.Image], supported_ratios: List[str], fallback: str = "9:16") -> str:
    concrete_ratios = [str(ratio) for ratio in supported_ratios if str(ratio or "").strip() and str(ratio) != "auto"]
    if not pil_images:
        return fallback if fallback in concrete_ratios or not concrete_ratios else concrete_ratios[0]
    try:
        width, height = pil_images[0].size
        if width <= 0 or height <= 0:
            return fallback if fallback in concrete_ratios or not concrete_ratios else concrete_ratios[0]
        image_ratio = width / height
        return min(
            concrete_ratios,
            key=lambda ratio: abs((float(ratio.split(":")[0]) / float(ratio.split(":")[1])) - image_ratio),
        )
    except Exception:
        return fallback if fallback in concrete_ratios or not concrete_ratios else concrete_ratios[0]


def default_chat_image_aspect_ratio(pil_images: list[Image.Image], api_format: str, model: str, nodes_module=None) -> str:
    fmt = normalize_image_api_format(api_format, model)
    supported: List[str] = []
    if nodes_module is not None:
        if fmt == "gpt_image":
            supported = list(getattr(nodes_module, "GPT_IMAGE_ASPECT_RATIOS", []) or [])
        else:
            supported = list(getattr(nodes_module, "PRIVATE_GEMINI_ASPECT_RATIOS", []) or [])
    if not supported:
        supported = [
            "auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
            "9:16", "16:9", "21:9", "9:21", "1:4", "4:1", "1:8", "8:1",
        ]
    return nearest_chat_image_aspect_ratio(pil_images, supported, "9:16")


def is_grsai_image_config(config: Dict[str, Any]) -> bool:
    values = [
        config.get("source_channel"),
        config.get("channel_id"),
        config.get("channel_name"),
        config.get("api_url"),
    ]
    text = " ".join(str(value or "").lower() for value in values)
    return "grsai" in text or "grsai.dakka.com.cn" in text or "grsaiapi.com" in text


def run_gemini_image_chat_task(
    plugin_id: str,
    task_id: str,
    session_id: str,
    message_id: str,
    text_input: str,
    attachments: List[dict],
    config: Dict[str, Any],
) -> None:
    api_url = str(config.get("api_url") or "")
    api_key = str(config.get("api_key") or "")
    model = str(config.get("model") or "")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    parsed_options = parse_image_prompt_options(text_input)
    prompt_text = parsed_options.get("prompt") or text_input or " "
    pil_refs = image_attachments_to_pil(attachments)
    aspect_ratio = parsed_options.get("aspect_ratio") or default_chat_image_aspect_ratio(pil_refs, "gemini_image", model)
    parts: list[dict] = [{"text": prompt_text}]
    for att in attachments:
        if str(att.get("category") or "") != "image":
            continue
        parts.append(
            {
                "inlineData": {
                    "mimeType": att.get("mime_type") or "image/png",
                    "data": att.get("data") or "",
                }
            }
        )

    image_config: dict[str, str] = {
        "aspectRatio": aspect_ratio,
    }
    if gemini_image_supports_size(model):
        image_config["imageSize"] = parsed_options.get("image_size") or "2K"

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": config.get("temperature", 0.7),
            "maxOutputTokens": config.get("max_tokens", 20000),
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": image_config,
        },
    }

    safe_send_sync("cometapi_chat_image_start", {
        "plugin_id": plugin_id,
        "session_id": session_id,
        "message_id": message_id,
    }, plugin_id, task_id)

    response = requests.post(build_gemini_image_url(api_url, model), headers=headers, json=payload, timeout=(5, 300))
    if response.status_code >= 400:
        error_text = response.text
        try:
            error_data = response.json()
            error_text = (
                (error_data.get("error") or {}).get("message")
                if isinstance(error_data.get("error"), dict)
                else error_data.get("message")
            ) or response.text
        except Exception:
            pass
        raise RuntimeError(f"API错误: {error_text}")
    data = response.json()
    files, text = extract_gemini_image_response(data)
    if not files:
        raise RuntimeError(text or "Gemini 图片模型响应里没有图片数据")

    safe_send_sync("cometapi_chat_image_result", {
        "plugin_id": plugin_id,
        "session_id": session_id,
        "message_id": message_id,
        "files": files,
        "text": text,
    }, plugin_id, task_id)
    safe_send_sync("cometapi_chat_image_end", {
        "plugin_id": plugin_id,
        "session_id": session_id,
        "message_id": message_id,
        "cancelled": False,
    }, plugin_id, task_id)


def run_custom_image_chat_task(
    plugin_id: str,
    task_id: str,
    session_id: str,
    message_id: str,
    text_input: str,
    attachments: List[dict],
    config: Dict[str, Any],
) -> None:
    api_url = str(config.get("api_url") or "")
    api_key = str(config.get("api_key") or "")
    model = str(config.get("model") or "")
    api_format = normalize_image_api_format(config.get("api_format"), model)
    interface_mode = normalize_image_interface_mode(config.get("interface_mode"), api_format)
    parsed_options = parse_image_prompt_options(text_input)
    prompt = parsed_options.get("prompt") or text_input or " "
    image_size = parsed_options.get("image_size") or "1K"
    pixel_size = parsed_options.get("pixel_size") or ""
    pil_images = image_attachments_to_pil(attachments)
    nodes_module = _get_cometapi_nodes_module()
    aspect_ratio = parsed_options.get("aspect_ratio") or default_chat_image_aspect_ratio(pil_images, api_format, model, nodes_module)

    safe_send_sync("cometapi_chat_image_start", {
        "plugin_id": plugin_id,
        "session_id": session_id,
        "message_id": message_id,
    }, plugin_id, task_id)

    generated: list[Image.Image] = []
    errors: list[str] = []

    model_key = model.strip().lower()
    grsai_gpt_models = set(getattr(nodes_module, "GPT_IMAGE_MODELS", []))
    if api_format == "gpt_image" and is_grsai_image_config(config) and model_key in {str(item).lower() for item in grsai_gpt_models}:
        urls: list[str] = []
        upload_image_grsai = getattr(nodes_module, "upload_image_grsai", None)
        max_refs = int(getattr(nodes_module, "GPT_IMAGE_MAX_IMAGES", 16) or 16)
        if upload_image_grsai:
            for image in pil_images[:max_refs]:
                url = upload_image_grsai(api_key, image)
                if url:
                    urls.append(url)
        valid_ratios = set(getattr(nodes_module, "GPT_IMAGE_ASPECT_RATIOS", []))
        safe_aspect = aspect_ratio if aspect_ratio in valid_ratios else "1:1"
        generated, errors = nodes_module.GrsaiAPI(api_key).gpt_image_generate(
            prompt=prompt,
            model=model,
            urls=urls,
            aspect_ratio=safe_aspect,
            image_size=image_size,
            quality="medium",
        )
    else:
        generated, errors = nodes_module.CustomOpenAIImageAPI(api_key, api_url).generate_image(
            prompt=prompt,
            model=model,
            pil_images=pil_images,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            quality="medium",
            api_format=api_format,
            interface_mode=interface_mode,
            pixel_size=pixel_size,
        )

    if not generated:
        raise RuntimeError("; ".join(errors) if errors else "图片接口没有返回图片")

    files = [save_generated_chat_pil_image(image, index) for index, image in enumerate(generated, start=1)]
    safe_send_sync("cometapi_chat_image_result", {
        "plugin_id": plugin_id,
        "session_id": session_id,
        "message_id": message_id,
        "files": files,
        "text": "",
    }, plugin_id, task_id)
    safe_send_sync("cometapi_chat_image_end", {
        "plugin_id": plugin_id,
        "session_id": session_id,
        "message_id": message_id,
        "cancelled": False,
    }, plugin_id, task_id)


def run_chat_task(plugin_id: str, task_id: str, session_id: str, text_input: str, pending_files: List[dict], config: Dict[str, Any], history_data: List[dict], response_message_id: str = "") -> None:
    attachments: List[dict] = []
    api_url = str(config.get("api_url") or "")
    api_key = str(config.get("api_key") or "")
    model = str(config.get("model") or "")
    api_format = config.get("api_format", "openai")
    model_category = str(config.get("model_category") or config.get("category") or "").strip().lower()
    is_image_chat = is_image_chat_model(api_format, model, model_category)

    config_error = validate_required_config(api_url, api_key, model)

    if isinstance(pending_files, list):
        for item in pending_files:
            file_info = resolve_file_to_base64(item)
            if file_info:
                attachments.append(file_info)

    if not text_input and not attachments:
        return

    if config_error:
        safe_send_sync("cometapi_chat_image_error" if is_image_chat else "cometapi_chat_stream_error", {
            "plugin_id": plugin_id,
            "session_id": session_id,
            "message_id": response_message_id,
            "error": config_error,
        }, plugin_id, task_id)
        return

    preset = next((p for p in config.get("presets", []) if p.get("id") == api_format), None)

    if is_image_chat:
        unsupported = [attachment_category_label(str(att.get("category") or "unknown")) for att in attachments if str(att.get("category") or "") != "image"]
        if unsupported:
            safe_send_sync("cometapi_chat_image_error", {
                "plugin_id": plugin_id,
                "session_id": session_id,
                "message_id": response_message_id,
                "error": f"图片模型当前只支持图片参考图，不支持 {'、'.join(unsupported)} 附件。",
            }, plugin_id, task_id)
            return
        try:
            if is_gemini_image_chat_model(api_format, model) and str(api_format or "").strip().lower().replace("-", "_") != "gemini_image":
                run_gemini_image_chat_task(
                    plugin_id=plugin_id,
                    task_id=task_id,
                    session_id=session_id,
                    message_id=response_message_id,
                    text_input=text_input,
                    attachments=attachments,
                    config=config,
                )
            else:
                run_custom_image_chat_task(
                    plugin_id=plugin_id,
                    task_id=task_id,
                    session_id=session_id,
                    message_id=response_message_id,
                    text_input=text_input,
                    attachments=attachments,
                    config=config,
                )
        except Exception as exc:
            print(f"[CometChat Image Error] 图片生成失败:\n{traceback.format_exc()}")
            safe_send_sync("cometapi_chat_image_error", {
                "plugin_id": plugin_id,
                "session_id": session_id,
                "message_id": response_message_id,
                "error": str(exc),
            }, plugin_id, task_id)
        return

    attachment_error = get_attachment_support_error(api_format, model, attachments, preset=preset)
    if attachment_error:
        safe_send_sync("cometapi_chat_stream_error", {
            "plugin_id": plugin_id,
            "session_id": session_id,
            "error": attachment_error,
        }, plugin_id, task_id)
        return

    if preset:
        provider: BaseChatProvider = PresetProvider(preset)
    else:
        normalized_api_format = normalize_api_format(api_format)
        if normalized_api_format == "claude":
            provider = AnthropicProvider()
        elif normalized_api_format == "gemini":
            provider = GeminiNativeProvider()
        else:
            provider = OpenAICompatibleProvider()

    chat_history = []
    for msg in history_data:
        r = msg.get("role")
        t = msg.get("text", "")
        fs = msg.get("files", [])
        
        if r == "user":
            resolved_fs = []
            for f in fs:
                rf = resolve_file_to_base64(f)
                if rf:
                    resolved_fs.append(rf)
            chat_history.append(provider.build_user_history_message(t, resolved_fs))
        elif r == "assistant":
            chat_history.append(provider.build_assistant_message(t))

    url = provider.get_url(api_url, model)
    headers = provider.get_headers(api_key)
    payload_kwargs = dict(config)
    payload_kwargs["api_url"] = api_url
    payload_kwargs["api_key"] = api_key
    
    payload = provider.format_payload(
        model,
        str(config.get("system_prompt") or ""),
        chat_history,
        text_input,
        attachments,
        payload_kwargs,
    )

    full_response_text = ""
    stream_state: Dict[str, Any] = {"in_think": False, "gemini_last_full": ""}
    cancelled = False
    stream_end_sent = False

    try:
        safe_send_sync("cometapi_chat_stream_start", {"plugin_id": plugin_id, "session_id": session_id}, plugin_id, task_id)

        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=(5, 180))
        attach_stream_response(plugin_id, task_id, response)
        response.raise_for_status()
        response.encoding = "utf-8"

        for raw_line in response.iter_lines(decode_unicode=True):
            if is_stream_cancelled(plugin_id, task_id):
                cancelled = True
                break

            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            chunk = extract_stream_chunk(api_format, data, stream_state, preset=preset)
            if chunk:
                full_response_text += chunk
                safe_send_sync("cometapi_chat_stream_chunk", {
                    "plugin_id": plugin_id,
                    "session_id": session_id,
                    "chunk": chunk,
                }, plugin_id, task_id)

        if stream_state.get("in_think") and not cancelled:
            full_response_text += "\n</think>\n"
            safe_send_sync("cometapi_chat_stream_chunk", {
                "plugin_id": plugin_id,
                "session_id": session_id,
                "chunk": "\n</think>\n",
            }, plugin_id, task_id)

        safe_send_sync("cometapi_chat_stream_end", {
            "plugin_id": plugin_id,
            "session_id": session_id,
            "cancelled": cancelled,
        }, plugin_id, task_id)
        stream_end_sent = True

    except Exception as e:
        cancelled = cancelled or is_stream_cancelled(plugin_id, task_id)
        if cancelled:
            if not stream_end_sent:
                safe_send_sync("cometapi_chat_stream_end", {
                    "plugin_id": plugin_id,
                    "session_id": session_id,
                    "cancelled": True,
                }, plugin_id, task_id)
        else:
            print(f"[CometChat Error] API 请求失败:\n{traceback.format_exc()}")
            safe_send_sync("cometapi_chat_stream_error", {
                "plugin_id": plugin_id,
                "session_id": session_id,
                "error": str(e),
            }, plugin_id, task_id)

# ======================================================================
# 导出空对象防止 ComfyUI 报错
# ======================================================================

_register_deferred_chat_routes_when_ready()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

