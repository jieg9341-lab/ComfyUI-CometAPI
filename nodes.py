from __future__ import annotations

import concurrent.futures
import base64
import asyncio
import copy
import importlib.util
import json
import logging
import mimetypes
import os
import re
import secrets
import tempfile
import threading
import time
import traceback
import uuid
import wave
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image, ImageFilter, ImageOps

try:
    from comfy.comfy_types import IO
except Exception:
    class _ComfyIOFallback:
        VIDEO = "VIDEO"
        AUDIO = "AUDIO"

    IO = _ComfyIOFallback()


class CometAnyType(str):
    def __ne__(self, _value):
        return False


COMET_ANY = CometAnyType("*")
COMET_BATCH_TEXT = "COMET_BATCH_TEXT"
COMET_BATCH_IMAGE_MODE_REGULAR = "常规批量"
COMET_BATCH_IMAGE_MODE_FOLDER = "文件夹批量"
COMET_BATCH_IMAGE_MODES = [COMET_BATCH_IMAGE_MODE_REGULAR, COMET_BATCH_IMAGE_MODE_FOLDER]
COMET_BATCH_PAIRING_ALL_REFS = "全部图片作为一组参考，跑每条提示词"
COMET_BATCH_PAIRING_EACH_REF = "每张图跑全部提示词"
COMET_BATCH_PAIRING_ORDERED = "图片和提示词一一配对"
COMET_BATCH_IMAGE_PAIRING_MODES = [
    COMET_BATCH_PAIRING_ALL_REFS,
    COMET_BATCH_PAIRING_EACH_REF,
    COMET_BATCH_PAIRING_ORDERED,
]
COMET_BATCH_FOLDER_PAIRING_MODE = COMET_BATCH_PAIRING_EACH_REF
COMET_BATCH_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
BATCH_IMAGE_FOLDER_PREVIEW_LIMIT = 50
BATCH_IMAGE_FOLDER_CIRCUIT_BREAKER_FAILURES = 10
BATCH_IMAGE_FAILURE_POLICY_WHITE = "生成原比例白图(保持编号)"
BATCH_IMAGE_FAILURE_POLICY_SKIP = "直接跳过"
BATCH_IMAGE_FAILURE_POLICIES = [BATCH_IMAGE_FAILURE_POLICY_WHITE, BATCH_IMAGE_FAILURE_POLICY_SKIP]
BATCH_IMAGE_FAILURE_POLICY_ALIASES = {
    "white": BATCH_IMAGE_FAILURE_POLICY_WHITE,
    "placeholder": BATCH_IMAGE_FAILURE_POLICY_WHITE,
    "fallback": BATCH_IMAGE_FAILURE_POLICY_WHITE,
    "生成原比例白图": BATCH_IMAGE_FAILURE_POLICY_WHITE,
    "生成原比例白图(保持编号)": BATCH_IMAGE_FAILURE_POLICY_WHITE,
    "skip": BATCH_IMAGE_FAILURE_POLICY_SKIP,
    "direct_skip": BATCH_IMAGE_FAILURE_POLICY_SKIP,
    "直接跳过": BATCH_IMAGE_FAILURE_POLICY_SKIP,
}


SUPPORTED_ASPECT_RATIOS = [
    "auto",
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "5:4",
    "4:5",
    "21:9",
    "9:21",
    "1:3",
    "3:1",
    "2:1",
    "1:2",
    "1:4",
    "1:8",
    "4:1",
    "8:1",
]
LLM_CHANNELS = ["grsai", "apimart", "runninghub", "modelverse"]
IMAGE_CHANNELS = ["grsai", "runninghub", "modelverse", "apimart"]
VIDEO_CHANNELS = ["apimart", "runninghub", "modelverse"]
MUSIC_CHANNELS = ["runninghub"]
PRESET_CHANNELS = ("grsai", "runninghub", "modelverse", "apimart")
CUSTOM_CHANNEL_MODEL_CATEGORIES = {"llm", "image"}
PRIVATE_MODEL_PASSTHROUGH_KEYS = (
    "family",
    "api_format",
    "interface_mode",
    "capabilities",
    "ui",
    "runtime",
    "adapter",
    "endpoint",
)
LLM_API_FORMATS = {"openai", "gemini", "claude"}
IMAGE_API_FORMATS = {"gemini_image", "gpt_image"}
IMAGE_INTERFACE_MODES_GEMINI = {"native", "openai_compat"}
IMAGE_INTERFACE_MODES_GPT = {"unified", "split"}
GEMINI_SUB_FAMILY_2_5 = "gemini_2_5"
GEMINI_SUB_FAMILY_3 = "gemini_3"
# 自定义渠道异步任务轮询参数
CUSTOM_IMAGE_ASYNC_POLL_INTERVAL_SEC = 3
CUSTOM_IMAGE_ASYNC_POLL_TIMEOUT_SEC = 600  # 10 分钟
MUSIC_MODE_GENERATE = "\u751f\u6210\u6b4c\u66f2"
MUSIC_MODE_LYRICS = "\u751f\u6210\u6b4c\u8bcd"
MUSIC_MODES = [MUSIC_MODE_GENERATE, MUSIC_MODE_LYRICS]
MUSIC_SUBMODE_INSPIRE = "\u7075\u611f\u6a21\u5f0f"
MUSIC_SUBMODE_CUSTOM = "\u81ea\u5b9a\u4e49\u6b4c\u8bcd"
MUSIC_SUBMODE_INSTRUMENTAL = "\u7eaf\u97f3\u4e50"
MUSIC_SUBMODE_LYRICS = "\u751f\u6210\u6b4c\u8bcd"
MUSIC_SUBMODES = [
    MUSIC_SUBMODE_INSPIRE,
    MUSIC_SUBMODE_CUSTOM,
    MUSIC_SUBMODE_INSTRUMENTAL,
    MUSIC_SUBMODE_LYRICS,
]
MUSIC_MODEL_V45 = "chirp-auk"
MUSIC_MODEL_V5 = "chirp-v5"
MUSIC_MODEL_V55 = "chirp-fenix"
MUSIC_MODEL_UPLOAD_EXTEND = "chirp-v3-5-upload"
MUSIC_MODEL_UPLOAD_COVER = "chirp-v3-5-tau"
MUSIC_MODELS = [MUSIC_MODEL_V55, MUSIC_MODEL_V5, MUSIC_MODEL_V45]
MUSIC_MODEL_ALIASES = {
    MUSIC_MODEL_V55: "Suno v5.5",
    MUSIC_MODEL_V5: "Suno v5",
    MUSIC_MODEL_V45: "Suno v4.5",
}
RUNNINGHUB_MUSIC_MODEL_V55 = "suno-v5.5"
RUNNINGHUB_MUSIC_MODEL_V5 = "suno-v5"
RUNNINGHUB_MUSIC_MODEL_V45 = "suno-v4.5"
RUNNINGHUB_MUSIC_MODELS = [RUNNINGHUB_MUSIC_MODEL_V55, RUNNINGHUB_MUSIC_MODEL_V5, RUNNINGHUB_MUSIC_MODEL_V45]
RUNNINGHUB_MUSIC_MODEL_ALIASES = {
    RUNNINGHUB_MUSIC_MODEL_V55: "Suno v5.5",
    RUNNINGHUB_MUSIC_MODEL_V5: "Suno v5",
    RUNNINGHUB_MUSIC_MODEL_V45: "Suno v4.5",
}
RUNNINGHUB_MUSIC_MODEL_ALIAS_TO_ID = {alias.lower(): model for model, alias in RUNNINGHUB_MUSIC_MODEL_ALIASES.items()}
MUSIC_INPUT_MODELS = list(dict.fromkeys(RUNNINGHUB_MUSIC_MODELS))
MUSIC_VOCAL_GENDERS = ["\u81ea\u52a8", "\u7537\u58f0", "\u5973\u58f0"]
NANO_BANANA_MODELS = [
    "nano-banana-fast",
    "nano-banana-pro",
    "nano-banana-pro-vt",
    "nano-banana-pro-cl",
    "nano-banana-2",
    "nano-banana-2-cl",
]
GPT_IMAGE_MODELS = ["gpt-image-2", "gpt-image-2-vip"]
PRIVATE_MODELS = [
    "gpt-image-2",
    "gpt-image-2-all",
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
    "doubao-seedream-5-0-260128",
    "doubao-seedream-4-5-251128",
    "doubao-seedream-4-0-250828",
    "grok-4.2-image",
]
MODELVERSE_IMAGE_MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
    "gpt-image-2",
    "doubao-seedream-4.5",
    "doubao-seedream-5-0-260128",
]
APIMART_IMAGE_MODELS = [
    "gpt-image-2",
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image-preview",
    "gpt-image-2-official",
    "gemini-3.1-flash-image-preview-official",
    "gemini-3-pro-image-preview-official",
    "gemini-2.5-flash-image-preview-official",
]
APIMART_IMAGE_MODEL_ALIASES = {
    "gemini-3.1-flash-image-preview": "banana-2",
    "gemini-3.1-flash-image-preview-official": "banana-2-官方",
    "gemini-3-pro-image-preview": "banana-pro",
    "gemini-3-pro-image-preview-official": "banana-pro-官方",
    "gemini-2.5-flash-image-preview": "banana-1",
    "gemini-2.5-flash-image-preview-official": "banana-1-官方",
    "gpt-image-2": "gpt-image-2",
    "gpt-image-2-official": "gpt-image-2-官方",
}
APIMART_GEMINI_IMAGE_MODELS = {
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-image-preview-official",
    "gemini-3-pro-image-preview",
    "gemini-3-pro-image-preview-official",
    "gemini-2.5-flash-image-preview",
    "gemini-2.5-flash-image-preview-official",
}
APIMART_GEMINI_31_IMAGE_MODELS = {"gemini-3.1-flash-image-preview", "gemini-3.1-flash-image-preview-official"}
APIMART_GEMINI_25_IMAGE_MODELS = {"gemini-2.5-flash-image-preview", "gemini-2.5-flash-image-preview-official"}
APIMART_GPT_IMAGE_MODELS = {"gpt-image-2", "gpt-image-2-official"}
RUNNINGHUB_IMAGE_MODELS = [
    "rhart-image-g-2",
    "rhart-image-g-2-official",
    "rhart-image-v1",
    "rhart-image-n-g31-flash",
    "rhart-image-n-pro",
    "rhart-image-v1-official",
    "rhart-image-n-g31-flash-official",
    "rhart-image-n-pro-official",
    "rhart-image-n-pro-official-ultra",
]
RUNNINGHUB_IMAGE_MODEL_ALIASES = {
    "rhart-image-g-2": "gpt-image2低价",
    "rhart-image-g-2-official": "gpt-image2官方",
    "rhart-image-v1": "banana1-低价",
    "rhart-image-n-g31-flash": "banana2-低价",
    "rhart-image-n-pro": "bananapro-低价",
    "rhart-image-v1-official": "banana1-官方",
    "rhart-image-n-g31-flash-official": "banana2-官方",
    "rhart-image-n-pro-official": "bananapro-官方",
    "rhart-image-n-pro-official-ultra": "bananapro-ultra-官方",
}
RUNNINGHUB_IMAGE_MODEL_ALIAS_TO_ID = {alias.lower(): model for model, alias in RUNNINGHUB_IMAGE_MODEL_ALIASES.items()}
RUNNINGHUB_IMAGE_V1_MODELS = {"rhart-image-v1", "rhart-image-v1-official"}
RUNNINGHUB_IMAGE_QUALITY_MODELS = {"rhart-image-g-2-official"}
RUNNINGHUB_IMAGE_ULTRA_MODELS = {"rhart-image-n-pro-official-ultra"}
RUNNINGHUB_IMAGE_MAX_IMAGES = 10
RUNNINGHUB_IMAGE_V1_MAX_IMAGES = 5
RUNNINGHUB_IMAGE_RESOLUTIONS = ["1k", "2k", "4k"]
RUNNINGHUB_IMAGE_ULTRA_RESOLUTIONS = ["4k", "8k"]
SUPPORTED_MODELS = list(dict.fromkeys(NANO_BANANA_MODELS + GPT_IMAGE_MODELS + RUNNINGHUB_IMAGE_MODELS + MODELVERSE_IMAGE_MODELS + APIMART_IMAGE_MODELS))

# LLM 文本模型配置
LLM_MODELS = [
    "gemini-3-flash",
    "gemini-3.5-flash",
    "gemini-3.1-pro",
    "gemini-3-pro",
    "gemini-3.1-flash-lite",
    "gpt-5.5",
    "gpt-5.4",
]
# grsai 渠道下各 LLM 模型的接口格式：默认 gemini，gpt-5.x 走 openai 兼容
GRSAI_LLM_API_FORMATS = {
    "gpt-5.5": "openai",
    "gpt-5.4": "openai",
}
PRIVATE_LLM_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
]
PRIVATE_LLM_MODEL_ALIASES = {
    "gemini-3.1-pro-preview": "gemini-3.1-pro",
    "gemini-3-pro-preview": "gemini-3-pro",
    "gemini-3-flash-preview": "gemini-3-flash",
}
PRIVATE_LLM_API_FORMATS = {model: "gemini" for model in PRIVATE_LLM_MODELS}
RUNNINGHUB_LLM_MODELS = [
    "google/gemini-3-flash-preview",
    "google/gemini-3.5-flash",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-5.5",
    "qwen/qwen3.6-plus",
    "bytedance/doubao-seed-2.0-pro",
    "bytedance/doubao-seed-2.0-lite",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
]
RUNNINGHUB_LLM_MODEL_ALIASES = {
    "google/gemini-3-flash-preview": "gemini-3-flash",
    "google/gemini-3.5-flash": "gemini-3.5-flash",
    "google/gemini-3.1-pro-preview": "gemini-3.1-pro",
    "google/gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
    "openai/gpt-5.5": "gpt-5.5",
    "qwen/qwen3.6-plus": "qwen3.6-plus",
    "bytedance/doubao-seed-2.0-pro": "doubao-seed-2.0-pro",
    "bytedance/doubao-seed-2.0-lite": "doubao-seed-2.0-lite",
    "deepseek/deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek/deepseek-v4-pro": "deepseek-v4-pro",
}
RUNNINGHUB_LLM_API_FORMATS = {model: "openai" for model in RUNNINGHUB_LLM_MODELS}
MODELVERSE_LLM_GEMINI_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]
MODELVERSE_LLM_OPENAI_MODELS = [
    "gpt-5.5",
    "gpt-5.4-mini",
    "qwen3.6-plus",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
]
MODELVERSE_LLM_MODELS = MODELVERSE_LLM_GEMINI_MODELS + MODELVERSE_LLM_OPENAI_MODELS
MODELVERSE_LLM_MODEL_ALIASES = {
    "gemini-3.1-pro-preview": "gemini-3.1-pro",
    "gemini-3-flash-preview": "gemini-3-flash",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
}
MODELVERSE_LLM_API_FORMATS = {
    **{model: "gemini" for model in MODELVERSE_LLM_GEMINI_MODELS},
    **{model: "openai" for model in MODELVERSE_LLM_OPENAI_MODELS},
}
APIMART_LLM_GEMINI_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
]
APIMART_LLM_OPENAI_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
]
APIMART_LLM_MODELS = APIMART_LLM_GEMINI_MODELS + APIMART_LLM_OPENAI_MODELS
APIMART_LLM_MODEL_ALIASES = {
    "gemini-3-flash-preview": "gemini-3-flash",
    "gemini-3.1-pro-preview": "gemini-3.1-pro",
}
APIMART_LLM_API_FORMATS = {
    **{model: "gemini" for model in APIMART_LLM_GEMINI_MODELS},
    **{model: "openai" for model in APIMART_LLM_OPENAI_MODELS},
}
MAX_LLM_IMAGE_INPUTS = 10
MAX_LLM_VIDEO_INPUTS = 3
MAX_LLM_AUDIO_INPUTS = 3
DEFAULT_LLM_MAX_OUTPUT_TOKENS = 20000
PRIVATE_VIDEO_MODELS = [
    "sora-2",
    "grok-video-3",
    "veo3.1-fast",
    "veo3.1-lite",
    "veo3.1",
    "omni-flash",
    "viduq3",
    "viduq3-pro",
    "viduq3-turbo",
    "viduq3-mix",
    "viduq2",
    "viduq2-turbo",
    "viduq2-pro",
    "MiniMax-Hailuo-2.3",
    "MiniMax-Hailuo-02",
    "kling-v2-6",
    "kling-v3",
    "kling-video-o1",
    "kling-v3-omni",
    "happyhorse-1.0",
]
PRIVATE_VIDEO_MODEL_FAMILIES = ["grok", "veo", "vidu", "hailuo", "kling", "happyhorse", "sora"]
RUNNINGHUB_VIDEO_MODELS = [
    "sparkvideo-2.0",
    "sparkvideo-2.0-fast",
    "gemini-omni-flash",
    "grok-video-1.5",
    "happyhorse-1.0",
]
MODELVERSE_VIDEO_MODELS = [
    "doubao-seedance-2-0-260128",
    "happyhorse-1.0",
    "sora-2",
    "kling-v2-6",
    "kling-v3",
    "kling-video-o1",
    "kling-v3-omni",
]
APIMART_VIDEO_MODELS = [
    "doubao-seedance-2.0",
    "doubao-seedance-2.0-fast",
    "grok-imagine-1.0-video-apimart",
    "sora-2",
    "sora-2-pro",
    "veo3.1-fast",
    "veo3.1-quality",
    "veo3.1-lite",
    "Omni-Flash-Ext",
    "MiniMax-Hailuo-2.3",
    "happyhorse-1.0",
    "kling-v3",
    "kling-v3-omni",
    "kling-video-o1",
    "viduq3-pro",
    "viduq3-turbo",
    "viduq3",
    "viduq3-mix",
    "wan2.7",
    "wan2.7-r2v",
    "wan2.7-videoedit",
]
RUNNINGHUB_VIDEO_MODEL_FAMILIES = ["seedance", "google", "grok", "happyhorse"]
MODELVERSE_VIDEO_MODEL_FAMILIES = ["seedance", "happyhorse", "sora", "kling"]
APIMART_VIDEO_MODEL_FAMILIES = ["seedance", "grok", "sora", "veo", "hailuo", "happyhorse", "kling", "vidu", "wan"]
RUNNINGHUB_VIDEO_MODEL_ALIASES = {
    "sparkvideo-2.0": "seedance-2",
    "sparkvideo-2.0-fast": "seedance-2-fast",
    "gemini-omni-flash": "Gemini Omni Flash",
    "grok-video-1.5": "Grok Video 1.5",
    "happyhorse-1.0": "happyhorse-1.0",
}
RUNNINGHUB_VIDEO_MODEL_ALIAS_TO_ID = {alias.lower(): model for model, alias in RUNNINGHUB_VIDEO_MODEL_ALIASES.items()}
MODELVERSE_VIDEO_MODEL_ALIASES = {
    "doubao-seedance-2-0-260128": "seedance-2.0",
    "happyhorse-1.0": "happyhorse-1.0",
    "sora-2": "sora-2",
    "kling-v2-6": "kling-v2-6",
    "kling-v3": "kling-v3",
    "kling-video-o1": "kling-video-o1",
    "kling-v3-omni": "kling-v3-omni",
}
MODELVERSE_VIDEO_MODEL_ALIAS_TO_ID = {alias.lower(): model for model, alias in MODELVERSE_VIDEO_MODEL_ALIASES.items()}
APIMART_VIDEO_MODEL_ALIASES = {
    "doubao-seedance-2.0": "seedance-2.0",
    "doubao-seedance-2.0-fast": "seedance-2.0-fast",
    "doubao-seedance-2.0-face": "seedance-2.0",
    "doubao-seedance-2.0-fast-face": "seedance-2.0-fast",
    "grok-imagine-1.0-video-apimart": "grok-imagine",
    "sora-2": "sora-2",
    "sora-2-pro": "sora-2-pro",
    "veo3.1-fast": "veo-3.1-fast",
    "veo3.1-quality": "veo-3.1-quality",
    "veo3.1-lite": "veo-3.1-lite",
    "Omni-Flash-Ext": "omni-flash",
    "MiniMax-Hailuo-2.3": "hailuo-2.3",
    "happyhorse-1.0": "happyhorse-1.0",
    "kling-v3": "kling-v3",
    "kling-v3-omni": "kling-v3-omni",
    "kling-video-o1": "kling-video-o1",
    "viduq3-pro": "viduq3-pro",
    "viduq3-turbo": "viduq3-turbo",
    "viduq3": "viduq3",
    "viduq3-mix": "viduq3-mix",
    "wan2.7": "wan2.7",
    "wan2.7-r2v": "wan2.7-r2v",
    "wan2.7-videoedit": "wan2.7-videoedit",
}
APIMART_VIDEO_MODEL_ALIAS_TO_ID = {alias.lower(): model for model, alias in APIMART_VIDEO_MODEL_ALIASES.items()}
RUNNINGHUB_SEEDANCE_MODELS = {"sparkvideo-2.0", "sparkvideo-2.0-fast"}
RUNNINGHUB_GOOGLE_MODELS = {"gemini-omni-flash"}
RUNNINGHUB_GROK_MODELS = {"grok-video-1.5"}
RUNNINGHUB_HAPPYHORSE_MODELS = {"happyhorse-1.0"}
RUNNINGHUB_SEEDANCE_MODES = ["文生", "首尾帧", "全能参考"]
RUNNINGHUB_GOOGLE_MODES = ["文生", "多参", "视频编辑"]
RUNNINGHUB_GROK_MODES = ["文生", "图生"]
RUNNINGHUB_HAPPYHORSE_MODES = ["文生", "图生", "多图参考", "视频编辑"]
MODELVERSE_SEEDANCE_MODELS = {"doubao-seedance-2-0-260128"}
MODELVERSE_HAPPYHORSE_MODELS = {"happyhorse-1.0"}
MODELVERSE_SORA_VIDEO_MODELS = {"sora-2"}
MODELVERSE_KLING_VIDEO_MODELS = {"kling-v2-6", "kling-v3", "kling-video-o1", "kling-v3-omni"}
MODELVERSE_KLING_OMNI_VIDEO_MODELS = {"kling-video-o1", "kling-v3-omni"}
MODELVERSE_SEEDANCE_MODES = ["文生", "首尾帧", "全能参考"]
MODELVERSE_HAPPYHORSE_MODES = ["文生", "图生", "多图参考", "视频编辑"]
MODELVERSE_SORA_VIDEO_MODES = ["文生", "图生"]
MODELVERSE_KLING_VIDEO_MODES = ["文生", "图生", "首尾帧", "运动控制"]
MODELVERSE_KLING_OMNI_VIDEO_MODES = ["Omni"]
APIMART_SEEDANCE_VIDEO_MODELS = {
    "doubao-seedance-2.0",
    "doubao-seedance-2.0-fast",
}
APIMART_GROK_VIDEO_MODELS = {"grok-imagine-1.0-video-apimart"}
APIMART_SORA_VIDEO_MODELS = {"sora-2", "sora-2-pro"}
APIMART_VEO_VIDEO_MODELS = {"veo3.1-fast", "veo3.1-quality", "veo3.1-lite", "Omni-Flash-Ext"}
APIMART_HAILUO_VIDEO_MODELS = {"MiniMax-Hailuo-2.3"}
APIMART_HAPPYHORSE_VIDEO_MODELS = {"happyhorse-1.0"}
APIMART_KLING_VIDEO_MODELS = {"kling-v3", "kling-v3-omni", "kling-video-o1"}
APIMART_KLING_OMNI_VIDEO_MODELS = {"kling-v3-omni", "kling-video-o1"}
APIMART_VIDU_VIDEO_MODELS = {"viduq3-pro", "viduq3-turbo", "viduq3", "viduq3-mix"}
APIMART_WAN_VIDEO_MODELS = {"wan2.7", "wan2.7-r2v", "wan2.7-videoedit"}
APIMART_SEEDANCE_VIDEO_MODES = ["文生", "首尾帧", "全能参考"]
APIMART_GROK_VIDEO_MODES = ["文生", "多参"]
APIMART_SORA_VIDEO_MODES = ["文生", "图生"]
APIMART_VEO_VIDEO_MODES = ["文生", "首尾帧", "多参"]
APIMART_OMNI_FLASH_VIDEO_MODES = ["文生", "多参", "视频参考"]
APIMART_HAILUO_VIDEO_MODES = ["文生", "图生"]
APIMART_HAPPYHORSE_VIDEO_MODES = ["文生", "图生", "多图参考", "视频编辑"]
APIMART_KLING_VIDEO_MODES = ["文生", "图生", "首尾帧", "视频参考"]
APIMART_KLING_OMNI_VIDEO_MODES = ["文生", "多参", "视频参考"]
APIMART_VIDU_VIDEO_MODES = ["参考", "图生", "首尾帧", "文生"]
APIMART_WAN_VIDEO_MODES = ["文生", "图生", "首尾帧", "视频参考", "视频编辑", "多图参考"]
RUNNINGHUB_SEEDANCE_ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
RUNNINGHUB_GOOGLE_ASPECT_RATIOS = ["16:9", "9:16"]
RUNNINGHUB_GROK_ASPECT_RATIOS = ["2:3", "3:2", "1:1", "16:9", "9:16"]
RUNNINGHUB_HAPPYHORSE_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
MODELVERSE_SEEDANCE_ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
MODELVERSE_VIDEO_ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
APIMART_SEEDANCE_ASPECT_RATIOS = ["adaptive", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
APIMART_GROK_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "3:2", "2:3"]
APIMART_VIDEO_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
APIMART_KLING_ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
RUNNINGHUB_SEEDANCE_RESOLUTIONS = ["480p", "720p", "native1080p", "1080p", "2k", "4k"]
RUNNINGHUB_SEEDANCE_FAST_RESOLUTIONS = ["480p", "720p", "1080p", "2k", "4k"]
RUNNINGHUB_GOOGLE_RESOLUTIONS = ["720p", "1080p", "4k"]
RUNNINGHUB_GROK_RESOLUTIONS = ["720p", "480p"]
RUNNINGHUB_HAPPYHORSE_RESOLUTIONS = ["720p", "1080p"]
MODELVERSE_SEEDANCE_RESOLUTIONS = ["480p", "720p", "1080p"]
MODELVERSE_HAPPYHORSE_RESOLUTIONS = ["720P", "1080P"]
MODELVERSE_KLING_QUALITY_MODES = ["std", "pro"]
APIMART_SEEDANCE_RESOLUTIONS = ["480p", "720p", "1080p"]
APIMART_SEEDANCE_FAST_RESOLUTIONS = ["480p", "720p"]
APIMART_GROK_RESOLUTIONS = ["480p", "720p"]
APIMART_SORA_RESOLUTIONS = ["720p", "1024p", "1080p"]
APIMART_VEO_RESOLUTIONS = ["720p", "1080p", "4k"]
APIMART_HAILUO_RESOLUTIONS = ["768p", "1080p"]
APIMART_HAPPYHORSE_RESOLUTIONS = ["720P", "1080P"]
APIMART_KLING_QUALITY_MODES = ["std", "pro", "4k"]
APIMART_VIDU_RESOLUTIONS = ["540p", "720p", "1080p"]
APIMART_WAN_RESOLUTIONS = ["720P", "1080P"]
APIMART_OMNI_FLASH_RESOLUTIONS = ["720p", "1080p", "4k"]
RUNNINGHUB_SEEDANCE_DURATIONS = [str(value) for value in range(4, 16)]
RUNNINGHUB_GOOGLE_DURATIONS = ["4", "6", "8", "10"]
RUNNINGHUB_GROK_DURATIONS = [str(value) for value in range(6, 31)]
RUNNINGHUB_HAPPYHORSE_DURATIONS = [str(value) for value in range(3, 16)]
MODELVERSE_SEEDANCE_DURATIONS = [str(value) for value in range(4, 16)]
MODELVERSE_HAPPYHORSE_DURATIONS = [str(value) for value in range(3, 16)]
MODELVERSE_SORA_VIDEO_DURATIONS = ["4", "8", "12"]
MODELVERSE_KLING_V26_DURATIONS = ["5", "10"]
MODELVERSE_KLING_O1_DURATIONS = ["5", "10"]
MODELVERSE_KLING_V3_DURATIONS = [str(value) for value in range(3, 16)]
APIMART_SEEDANCE_DURATIONS = [str(value) for value in range(4, 16)]
APIMART_GROK_DURATIONS = [str(value) for value in range(6, 31)]
APIMART_SORA_DURATIONS = ["4", "8", "12", "16", "20"]
APIMART_VEO_DURATIONS = ["8"]
APIMART_HAILUO_DURATIONS = ["6", "10"]
APIMART_HAPPYHORSE_DURATIONS = [str(value) for value in range(3, 16)]
APIMART_KLING_V3_DURATIONS = [str(value) for value in range(3, 16)]
APIMART_KLING_O1_DURATIONS = ["5", "10"]
APIMART_VIDU_DURATIONS = [str(value) for value in range(1, 17)]
APIMART_WAN_DURATIONS = [str(value) for value in range(2, 16)]
APIMART_WAN_VIDEO_EDIT_DURATIONS = ["0"] + [str(value) for value in range(2, 11)]
APIMART_OMNI_FLASH_DURATIONS = ["4", "6", "8", "10"]
RUNNINGHUB_MAX_IMAGES = 9
RUNNINGHUB_GROK_MAX_IMAGES = 7
RUNNINGHUB_MAX_VIDEOS = 3
RUNNINGHUB_MAX_AUDIOS = 3
MODELVERSE_VIDEO_MAX_IMAGES = 9
MODELVERSE_VIDEO_MAX_VIDEOS = 3
MODELVERSE_VIDEO_MAX_AUDIOS = 3
APIMART_MAX_IMAGES = 9
APIMART_MAX_VIDEOS = 5
APIMART_MAX_AUDIOS = 3
PRIVATE_VEO_VIDEO_REAL_MODELS = {
    "veo3.1-fast",
    "veo3.1-lite",
    "veo3.1",
    "veo3.1-fast-4K",
    "veo3.1-lite-4K",
    "veo3.1-4K",
    "veo3.1-components",
    "veo3.1-components-4K",
    "veo3.1-fast-components-4K",
    "omni-flash",
    "omni-flash-components",
}
PRIVATE_VIDEO_MODEL_ALIASES = {
    "sora-2": "sora-2",
    "grok-videos": "grok-videos",
    "grok-video-3": "grok-video-3",
    "veo3.1-fast": "veo-3.1-fast",
    "veo3.1-lite": "veo-3.1-lite",
    "veo3.1": "veo-3.1",
    "omni-flash": "omni-flash",
    "viduq3": "viduq3",
    "viduq3-pro": "viduq3-pro",
    "viduq3-turbo": "viduq3-turbo",
    "viduq3-mix": "viduq3-mix",
    "viduq2": "viduq2",
    "viduq2-turbo": "viduq2-turbo",
    "viduq2-pro": "viduq2-pro",
    "MiniMax-Hailuo-2.3": "hailuo-2.3",
    "MiniMax-Hailuo-02": "hailuo-02",
    "kling-v2-6": "kling-v2-6",
    "kling-v3": "kling-v3",
    "kling-video-o1": "kling-video-o1",
    "kling-v3-omni": "kling-v3-omni",
    "happyhorse-1.0": "happyhorse-1.0",
}
PRIVATE_KLING_LEGACY_MODEL_ALIASES = {
    "kling-v2-6": "v2.6",
    "kling-v3": "v3",
    "kling-video-o1": "o1",
    "kling-v3-omni": "v3-omni",
}
PRIVATE_SORA_VIDEO_MODELS = {"sora-2"}
PRIVATE_GROK_VIDEO_MODELS = {
    "grok-videos",
    "grok-videos-10s",
    "grok-videos-15s",
    "grok-video-3",
    "grok-video-3-10s",
    "grok-video-3-15s",
}
PRIVATE_VEO_VIDEO_MODELS = set(PRIVATE_VEO_VIDEO_REAL_MODELS)
PRIVATE_GROK_VIDEO_ASPECT_RATIOS = ["2:3", "3:2", "1:1", "9:16", "16:9"]
PRIVATE_VIDEO_ASPECT_RATIOS = ["16:9", "9:16"]
PRIVATE_VEO_VIDEO_RESOLUTIONS = ["720P", "4K"]
PRIVATE_VEO_VIDEO_MODES = ["首尾帧", "多参"]
PRIVATE_OMNI_FLASH_VIDEO_MODES = ["文生", "多参"]
PRIVATE_VIDU_VIDEO_MODELS = {
    "viduq3",
    "viduq3-pro",
    "viduq3-turbo",
    "viduq3-mix",
    "viduq2",
    "viduq2-turbo",
    "viduq2-pro",
}
PRIVATE_VIDU_VIDEO_MODES = ["参考", "图生", "首尾帧", "文生"]
PRIVATE_VIDU_VIDEO_ASPECT_RATIOS = ["auto", "16:9", "9:16", "1:1", "4:3", "3:4"]
PRIVATE_VIDU_VIDEO_RESOLUTIONS = ["540p", "720p", "1080p"]
PRIVATE_VIDU_HOST = ""
PRIVATE_HAILUO_VIDEO_MODELS = {"MiniMax-Hailuo-2.3", "MiniMax-Hailuo-02"}
PRIVATE_HAILUO_VIDEO_MODES = ["文生", "图生", "首尾帧"]
PRIVATE_HAILUO_VIDEO_DURATIONS = ["6", "10"]
PRIVATE_HAILUO_VIDEO_RESOLUTIONS = ["768P", "1080P"]
PRIVATE_KLING_VIDEO_MODELS = {"kling-v2-6", "kling-v3", "kling-video-o1", "kling-v3-omni"}
PRIVATE_KLING_OMNI_VIDEO_MODELS = {"kling-video-o1", "kling-v3-omni"}
PRIVATE_KLING_VIDEO_MODES = ["文生", "图生", "首尾帧", "多图"]
PRIVATE_KLING_OMNI_VIDEO_MODES = ["Omni"]
PRIVATE_KLING_VIDEO_ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
PRIVATE_KLING_VIDEO_QUALITY_MODES = ["std", "pro"]
PRIVATE_VIDEO_DURATIONS = ["6", "10"]
PRIVATE_SORA_VIDEO_DURATIONS = ["4", "8", "12"]
PRIVATE_SORA_VIDEO_SIZES = ["1280x720", "720x1280"]
PRIVATE_HAPPYHORSE_VIDEO_MODELS = {"happyhorse-1.0"}
PRIVATE_HAPPYHORSE_VIDEO_MODES = ["文生", "图生", "多图参考", "视频编辑"]
PRIVATE_HAPPYHORSE_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
PRIVATE_HAPPYHORSE_RESOLUTIONS = ["720P", "1080P"]
PRIVATE_HAPPYHORSE_DURATIONS = [str(value) for value in range(3, 16)]
PRIVATE_HAPPYHORSE_HOST = ""
PRIVATE_HAPPYHORSE_MAX_REFERENCE_IMAGES = 5
VIDEO_INPUT_ASPECT_RATIOS = list(dict.fromkeys(PRIVATE_GROK_VIDEO_ASPECT_RATIOS + PRIVATE_VIDEO_ASPECT_RATIOS + PRIVATE_VIDU_VIDEO_ASPECT_RATIOS + PRIVATE_KLING_VIDEO_ASPECT_RATIOS + PRIVATE_HAPPYHORSE_ASPECT_RATIOS + RUNNINGHUB_SEEDANCE_ASPECT_RATIOS + RUNNINGHUB_GROK_ASPECT_RATIOS + RUNNINGHUB_HAPPYHORSE_ASPECT_RATIOS + MODELVERSE_SEEDANCE_ASPECT_RATIOS + MODELVERSE_VIDEO_ASPECT_RATIOS + APIMART_SEEDANCE_ASPECT_RATIOS + APIMART_GROK_ASPECT_RATIOS + APIMART_VIDEO_ASPECT_RATIOS))
VIDEO_INPUT_DURATIONS = [str(value) for value in range(0, 31)]
VIDEO_INPUT_RESOLUTIONS = list(dict.fromkeys(["480P"] + PRIVATE_VEO_VIDEO_RESOLUTIONS + PRIVATE_VIDU_VIDEO_RESOLUTIONS + PRIVATE_HAILUO_VIDEO_RESOLUTIONS + PRIVATE_KLING_VIDEO_QUALITY_MODES + PRIVATE_HAPPYHORSE_RESOLUTIONS + RUNNINGHUB_SEEDANCE_RESOLUTIONS + RUNNINGHUB_GROK_RESOLUTIONS + RUNNINGHUB_HAPPYHORSE_RESOLUTIONS + MODELVERSE_SEEDANCE_RESOLUTIONS + MODELVERSE_HAPPYHORSE_RESOLUTIONS + MODELVERSE_KLING_QUALITY_MODES + APIMART_SEEDANCE_RESOLUTIONS + APIMART_SORA_RESOLUTIONS + APIMART_VEO_RESOLUTIONS + APIMART_HAILUO_RESOLUTIONS + APIMART_HAPPYHORSE_RESOLUTIONS + APIMART_KLING_QUALITY_MODES + APIMART_VIDU_RESOLUTIONS + APIMART_WAN_RESOLUTIONS + APIMART_OMNI_FLASH_RESOLUTIONS))
VIDEO_INPUT_MODES = list(dict.fromkeys(PRIVATE_VEO_VIDEO_MODES + PRIVATE_OMNI_FLASH_VIDEO_MODES + PRIVATE_VIDU_VIDEO_MODES + PRIVATE_HAILUO_VIDEO_MODES + PRIVATE_KLING_VIDEO_MODES + PRIVATE_KLING_OMNI_VIDEO_MODES + PRIVATE_HAPPYHORSE_VIDEO_MODES + RUNNINGHUB_SEEDANCE_MODES + RUNNINGHUB_GROK_MODES + RUNNINGHUB_HAPPYHORSE_MODES + MODELVERSE_SEEDANCE_MODES + MODELVERSE_HAPPYHORSE_MODES + MODELVERSE_SORA_VIDEO_MODES + MODELVERSE_KLING_VIDEO_MODES + MODELVERSE_KLING_OMNI_VIDEO_MODES + APIMART_SEEDANCE_VIDEO_MODES + APIMART_GROK_VIDEO_MODES + APIMART_SORA_VIDEO_MODES + APIMART_VEO_VIDEO_MODES + APIMART_OMNI_FLASH_VIDEO_MODES + APIMART_HAILUO_VIDEO_MODES + APIMART_HAPPYHORSE_VIDEO_MODES + APIMART_KLING_VIDEO_MODES + APIMART_KLING_OMNI_VIDEO_MODES + APIMART_VIDU_VIDEO_MODES + APIMART_WAN_VIDEO_MODES))
PRIVATE_VIDEO_MAX_IMAGES = 7
MAX_VIDEO_IMAGE_INPUTS = max(PRIVATE_VIDEO_MAX_IMAGES, RUNNINGHUB_MAX_IMAGES, APIMART_MAX_IMAGES)
MAX_VIDEO_FILE_INPUTS = max(RUNNINGHUB_MAX_VIDEOS, APIMART_MAX_VIDEOS)
MAX_AUDIO_INPUTS = max(RUNNINGHUB_MAX_AUDIOS, APIMART_MAX_AUDIOS)
MAX_VIDEO_MEDIA_INPUTS = MAX_VIDEO_IMAGE_INPUTS + MAX_VIDEO_FILE_INPUTS + MAX_AUDIO_INPUTS
PRIVATE_SORA_MAX_IMAGES = 1
PRIVATE_GROK_MAX_IMAGES = 7
PRIVATE_VEO_MAX_IMAGES = 3
PRIVATE_GEMINI_MODELS = {
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
}
PRIVATE_SEEDREAM_MODELS = {
    "doubao-seedream-5-0-260128",
    "doubao-seedream-4-5-251128",
    "doubao-seedream-4-0-250828",
}
PRIVATE_SEEDREAM_SIZE_VALUES = {
    "doubao-seedream-5-0-260128": {"2K", "3K"},
    "doubao-seedream-4-5-251128": {"2K", "4K"},
    "doubao-seedream-4-0-250828": {"1K", "2K", "4K"},
}
PRIVATE_GPT_IMAGE_MODELS = {"gpt-image-2"}
PRIVATE_OPENAI_COMPAT_MODELS = {"grok-4.2-image", "gpt-image-2-all"}
MODELVERSE_GEMINI_MODELS = {
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
}
MODELVERSE_GPT_IMAGE_MODELS = {"gpt-image-2"}
MODELVERSE_SEEDREAM_MODELS = {"doubao-seedream-4.5", "doubao-seedream-5-0-260128"}
MODELVERSE_SEEDREAM_SIZE_VALUES = {
    "doubao-seedream-4.5": {"2K", "4K"},
    "doubao-seedream-5-0-260128": {"2K", "4K"},
}
GPT_IMAGE_ASPECT_RATIOS = [
    "auto",
    "1:1",
    "3:2",
    "2:3",
    "16:9",
    "9:16",
    "5:4",
    "4:5",
    "4:3",
    "3:4",
    "21:9",
    "9:21",
    "1:3",
    "3:1",
    "2:1",
    "1:2",
]
GPT_IMAGE_VIP_SIZE_MAP = {
    "1:1": {"1K": "1248x1248", "2K": "2048x2048", "4K": "2880x2880"},
    "3:2": {"1K": "1536x1024", "2K": "2496x1664", "4K": "3504x2336"},
    "2:3": {"1K": "1024x1536", "2K": "1664x2496", "4K": "2336x3504"},
    "16:9": {"1K": "1792x1008", "2K": "2816x1584", "4K": "3840x2160"},
    "9:16": {"1K": "1008x1792", "2K": "1584x2816", "4K": "2160x3840"},
    "4:3": {"1K": "1472x1104", "2K": "2368x1776", "4K": "3264x2448"},
    "3:4": {"1K": "1104x1472", "2K": "1776x2368", "4K": "2448x3264"},
    "5:4": {"1K": "1440x1152", "2K": "2320x1856", "4K": "3200x2560"},
    "4:5": {"1K": "1152x1440", "2K": "1856x2320", "4K": "2560x3200"},
    "21:9": {"1K": "2016x864", "2K": "3024x1296", "4K": "3696x1584"},
    "9:21": {"1K": "864x2016", "2K": "1296x3024", "4K": "1584x3696"},
    "1:3": {"1K": "720x2160", "2K": "1184x3552", "4K": "1280x3840"},
    "3:1": {"1K": "2160x720", "2K": "3552x1184", "4K": "3840x1280"},
    "2:1": {"1K": "1760x880", "2K": "2912x1456", "4K": "3840x1920"},
    "1:2": {"1K": "880x1760", "2K": "1456x2912", "4K": "1920x3840"},
}
GPT_IMAGE_QUALITY_VALUES = ["low", "medium", "high"]
PRIVATE_GEMINI_ASPECT_RATIOS = ["auto", "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"]
PRIVATE_SEEDREAM_ASPECT_RATIOS = ["auto", "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"]
PRIVATE_GPT_IMAGE_ASPECT_RATIOS = GPT_IMAGE_ASPECT_RATIOS
PRIVATE_OPENAI_ASPECT_RATIOS = ["auto", "1:1", "3:2", "2:3"]
MODELVERSE_GEMINI_ASPECT_RATIOS = PRIVATE_GEMINI_ASPECT_RATIOS
MODELVERSE_SEEDREAM_ASPECT_RATIOS = PRIVATE_SEEDREAM_ASPECT_RATIOS
MODELVERSE_OPENAI_ASPECT_RATIOS = GPT_IMAGE_ASPECT_RATIOS
APIMART_GEMINI_ASPECT_RATIOS = ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
APIMART_GEMINI_31_ASPECT_RATIOS = APIMART_GEMINI_ASPECT_RATIOS + ["1:4", "4:1", "1:8", "8:1"]
APIMART_GPT_IMAGE_ASPECT_RATIOS = GPT_IMAGE_ASPECT_RATIOS
ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u180e", "\u200e", "\u200f"]
MAX_IMAGE_INPUTS = 16
NANO_BANANA_MAX_IMAGES = 14
GPT_IMAGE_MAX_IMAGES = 16
PRIVATE_MAX_IMAGES = 16
MODELVERSE_MAX_IMAGES = 16
APIMART_GEMINI_MAX_IMAGES = 14
APIMART_GPT_IMAGE_MAX_IMAGES = 16
PRO_SIZE_MODELS = {
    "nano-banana-fast",
    "nano-banana-pro",
    "nano-banana-pro-vt",
    "nano-banana-pro-cl",
    "nano-banana-2",
    "nano-banana-2-cl",
    "nano-banana-2-4k-cl",
}

ROUTE_PREFIX = "/cometapi/virtual_wire_proto"
RUN_NODE_WORKERS = max(1, int(os.environ.get("NKXX_VWIRE_RUN_NODE_WORKERS", "8")))
ASSET_SUBFOLDER = "cometapi_assets"

logger = logging.getLogger("CometAPI-VirtualWireProto")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
logger.setLevel(logging.INFO)

_EXECUTOR_POOL = None
_RUNNING_PROMPTS = {}
_RUNNING_LOCK = threading.Lock()
_ROUTES_REGISTERED = False
_PRIVATE_HOST_CACHE = None
_SETTINGS_LOCK = threading.Lock()
# 进程内缓存：batch 节点最近一次执行产生的 asset_refs；按 batch 节点 ID 索引。
# 图像卡片在多图场景下嗅探这里，按张数对得上就直接复制原图落盘，避免 IMAGE tensor pad 出来的黑边。
_LAST_BATCH_ASSET_REFS: dict[str, dict] = {}
_LAST_BATCH_ASSET_REFS_LOCK = threading.Lock()
# 缓存条目过期阈值（秒）。设到 60 分钟覆盖"批量被限流跑很久才轮到下游"的工作流。
# 命中后会立刻从缓存里 pop 掉，所以正常情况下不会留陈旧条目。
_LAST_BATCH_ASSET_REFS_TTL = 3600
PLUGIN_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(PLUGIN_DIR, "data")
SETTINGS_FILE = os.path.join(DATA_DIR, "cometapi_settings.json")
ANNOUNCEMENT_URL = "https://cnb.cool/nkxx666/comfyui_cpu_nkxx/-/git/raw/main/announcement.md"
ANNOUNCEMENT_CACHE_FILE = os.path.join(DATA_DIR, "cometapi_announcement_cache.json")
ANNOUNCEMENT_CACHE_TTL_SECONDS = 24 * 60 * 60
ANNOUNCEMENT_REQUEST_TIMEOUT = (3, 5)

GRSAI_MODEL_ALIASES = {model: model.replace("nano-", "") if model.startswith("nano-") else model for model in (GPT_IMAGE_MODELS + NANO_BANANA_MODELS)}
PRIVATE_MODEL_ALIASES = {
    "gemini-3.1-flash-image-preview": "banana-2",
    "gemini-3-pro-image-preview": "banana-pro",
    "gemini-2.5-flash-image": "banana-1",
    "doubao-seedream-5-0-260128": "seedream-5-lite",
    "doubao-seedream-4-5-251128": "seedream-4.5",
    "doubao-seedream-4-0-250828": "seedream-4",
}
MODELVERSE_IMAGE_MODEL_ALIASES = {
    "gemini-3-pro-image-preview": "banana-pro",
    "gemini-3.1-flash-image-preview": "banana-2",
    "gpt-image-2": "gpt-image-2",
    "doubao-seedream-4.5": "seedream-4.5",
    "doubao-seedream-5-0-260128": "seedream-5",
}


class CometAPIError(Exception):
    pass


class SubmitResponseLostError(CometAPIError):
    pass


_SENSITIVE_REPLACEMENT = "***"
_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"(?i)((?:[?&]|%3[fF]|%26)(?:api[_-]?key|apikey|key|token|access[_-]?token|auth|authorization|secret)=)[^&\s\"'<>]+"),
    re.compile(r"(?i)(\bauthorization\b[\"']?\s*[:=]\s*[\"']?Bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(\b(?:api[_-]?key|apikey|key|token|access[_-]?token|x-goog-api-key|x-api-key)\b\s*[:=]\s*[\"']?)[^\"'\s,;)&}]+"),
    re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(data:[^,\s]+;base64,)[A-Za-z0-9+/=]{32,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?<![A-Za-z0-9_-])(?=[A-Za-z0-9_-]{32,}\b)(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])"),
)


def redact_sensitive_text(value) -> str:
    text = str(value or "")
    for pattern in _SENSITIVE_TEXT_PATTERNS:
        def _replace(match):
            if match.lastindex:
                return f"{match.group(1)}{_SENSITIVE_REPLACEMENT}"
            return _SENSITIVE_REPLACEMENT

        text = pattern.sub(_replace, text)
    return text


def print_sanitized_exception(error: Exception) -> None:
    try:
        traceback_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        print(redact_sensitive_text(traceback_text))
    except Exception:
        print(format_error_message(error))


def format_error_message(error: Exception) -> str:
    message = redact_sensitive_text(str(error).strip())
    if isinstance(error, CometAPIError):
        return _extract_core_error_message(message) or "API 调用失败"
    return f"插件运行出错：{type(error).__name__}: {_extract_core_error_message(message)}"


def _extract_core_error_message(raw: str) -> str:
    """从可能嵌套 JSON 的错误文本中提取核心可读信息。"""
    if not raw:
        return raw
    # 尝试从 JSON 结构中提取 message 字段
    core = _try_extract_json_message(raw)
    if core and core != raw:
        return core
    # 截断过长的原始消息
    if len(raw) > 200:
        return raw[:200] + "..."
    return raw


def _try_extract_json_message(text: str) -> str:
    """递归尝试从文本中找到 JSON 并提取最深层的 message/error 字段。"""
    # 找第一个 { 开始的 JSON
    start = text.find("{")
    if start < 0:
        return ""
    prefix = text[:start].strip().rstrip(":").strip()
    json_part = text[start:]
    try:
        data = json.loads(json_part)
    except (json.JSONDecodeError, ValueError):
        # 可能 JSON 后面还有多余字符，尝试逐步截断
        for end_offset in range(len(json_part), max(0, len(json_part) - 100), -1):
            try:
                data = json.loads(json_part[:end_offset])
                break
            except (json.JSONDecodeError, ValueError):
                continue
        else:
            return ""

    # 从 data 中递归提取最有意义的 message
    core = _dig_message(data)
    if core:
        result = f"{prefix}: {core}" if prefix else core
        # 去掉重复的前缀
        if result.startswith(": "):
            result = result[2:]
        return result.strip()
    return ""


def _dig_message(data, depth: int = 0) -> str:
    """从嵌套 dict 中找最深层的 message 字段。"""
    if depth > 5 or not isinstance(data, dict):
        return ""
    # 优先找嵌套的 error.message 或 data.message
    for key in ("error", "data", "original_error", "vclm_response"):
        child = data.get(key)
        if isinstance(child, str):
            # 可能是嵌套 JSON 字符串
            try:
                nested = json.loads(child)
                if isinstance(nested, dict):
                    deeper = _dig_message(nested, depth + 1)
                    if deeper:
                        return deeper
            except (json.JSONDecodeError, ValueError):
                pass
            # 纯文本 error
            if key in ("error", "original_error") and child.strip():
                return child.strip()[:200]
        elif isinstance(child, dict):
            deeper = _dig_message(child, depth + 1)
            if deeper:
                return deeper
    # 直接取 message 字段
    msg = data.get("message") or data.get("Message") or data.get("msg")
    if isinstance(msg, str) and msg.strip():
        # 如果 message 本身也是嵌套 JSON
        inner = _try_extract_json_message(msg)
        return inner if inner else msg.strip()[:200]
    return ""


def combine_failure_reason_and_error(data: dict, fallback: str = "未知原因") -> str:
    parts = []
    for key in ("failure_reason", "error"):
        text = str(data.get(key) or "").strip()
        if text and text not in parts:
            parts.append(text)
    return ", ".join(parts) or fallback


def media_type_label(media_type: str) -> str:
    return {
        "image": "图片",
        "video": "视频",
        "audio": "音频",
    }.get(str(media_type or "").lower(), str(media_type or "素材"))


_PRIVATE_CHANNEL_SPECS_CACHE: list[dict] | None = None
_PRIVATE_ADAPTER_CACHE: dict[str, object] = {}


def _sanitize_private_name(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    return text.strip("_-")


def _private_channel_dirs() -> list[str]:
    dirs = [os.path.join(PLUGIN_DIR, "private_channels")]
    extra = os.environ.get("COMETAPI_PRIVATE_CHANNELS_DIR", "")
    for item in re.split(r"[;]", extra):
        path = item.strip().strip('"')
        if path:
            dirs.append(path)

    unique = []
    seen = set()
    for path in dirs:
        full = os.path.abspath(path)
        if full not in seen:
            unique.append(full)
            seen.add(full)
    return unique


def _normalize_private_channel_spec(raw: dict, source_path: str) -> dict | None:
    if not isinstance(raw, dict) or raw.get("enabled") is False:
        return None
    channel_id = _sanitize_private_name(raw.get("id") or raw.get("channel") or os.path.splitext(os.path.basename(source_path))[0])
    if not channel_id or channel_id in PRESET_CHANNELS:
        return None
    models = []
    for item in raw.get("models") or []:
        if not isinstance(item, dict) or not str(item.get("model") or "").strip():
            continue
        category = str(item.get("category") or "video").strip().lower()
        model_entry = {
            "model": str(item.get("model")).strip(),
            "alias": str(item.get("alias") or item.get("model")).strip(),
            "category": category,
            "family": str(item.get("family") or "").strip().lower(),
            "api_key": str(item.get("api_key") or "").strip(),
        }
        for passthrough_key in PRIVATE_MODEL_PASSTHROUGH_KEYS:
            if passthrough_key in item:
                model_entry[passthrough_key] = item.get(passthrough_key)
        models.append(model_entry)
    if not models:
        return None
    return {
        "id": channel_id,
        "name": str(raw.get("name") or channel_id).strip() or channel_id,
        "api_url": str(raw.get("api_url") or raw.get("base_url") or "").strip(),
        "api_key": str(raw.get("api_key") or "").strip(),
        "api_key_env": str(raw.get("api_key_env") or "").strip(),
        "adapter": _sanitize_private_name(raw.get("adapter") or channel_id),
        "models": models,
        "__file": source_path,
        "__dir": os.path.dirname(source_path),
    }


def load_private_channel_specs(force: bool = False) -> list[dict]:
    global _PRIVATE_CHANNEL_SPECS_CACHE
    if _PRIVATE_CHANNEL_SPECS_CACHE is not None and not force:
        return _PRIVATE_CHANNEL_SPECS_CACHE

    specs = []
    used = set()
    for folder in _private_channel_dirs():
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(folder, name)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    spec = _normalize_private_channel_spec(json.load(handle), path)
            except Exception as exc:
                logger.warning(f"Failed to load private channel {path}: {redact_sensitive_text(exc)}")
                continue
            if not spec or spec["id"] in used:
                continue
            specs.append(spec)
            used.add(spec["id"])
    _PRIVATE_CHANNEL_SPECS_CACHE = specs
    return specs


def get_private_channel_spec(channel: str) -> dict | None:
    key = _sanitize_private_name(channel)
    for spec in load_private_channel_specs():
        if spec.get("id") == key:
            return spec
    return None


def get_private_channel_model_settings(channel: str, model: str, category: str = "") -> dict:
    spec = get_private_channel_spec(channel)
    if not spec:
        return {}
    model_key = str(model or "").strip().lower()
    category_key = str(category or "").strip().lower()
    for item in spec.get("models") or []:
        if str(item.get("model") or "").strip().lower() != model_key:
            continue
        if category_key and str(item.get("category") or "").lower() != category_key:
            continue
        return item
    return {}


def private_model_ui(channel: str, model: str, category: str = "") -> dict:
    settings = get_private_channel_model_settings(channel, model, category)
    ui = settings.get("ui") if isinstance(settings, dict) else None
    capabilities = settings.get("capabilities") if isinstance(settings, dict) else None
    if isinstance(ui, dict):
        return ui
    if isinstance(capabilities, dict):
        return capabilities
    return {}


def _media_limit_int(value, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return default


def private_media_limits_for_model(channel: str, model: str, category: str = "video", mode: str = "") -> dict[str, int]:
    ui = private_model_ui(channel, model, category)
    limits = ui.get("media_limits") if isinstance(ui.get("media_limits"), dict) else ui
    if not isinstance(limits, dict):
        limits = {}

    mode_key = str(mode or "").strip()
    merged = {
        "image": _media_limit_int(limits.get("image"), 0),
        "video": _media_limit_int(limits.get("video"), 0),
        "audio": _media_limit_int(limits.get("audio"), 0),
    }
    by_mode = limits.get("by_mode") or limits.get("media_limits_by_mode")
    if isinstance(by_mode, dict) and mode_key:
        mode_limits = by_mode.get(mode_key)
        if not isinstance(mode_limits, dict):
            for key, value in by_mode.items():
                if str(key).strip().lower() == mode_key.lower() and isinstance(value, dict):
                    mode_limits = value
                    break
        if isinstance(mode_limits, dict):
            for media_type in ("image", "video", "audio"):
                if media_type in mode_limits:
                    merged[media_type] = _media_limit_int(mode_limits.get(media_type), merged[media_type])
    if not any(merged.values()) and category == "video":
        # Backward-compatible fallback for older private video specs.
        merged["image"] = PRIVATE_VIDEO_MAX_IMAGES
    return merged


def private_allowed_media_types(channel: str, model: str, category: str = "video", mode: str = "") -> set[str]:
    limits = private_media_limits_for_model(channel, model, category, mode)
    return {media_type for media_type, limit in limits.items() if int(limit or 0) > 0}


def canonical_apimart_video_model(model: str) -> str:
    raw = str(model or "").strip()
    if not raw:
        return raw
    normalized = raw.lower()
    for model_id in APIMART_VIDEO_MODELS:
        if normalized == model_id.lower():
            return model_id
    return APIMART_VIDEO_MODEL_ALIAS_TO_ID.get(normalized, raw)


def apimart_video_family(model: str) -> str:
    raw = canonical_apimart_video_model(model)
    raw_lower = raw.lower()
    if raw_lower in APIMART_SEEDANCE_VIDEO_MODELS:
        return "seedance"
    if raw_lower in APIMART_GROK_VIDEO_MODELS:
        return "grok"
    if raw_lower in APIMART_SORA_VIDEO_MODELS:
        return "sora"
    if raw in APIMART_VEO_VIDEO_MODELS or raw_lower in {item.lower() for item in APIMART_VEO_VIDEO_MODELS}:
        return "veo"
    if raw in APIMART_HAILUO_VIDEO_MODELS:
        return "hailuo"
    if raw_lower in APIMART_HAPPYHORSE_VIDEO_MODELS:
        return "happyhorse"
    if raw_lower in APIMART_KLING_VIDEO_MODELS:
        return "kling"
    if raw_lower in APIMART_VIDU_VIDEO_MODELS:
        return "vidu"
    if raw_lower in APIMART_WAN_VIDEO_MODELS:
        return "wan"
    return "seedance"


def runninghub_video_family(model: str) -> str:
    raw = str(model or "").strip()
    normalized = raw.lower()
    for model_id in RUNNINGHUB_VIDEO_MODELS:
        if normalized == model_id.lower():
            raw = model_id
            break
    else:
        raw = RUNNINGHUB_VIDEO_MODEL_ALIAS_TO_ID.get(normalized, raw)
    raw_lower = raw.lower()
    if raw_lower in RUNNINGHUB_SEEDANCE_MODELS:
        return "seedance"
    if raw_lower in RUNNINGHUB_GOOGLE_MODELS:
        return "google"
    if raw_lower in RUNNINGHUB_GROK_MODELS:
        return "grok"
    if raw_lower in RUNNINGHUB_HAPPYHORSE_MODELS:
        return "happyhorse"
    return "seedance"


def get_model_catalog() -> dict:
    catalog = {
        "grsai": {
            "image": [{"model": model, "alias": GRSAI_MODEL_ALIASES.get(model, model)} for model in (GPT_IMAGE_MODELS + NANO_BANANA_MODELS)],
            "llm": [
                {
                    "model": model,
                    "alias": model,
                    "api_format": GRSAI_LLM_API_FORMATS.get(model, "gemini"),
                }
                for model in LLM_MODELS
            ],
            "video": [],
            "music": [],
        },
        "runninghub": {
            "image": [{"model": model, "alias": RUNNINGHUB_IMAGE_MODEL_ALIASES.get(model, model)} for model in RUNNINGHUB_IMAGE_MODELS],
            "llm": [
                {
                    "model": model,
                    "alias": RUNNINGHUB_LLM_MODEL_ALIASES.get(model, model),
                    "api_format": RUNNINGHUB_LLM_API_FORMATS.get(model, "openai"),
                }
                for model in RUNNINGHUB_LLM_MODELS
            ],
            "video": [
                {
                    "model": model,
                    "alias": RUNNINGHUB_VIDEO_MODEL_ALIASES.get(model, model),
                    "family": runninghub_video_family(model),
                }
                for model in RUNNINGHUB_VIDEO_MODELS
            ],
            "music": [{"model": model, "alias": RUNNINGHUB_MUSIC_MODEL_ALIASES.get(model, model)} for model in RUNNINGHUB_MUSIC_MODELS],
        },
        "modelverse": {
            "image": [{"model": model, "alias": MODELVERSE_IMAGE_MODEL_ALIASES.get(model, model)} for model in MODELVERSE_IMAGE_MODELS],
            "llm": [
                {
                    "model": model,
                    "alias": MODELVERSE_LLM_MODEL_ALIASES.get(model, model),
                    "api_format": MODELVERSE_LLM_API_FORMATS.get(model, "openai"),
                }
                for model in MODELVERSE_LLM_MODELS
            ],
            "video": [{"model": model, "alias": MODELVERSE_VIDEO_MODEL_ALIASES.get(model, model)} for model in MODELVERSE_VIDEO_MODELS],
            "music": [],
        },
        "apimart": {
            "image": [{"model": model, "alias": APIMART_IMAGE_MODEL_ALIASES.get(model, model)} for model in APIMART_IMAGE_MODELS],
            "llm": [
                {
                    "model": model,
                    "alias": APIMART_LLM_MODEL_ALIASES.get(model, model),
                    "api_format": APIMART_LLM_API_FORMATS.get(model, "gemini"),
                }
                for model in APIMART_LLM_MODELS
            ],
            "video": [
                {
                    "model": model,
                    "alias": APIMART_VIDEO_MODEL_ALIASES.get(model, model),
                    "family": apimart_video_family(model),
                }
                for model in APIMART_VIDEO_MODELS
            ],
            "music": [],
        },
    }
    for spec in load_private_channel_specs():
        grouped = {"image": [], "llm": [], "video": [], "music": []}
        for item in spec.get("models") or []:
            category = item.get("category")
            if category in grouped:
                entry = {"model": item.get("model"), "alias": item.get("alias") or item.get("model")}
                for passthrough_key in PRIVATE_MODEL_PASSTHROUGH_KEYS:
                    if item.get(passthrough_key):
                        entry[passthrough_key] = item.get(passthrough_key)
                grouped[category].append(entry)
        grouped["_meta"] = {"name": spec.get("name") or spec["id"], "private": True}
        catalog[spec["id"]] = grouped
    return catalog


def _unique_non_empty(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if str(value or "").strip()))


def get_video_channel_choices() -> list[str]:
    private_channels = [
        spec["id"]
        for spec in load_private_channel_specs()
        if any(item.get("category") == "video" for item in spec.get("models") or [])
    ]
    return _unique_non_empty(VIDEO_CHANNELS + private_channels)


def get_music_channel_choices() -> list[str]:
    private_channels = [
        spec["id"]
        for spec in load_private_channel_specs()
        if any(item.get("category") == "music" for item in spec.get("models") or [])
    ]
    return _unique_non_empty(MUSIC_CHANNELS + private_channels)


def get_default_video_channel() -> str:
    choices = get_video_channel_choices()
    for candidate in ("apimart", "runninghub", "modelverse", "grsai"):
        if candidate in choices:
            return candidate
    return choices[0] if choices else "apimart"


def get_default_music_channel() -> str:
    choices = get_music_channel_choices()
    for candidate in ("runninghub",):
        if candidate in choices:
            return candidate
    return choices[0] if choices else "runninghub"


def get_private_video_model_choices() -> list[str]:
    return _unique_non_empty(
        [
            str(item.get("model"))
            for spec in load_private_channel_specs()
            for item in (spec.get("models") or [])
            if item.get("category") == "video" and item.get("model")
        ]
    )


def get_video_model_choices() -> list[str]:
    return _unique_non_empty(APIMART_VIDEO_MODELS + RUNNINGHUB_VIDEO_MODELS + MODELVERSE_VIDEO_MODELS + get_private_video_model_choices())


def get_private_music_model_choices() -> list[str]:
    return _unique_non_empty(
        [
            str(item.get("model"))
            for spec in load_private_channel_specs()
            for item in (spec.get("models") or [])
            if item.get("category") == "music" and item.get("model")
        ]
    )


def get_music_model_choices() -> list[str]:
    return _unique_non_empty(MUSIC_INPUT_MODELS + get_private_music_model_choices())


def get_video_model_family_choices() -> list[str]:
    private_families = _unique_non_empty(
        [
            str(item.get("family"))
            for spec in load_private_channel_specs()
            for item in (spec.get("models") or [])
            if item.get("category") == "video" and item.get("family")
        ]
    )
    return _unique_non_empty(PRIVATE_VIDEO_MODEL_FAMILIES + APIMART_VIDEO_MODEL_FAMILIES + RUNNINGHUB_VIDEO_MODEL_FAMILIES + MODELVERSE_VIDEO_MODEL_FAMILIES + private_families)


def normalize_api_base_url(value: str, fallback: str = "https://api.openai.com") -> str:
    raw = str(value or "").strip()
    if not raw:
        raw = fallback
    raw = raw.rstrip("/")
    if not re.match(r"^https?://", raw, flags=re.I):
        raw = f"https://{raw}"
    return raw.rstrip("/")


# 已知 endpoint 模式。当用户的 api_url 已经填到完整 endpoint，遇到不同业务的请求需要剥离再拼。
# 元组顺序：从最长 suffix 排到最短，避免 /messages 误匹配 /v1/messages。
_KNOWN_ENDPOINT_SUFFIXES = (
    "/v1/chat/completions",
    "/v1/messages",
    "/v1/images/generations",
    "/v1/images/edits",
    "/chat/completions",
    "/messages",
    "/images/generations",
    "/images/edits",
)
# 一个 endpoint 归属哪种"业务种类"。当 base 与 path 同种类时，直接复用用户填的 base，不再拼。
_ENDPOINT_KIND_PATTERNS = (
    ("chat", ("/chat/completions",)),
    ("messages", ("/messages",)),
    ("images-gen", ("/images/generations",)),
    ("images-edit", ("/images/edits",)),
)
# Gemini endpoint：/v1beta/models/{model}:operation，单独识别。
_GEMINI_ENDPOINT_RE = re.compile(r"/v1beta/models/[^/]+?:[A-Za-z]+$", flags=re.I)


def _classify_endpoint_kind(url_lower: str) -> str:
    """识别 url 末尾属于哪种业务，未识别返回空串。"""
    for kind, tails in _ENDPOINT_KIND_PATTERNS:
        for tail in tails:
            if url_lower.endswith(tail):
                return kind
    if _GEMINI_ENDPOINT_RE.search(url_lower):
        return "gemini"
    return ""


def _strip_known_endpoint_suffix(base_lower: str, base: str) -> str:
    """如果 base 以已知 endpoint 结尾就剥掉它。保留前缀路径（例如 /openai 这类反向代理 prefix）。"""
    for suffix in _KNOWN_ENDPOINT_SUFFIXES:
        if base_lower.endswith(suffix):
            return base[: len(base) - len(suffix)].rstrip("/")
    match = _GEMINI_ENDPOINT_RE.search(base_lower)
    if match and base_lower.endswith(match.group(0)):
        return base[: match.start()].rstrip("/")
    return base


def build_api_url(api_url: str, path: str, fallback: str) -> str:
    """
    把渠道 api_url 与目标子路径组装成完整请求 URL。

    支持以下用户填法：
      1) 服务根地址：https://api.example.com  → 直接拼 path
      2) 服务根地址带反向代理前缀：https://host/openai → 拼成 https://host/openai{path}
      3) 完整 endpoint URL（带或不带 /v1）：例如 https://host/v1/chat/completions、https://host/chat/completions
         - 当 path 与 base 属于同一业务种类（例如都是 chat completions）：直接返回 base
         - 当 path 是另一业务（例如图片）：剥掉 base 尾部的已知 endpoint，再拼 path
      4) base 已经带 /v1 而 path 也以 /v1/ 开头：去掉 path 的 /v1 避免重复
    Gemini 的 /v1beta/models/{model}:operation 单独识别：同模型同操作直接复用，否则剥到 base 再拼新 path。
    """
    base = normalize_api_base_url(api_url, fallback)
    endpoint = str(path or "")
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    base_lower = base.lower()
    endpoint_lower = endpoint.lower()

    base_kind = _classify_endpoint_kind(base_lower)
    endpoint_kind = _classify_endpoint_kind(endpoint_lower)

    # 同业务种类（chat/messages/images-gen/images-edit）：直接复用用户填的 base
    if base_kind and base_kind == endpoint_kind and base_kind != "gemini":
        return base

    # Gemini 精确匹配：模型 + 操作完全一致才直接复用
    if base_kind == "gemini" and endpoint_kind == "gemini":
        gm_base = _GEMINI_ENDPOINT_RE.search(base_lower)
        if gm_base and base_lower.endswith(gm_base.group(0)) and gm_base.group(0) == endpoint_lower:
            return base

    # 跨业务或不同 Gemini 模型：剥掉 base 尾部的已知 endpoint
    stripped = _strip_known_endpoint_suffix(base_lower, base)
    if stripped != base:
        base = stripped
        base_lower = base.lower()

    # base 包含 /v1 且 endpoint 也以 /v1 开头：去掉 endpoint 的 /v1 避免重复
    if base_lower.endswith("/v1") and endpoint_lower.startswith("/v1/"):
        endpoint = endpoint[3:]
    return base + endpoint


def _normalize_image_model_id_for_match(value: str) -> str:
    """把模型 ID 归一化为小写并把下划线视作连字符，方便做关键词匹配。"""
    return str(value or "").strip().lower().replace("_", "-")


# 精确名匹配集合（已统一使用 - 作为分隔符）
_GEMINI_2_5_EXACT = {
    "nanobanana",
    "nano-banana",
    "banana",
    "nano-banana-1",
    "nanobanana-1",
    "banana-1",
}
_GEMINI_3_EXACT = {
    "nano-banana-pro",
    "banana-pro",
    "nanobananapro",
    "nano-banana-2",
    "banana-2",
    "nanobanana2",
    "nano-banana-2-cl",
}


def detect_gemini_sub_family(model_id: str) -> str | None:
    """从模型 ID 字符串里推断 Gemini 子族。命中返回 GEMINI_SUB_FAMILY_2_5/3，未命中返回 None。

    优先级：精确名 → 子串规则 → fallback。
    """
    text = _normalize_image_model_id_for_match(model_id)
    if not text:
        return None
    # 精确名优先：先 2.5 系列具体版本，再 2.5 通用名，再 3 系列
    if text in _GEMINI_2_5_EXACT:
        return GEMINI_SUB_FAMILY_2_5
    if text in _GEMINI_3_EXACT:
        return GEMINI_SUB_FAMILY_3
    # 子串规则：2.5 + image 优先（覆盖官方名 gemini-2.5-flash-image / -preview）
    if "2.5" in text and "image" in text:
        return GEMINI_SUB_FAMILY_2_5
    if "3-pro-image" in text or "3.1-flash-image" in text:
        return GEMINI_SUB_FAMILY_3
    # fallback：含 gemini + image 的其它命名一律按 3 代处理（保守显示 image_size）
    if "gemini" in text and "image" in text:
        return GEMINI_SUB_FAMILY_3
    return None


def detect_image_api_format(model_id: str) -> str:
    """根据用户填入的模型 ID 推断默认 api_format。命中 Gemini 返回 gemini_image，否则 gpt_image。"""
    return "gemini_image" if detect_gemini_sub_family(model_id) is not None else "gpt_image"


def normalize_image_interface_mode(value: str, api_format: str) -> str:
    """把图像接口模式归一化。不同 api_format 下默认值不同，非法值统一回退到默认。"""
    text = str(value or "").strip().lower().replace("-", "_")
    fmt = api_format if api_format in IMAGE_API_FORMATS else "gpt_image"
    valid = IMAGE_INTERFACE_MODES_GEMINI if fmt == "gemini_image" else IMAGE_INTERFACE_MODES_GPT
    default = "native" if fmt == "gemini_image" else "unified"
    if text in valid:
        return text
    return default


def normalize_api_format_value(value: str, category: str = "llm", default: str = "openai") -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if category == "image":
        if text in IMAGE_API_FORMATS:
            return text
        # 历史兼容：之前可能存过 openai / openai_image / images
        if text in {"openai", "openai_image", "openai_images", "images"}:
            return "gpt_image"
        if "gemini" in text:
            return "gemini_image"
        return "gpt_image"
    if text in LLM_API_FORMATS:
        return text
    if "claude" in text or "anthropic" in text:
        return "claude"
    if "gemini" in text:
        return "gemini"
    if "openai" in text or "chat" in text:
        return "openai"
    return default if default in LLM_API_FORMATS else "openai"


def _normalize_custom_channel_id(value: str, index: int, used: set[str]) -> str:
    raw = str(value or "").strip().lower()
    raw = re.sub(r"[^a-z0-9_-]+", "_", raw).strip("_-")
    if not raw or raw in PRESET_CHANNELS:
        raw = f"custom_{index + 1}"
    base = raw
    suffix = 2
    while raw in used:
        raw = f"{base}_{suffix}"
        suffix += 1
    used.add(raw)
    return raw


def _normalize_custom_channel_models(raw_models, default_api_format: str = "openai") -> list[dict]:
    if isinstance(raw_models, dict):
        iterable = []
        for model_id, config in raw_models.items():
            item = dict(config) if isinstance(config, dict) else {}
            item.setdefault("model", model_id)
            iterable.append(item)
    elif isinstance(raw_models, list):
        iterable = raw_models
    else:
        iterable = []

    normalized = []
    used: set[tuple[str, str]] = set()
    for item in iterable:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or item.get("id") or item.get("name") or "").strip()
        if not model:
            continue
        category = str(item.get("category") or "llm").strip().lower()
        if category not in CUSTOM_CHANNEL_MODEL_CATEGORIES:
            category = "llm"
        key = (category, model.lower())
        if key in used:
            continue
        used.add(key)
        alias = str(item.get("alias") or item.get("label") or item.get("name") or model).strip() or model
        api_format = normalize_api_format_value(item.get("api_format") or default_api_format, category, default_api_format)
        entry = {
            "model": model,
            "alias": alias,
            "category": category,
            "api_format": api_format,
            "api_key": str(item.get("api_key") or item.get("override_api_key") or ""),
        }
        if category == "image":
            # 兼容旧字段名 endpoint_mode
            raw_mode = item.get("interface_mode") or item.get("endpoint_mode")
            entry["interface_mode"] = normalize_image_interface_mode(raw_mode, api_format)
        normalized.append(entry)
    return normalized


def normalize_custom_channels(raw_channels) -> list[dict]:
    if not isinstance(raw_channels, list):
        return []
    normalized = []
    used = set(PRESET_CHANNELS)
    for index, item in enumerate(raw_channels):
        if not isinstance(item, dict):
            continue
        channel_id = _normalize_custom_channel_id(item.get("id") or item.get("key"), index, used)
        default_api_format = normalize_api_format_value(item.get("api_format") or item.get("default_api_format") or "openai")
        models = _normalize_custom_channel_models(item.get("models"), default_api_format)
        normalized.append(
            {
                "id": channel_id,
                "name": str(item.get("name") or item.get("label") or channel_id).strip() or channel_id,
                "api_url": normalize_api_base_url(item.get("api_url") or item.get("base_url") or "https://api.openai.com"),
                "api_key": str(item.get("api_key") or ""),
                "enabled": item.get("enabled") is not False,
                "api_format": default_api_format,
                "models": models,
            }
        )
    return normalized


def get_custom_channel_settings(settings: dict, channel: str) -> dict | None:
    channel_key = str(channel or "").lower()
    for item in settings.get("custom_channels") or []:
        if isinstance(item, dict) and str(item.get("id") or "").lower() == channel_key:
            return item
    return None


def get_custom_model_settings(channel_settings: dict | None, model: str, category: str = "") -> dict:
    if not isinstance(channel_settings, dict):
        return {}
    raw = str(model or "").strip()
    normalized = raw.lower()
    category_key = str(category or "").lower()
    for item in channel_settings.get("models") or []:
        if not isinstance(item, dict):
            continue
        if category_key and str(item.get("category") or "").lower() != category_key:
            continue
        model_id = str(item.get("model") or "")
        alias = str(item.get("alias") or "").strip()
        if model_id == raw or model_id.lower() == normalized:
            return item
        if alias and alias.lower() == normalized:
            return item
    return {}


def channel_has_custom_model_category(channel_settings: dict | None, category: str) -> bool:
    if not isinstance(channel_settings, dict) or channel_settings.get("enabled") is False:
        return False
    category_key = str(category or "").lower()
    return any(isinstance(item, dict) and str(item.get("category") or "").lower() == category_key for item in channel_settings.get("models") or [])


def channel_has_settings_model_category(channel_settings: dict | None, category: str) -> bool:
    if not isinstance(channel_settings, dict):
        return False
    category_key = str(category or "").lower()
    for config in (channel_settings.get("models") or {}).values():
        if isinstance(config, dict) and str(config.get("category") or "").lower() == category_key:
            return True
    for item in channel_settings.get("custom_models") or []:
        if isinstance(item, dict) and str(item.get("category") or "").lower() == category_key:
            return True
    return False


def get_llm_channel_choices() -> list[str]:
    choices = list(LLM_CHANNELS)
    try:
        settings = load_comet_settings()
        for channel, channel_settings in (settings.get("channels") or {}).items():
            if channel_has_settings_model_category(channel_settings, "llm"):
                choices.append(str(channel))
        for channel in settings.get("custom_channels") or []:
            if channel_has_custom_model_category(channel, "llm"):
                choices.append(str(channel.get("id") or ""))
    except Exception:
        pass
    return _unique_non_empty(choices)


def get_image_channel_choices() -> list[str]:
    choices = list(IMAGE_CHANNELS)
    try:
        settings = load_comet_settings()
        for channel, channel_settings in (settings.get("channels") or {}).items():
            if channel_has_settings_model_category(channel_settings, "image"):
                choices.append(str(channel))
        for channel in settings.get("custom_channels") or []:
            if channel_has_custom_model_category(channel, "image"):
                choices.append(str(channel.get("id") or ""))
    except Exception:
        pass
    return _unique_non_empty(choices)


def get_llm_model_choices() -> list[str]:
    models = list(LLM_MODELS + RUNNINGHUB_LLM_MODELS + MODELVERSE_LLM_MODELS + APIMART_LLM_MODELS)
    try:
        settings = load_comet_settings()
        for channel_settings in settings.get("channels", {}).values():
            for item in channel_settings.get("custom_llm_models") or []:
                model = str(item.get("model") or "").strip() if isinstance(item, dict) else ""
                if model:
                    models.append(model)
            for model, config in (channel_settings.get("models") or {}).items():
                if isinstance(config, dict) and config.get("category") == "llm":
                    models.append(str(model))
        for channel_settings in settings.get("custom_channels") or []:
            for item in channel_settings.get("models") or []:
                if isinstance(item, dict) and item.get("category") == "llm" and item.get("model"):
                    models.append(str(item.get("model")))
    except Exception:
        pass
    return _unique_non_empty(models)


def get_image_model_choices() -> list[str]:
    models = list(SUPPORTED_MODELS)
    try:
        settings = load_comet_settings()
        for channel_settings in settings.get("channels", {}).values():
            for model, config in (channel_settings.get("models") or {}).items():
                if isinstance(config, dict) and config.get("category") == "image":
                    models.append(str(model))
        for channel_settings in settings.get("custom_channels") or []:
            for item in channel_settings.get("models") or []:
                if isinstance(item, dict) and item.get("category") == "image" and item.get("model"):
                    models.append(str(item.get("model")))
    except Exception:
        pass
    return _unique_non_empty(models)


def resolve_model_id(channel: str, model: str, category: str = "") -> str:
    raw = str(model or "").strip()
    if not raw:
        return raw

    channel_key = str(channel or "grsai").lower()
    category_key = str(category or "").lower()
    normalized = raw.lower()

    try:
        settings = load_comet_settings()
        channel_settings = settings.get("channels", {}).get(channel_key, {})
        for model_id, config in (channel_settings.get("models") or {}).items():
            if category_key and isinstance(config, dict) and config.get("category") != category_key:
                continue
            model_id = str(model_id)
            if model_id == raw or model_id.lower() == normalized:
                return model_id
            alias = str(config.get("alias") or "").strip() if isinstance(config, dict) else ""
            if alias and alias.lower() == normalized:
                return model_id
        custom_channel = get_custom_channel_settings(settings, channel_key)
        if custom_channel:
            for item in custom_channel.get("models") or []:
                if not isinstance(item, dict):
                    continue
                if category_key and str(item.get("category") or "").lower() != category_key:
                    continue
                model_id = str(item.get("model") or "")
                alias = str(item.get("alias") or "").strip()
                if model_id == raw or model_id.lower() == normalized:
                    return model_id
                if alias and alias.lower() == normalized:
                    return model_id
    except Exception:
        pass

    catalog = get_model_catalog().get(channel_key, {})
    categories = [category_key] if category_key else list(catalog.keys())
    for catalog_category in categories:
        for item in catalog.get(catalog_category, []) or []:
            model_id = str(item.get("model") or "")
            alias = str(item.get("alias") or "").strip()
            if model_id == raw or model_id.lower() == normalized:
                return model_id
            if alias and alias.lower() == normalized:
                return model_id
    return raw


def default_comet_settings() -> dict:
    settings = {
        "version": 2,
        "advanced": {
            "batch_image": {
                "batch_concurrency": 20,
                "max_tasks": 200,
                "failure_image_policy": BATCH_IMAGE_FAILURE_POLICY_SKIP,
                "folder_preview_limit": BATCH_IMAGE_FOLDER_PREVIEW_LIMIT,
            }
        },
        "channels": {},
        "custom_channels": [],
    }
    catalog = get_model_catalog()
    for channel, categories in catalog.items():
        channel_models = {}
        for category, models in categories.items():
            if category.startswith("_") or not isinstance(models, list):
                continue
            for item in models:
                channel_models[item["model"]] = {
                    "api_key": "",
                    "category": category,
                    "api_format": item.get("api_format", "gemini") if category == "llm" else "",
                    "alias": item.get("alias", item["model"]),
                }
                if category == "image":
                    channel_models[item["model"]]["api_format"] = item.get("api_format", "")
                    channel_models[item["model"]]["interface_mode"] = item.get("interface_mode", "")
        settings["channels"][channel] = {
            "api_key": "",
            "models": channel_models,
            "custom_llm_models": [],
            "custom_models": [],
        }
    return settings


def normalize_comet_settings(data: dict | None) -> dict:
    settings = default_comet_settings()
    if not isinstance(data, dict):
        return settings

    input_advanced = data.get("advanced") if isinstance(data.get("advanced"), dict) else {}
    input_batch_image = input_advanced.get("batch_image") if isinstance(input_advanced.get("batch_image"), dict) else {}
    settings["advanced"]["batch_image"]["batch_concurrency"] = _coerce_int(
        input_batch_image.get("batch_concurrency"),
        settings["advanced"]["batch_image"]["batch_concurrency"],
        1,
        64,
    )
    settings["advanced"]["batch_image"]["max_tasks"] = _coerce_int(
        input_batch_image.get("max_tasks"),
        settings["advanced"]["batch_image"]["max_tasks"],
        1,
        10000,
    )
    settings["advanced"]["batch_image"]["failure_image_policy"] = normalize_batch_image_failure_policy(
        input_batch_image.get("failure_image_policy"),
    )
    settings["advanced"]["batch_image"]["folder_preview_limit"] = _coerce_int(
        input_batch_image.get("folder_preview_limit"),
        settings["advanced"]["batch_image"]["folder_preview_limit"],
        1,
        500,
    )

    input_channels = data.get("channels") if isinstance(data.get("channels"), dict) else {}
    settings["custom_channels"] = normalize_custom_channels(data.get("custom_channels"))
    catalog = get_model_catalog()
    for channel, categories in catalog.items():
        catalog_model_keys = {
            str(item.get("model") or "").strip().lower()
            for category_items in categories.values()
            if isinstance(category_items, list)
            for item in category_items
            if isinstance(item, dict) and item.get("model")
        }
        input_channel = input_channels.get(channel) if isinstance(input_channels.get(channel), dict) else {}
        settings["channels"][channel]["api_key"] = str(input_channel.get("api_key") or "")
        is_private_channel = bool(get_private_channel_spec(channel))
        
        # 处理自定义LLM模型
        custom_llm_models = input_channel.get("custom_llm_models") or []
        if isinstance(custom_llm_models, list):
            settings["channels"][channel]["custom_llm_models"] = custom_llm_models
        
        input_models = input_channel.get("models") if isinstance(input_channel.get("models"), dict) else {}
        for category, models in categories.items():
            if category.startswith("_") or not isinstance(models, list):
                continue
            for item in models:
                model = item["model"]
                input_model = input_models.get(model) if isinstance(input_models.get(model), dict) else {}
                settings["channels"][channel]["models"][model]["api_key"] = str(input_model.get("api_key") or "")
                alias = str(input_model.get("alias") or item.get("alias") or model)
                if category == "video" and model in PRIVATE_KLING_VIDEO_MODELS and alias == PRIVATE_KLING_LEGACY_MODEL_ALIASES.get(model):
                    alias = str(item.get("alias") or model)
                if category == "music" and alias in {"v5.5", "v5", "v4.5"}:
                    alias = str(item.get("alias") or model)
                settings["channels"][channel]["models"][model]["alias"] = alias
                if category == "llm":
                    settings["channels"][channel]["models"][model]["api_format"] = str(input_model.get("api_format") or item.get("api_format") or "gemini")
                if category == "image":
                    api_format = normalize_api_format_value(input_model.get("api_format") or item.get("api_format") or detect_image_api_format(model), "image")
                    settings["channels"][channel]["models"][model]["api_format"] = api_format
                    settings["channels"][channel]["models"][model]["interface_mode"] = normalize_image_interface_mode(
                        input_model.get("interface_mode") or input_model.get("endpoint_mode") or item.get("interface_mode"),
                        api_format,
                    )
        
        # 处理自定义模型的配置
        for custom_model in custom_llm_models:
            model_id = custom_model.get("model")
            if not model_id:
                continue
            input_model = input_models.get(model_id) if isinstance(input_models.get(model_id), dict) else {}
            if model_id not in settings["channels"][channel]["models"]:
                settings["channels"][channel]["models"][model_id] = {
                    "api_key": "",
                    "category": "llm",
                    "api_format": "gemini",
                    "alias": model_id,
                }
            settings["channels"][channel]["models"][model_id]["api_key"] = str(input_model.get("api_key") or "")
            settings["channels"][channel]["models"][model_id]["api_format"] = str(input_model.get("api_format") or "gemini")
            settings["channels"][channel]["models"][model_id]["alias"] = str(input_model.get("alias") or custom_model.get("alias") or model_id)

        if is_private_channel:
            raw_custom_models = input_channel.get("custom_models")
            custom_models = _normalize_custom_channel_models(raw_custom_models, "openai")
            custom_keys = {(item["category"], item["model"].lower()) for item in custom_models}
            legacy_custom_llm_models = input_channel.get("custom_llm_models")
            if isinstance(legacy_custom_llm_models, list):
                for item in legacy_custom_llm_models:
                    if not isinstance(item, dict):
                        continue
                    legacy_item = dict(item)
                    legacy_item["category"] = "llm"
                    normalized_legacy = _normalize_custom_channel_models([legacy_item], "openai")
                    if not normalized_legacy:
                        continue
                    key = (normalized_legacy[0]["category"], normalized_legacy[0]["model"].lower())
                    if key in custom_keys:
                        continue
                    custom_models.append(normalized_legacy[0])
                    custom_keys.add(key)
            for model_id, config in input_models.items():
                if not isinstance(config, dict) or str(model_id).lower() in catalog_model_keys:
                    continue
                category = str(config.get("category") or "").strip().lower()
                if category not in CUSTOM_CHANNEL_MODEL_CATEGORIES:
                    continue
                key = (category, str(model_id).lower())
                if key in custom_keys:
                    continue
                item = {"model": model_id, **config}
                normalized_extra = _normalize_custom_channel_models([item], "openai")
                if not normalized_extra:
                    continue
                custom_models.append(normalized_extra[0])
                custom_keys.add((normalized_extra[0]["category"], normalized_extra[0]["model"].lower()))
            settings["channels"][channel]["custom_llm_models"] = []
            settings["channels"][channel]["custom_models"] = custom_models
            for custom_model in custom_models:
                model_id = custom_model.get("model")
                if not model_id:
                    continue
                input_model = input_models.get(model_id) if isinstance(input_models.get(model_id), dict) else {}
                if custom_model.get("category") not in CUSTOM_CHANNEL_MODEL_CATEGORIES:
                    continue
                api_format = normalize_api_format_value(
                    custom_model.get("api_format") or input_model.get("api_format") or "openai",
                    custom_model.get("category") or "llm",
                )
                settings["channels"][channel]["models"][model_id] = {
                    "api_key": str(custom_model.get("api_key") or input_model.get("api_key") or ""),
                    "category": custom_model.get("category") or "llm",
                    "api_format": api_format,
                    "alias": str(custom_model.get("alias") or input_model.get("alias") or model_id),
                }
                if custom_model.get("category") == "image":
                    settings["channels"][channel]["models"][model_id]["interface_mode"] = normalize_image_interface_mode(
                        custom_model.get("interface_mode") or input_model.get("interface_mode") or input_model.get("endpoint_mode"),
                        api_format,
                    )
    
    return settings


def _coerce_int(value, default: int, min_value: int, max_value: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = default
    return max(min_value, min(max_value, number))


def normalize_batch_image_failure_policy(value) -> str:
    raw = str(value or "").strip()
    return BATCH_IMAGE_FAILURE_POLICY_ALIASES.get(raw, BATCH_IMAGE_FAILURE_POLICY_SKIP)


def get_batch_image_advanced_settings() -> dict:
    batch_settings = load_comet_settings().get("advanced", {}).get("batch_image", {})
    return {
        "batch_concurrency": _coerce_int(batch_settings.get("batch_concurrency"), 20, 1, 64),
        "max_tasks": _coerce_int(batch_settings.get("max_tasks"), 200, 1, 10000),
        "failure_image_policy": normalize_batch_image_failure_policy(batch_settings.get("failure_image_policy")),
        "folder_preview_limit": _coerce_int(batch_settings.get("folder_preview_limit"), BATCH_IMAGE_FOLDER_PREVIEW_LIMIT, 1, 500),
    }


def load_comet_settings() -> dict:
    with _SETTINGS_LOCK:
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
                    return normalize_comet_settings(json.load(handle))
        except Exception as exc:
            logger.warning(f"Failed to load settings: {redact_sensitive_text(exc)}")
    return default_comet_settings()


def save_comet_settings(data: dict) -> dict:
    settings = normalize_comet_settings(data)
    with _SETTINGS_LOCK:
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
                json.dump(settings, handle, ensure_ascii=False, indent=2)
        except Exception as exc:
            raise CometAPIError(f"保存设置失败：{exc}") from exc
    return settings


def _trim_announcement_text(value, limit: int = 2000) -> str:
    text = str(value if value is not None else "").strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _safe_announcement_url(value: str) -> str:
    url = str(value or "").strip().strip('"').strip("'")
    if not url or not re.match(r"^https?://", url, flags=re.IGNORECASE):
        return ""
    return url


def get_announcement_url() -> str:
    return _safe_announcement_url(ANNOUNCEMENT_URL)


def _default_announcement() -> dict:
    return {
        "title": "CometAPI 公告",
        "version": "",
        "date": "",
        "summary": "公告源配置后，这里会显示最新更新和下载指引。",
        "content": "## 暂无公告\n\n当前还没有读取到远程公告。配置 CNB 公告地址后，用户可以在这里查看更新内容并手动检查更新。",
        "download_url": "",
        "docs_url": "",
        "level": "normal",
    }


def _normalize_announcement_payload(payload, source_url: str = "") -> dict:
    if isinstance(payload, str):
        return {
            **_default_announcement(),
            "title": "",
            "summary": "",
            "content": _trim_announcement_text(payload, 50000),
            "raw_markdown": True,
            "source_url": _safe_announcement_url(source_url),
        }

    data = payload if isinstance(payload, dict) else {}
    content = (
        data.get("content")
        or data.get("markdown")
        or data.get("body")
        or data.get("changelog")
        or data.get("description")
        or ""
    )
    if not content and isinstance(data.get("items"), list):
        content = "\n".join(f"- {item}" for item in data.get("items") if str(item).strip())
    level = str(data.get("level") or "normal").strip().lower()
    if level not in {"normal", "important", "critical"}:
        level = "normal"
    return {
        "title": _trim_announcement_text(data.get("title") or "CometAPI 公告", 120),
        "version": _trim_announcement_text(data.get("version") or "", 64),
        "date": _trim_announcement_text(data.get("date") or "", 64),
        "summary": _trim_announcement_text(data.get("summary") or "", 500),
        "content": _trim_announcement_text(content or _default_announcement()["content"], 50000),
        "download_url": _safe_announcement_url(data.get("download_url") or data.get("downloadUrl") or ""),
        "docs_url": _safe_announcement_url(data.get("docs_url") or data.get("docsUrl") or ""),
        "level": level,
        "raw_markdown": False,
        "source_url": _safe_announcement_url(source_url),
    }


def _load_announcement_cache() -> dict | None:
    try:
        if not os.path.exists(ANNOUNCEMENT_CACHE_FILE):
            return None
        with open(ANNOUNCEMENT_CACHE_FILE, "r", encoding="utf-8") as handle:
            cache = json.load(handle)
        if not isinstance(cache, dict):
            return None
        announcement = _normalize_announcement_payload(cache.get("announcement") or {}, cache.get("source_url") or "")
        return {
            "announcement": announcement,
            "fetched_at": float(cache.get("fetched_at") or 0),
            "checked_at": float(cache.get("checked_at") or cache.get("fetched_at") or 0),
            "source_url": _safe_announcement_url(cache.get("source_url") or announcement.get("source_url") or ""),
        }
    except Exception as exc:
        logger.warning(f"Failed to load announcement cache: {redact_sensitive_text(exc)}")
        return None


def _save_announcement_cache(announcement: dict, source_url: str) -> dict:
    now = time.time()
    cache = {
        "announcement": announcement,
        "fetched_at": now,
        "checked_at": now,
        "source_url": _safe_announcement_url(source_url),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ANNOUNCEMENT_CACHE_FILE, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)
    return cache


def _fetch_remote_announcement(url: str) -> dict:
    response = requests.get(
        url,
        headers={
            "Accept": "application/json, text/markdown, text/plain;q=0.9, */*;q=0.5",
            "User-Agent": "ComfyUI-CometAPI/announcement",
        },
        timeout=ANNOUNCEMENT_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    text = response.text or ""
    content_type = response.headers.get("Content-Type", "").lower()
    payload = None
    if "json" in content_type or text.lstrip().startswith(("{", "[")):
        payload = response.json()
    else:
        payload = text
    return _normalize_announcement_payload(payload, url)


def get_comet_announcement(force: bool = False) -> dict:
    now = time.time()
    url = get_announcement_url()
    cache = _load_announcement_cache()
    cached_announcement = cache.get("announcement") if cache else None
    fetched_at = float(cache.get("fetched_at") or 0) if cache else 0
    should_fetch = bool(url) and (force or not cache or now - fetched_at >= ANNOUNCEMENT_CACHE_TTL_SECONDS)

    if should_fetch:
        try:
            announcement = _fetch_remote_announcement(url)
            cache = _save_announcement_cache(announcement, url)
            return {
                "ok": True,
                "remote_configured": True,
                "from_cache": False,
                "announcement": announcement,
                "fetched_at": cache["fetched_at"],
                "checked_at": cache["checked_at"],
                "source_url": url,
                "error": "",
            }
        except Exception as exc:
            error = format_error_message(exc)
            if cached_announcement:
                return {
                    "ok": False,
                    "remote_configured": True,
                    "from_cache": True,
                    "announcement": cached_announcement,
                    "fetched_at": fetched_at,
                    "checked_at": now,
                    "source_url": cache.get("source_url") or url,
                    "error": error,
                }
            return {
                "ok": False,
                "remote_configured": True,
                "from_cache": False,
                "announcement": _default_announcement(),
                "fetched_at": 0,
                "checked_at": now,
                "source_url": url,
                "error": error,
            }

    return {
        "ok": True,
        "remote_configured": bool(url),
        "from_cache": bool(cache),
        "announcement": cached_announcement or _default_announcement(),
        "fetched_at": fetched_at,
        "checked_at": float(cache.get("checked_at") or fetched_at or 0) if cache else 0,
        "source_url": (cache.get("source_url") if cache else "") or url,
        "error": "" if url or cache else "公告源未配置",
    }


def get_channel_api_key(api_key_from_node: str = "", channel: str = "grsai", model: str = "", category: str = "") -> str:
    if api_key_from_node and api_key_from_node.strip():
        return api_key_from_node.strip()

    settings = load_comet_settings()
    channel_key_name = str(channel or "grsai").lower()
    custom_channel = get_custom_channel_settings(settings, channel_key_name)
    if custom_channel:
        if model:
            model_id = resolve_model_id(channel, model, category)
            model_key = get_custom_model_settings(custom_channel, model_id, category).get("api_key", "")
            if model_key and str(model_key).strip():
                return str(model_key).strip()
        channel_key = custom_channel.get("api_key", "")
        return str(channel_key).strip() if channel_key and str(channel_key).strip() else ""

    channel_settings = settings.get("channels", {}).get(channel_key_name, {})
    if model:
        model_id = resolve_model_id(channel, model, category)
        model_entry = channel_settings.get("models", {}).get(model_id, {})
        # 同名模型若同时存在多分类，优先匹配指定 category 的那条；否则任选第一个
        if category and isinstance(model_entry, dict) and model_entry.get("category") and model_entry.get("category") != category:
            model_key = ""
        else:
            model_key = model_entry.get("api_key", "") if isinstance(model_entry, dict) else ""
        if model_key and str(model_key).strip():
            return str(model_key).strip()
    channel_key = channel_settings.get("api_key", "")
    if channel_key and str(channel_key).strip():
        return str(channel_key).strip()

    private_channel = get_private_channel_spec(channel_key_name)
    if private_channel:
        if model:
            private_model = get_private_channel_model_settings(channel_key_name, model, category)
            private_model_key = str(private_model.get("api_key") or "").strip()
            if private_model_key:
                return private_model_key
        private_env = str(private_channel.get("api_key_env") or "").strip()
        if private_env and os.getenv(private_env, "").strip():
            return os.getenv(private_env, "").strip()
        private_key = str(private_channel.get("api_key") or "").strip()
        if private_key:
            return private_key
        return ""

    try:
        import builtins

        getter = getattr(builtins, "get_api_key", None)
        if callable(getter):
            try:
                key = getter("", channel)
            except TypeError:
                key = getter("")
            if key:
                return key.strip()
    except Exception:
        pass

    if channel_key_name == "runninghub":
        return os.getenv("RUNNINGHUB_KEY", "").strip()
    if channel_key_name == "modelverse":
        return (os.getenv("MODELVERSE_API_KEY", "") or os.getenv("YOUYUN_KEY", "")).strip()
    if channel_key_name == "apimart":
        return os.getenv("APIMART_KEY", "").strip()
    return os.getenv("GRSAI_KEY", "").strip()


def get_channel_api_url(channel: str = "grsai") -> str:
    settings = load_comet_settings()
    custom_channel = get_custom_channel_settings(settings, channel)
    if custom_channel:
        return normalize_api_base_url(custom_channel.get("api_url") or "https://api.openai.com")
    channel_key = str(channel or "grsai").lower()
    private_channel = get_private_channel_spec(channel_key)
    if private_channel:
        return normalize_api_base_url(private_channel.get("api_url") or "", "https://api.openai.com")
    if channel_key == "runninghub":
        return "https://llm.runninghub.cn/v1"
    if channel_key == "modelverse":
        return "https://api.modelverse.cn"
    if channel_key == "apimart":
        return "https://api.apimart.ai"
    return ""


def get_model_api_format(channel: str = "grsai", model: str = "") -> str:
    """获取LLM模型的接口格式设置"""
    if not model:
        return "gemini"
    
    settings = load_comet_settings()
    channel_key = str(channel or "grsai").lower()
    custom_channel = get_custom_channel_settings(settings, channel_key)
    if custom_channel:
        model_id = resolve_model_id(channel, model, "llm")
        model_settings = get_custom_model_settings(custom_channel, model_id, "llm")
        return normalize_api_format_value(model_settings.get("api_format") or custom_channel.get("api_format") or "openai")

    channel_settings = settings.get("channels", {}).get(channel_key, {})
    model_id = resolve_model_id(channel, model, "llm")
    model_settings = channel_settings.get("models", {}).get(model_id, {})
    if model_settings.get("api_format"):
        return str(model_settings.get("api_format"))
    private_channel = get_private_channel_spec(channel_key)
    if private_channel:
        private_model = get_private_channel_model_settings(channel_key, model_id, "llm")
        private_api_format = str(private_model.get("api_format") or private_channel.get("api_format") or "").strip()
        if private_api_format:
            return normalize_api_format_value(private_api_format, "llm")
    if channel_key == "grsai":
        return GRSAI_LLM_API_FORMATS.get(model_id, "gemini")
    if channel_key == "runninghub":
        return RUNNINGHUB_LLM_API_FORMATS.get(model_id, "openai")
    if channel_key == "modelverse":
        return MODELVERSE_LLM_API_FORMATS.get(model_id, "openai")
    if channel_key == "apimart":
        return APIMART_LLM_API_FORMATS.get(model_id, "gemini")
    return "gemini"


def get_grsai_api_key(api_key_from_node: str = "") -> str:
    return get_channel_api_key(api_key_from_node, "grsai")


def safe_pil_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def tensor_to_pil(tensor: torch.Tensor) -> list[Image.Image]:
    if not isinstance(tensor, torch.Tensor):
        return []
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    images = []
    for i in range(tensor.shape[0]):
        arr = (torch.clamp(tensor[i], 0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            images.append(Image.fromarray(arr, "RGBA"))
        else:
            images.append(Image.fromarray(arr[:, :, :3], "RGB"))
    return images


def pil_to_tensor(pil_images: Image.Image | list[Image.Image]) -> torch.Tensor:
    if not isinstance(pil_images, list):
        pil_images = [pil_images]

    tensors = []
    for pil_image in pil_images:
        pil_image = safe_pil_to_rgb(pil_image)
        arr = np.array(pil_image).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr)[None,])

    if not tensors:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)
    return torch.cat(tensors, dim=0)


def get_folder_paths():
    try:
        import folder_paths

        return folder_paths
    except Exception as exc:
        raise RuntimeError(f"cannot import ComfyUI folder_paths: {exc}") from exc


def tensor_first_to_pil(image: torch.Tensor) -> Image.Image:
    pil_images = tensor_to_pil(image)
    if not pil_images:
        raise ValueError("输入图片为空。")
    return safe_pil_to_rgb(pil_images[0])


def _strip_wrapping_path_quotes(value: str) -> str:
    raw = str(value or "").strip()
    quote_pairs = {
        '"': '"',
        "'": "'",
        "\u201c": "\u201d",
        "\u2018": "\u2019",
    }
    changed = True
    while changed and len(raw) >= 2:
        changed = False
        first = raw[0]
        last = raw[-1]
        if quote_pairs.get(first) == last:
            raw = raw[1:-1].strip()
            changed = True
    return raw


def _looks_like_windows_absolute_path(value: str) -> bool:
    raw = _strip_wrapping_path_quotes(value)
    return bool(re.match(r"^[A-Za-z]:[\\/]", raw) or raw.startswith("\\\\"))


def _normalize_user_path(value: str) -> str:
    raw = os.path.expanduser(_strip_wrapping_path_quotes(value))
    if os.name == "nt":
        return raw.replace("/", os.sep)
    return raw.replace("\\", "/")


def _is_current_platform_absolute_path(value: str) -> bool:
    if os.name != "nt" and _looks_like_windows_absolute_path(value):
        return False
    return os.path.isabs(_normalize_user_path(value))


def _normalize_absolute_asset_path(value: str) -> str:
    raw = _strip_wrapping_path_quotes(value)
    if os.name != "nt" and _looks_like_windows_absolute_path(raw):
        raise CometAPIError("这个素材使用的是 Windows 绝对路径，当前 Linux/macOS 环境无法直接读取，请改成当前系统的绝对路径或重新生成素材。")
    path = _normalize_user_path(raw)
    if not os.path.isabs(path):
        raise CometAPIError("绝对路径素材缺少有效的绝对路径。")
    return os.path.abspath(path)


def normalize_asset_ref(asset_ref: str | dict | None) -> dict | None:
    if not asset_ref:
        return None
    if isinstance(asset_ref, dict):
        data = asset_ref
    else:
        raw = str(asset_ref).strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except Exception:
            if os.name != "nt" and _looks_like_windows_absolute_path(raw):
                _normalize_absolute_asset_path(raw)
            if _is_current_platform_absolute_path(raw):
                absolute_path = _normalize_absolute_asset_path(raw)
                return {
                    "filename": os.path.basename(absolute_path),
                    "subfolder": "",
                    "type": "absolute",
                    "absolute_path": absolute_path,
                }
            return {"filename": os.path.basename(raw), "subfolder": ASSET_SUBFOLDER, "type": "output"}

    absolute_path = _strip_wrapping_path_quotes(data.get("absolute_path") or data.get("abs_path") or data.get("path") or "")
    asset_type = str(data.get("type", "output") or "output")
    if asset_type == "absolute" or absolute_path:
        absolute_path = _normalize_absolute_asset_path(absolute_path or str(data.get("filename", "")).strip())
        filename = str(data.get("filename") or os.path.basename(absolute_path)).strip()
        if not filename:
            return None
        return {
            "filename": filename,
            "subfolder": "",
            "type": "absolute",
            "absolute_path": absolute_path,
        }

    filename = str(data.get("filename", "")).strip()
    if not filename:
        return None
    return {
        "filename": filename,
        "subfolder": str(data.get("subfolder", ASSET_SUBFOLDER) or ""),
        "type": asset_type,
    }


def normalize_asset_refs(asset_ref: str | dict | list | None) -> list[dict]:
    if not asset_ref:
        return []
    if isinstance(asset_ref, list):
        return [ref for item in asset_ref if (ref := normalize_asset_ref(item))]
    if isinstance(asset_ref, str):
        raw = asset_ref.strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except Exception:
            data = raw
        if isinstance(data, list):
            return [ref for item in data if (ref := normalize_asset_ref(item))]
        return [ref] if (ref := normalize_asset_ref(data)) else []
    return [ref] if (ref := normalize_asset_ref(asset_ref)) else []


def asset_ref_to_json(asset_ref: dict) -> str:
    data = {
        "filename": asset_ref["filename"],
        "subfolder": asset_ref.get("subfolder", ""),
        "type": asset_ref.get("type", "output"),
    }
    if asset_ref.get("absolute_path"):
        data["absolute_path"] = asset_ref["absolute_path"]
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def asset_refs_to_json(asset_refs: list[dict]) -> str:
    if len(asset_refs) == 1:
        return asset_ref_to_json(asset_refs[0])
    data = []
    for asset_ref in asset_refs:
        item = {
            "filename": asset_ref["filename"],
            "subfolder": asset_ref.get("subfolder", ""),
            "type": asset_ref.get("type", "output"),
        }
        if asset_ref.get("absolute_path"):
            item["absolute_path"] = asset_ref["absolute_path"]
        data.append(item)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def remember_batch_asset_refs(node_id: str | int, asset_refs: list[dict]) -> None:
    """记录 batch 节点本次执行产生的 asset_refs。下游图像卡片可以靠这个免去对 padded tensor 落盘。"""
    if not asset_refs:
        return
    key = str(node_id or "").strip()
    if not key:
        return
    snapshot = []
    for ref in asset_refs:
        if not isinstance(ref, dict):
            continue
        item = {
            "filename": ref.get("filename"),
            "subfolder": ref.get("subfolder", ""),
            "type": ref.get("type", "output"),
        }
        if ref.get("absolute_path"):
            item["absolute_path"] = ref["absolute_path"]
        snapshot.append(item)
    if not snapshot:
        return
    with _LAST_BATCH_ASSET_REFS_LOCK:
        _LAST_BATCH_ASSET_REFS[key] = {
            "refs": snapshot,
            "ts": time.time(),
        }
        # 顺手清理一下过期条目，避免长时间运行后内存膨胀
        threshold = time.time() - _LAST_BATCH_ASSET_REFS_TTL
        for stale_key in [k for k, v in _LAST_BATCH_ASSET_REFS.items() if v.get("ts", 0) < threshold]:
            _LAST_BATCH_ASSET_REFS.pop(stale_key, None)


def consume_batch_asset_refs_for_count(expected_count: int, preferred_node_id: str | int | None = None) -> list[dict]:
    """图像卡片嗅探：返回最近 batch 缓存里**张数刚好等于 expected_count** 的那一份原图列表。

    匹配优先级：
      1) 张数相等 + 节点 ID 相等（`preferred_node_id` 命中）→ 最稳；
      2) 张数相等的全部条目里，时间最新的一份。
    命中后立刻把那条从缓存 pop 掉，避免下次"重跑同一个工作流"时拿到陈旧数据。

    `expected_count <= 0` 时返回空列表。
    """
    if expected_count <= 0:
        return []
    pref_key = str(preferred_node_id or "").strip()
    with _LAST_BATCH_ASSET_REFS_LOCK:
        threshold = time.time() - _LAST_BATCH_ASSET_REFS_TTL
        candidates = [
            (key, entry)
            for key, entry in _LAST_BATCH_ASSET_REFS.items()
            if entry.get("ts", 0) >= threshold and len(entry.get("refs") or []) == expected_count
        ]
        if not candidates:
            return []
        # 优先用 preferred_node_id 命中
        chosen = None
        if pref_key:
            for key, entry in candidates:
                if key == pref_key:
                    chosen = (key, entry)
                    break
        # 否则取最新
        if chosen is None:
            candidates.sort(key=lambda item: item[1].get("ts", 0), reverse=True)
            chosen = candidates[0]
        chosen_key, chosen_entry = chosen
        # 命中即消费：从缓存里 pop，避免重跑工作流时被陈旧数据"污染"
        _LAST_BATCH_ASSET_REFS.pop(chosen_key, None)
        return list(chosen_entry["refs"])


def asset_ref_to_native_view_ref(asset_ref: dict) -> dict | None:
    if not asset_ref or asset_ref.get("type") == "absolute" or asset_ref.get("absolute_path"):
        return None
    return {
        "filename": asset_ref["filename"],
        "subfolder": asset_ref.get("subfolder", ""),
        "type": asset_ref.get("type", "output"),
    }


def video_card_ui(asset_ref: dict) -> dict:
    ui = {
        "asset_ref": [asset_ref_to_json(asset_ref)],
        "videos": [asset_ref],
    }
    native_ref = asset_ref_to_native_view_ref(asset_ref)
    if native_ref:
        ui["images"] = [native_ref]
        ui["animated"] = [True]
    return ui


def asset_abs_path(asset_ref: dict) -> str:
    if asset_ref.get("type") == "absolute" or asset_ref.get("absolute_path"):
        path = asset_ref.get("absolute_path") or asset_ref.get("filename", "")
        return os.path.abspath(_normalize_user_path(path))

    folder_paths = get_folder_paths()
    asset_type = asset_ref.get("type", "output")
    if asset_type == "temp":
        base_dir = folder_paths.get_temp_directory()
    elif asset_type == "input":
        base_dir = folder_paths.get_input_directory()
    else:
        base_dir = folder_paths.get_output_directory()

    subfolder = asset_ref.get("subfolder", "") or ""
    return os.path.abspath(os.path.join(base_dir, subfolder, asset_ref["filename"]))


def asset_ref_from_path(path: str) -> dict:
    absolute_path = os.path.abspath(path)
    try:
        folder_paths = get_folder_paths()
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        if os.path.commonpath([output_dir, absolute_path]) == output_dir:
            rel = os.path.relpath(absolute_path, output_dir)
            subfolder = os.path.dirname(rel).replace("\\", "/")
            return {"filename": os.path.basename(absolute_path), "subfolder": "" if subfolder == "." else subfolder, "type": "output"}
    except Exception:
        pass
    return {
        "filename": os.path.basename(absolute_path),
        "subfolder": "",
        "type": "absolute",
        "absolute_path": absolute_path,
    }


def _safe_filename_stem(value: str, fallback: str) -> str:
    stem = os.path.splitext(str(value or "").strip())[0].strip()
    stem = re.sub(r'[<>:"|?*\x00-\x1f]+', "_", stem)
    stem = stem.strip(" .")
    return stem or fallback


def _resolve_asset_output_target(prefix: str, fallback: str, extension: str) -> tuple[str, str, str, str | None]:
    folder_paths = get_folder_paths()
    output_dir = os.path.abspath(folder_paths.get_output_directory())
    raw_input = _strip_wrapping_path_quotes(prefix) or fallback
    if os.name != "nt" and _looks_like_windows_absolute_path(raw_input):
        raise CometAPIError("当前是 Linux/macOS 环境，不能使用 Windows 盘符路径。请改成当前系统的绝对路径，比如 /home/用户名/输出目录。")
    raw = _normalize_user_path(raw_input)
    ends_with_sep = raw.endswith(os.sep)

    if os.path.isabs(raw):
        # 绝对路径一律视为目标目录，不再把末段当文件名前缀。
        # 这样从资源管理器复制路径直接粘进来就能用，符合 Windows 习惯。
        target_dir = os.path.abspath(raw.rstrip(os.sep)) if not ends_with_sep else os.path.abspath(raw)
        return target_dir, fallback, "", target_dir

    normalized = os.path.normpath(raw)
    if normalized in {"", "."}:
        normalized = fallback
    rel_dir = normalized if ends_with_sep else os.path.dirname(normalized)
    if rel_dir in {"", "."}:
        rel_dir = ""
    stem_source = fallback if ends_with_sep else os.path.basename(normalized)
    stem = _safe_filename_stem(stem_source, fallback)

    target_dir = os.path.abspath(os.path.join(output_dir, rel_dir))
    if os.path.commonpath([output_dir, target_dir]) != output_dir:
        raise CometAPIError("相对保存路径不能跳出 ComfyUI 的 output 目录；如果要跳出，请填写绝对路径。")
    subfolder = os.path.relpath(target_dir, output_dir)
    if subfolder == ".":
        subfolder = ""
    return target_dir, stem, subfolder.replace("\\", "/"), None


def _make_asset_ref(filename: str, subfolder: str, absolute_dir: str | None) -> dict:
    if absolute_dir:
        return {
            "filename": filename,
            "subfolder": "",
            "type": "absolute",
            "absolute_path": os.path.abspath(os.path.join(absolute_dir, filename)),
        }
    return {"filename": filename, "subfolder": subfolder, "type": "output"}


def save_asset_image(image: torch.Tensor, prefix: str = "CometAPIImageCard") -> tuple[dict, Image.Image]:
    target_dir, safe_prefix, subfolder, absolute_dir = _resolve_asset_output_target(prefix, "CometAPIImageCard", ".png")
    os.makedirs(target_dir, exist_ok=True)

    pil_image = tensor_first_to_pil(image)
    filename = f"{safe_prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
    pil_image.save(os.path.join(target_dir, filename), "PNG")
    return _make_asset_ref(filename, subfolder, absolute_dir), pil_image


def save_asset_images(image: torch.Tensor, prefix: str = "CometAPIImageCard") -> tuple[list[dict], list[Image.Image]]:
    target_dir, safe_prefix, subfolder, absolute_dir = _resolve_asset_output_target(prefix, "CometAPIImageCard", ".png")
    os.makedirs(target_dir, exist_ok=True)

    pil_images = tensor_to_pil(image)
    saved_refs = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for index, pil_image in enumerate(pil_images, start=1):
        pil_image = safe_pil_to_rgb(pil_image)
        filename = f"{safe_prefix}_{timestamp}_{index:02d}_{uuid.uuid4().hex[:8]}.png"
        pil_image.save(os.path.join(target_dir, filename), "PNG")
        saved_refs.append(_make_asset_ref(filename, subfolder, absolute_dir))
    return saved_refs, pil_images


def load_asset_image(asset_ref: dict) -> Image.Image:
    path = asset_abs_path(asset_ref)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return safe_pil_to_rgb(Image.open(path))


IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 300


def download_image(url: str, timeout: int = IMAGE_DOWNLOAD_TIMEOUT_SECONDS) -> Image.Image | None:
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        res.raise_for_status()
        return Image.open(BytesIO(res.content))
    except Exception as exc:
        print(f"[CometAPI] image download failed: {redact_sensitive_text(exc)}")
        return None


def upload_image_grsai(api_key: str, image_input: torch.Tensor | Image.Image) -> str | None:
    temp_path = None
    try:
        if isinstance(image_input, torch.Tensor):
            pil_images = tensor_to_pil(image_input)
            if not pil_images:
                return None
            pil_image = pil_images[0]
        else:
            pil_image = image_input

        rgb_pil = safe_pil_to_rgb(pil_image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            rgb_pil.save(temp_file, "PNG")
            temp_path = temp_file.name

        host = "https://grsai.dakka.com.cn"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        token_res = requests.post(
            f"{host}/client/resource/newUploadTokenZH",
            headers=headers,
            json={"sux": "png"},
            timeout=30,
        )
        token_res.raise_for_status()
        token_body = token_res.json()
        token_data = token_body.get("data", {}) if isinstance(token_body, dict) else {}
        if not isinstance(token_data, dict):
            raise CometAPIError(f"grsai 上传凭证响应异常：{str(token_body)[:300]}")

        with open(temp_path, "rb") as image_file:
            upload_res = requests.post(
                url=token_data["url"],
                data={"token": token_data["token"], "key": token_data["key"]},
                files={"file": image_file},
                timeout=120,
            )
            upload_res.raise_for_status()

        return f"{token_data['domain']}/{token_data['key']}"
    except Exception as exc:
        print(f"[CometAPI] grsai upload failed: {redact_sensitive_text(exc)}")
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


class GrsaiMediaUploadAPI:
    host = "https://grsai.dakka.com.cn"
    label = "grsai"

    def __init__(self, api_key: str):
        if not api_key:
            raise CometAPIError("缺少 grsai API Key，请先在设置中心填写。")
        self.api_key = api_key

    def _upload_file(self, path: str, suffix: str = "") -> str:
        if not path or not os.path.exists(path):
            raise CometAPIError("grsai 媒体文件不存在或无法读取。")

        ext = (os.path.splitext(path)[1] or suffix or ".bin").lstrip(".").lower()
        filename = os.path.basename(path) or f"comet-media.{ext}"
        mime = mimetypes.guess_type(filename)[0] or mimetypes.guess_type(f"file.{ext}")[0] or "application/octet-stream"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        token_res = requests.post(
            f"{self.host}/client/resource/newUploadTokenZH",
            headers=headers,
            json={"sux": ext},
            timeout=30,
        )
        token_res.raise_for_status()
        token_body = token_res.json()
        token_data = token_body.get("data", {}) if isinstance(token_body, dict) else {}
        if not isinstance(token_data, dict):
            raise CometAPIError(f"grsai 上传凭证响应异常：{str(token_body)[:300]}")
        required_fields = {"url", "token", "key", "domain"}
        missing = sorted(field for field in required_fields if not token_data.get(field))
        if missing:
            raise CometAPIError(f"grsai 上传凭证缺少字段：{', '.join(missing)}")

        with open(path, "rb") as media_file:
            upload_res = requests.post(
                url=token_data["url"],
                data={"token": token_data["token"], "key": token_data["key"]},
                files={"file": (filename, media_file, mime)},
                timeout=180,
            )
            upload_res.raise_for_status()

        domain = str(token_data["domain"]).rstrip("/")
        key = str(token_data["key"]).lstrip("/")
        return f"{domain}/{key}"

    def _upload_media_input(self, media_input, suffix: str) -> str:
        path, should_delete = _media_input_to_path(media_input, suffix)
        if not path:
            raise CometAPIError("不支持这个 grsai 媒体输入，请检查连接的素材类型。")
        try:
            return self._upload_file(path, suffix)
        finally:
            if should_delete:
                try:
                    os.unlink(path)
                except Exception:
                    pass


def upload_image_private(api_key: str, image_input: torch.Tensor | Image.Image) -> str | None:
    try:
        if isinstance(image_input, torch.Tensor):
            pil_images = tensor_to_pil(image_input)
            if not pil_images:
                return None
            pil_image = pil_images[0]
        else:
            pil_image = image_input

        buffered = BytesIO()
        safe_pil_to_rgb(pil_image).save(buffered, format="PNG")
        headers = {"Authorization": f"Bearer {api_key}"}
        for _ in range(3):
            try:
                buffered.seek(0)
                response = requests.post(
                    "https://imageproxy.zhongzhuan.chat/api/upload",
                    headers=headers,
                    files={"file": ("image.png", buffered, "image/png")},
                    timeout=45,
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("data") or data.get("url")
            except Exception:
                time.sleep(1)
        return None
    except Exception as exc:
        print(f"[CometAPI] private image upload failed: {redact_sensitive_text(exc)}")
        return None


class VideoAdapter:
    def __init__(self, video_path: str | None, comet_error: str = "", video_url: str = ""):
        self.video_path = video_path or ""
        self.comet_error = str(comet_error or "")
        self.video_url = str(video_url or "")

    def get_dimensions(self):
        return 1280, 720

    def save_to(self, output_path, **kwargs):
        if self.video_path and os.path.exists(self.video_path):
            import shutil

            shutil.copyfile(self.video_path, output_path)
            return True
        return False


def _safe_asset_prefix(prefix: str, fallback: str) -> str:
    safe_prefix = (prefix.strip() or fallback).replace(os.sep, "_").replace("/", "_")
    return safe_prefix or fallback


def _resolve_existing_media_path(value) -> str:
    if value is None:
        return ""
    if isinstance(value, os.PathLike):
        raw = os.fspath(value)
    elif isinstance(value, str):
        raw = value
    else:
        return ""

    raw = _strip_wrapping_path_quotes(raw)
    if not raw:
        return ""
    if raw.startswith("file://"):
        raw = raw[7:]

    candidates = []
    try:
        candidates.append(_normalize_user_path(raw))
    except Exception:
        candidates.append(raw)
    candidates.append(raw)

    if os.name == "nt" or not _looks_like_windows_absolute_path(raw):
        try:
            folder_paths = get_folder_paths()
            candidates.append(folder_paths.get_annotated_filepath(raw))
            for getter_name in ("get_input_directory", "get_output_directory", "get_temp_directory"):
                getter = getattr(folder_paths, getter_name, None)
                if callable(getter):
                    candidates.append(os.path.join(getter(), raw))
        except Exception:
            pass

    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = os.path.abspath(os.path.expanduser(str(candidate)))
        except Exception:
            path = str(candidate)
        if os.path.exists(path):
            return path
    return ""


def _extract_object_media_path(value) -> str:
    for attr in ("video_path", "audio_path", "path", "file", "filename", "filepath", "source"):
        path = _resolve_existing_media_path(getattr(value, attr, None))
        if path:
            return path
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        for key in ("video_path", "audio_path", "path", "file", "filename", "filepath", "source", "_VideoFromFile__file"):
            path = _resolve_existing_media_path(data.get(key))
            if path:
                return path
    return ""


def _copy_or_save_video(video, output_path: str) -> None:
    if video is None:
        raise CometAPIError("没有可保存的视频。")

    if isinstance(video, VideoAdapter) and not video.video_path:
        upstream_error = str(getattr(video, "comet_error", "") or "").strip()
        if upstream_error:
            raise CometAPIError(upstream_error)
        raise CometAPIError("上游视频节点没有输出可保存的视频，请先查看上游节点的报错。")

    source_path = _resolve_existing_media_path(video) or _extract_object_media_path(video)

    if source_path:
        import shutil

        shutil.copyfile(source_path, output_path)
        return

    save_to = getattr(video, "save_to", None)
    if callable(save_to):
        try:
            save_to(output_path)
        except TypeError:
            try:
                from comfy_api.latest import Types

                save_to(output_path, format=Types.VideoContainer("mp4"), codec="auto")
            except Exception:
                save_to(output_path, format="mp4")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return

    raise CometAPIError(f"不支持的视频输入类型：{type(video).__name__}")


def save_asset_video(video, prefix: str = "CometAPIVideoCard") -> tuple[dict, VideoAdapter]:
    target_dir, safe_prefix, subfolder, absolute_dir = _resolve_asset_output_target(prefix, "CometAPIVideoCard", ".mp4")
    os.makedirs(target_dir, exist_ok=True)

    filename = f"{safe_prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
    path = os.path.join(target_dir, filename)
    _copy_or_save_video(video, path)
    return _make_asset_ref(filename, subfolder, absolute_dir), VideoAdapter(path)


def load_asset_video(asset_ref: dict) -> VideoAdapter:
    path = asset_abs_path(asset_ref)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return VideoAdapter(path)


def save_asset_audio(audio, prefix: str = "CometAPIAudioCard") -> list[dict]:
    if not isinstance(audio, dict) or not isinstance(audio.get("waveform"), torch.Tensor):
        raise CometAPIError("没有可保存的音频。")

    waveform = audio["waveform"].detach().cpu().float().clamp(-1.0, 1.0)
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3:
        raise CometAPIError("音频数据格式不正确。")

    sample_rate = int(audio.get("sample_rate") or 44100)
    target_dir, safe_prefix, subfolder, absolute_dir = _resolve_asset_output_target(prefix, "CometAPIAudioCard", ".wav")
    os.makedirs(target_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    refs = []
    for index, item in enumerate(waveform, start=1):
        if item.ndim != 2:
            continue
        channels = int(item.shape[0])
        samples = (item.transpose(0, 1).numpy() * 32767.0).astype(np.int16)
        filename = f"{safe_prefix}_{timestamp}_{index:02d}_{uuid.uuid4().hex[:8]}.wav"
        path = os.path.join(target_dir, filename)
        with wave.open(path, "wb") as handle:
            handle.setnchannels(max(1, channels))
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(samples.tobytes())
        refs.append(_make_asset_ref(filename, subfolder, absolute_dir))
    if not refs:
        raise CometAPIError("没有可保存的音频。")
    return refs


def video_thumbnail_image(video_path: str, max_side: int = 160) -> Image.Image:
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    resampling = getattr(Image, "Resampling", Image)
    bilinear = getattr(resampling, "BILINEAR", Image.BILINEAR)
    lanczos = getattr(resampling, "LANCZOS", Image.LANCZOS)

    def _score_frame(image: Image.Image) -> tuple[float, bool]:
        sample = safe_pil_to_rgb(image).resize((32, 32), bilinear)
        arr = np.asarray(sample, dtype=np.float32)
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
        avg = float(lum.mean())
        bright_ratio = float((lum > 24).mean())
        return avg + bright_ratio * 90.0, avg > 18.0 or bright_ratio > 0.035

    def _finish(image: Image.Image) -> Image.Image:
        thumb = safe_pil_to_rgb(image)
        thumb.thumbnail((max_side, max_side), lanczos)
        return thumb

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            indices = [0]
            if frame_count > 1:
                indices.extend(
                    max(0, min(frame_count - 1, int(frame_count * ratio)))
                    for ratio in (0.02, 0.12, 0.3, 0.55, 0.8)
                )
            best: tuple[float, Image.Image] | None = None
            for index in dict.fromkeys(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                score, usable = _score_frame(image)
                if best is None or score > best[0]:
                    best = (score, image)
                if usable:
                    cap.release()
                    return _finish(image)
            cap.release()
            if best is not None:
                return _finish(best[1])
    except Exception as exc:
        logger.debug(f"cv2 video thumbnail failed: {redact_sensitive_text(exc)}")

    try:
        import imageio.v3 as iio  # type: ignore

        frame = iio.imread(video_path, index=0)
        image = Image.fromarray(np.asarray(frame)[..., :3])
        return _finish(image)
    except Exception as exc:
        logger.debug(f"imageio video thumbnail failed: {redact_sensitive_text(exc)}")

    try:
        import shutil
        import subprocess

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            handle = tempfile.NamedTemporaryFile(prefix="cometapi_video_thumb_", suffix=".jpg", delete=False)
            thumb_path = handle.name
            handle.close()
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-ss", "0", "-i", video_path, "-frames:v", "1", "-q:v", "3", thumb_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=12,
                    check=False,
                )
                if os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0:
                    return _finish(Image.open(thumb_path))
            finally:
                try:
                    os.remove(thumb_path)
                except OSError:
                    pass
    except Exception as exc:
        logger.debug(f"ffmpeg video thumbnail failed: {redact_sensitive_text(exc)}")

    raise RuntimeError("cannot extract video thumbnail; install cv2, imageio, or ffmpeg")


def video_input_to_path(video) -> str:
    if video is None:
        return ""
    source_path = ""
    if isinstance(video, (str, os.PathLike)):
        source_path = os.fspath(video)
    else:
        source_path = str(getattr(video, "video_path", "") or getattr(video, "path", "") or "")
    if source_path and os.path.exists(source_path):
        return source_path

    handle = tempfile.NamedTemporaryFile(prefix="cometapi_llm_video_", suffix=".mp4", delete=False)
    temp_path = handle.name
    handle.close()
    try:
        _copy_or_save_video(video, temp_path)
        return temp_path
    except Exception:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise


def download_video_asset(video_url: str, prefix: str = "CometAPIVideo", timeout: int = 600) -> str:
    if not video_url:
        raise CometAPIError("API 没有返回视频地址")

    folder_paths = get_folder_paths()
    output_dir = folder_paths.get_output_directory()
    target_dir = os.path.join(output_dir, ASSET_SUBFOLDER)
    os.makedirs(target_dir, exist_ok=True)
    safe_prefix = _safe_asset_prefix(prefix, "CometAPIVideo")
    filename = f"{safe_prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
    path = os.path.join(target_dir, filename)

    headers = {"User-Agent": "Mozilla/5.0"}
    last_error = None
    for _ in range(3):
        try:
            with requests.get(video_url, headers=headers, stream=True, timeout=(20, timeout)) as response:
                response.raise_for_status()
                with open(path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if os.path.exists(path) and os.path.getsize(path) > 1024:
                return path
        except Exception as exc:
            last_error = exc
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass
            time.sleep(2)

    try:
        import shutil
        import subprocess

        curl_path = shutil.which("curl")
        if curl_path:
            subprocess.run(
                [curl_path, "-k", "-L", "--connect-timeout", "20", "--max-time", str(timeout), "-o", path, video_url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout + 10,
                check=True,
            )
            if os.path.exists(path) and os.path.getsize(path) > 1024:
                return path
    except Exception as exc:
        last_error = exc

    raise CometAPIError(f"视频下载失败：{last_error}")


def silent_audio(duration_seconds: float = 1.0, sample_rate: int = 44100) -> dict:
    samples = max(1, int(float(duration_seconds or 1.0) * sample_rate))
    return {"waveform": torch.zeros((1, 2, samples), dtype=torch.float32), "sample_rate": sample_rate}


def _audio_tensor_to_float(wav: torch.Tensor) -> torch.Tensor:
    if wav.dtype.is_floating_point:
        return wav.float()
    if wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    if wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    if wav.dtype == torch.uint8:
        return (wav.float() - 128.0) / 128.0
    return wav.float()


def load_audio_file(path: str) -> dict:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        import av

        with av.open(path) as container:
            if not container.streams.audio:
                raise CometAPIError("\u97f3\u9891\u6587\u4ef6\u4e2d\u6ca1\u6709\u53ef\u89e3\u7801\u7684\u97f3\u8f68\u3002")
            stream = container.streams.audio[0]
            sample_rate = int(stream.codec_context.sample_rate or 44100)
            channel_count = int(getattr(stream, "channels", 0) or 0)
            frames = []
            for frame in container.decode(stream):
                arr = torch.from_numpy(frame.to_ndarray())
                if channel_count and arr.ndim == 1:
                    arr = arr.view(-1, channel_count).t()
                elif channel_count and arr.ndim == 2 and arr.shape[0] != channel_count and arr.shape[1] == channel_count:
                    arr = arr.t()
                frames.append(_audio_tensor_to_float(arr))
            if not frames:
                raise CometAPIError("\u97f3\u9891\u6587\u4ef6\u6ca1\u6709\u89e3\u7801\u51fa\u6570\u636e\u3002")
            waveform = torch.cat(frames, dim=1).clamp(-1.0, 1.0)
            return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    except ImportError:
        if os.path.splitext(path)[1].lower() != ".wav":
            raise CometAPIError("\u5f53\u524d ComfyUI \u73af\u5883\u6ca1\u6709 av\uff0c\u53ea\u80fd\u89e3\u7801 wav \u97f3\u9891\u3002")
        with wave.open(path, "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
        if sample_width != 2:
            raise CometAPIError("\u5f53\u524d wav \u56de\u9000\u89e3\u7801\u53ea\u652f\u6301 16bit PCM\u3002")
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        arr = arr.reshape(-1, channels).T
        return {"waveform": torch.from_numpy(arr).unsqueeze(0), "sample_rate": int(sample_rate)}


def _guess_audio_extension(url: str, requested_format: str = "", content_type: str = "") -> str:
    mime = str(content_type or "").split(";", 1)[0].strip().lower()
    mime_exts = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/vnd.wave": ".wav",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/aac": ".aac",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
    }
    if mime in mime_exts:
        return mime_exts[mime]
    mime_ext = mimetypes.guess_extension(mime)
    if mime_ext in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}:
        return mime_ext
    path_ext = os.path.splitext(str(url or "").split("?", 1)[0].lower())[1]
    if path_ext in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}:
        return path_ext
    requested = str(requested_format or "").strip().lower()
    return f".{requested}" if requested in {"mp3", "wav", "m4a", "aac", "ogg", "flac"} else ".mp3"


def _sniff_audio_extension(data: bytes) -> str:
    if not data:
        return ""
    head = data[:16]
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return ".wav"
    if head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return ".mp3"
    if head.startswith(b"OggS"):
        return ".ogg"
    if head.startswith(b"fLaC"):
        return ".flac"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return ".m4a"
    return ""


def download_audio_asset(audio_url: str, prefix: str = "CometAPIMusic", audio_format: str = "", timeout: int = 600) -> str:
    if not audio_url:
        raise CometAPIError("\u97f3\u4e50 API \u6ca1\u6709\u8fd4\u56de\u97f3\u9891\u5730\u5740")

    folder_paths = get_folder_paths()
    output_dir = folder_paths.get_output_directory()
    target_dir = os.path.join(output_dir, ASSET_SUBFOLDER)
    os.makedirs(target_dir, exist_ok=True)
    safe_prefix = _safe_asset_prefix(prefix, "CometAPIMusic")

    headers = {"User-Agent": "Mozilla/5.0"}
    last_error = None
    for _ in range(3):
        path = ""
        try:
            with requests.get(audio_url, headers=headers, stream=True, timeout=(20, timeout)) as response:
                response.raise_for_status()
                ext = _guess_audio_extension(audio_url, audio_format, response.headers.get("Content-Type", ""))
                chunks = response.iter_content(chunk_size=1024 * 1024)
                first_chunk = b""
                for chunk in chunks:
                    if chunk:
                        first_chunk = chunk
                        break
                sniffed_ext = _sniff_audio_extension(first_chunk)
                if sniffed_ext:
                    ext = sniffed_ext
                filename = f"{safe_prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
                path = os.path.join(target_dir, filename)
                with open(path, "wb") as handle:
                    if first_chunk:
                        handle.write(first_chunk)
                    for chunk in chunks:
                        if chunk:
                            handle.write(chunk)
            if os.path.exists(path) and os.path.getsize(path) > 256:
                return path
        except Exception as exc:
            last_error = exc
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass
            time.sleep(2)
    raise CometAPIError(f"\u97f3\u9891\u4e0b\u8f7d\u5931\u8d25\uff1a{last_error}")


def _match_audio_channels(waveform: torch.Tensor, channels: int) -> torch.Tensor:
    if waveform.ndim != 3:
        raise CometAPIError("\u97f3\u9891\u6570\u636e\u683c\u5f0f\u4e0d\u6b63\u786e\u3002")
    current = waveform.shape[1]
    if current == channels:
        return waveform
    if current == 1 and channels == 2:
        return waveform.repeat(1, 2, 1)
    if current > channels:
        return waveform[:, :channels, :]
    repeats = (channels + current - 1) // current
    return waveform.repeat(1, repeats, 1)[:, :channels, :]


def concat_audio_results(audios: list[dict], silence_seconds: float = 0.6) -> dict:
    valid = [audio for audio in audios if isinstance(audio, dict) and isinstance(audio.get("waveform"), torch.Tensor)]
    if not valid:
        return silent_audio()
    if len(valid) == 1:
        return valid[0]

    sample_rate = int(valid[0].get("sample_rate") or 44100)
    channels = max(1, int(valid[0]["waveform"].shape[1]))
    silence_len = max(1, int(sample_rate * silence_seconds))
    pieces = []
    for index, audio in enumerate(valid):
        waveform = audio["waveform"].detach().cpu().float()
        if int(audio.get("sample_rate") or sample_rate) != sample_rate:
            raise CometAPIError("\u5019\u9009\u97f3\u9891\u91c7\u6837\u7387\u4e0d\u4e00\u81f4\uff0c\u6682\u65f6\u65e0\u6cd5\u81ea\u52a8\u5408\u5e76\u3002")
        pieces.append(_match_audio_channels(waveform, channels))
        if index < len(valid) - 1:
            pieces.append(torch.zeros((1, channels, silence_len), dtype=waveform.dtype))
    return {"waveform": torch.cat(pieces, dim=2).clamp(-1.0, 1.0), "sample_rate": sample_rate}


GRSAI_IMAGE_REQUEST_TIMEOUT_SECONDS = 900


class GrsaiAPI:
    def __init__(self, api_key: str):
        if not api_key:
            raise CometAPIError("缺少 grsai API Key，请先在设置中心填写。")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "ComfyUI-CometAPI/0.1",
                "Authorization": f"Bearer {api_key}",
            }
        )

    def _make_request(self, method: str, endpoint: str, data: dict | None = None, timeout: int = GRSAI_IMAGE_REQUEST_TIMEOUT_SECONDS) -> dict:
        url = f"https://grsai.dakka.com.cn{endpoint}"
        response = self.session.request(method, url, json=data, timeout=timeout)
        response.raise_for_status()
        text = response.text.strip()
        try:
            return json.loads(text[6:].strip() if text.startswith("data: ") and "\n" not in text else text)
        except json.JSONDecodeError:
            pass

        last_valid_json = None
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            json_text = line[5:].strip()
            if not json_text or json_text == "[DONE]":
                continue
            try:
                last_valid_json = json.loads(json_text)
            except json.JSONDecodeError:
                continue
        if last_valid_json is not None:
            return last_valid_json
            raise CometAPIError(f"无法解析 API 响应：{text[:200]}")

    def nano_banana_generate_image(
        self,
        prompt: str,
        model: str,
        urls: list[str],
        aspect_ratio: str,
        image_size: str = "1K",
        include_error_detail: bool = False,
    ) -> tuple[list[Image.Image], list[str]]:
        payload = {
            "model": model,
            "prompt": prompt,
            "urls": urls,
            "shutProgress": True,
            "aspectRatio": aspect_ratio,
        }
        if model in PRO_SIZE_MODELS:
            payload["imageSize"] = image_size

        data = self._make_request("POST", "/v1/draw/nano-banana", data=payload, timeout=GRSAI_IMAGE_REQUEST_TIMEOUT_SECONDS)
        if data.get("status") != "succeeded":
            if include_error_detail:
                raise CometAPIError(f"图片生成失败：{combine_failure_reason_and_error(data)}")
            fail_reason = data.get("failure_reason", "")
            err_msg = data.get("error", "unknown error")
            raise CometAPIError(f"图片生成失败：[{fail_reason}] {err_msg}")

        urls = [item["url"] for item in data.get("results", []) if item.get("url")]
        if not urls:
            raise CometAPIError("API 没有返回图片地址")

        pil_images = []
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(urls))) as executor:
            future_to_url = {executor.submit(download_image, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    image = future.result()
                    if image is None:
                        errors.append(f"图片下载失败：{url}")
                    else:
                        pil_images.append(image)
                except Exception as exc:
                    errors.append(f"download exception: {url}: {exc}")
        return pil_images, errors

    def gpt_image_generate(
        self,
        prompt: str,
        model: str,
        urls: list[str],
        aspect_ratio: str,
        image_size: str = "2K",
        quality: str = "medium",
        auto_aspect_ratio: str = "1:1",
        include_error_detail: bool = False,
    ) -> tuple[list[Image.Image], list[str]]:
        payload = {
            "model": model,
            "prompt": prompt,
            "urls": urls,
            "shutProgress": True,
        }
        if model == "gpt-image-2-vip":
            if aspect_ratio == "auto":
                aspect_ratio = auto_aspect_ratio if auto_aspect_ratio in GPT_IMAGE_VIP_SIZE_MAP else "1:1"
            size = image_size if image_size in {"1K", "2K", "4K"} else "2K"
            mapped_size = GPT_IMAGE_VIP_SIZE_MAP.get(aspect_ratio, {}).get(size)
            if not mapped_size:
                raise CometAPIError(f"gpt-image-2-vip 不支持这个尺寸：{aspect_ratio} / {size}")
            payload["aspectRatio"] = mapped_size
            payload["quality"] = quality if quality in GPT_IMAGE_QUALITY_VALUES else "medium"
        else:
            payload["aspectRatio"] = aspect_ratio if aspect_ratio in GPT_IMAGE_ASPECT_RATIOS else "auto"

        data = self._make_request("POST", "/v1/draw/completions", data=payload, timeout=GRSAI_IMAGE_REQUEST_TIMEOUT_SECONDS)
        if data.get("code") and data.get("code") != 0:
            raise CometAPIError(f"API 业务错误：{data.get('msg')}（code: {data.get('code')}）")
        if data.get("status") == "failed":
            if include_error_detail:
                raise CometAPIError(f"图片生成失败：{combine_failure_reason_and_error(data)}")
            raise CometAPIError(f"图片生成失败：{data.get('failure_reason', '未知原因')}")

        results_info = []
        if isinstance(data.get("results"), list):
            results_info = data["results"]
        elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("results"), list):
            results_info = data["data"]["results"]
        elif data.get("url"):
            results_info = [{"url": data["url"]}]
        elif isinstance(data.get("data"), dict) and data["data"].get("url"):
            results_info = [{"url": data["data"]["url"]}]

        urls = [item["url"] for item in results_info if item.get("url")]
        if not urls:
            raise CometAPIError("API 没有返回图片地址")

        pil_images = []
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(urls))) as executor:
            future_to_url = {executor.submit(download_image, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    image = future.result()
                    if image is None:
                        errors.append(f"图片下载失败：{url}")
                    else:
                        pil_images.append(image)
                except Exception as exc:
                    errors.append(f"download exception: {url}: {exc}")
        return pil_images, errors


def nearest_aspect_ratio(aspect_ratio: str, pil_images: list[Image.Image], supported_ratios: list[str], fallback: str = "1:1") -> str:
    if aspect_ratio != "auto":
        return aspect_ratio if aspect_ratio in supported_ratios else fallback
    if not pil_images:
        return fallback
    try:
        width, height = pil_images[0].size
        if width <= 0 or height <= 0:
            return fallback
        image_ratio = width / height
        concrete_ratios = [ratio for ratio in supported_ratios if ratio != "auto"]
        return min(
            concrete_ratios,
            key=lambda ratio: abs((float(ratio.split(":")[0]) / float(ratio.split(":")[1])) - image_ratio),
        )
    except Exception:
        return fallback


def add_prompt_variation(prompt: str, subtask_idx: int) -> str:
    base = "".join(secrets.choice(ZERO_WIDTH_CHARS) for _ in range(secrets.randbelow(4) + 1))
    return f"{prompt}{base}{secrets.choice(ZERO_WIDTH_CHARS) * subtask_idx}"


def convert_prompt_asset_mentions(
    prompt: str,
    image_count: int = 0,
    video_count: int = 0,
    audio_count: int = 0,
) -> str:
    text = str(prompt or "")
    specs = [
        ("图片", "Image", max(0, int(image_count or 0))),
        ("图像", "Image", max(0, int(image_count or 0))),
        ("视频", "Video", max(0, int(video_count or 0))),
        ("音频", "Audio", max(0, int(audio_count or 0))),
    ]

    for cn_label, api_label, max_count in specs:
        if max_count <= 0:
            continue

        pattern = re.compile(rf"(?:@\s*)?{re.escape(cn_label)}\s*(\d+)")

        def replace(match):
            index = int(match.group(1))
            if index < 1 or index > max_count:
                return match.group(0)
            return f"@{api_label} {index}"

        text = pattern.sub(replace, text)
    return text


def pil_to_data_url(pil_image: Image.Image, image_format: str = "PNG") -> str:
    buffered = BytesIO()
    safe_pil_to_rgb(pil_image).save(buffered, format=image_format)
    return f"data:image/{image_format.lower()};base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


def pil_to_base64(pil_image: Image.Image, image_format: str = "PNG") -> str:
    buffered = BytesIO()
    safe_pil_to_rgb(pil_image).save(buffered, format=image_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decode_base64_image(value: str) -> Image.Image | None:
    if not value:
        return None
    raw = value.split(",", 1)[-1] if value.startswith("data:image") else value
    try:
        return safe_pil_to_rgb(Image.open(BytesIO(base64.b64decode(raw))))
    except Exception:
        return None


def modelverse_image_max_images(model: str) -> int:
    raw = str(model or "").strip()
    if raw in MODELVERSE_GPT_IMAGE_MODELS:
        return GPT_IMAGE_MAX_IMAGES
    if raw in MODELVERSE_SEEDREAM_MODELS:
        return NANO_BANANA_MAX_IMAGES
    return MODELVERSE_MAX_IMAGES


def apimart_image_max_images(model: str) -> int:
    return APIMART_GPT_IMAGE_MAX_IMAGES if str(model or "").strip() in APIMART_GPT_IMAGE_MODELS else APIMART_GEMINI_MAX_IMAGES


class PrivateAPI:
    def __init__(self, api_key: str, api_url: str = ""):
        if not api_key:
            raise CometAPIError("缺少私有渠道 API Key，请先在设置中心填写。")
        self.api_key = api_key
        self.host = normalize_api_base_url(api_url or "", "")

    def _post_json(self, url: str, headers: dict, payload: dict, timeout: int = 720) -> dict:
        res = requests.post(url, headers=headers, json=payload, timeout=(15, timeout))
        res.raise_for_status()
        return res.json()

    def _post_files(self, url: str, headers: dict, payload: dict, files: list, timeout: int = 720) -> dict:
        res = requests.post(url, headers=headers, data=payload, files=files, timeout=(15, timeout))
        res.raise_for_status()
        return res.json()

    def _parse_image_response(self, data: dict, provider_name: str) -> tuple[list[Image.Image], list[str]]:
        errors = []
        pil_images = []
        if not isinstance(data, dict):
            return [], [f"{provider_name} 返回的不是 JSON 响应"]

        if "candidates" in data:
            returned_text = ""
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    inline = part.get("inlineData") or part.get("inline_data")
                    if isinstance(inline, dict) and inline.get("data"):
                        image = decode_base64_image(inline["data"])
                        if image:
                            pil_images.append(image)
                    if part.get("text"):
                        returned_text += str(part["text"]) + " "
            if pil_images:
                return pil_images, []
            return [], [f"{provider_name} 没有返回图片" + (f"：{returned_text[:120]}" if returned_text.strip() else "")]

        items = data.get("data")
        if not isinstance(items, list):
            message = data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else data.get("message")
            return [], [f"{provider_name} API error: {message or str(data)[:200]}"]

        for item in items:
            if item.get("b64_json"):
                image = decode_base64_image(item["b64_json"])
                if image:
                    pil_images.append(image)
                else:
                    errors.append("b64_json 图片解码失败")
            elif item.get("url"):
                image = download_image(item["url"])
                if image:
                    pil_images.append(image)
                else:
                    errors.append(f"图片下载失败：{item['url']}")

        if not pil_images and not errors:
            errors.append(f"{provider_name} response contained no images")
        return pil_images, errors

    def gemini_generate(self, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str, image_size: str, subtask_idx: int) -> tuple[list[Image.Image], list[str]]:
        resolved_ratio = nearest_aspect_ratio(aspect_ratio, pil_images, PRIVATE_GEMINI_ASPECT_RATIOS)
        parts = [{"text": add_prompt_variation(prompt, subtask_idx)}]
        for pil_image in pil_images[:PRIVATE_MAX_IMAGES]:
            buffered = BytesIO()
            safe_pil_to_rgb(pil_image).save(buffered, format="JPEG", quality=90)
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(buffered.getvalue()).decode("utf-8"),
                    }
                }
            )

        image_config = {"aspectRatio": resolved_ratio}
        if "2.5" not in model.lower():
            image_config["imageSize"] = image_size if image_size in {"1K", "2K", "3K", "4K"} else "1K"

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": image_config},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = self._post_json(f"{self.host}/v1beta/models/{model}:generateContent?key={self.api_key}", headers, payload)
        return self._parse_image_response(data, "Gemini")

    def seedream_generate(self, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str, image_size: str, subtask_idx: int) -> tuple[list[Image.Image], list[str]]:
        resolved_ratio = nearest_aspect_ratio(aspect_ratio, pil_images, PRIVATE_SEEDREAM_ASPECT_RATIOS)
        valid_sizes = PRIVATE_SEEDREAM_SIZE_VALUES.get(model, {"2K"})
        image_size = image_size if image_size in valid_sizes else "2K"
        sizes_1k = {"1:1": "1024x1024", "4:3": "1152x864", "3:4": "864x1152", "16:9": "1280x720", "9:16": "720x1280", "3:2": "1248x832", "2:3": "832x1248", "21:9": "1512x648"}
        sizes_2k = {"1:1": "2048x2048", "4:3": "2304x1728", "3:4": "1728x2304", "16:9": "2848x1600", "9:16": "1600x2848", "3:2": "2496x1664", "2:3": "1664x2496", "21:9": "3136x1344"}
        sizes_3k = {"1:1": "3072x3072", "4:3": "3456x2592", "3:4": "2592x3456", "16:9": "4096x2304", "9:16": "2304x4096", "3:2": "3744x2496", "2:3": "2496x3744", "21:9": "4704x2016"}
        sizes_4k = {"1:1": "4096x4096", "4:3": "4704x3520", "3:4": "3520x4704", "16:9": "5504x3040", "9:16": "3040x5504", "3:2": "4992x3328", "2:3": "3328x4992", "21:9": "6240x2656"}

        model_lower = model.lower()
        if "seedream-5" in model_lower:
            mapped_size = sizes_3k.get(resolved_ratio, "3072x3072") if image_size == "3K" else sizes_2k.get(resolved_ratio, "2048x2048")
        elif "seedream-4-5" in model_lower:
            mapped_size = sizes_4k.get(resolved_ratio, "4096x4096") if image_size == "4K" else sizes_2k.get(resolved_ratio, "2048x2048")
        else:
            if image_size == "1K":
                mapped_size = sizes_1k.get(resolved_ratio, "1024x1024")
            elif image_size == "4K":
                mapped_size = sizes_4k.get(resolved_ratio, "4096x4096")
            else:
                mapped_size = sizes_2k.get(resolved_ratio, "2048x2048")

        payload = {
            "model": model,
            "prompt": add_prompt_variation(prompt, subtask_idx),
            "size": mapped_size,
            "watermark": False,
            "output_format": "png",
            "response_format": "b64_json",
        }
        image_data = [pil_to_data_url(image) for image in pil_images[:14]]
        if image_data:
            payload["image"] = image_data if len(image_data) > 1 else image_data[0]

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json"}
        data = self._post_json(f"{self.host}/v1/images/generations", headers, payload)
        return self._parse_image_response(data, "Seedream")

    def gpt_image_generate(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        image_size: str,
        quality: str,
        subtask_idx: int,
    ) -> tuple[list[Image.Image], list[str]]:
        resolved_ratio = nearest_aspect_ratio(aspect_ratio, pil_images, PRIVATE_GPT_IMAGE_ASPECT_RATIOS)
        size = image_size if image_size in {"1K", "2K", "4K"} else "2K"
        mapped_size = GPT_IMAGE_VIP_SIZE_MAP.get(resolved_ratio, {}).get(size)
        if not mapped_size:
            raise CometAPIError(f"私有渠道 gpt-image-2 不支持这个尺寸：{resolved_ratio} / {size}")

        final_prompt = add_prompt_variation(prompt, subtask_idx)
        safe_quality = quality if quality in GPT_IMAGE_QUALITY_VALUES else "medium"
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

        if pil_images:
            payload = {
                "model": model,
                "prompt": final_prompt,
                "n": "1",
                "size": mapped_size,
                "quality": safe_quality,
            }
            files = []
            for index, pil_image in enumerate(pil_images[:16], start=1):
                buffered = BytesIO()
                safe_pil_to_rgb(pil_image).save(buffered, format="PNG")
                files.append(("image", (f"image_{index}.png", buffered.getvalue(), "image/png")))
            data = self._post_files(f"{self.host}/v1/images/edits", headers, payload, files)
        else:
            payload = {
                "model": model,
                "prompt": final_prompt,
                "n": 1,
                "size": mapped_size,
                "quality": safe_quality,
                "format": "png",
            }
            json_headers = {**headers, "Content-Type": "application/json"}
            data = self._post_json(f"{self.host}/v1/images/generations", json_headers, payload)
        return self._parse_image_response(data, "Private gpt-image-2")

    def openai_compatible_generate(self, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str, subtask_idx: int) -> tuple[list[Image.Image], list[str]]:
        size_map = {"1:1": "1024x1024", "3:2": "1536x1024", "2:3": "1024x1536", "auto": "auto"}
        size = size_map.get(aspect_ratio, "auto")
        final_prompt = add_prompt_variation(prompt, subtask_idx)

        if pil_images:
            payload = {"model": model, "prompt": final_prompt, "size": size, "response_format": "b64_json"}
            files = []
            for index, pil_image in enumerate(pil_images[:5], start=1):
                buffered = BytesIO()
                safe_pil_to_rgb(pil_image).save(buffered, format="PNG")
                files.append(("image", (f"image_{index}.png", buffered.getvalue(), "image/png")))
            headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
            data = self._post_files(f"{self.host}/v1/images/edits", headers, payload, files)
        else:
            payload = {"model": model, "prompt": final_prompt, "size": size, "response_format": "b64_json"}
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json"}
            data = self._post_json(f"{self.host}/v1/images/generations", headers, payload)
        return self._parse_image_response(data, "OpenAI-compatible")

    def generate_image(self, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str, image_size: str, quality: str, subtask_idx: int) -> tuple[list[Image.Image], list[str]]:
        if model in PRIVATE_GEMINI_MODELS:
            return self.gemini_generate(prompt, model, pil_images, aspect_ratio, image_size, subtask_idx)
        if model in PRIVATE_SEEDREAM_MODELS:
            return self.seedream_generate(prompt, model, pil_images, aspect_ratio, image_size, subtask_idx)
        if model in PRIVATE_GPT_IMAGE_MODELS:
            return self.gpt_image_generate(prompt, model, pil_images, aspect_ratio, image_size, quality, subtask_idx)
        if model in PRIVATE_OPENAI_COMPAT_MODELS:
            return self.openai_compatible_generate(prompt, model, pil_images, aspect_ratio, subtask_idx)
        raise CometAPIError(f"不支持的私有模型：{model}")


class ModelVerseImageAPI:
    host = "https://api.modelverse.cn"

    def __init__(self, api_key: str):
        if not api_key:
            raise CometAPIError("缺少优云智算 API Key，请先在设置中心填写。")
        self.api_key = api_key

    def _raise_for_status(self, response: requests.Response, label: str) -> None:
        if response.status_code < 400:
            return
        text = (response.text or "").strip()
        detail = f"：{text[:500]}" if text else ""
        raise CometAPIError(f"{label} 请求失败：HTTP {response.status_code} {response.reason}{detail}")

    def _post_json(self, path: str, payload: dict, timeout: int = 720) -> dict:
        response = requests.post(
            f"{self.host}{path}",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json"},
            json=payload,
            timeout=(15, timeout),
        )
        self._raise_for_status(response, "优云智算")
        return response.json()

    def _post_gemini_json(self, model: str, payload: dict, timeout: int = 720) -> dict:
        response = requests.post(
            f"{self.host}/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json", "Accept": "application/json"},
            json=payload,
            timeout=(15, timeout),
        )
        self._raise_for_status(response, "优云 Gemini")
        return response.json()

    def _post_files(self, path: str, payload: dict, files: list, timeout: int = 720) -> dict:
        response = requests.post(
            f"{self.host}{path}",
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
            data=payload,
            files=files,
            timeout=(15, timeout),
        )
        self._raise_for_status(response, "优云智算")
        return response.json()

    def _parse_image_response(self, data: dict, provider_name: str) -> tuple[list[Image.Image], list[str]]:
        errors = []
        pil_images = []
        if not isinstance(data, dict):
            return [], [f"{provider_name} 返回的不是 JSON 响应"]

        if "candidates" in data:
            returned_text = ""
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if part.get("thought"):
                        continue
                    inline = part.get("inlineData") or part.get("inline_data")
                    if isinstance(inline, dict) and inline.get("data"):
                        image = decode_base64_image(inline["data"])
                        if image:
                            pil_images.append(image)
                    if part.get("text"):
                        returned_text += str(part["text"]) + " "
            if pil_images:
                return pil_images, []
            return [], [f"{provider_name} 没有返回图片" + (f"：{returned_text[:120]}" if returned_text.strip() else "")]

        items = data.get("data")
        if not isinstance(items, list):
            message = data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else data.get("message")
            return [], [f"{provider_name} API error: {message or str(data)[:200]}"]

        for item in items:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("error"), dict):
                errors.append(str(item["error"].get("message") or item["error"]))
                continue
            if item.get("b64_json"):
                image = decode_base64_image(item["b64_json"])
                if image:
                    pil_images.append(image)
                else:
                    errors.append("b64_json 图片解码失败")
            elif item.get("url"):
                image = download_image(item["url"])
                if image:
                    pil_images.append(image)
                else:
                    errors.append(f"图片下载失败：{item['url']}")

        if not pil_images and not errors:
            errors.append(f"{provider_name} response contained no images")
        return pil_images, errors

    def _size_for_ratio(self, aspect_ratio: str, image_size: str, fallback: str = "2048x2048") -> str:
        safe_size = image_size if image_size in {"1K", "2K", "4K"} else "2K"
        if safe_size == "1K":
            return {
                "1:1": "1024x1024", "4:3": "1152x864", "3:4": "864x1152",
                "16:9": "1280x720", "9:16": "720x1280", "3:2": "1248x832",
                "2:3": "832x1248", "21:9": "1512x648",
            }.get(aspect_ratio, "1024x1024")
        if safe_size == "4K":
            return {
                "1:1": "4096x4096", "4:3": "4704x3520", "3:4": "3520x4704",
                "16:9": "5504x3040", "9:16": "3040x5504", "3:2": "4992x3328",
                "2:3": "3328x4992", "21:9": "6240x2656",
            }.get(aspect_ratio, "4096x4096")
        return {
            "1:1": "2048x2048", "4:3": "2304x1728", "3:4": "1728x2304",
            "16:9": "2848x1600", "9:16": "1600x2848", "3:2": "2496x1664",
            "2:3": "1664x2496", "21:9": "3136x1344",
        }.get(aspect_ratio, fallback)

    def gemini_generate(self, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str, image_size: str, subtask_idx: int) -> tuple[list[Image.Image], list[str]]:
        resolved_ratio = nearest_aspect_ratio(aspect_ratio, pil_images, MODELVERSE_GEMINI_ASPECT_RATIOS)
        parts = [{"text": add_prompt_variation(prompt, subtask_idx)}]
        for pil_image in pil_images[:modelverse_image_max_images(model)]:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": pil_to_base64(pil_image, "JPEG"),
                    }
                }
            )

        generation_config = {"responseModalities": ["TEXT", "IMAGE"]}
        image_config = {"aspectRatio": resolved_ratio}
        image_config["imageSize"] = image_size if image_size in {"1K", "2K", "4K"} else "1K"
        generation_config["imageConfig"] = image_config

        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation_config}
        data = self._post_gemini_json(model, payload)
        return self._parse_image_response(data, "优云 Gemini")

    def gpt_image_generate(self, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str, image_size: str, quality: str, subtask_idx: int) -> tuple[list[Image.Image], list[str]]:
        resolved_ratio = nearest_aspect_ratio(aspect_ratio, pil_images, GPT_IMAGE_ASPECT_RATIOS, "1:1")
        safe_size = image_size if image_size in {"1K", "2K", "4K"} else "2K"
        mapped_size = GPT_IMAGE_VIP_SIZE_MAP.get(resolved_ratio, {}).get(safe_size, "2048x2048")
        safe_quality = quality if quality in GPT_IMAGE_QUALITY_VALUES else "medium"
        final_prompt = add_prompt_variation(prompt, subtask_idx)

        if pil_images:
            payload = {
                "model": model,
                "prompt": final_prompt,
                "n": "1",
                "size": mapped_size,
                "quality": safe_quality,
                "output_format": "png",
                "output_compression": "100",
            }
            files = []
            for index, pil_image in enumerate(pil_images[:GPT_IMAGE_MAX_IMAGES], start=1):
                buffered = BytesIO()
                safe_pil_to_rgb(pil_image).save(buffered, format="PNG")
                files.append(("image", (f"image_{index}.png", buffered.getvalue(), "image/png")))
            data = self._post_files("/v1/images/edits", payload, files)
        else:
            payload = {
                "model": model,
                "prompt": final_prompt,
                "n": 1,
                "size": mapped_size,
                "quality": safe_quality,
                "output_format": "png",
                "output_compression": 100,
            }
            data = self._post_json("/v1/images/generations", payload)
        return self._parse_image_response(data, "优云 gpt-image-2")

    def seedream_generate(self, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str, image_size: str, subtask_idx: int) -> tuple[list[Image.Image], list[str]]:
        resolved_ratio = nearest_aspect_ratio(aspect_ratio, pil_images, MODELVERSE_SEEDREAM_ASPECT_RATIOS)
        valid_sizes = MODELVERSE_SEEDREAM_SIZE_VALUES.get(model, {"2K", "4K"})
        safe_size = image_size if image_size in valid_sizes else "2K"
        payload = {
            "model": model,
            "prompt": add_prompt_variation(prompt, subtask_idx),
            "images": [pil_to_data_url(image) for image in pil_images[:NANO_BANANA_MAX_IMAGES]],
            "size": self._size_for_ratio(resolved_ratio, safe_size),
            "watermark": False,
            "response_format": "b64_json",
            "output_format": "png",
        }
        if not payload["images"]:
            payload.pop("images", None)
        data = self._post_json("/v1/images/generations", payload)
        return self._parse_image_response(data, "优云 Seedream")

    def generate_image(self, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str, image_size: str, quality: str, subtask_idx: int) -> tuple[list[Image.Image], list[str]]:
        if model in MODELVERSE_GEMINI_MODELS:
            return self.gemini_generate(prompt, model, pil_images, aspect_ratio, image_size, subtask_idx)
        if model in MODELVERSE_GPT_IMAGE_MODELS:
            return self.gpt_image_generate(prompt, model, pil_images, aspect_ratio, image_size, quality, subtask_idx)
        if model in MODELVERSE_SEEDREAM_MODELS:
            return self.seedream_generate(prompt, model, pil_images, aspect_ratio, image_size, subtask_idx)
        raise CometAPIError(f"不支持的优云智算图片模型：{model}")


class APIMartImageAPI:
    host = "https://api.apimart.ai"

    def __init__(self, api_key: str):
        if not api_key:
            raise CometAPIError("缺少 Apimart API Key，请先在设置中心填写。")
        self.api_key = api_key

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _raise_for_status(self, response: requests.Response, label: str = "Apimart") -> None:
        if response.status_code < 400:
            return
        text = (response.text or "").strip()
        message = ""
        try:
            payload = response.json()
            if isinstance(payload.get("error"), dict):
                message = str(payload["error"].get("message") or "")
            message = message or str(payload.get("message") or "")
        except Exception:
            pass
        detail = message or text[:500]
        suffix = f"：{detail}" if detail else ""
        raise CometAPIError(f"{label} 请求失败：HTTP {response.status_code} {response.reason}{suffix}")

    def _post_json(self, path: str, payload: dict, timeout: int = 60) -> dict:
        response = requests.post(
            f"{self.host}{path}",
            headers=self._headers(),
            json=payload,
            timeout=(15, timeout),
        )
        self._raise_for_status(response)
        return response.json()

    def _get_json(self, path: str, timeout: int = 30) -> dict:
        response = requests.get(
            f"{self.host}{path}",
            headers=self._headers(),
            timeout=(15, timeout),
        )
        self._raise_for_status(response)
        return response.json()

    def _submit_task(self, payload: dict) -> str:
        data = self._post_json("/v1/images/generations", payload)
        items = data.get("data") if isinstance(data, dict) else None
        if isinstance(items, dict):
            items = [items]
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and (item.get("task_id") or item.get("taskId") or item.get("id")):
                    return str(item.get("task_id") or item.get("taskId") or item.get("id"))
        raise CometAPIError(f"Apimart API 没有返回图片任务 ID：{str(data)[:300]}")

    def _query_task(self, task_id: str) -> tuple[str, list[str], dict]:
        data = self._get_json(f"/v1/tasks/{task_id}")
        body = data.get("data") if isinstance(data, dict) and isinstance(data.get("data"), dict) else {}
        status = str(body.get("status") or data.get("status") or "submitted").strip().lower()
        result = body.get("result") if isinstance(body.get("result"), dict) else {}
        image_urls = []
        for item in result.get("images") or []:
            if not isinstance(item, dict):
                continue
            urls = item.get("url") or item.get("urls")
            if isinstance(urls, str):
                image_urls.append(urls)
            elif isinstance(urls, list):
                image_urls.extend(str(url) for url in urls if str(url or "").strip())
        return status, image_urls, body or data

    def _wait_for_image_urls(self, task_id: str, timeout: int = 900, interval: int = 5) -> list[str]:
        started = time.time()
        last_status = "submitted"
        while time.time() - started < timeout:
            time.sleep(interval)
            status, image_urls, data = self._query_task(task_id)
            last_status = status
            if status in {"completed", "success", "succeeded"}:
                if image_urls:
                    return image_urls
                raise CometAPIError(f"Apimart 任务已完成，但没有返回图片地址：{str(data)[:300]}")
            if status in {"failed", "failure", "error", "canceled", "cancelled"}:
                error = data.get("error") if isinstance(data, dict) else None
                if isinstance(error, dict):
                    error = error.get("message") or error
                message = error or (data.get("message") if isinstance(data, dict) else "") or data
                raise CometAPIError(f"Apimart 图片任务失败：{str(message)[:300]}")
        raise CometAPIError(f"Apimart 图片任务等待超时（{timeout} 秒），最后状态：{last_status}")

    def generate_image(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        image_size: str,
        quality: str,
        subtask_idx: int,
    ) -> tuple[list[Image.Image], list[str]]:
        if model not in APIMART_IMAGE_MODELS:
            raise CometAPIError(f"不支持的 Apimart 图片模型：{model}")

        is_gpt = model in APIMART_GPT_IMAGE_MODELS
        if is_gpt:
            aspect_choices = APIMART_GPT_IMAGE_ASPECT_RATIOS
        elif model in APIMART_GEMINI_31_IMAGE_MODELS:
            aspect_choices = APIMART_GEMINI_31_ASPECT_RATIOS
        else:
            aspect_choices = APIMART_GEMINI_ASPECT_RATIOS
        resolved_ratio = nearest_aspect_ratio(aspect_ratio, pil_images, aspect_choices, "1:1")
        safe_size = image_size if image_size in {"1K", "2K", "4K"} else "2K"
        if model in APIMART_GEMINI_25_IMAGE_MODELS:
            safe_size = "1K"

        payload = {
            "model": model,
            "prompt": add_prompt_variation(prompt, subtask_idx),
            "size": resolved_ratio,
            "resolution": safe_size.lower() if is_gpt else safe_size,
            "n": 1,
        }
        image_urls = [pil_to_data_url(image) for image in pil_images[: apimart_image_max_images(model)]]
        if image_urls:
            payload["image_urls"] = image_urls
        if model == "gpt-image-2-official":
            payload["quality"] = quality if quality in GPT_IMAGE_QUALITY_VALUES else "medium"

        task_id = self._submit_task(payload)
        urls = self._wait_for_image_urls(task_id)
        pil_results = []
        errors = []
        for url in urls:
            image = download_image(url, timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS)
            if image:
                pil_results.append(safe_pil_to_rgb(image))
            else:
                errors.append(f"图片下载失败：{url}")
        return pil_results, errors


def is_sora_video_model(model: str) -> bool:
    return model in PRIVATE_SORA_VIDEO_MODELS or model.startswith("sora-")


def is_grok_video_model(model: str) -> bool:
    return model in PRIVATE_GROK_VIDEO_MODELS or model.startswith("grok-video-") or model.startswith("grok-videos") or model.startswith("grok-imagine-video")


def is_veo_video_model(model: str) -> bool:
    raw = str(model or "")
    return raw in PRIVATE_VEO_VIDEO_MODELS or raw.startswith("veo3.") or raw.startswith("veo_") or raw.startswith("omni-flash")


def is_vidu_video_model(model: str) -> bool:
    return str(model or "").lower() in PRIVATE_VIDU_VIDEO_MODELS


def is_hailuo_video_model(model: str) -> bool:
    return str(model or "") in PRIVATE_HAILUO_VIDEO_MODELS


def is_kling_video_model(model: str) -> bool:
    return str(model or "").lower() in PRIVATE_KLING_VIDEO_MODELS


def is_private_happyhorse_video_model(model: str) -> bool:
    return str(model or "").lower() in {value.lower() for value in PRIVATE_HAPPYHORSE_VIDEO_MODELS}


def normalize_private_happyhorse_mode(mode: str) -> str:
    raw = str(mode or "").strip()
    return raw if raw in PRIVATE_HAPPYHORSE_VIDEO_MODES else PRIVATE_HAPPYHORSE_VIDEO_MODES[0]


def normalize_private_happyhorse_resolution(resolution: str) -> str:
    raw = str(resolution or "").strip().upper()
    return raw if raw in PRIVATE_HAPPYHORSE_RESOLUTIONS else "1080P"


def normalize_private_happyhorse_aspect(aspect_ratio: str, mode: str) -> str:
    safe_mode = normalize_private_happyhorse_mode(mode)
    if safe_mode in {"图生", "视频编辑"}:
        return ""
    raw = str(aspect_ratio or "").strip()
    return raw if raw in PRIVATE_HAPPYHORSE_ASPECT_RATIOS else "16:9"


def normalize_private_happyhorse_duration(duration: str) -> int:
    raw = str(duration or "").strip()
    if raw in PRIVATE_HAPPYHORSE_DURATIONS:
        try:
            return int(raw)
        except ValueError:
            return 5
    return 5


def is_kling_omni_video_model(model: str) -> bool:
    return str(model or "").lower() in PRIVATE_KLING_OMNI_VIDEO_MODELS


def video_max_images_for_model(model: str, mode: str = "") -> int:
    raw = canonical_runninghub_video_model(model).lower()
    if raw in RUNNINGHUB_GOOGLE_MODELS:
        safe_mode = normalize_runninghub_mode(model, mode)
        return 3 if safe_mode in {"多参", "视频编辑"} else 0
    if raw in RUNNINGHUB_GROK_MODELS:
        return RUNNINGHUB_GROK_MAX_IMAGES if normalize_runninghub_mode(model, mode) == "图生" else 0
    if raw in RUNNINGHUB_SEEDANCE_MODELS or raw in RUNNINGHUB_HAPPYHORSE_MODELS:
        return RUNNINGHUB_MAX_IMAGES
    if raw.startswith("omni-flash"):
        return PRIVATE_VIDEO_MAX_IMAGES if raw.endswith("components") or str(mode or "") == "多参" else 0
    if is_private_happyhorse_video_model(raw):
        return PRIVATE_HAPPYHORSE_MAX_REFERENCE_IMAGES
    return PRIVATE_VIDEO_MAX_IMAGES


def video_allowed_media_types(channel: str, model: str, mode: str) -> set[str]:
    channel_key = str(channel or "").lower()
    raw_model = canonical_runninghub_video_model(model).lower() if channel_key == "runninghub" else str(model or "").lower()

    if channel_key == "apimart":
        return apimart_allowed_media_types(raw_model, mode)

    if channel_key == "modelverse":
        safe_mode = normalize_modelverse_video_mode(raw_model, mode)
        if raw_model in MODELVERSE_SEEDANCE_MODELS:
            if safe_mode == "文生":
                return set()
            if safe_mode == "全能参考":
                return {"image", "video", "audio"}
            return {"image"}
        if raw_model in MODELVERSE_HAPPYHORSE_MODELS:
            if safe_mode == "文生":
                return set()
            if safe_mode == "视频编辑":
                return {"image", "video"}
            return {"image"}
        if raw_model in MODELVERSE_SORA_VIDEO_MODELS:
            return set() if safe_mode == "文生" else {"image"}
        if raw_model in MODELVERSE_KLING_VIDEO_MODELS:
            if safe_mode == "文生":
                return set()
            if safe_mode in {"运动控制", "Omni"}:
                return {"image", "video"}
            return {"image"}
        return {"image"}

    if channel_key == "runninghub":
        safe_mode = normalize_runninghub_mode(raw_model, mode)
        if is_runninghub_grok_model(raw_model):
            return {"image"} if safe_mode == "图生" else set()
        if is_runninghub_google_model(raw_model):
            if safe_mode == "文生":
                return set()
            if safe_mode == "视频编辑":
                return {"image", "video"}
            return {"image"}
        if is_runninghub_seedance_model(raw_model):
            if safe_mode == "文生":
                return set()
            if safe_mode == "全能参考":
                return {"image", "video", "audio"}
            return {"image"}
        if is_runninghub_happyhorse_model(raw_model):
            if safe_mode == "文生":
                return set()
            if safe_mode == "视频编辑":
                return {"image", "video"}
            return {"image"}
        return set()

    if is_vidu_video_model(raw_model):
        return set() if normalize_vidu_video_mode(raw_model, mode) == "文生" else {"image"}
    if is_hailuo_video_model(model):
        return set() if normalize_hailuo_video_mode(mode) == "文生" else {"image"}
    if is_kling_video_model(raw_model):
        return set() if normalize_kling_video_mode(raw_model, mode) == "文生" else {"image"}
    if is_private_happyhorse_video_model(raw_model):
        safe_mode = normalize_private_happyhorse_mode(mode)
        if safe_mode == "文生":
            return set()
        if safe_mode == "视频编辑":
            return {"image", "video"}
        return {"image"}
    if raw_model.startswith("omni-flash"):
        return {"image"} if raw_model.endswith("components") or str(mode or "") == "多参" else set()
    if is_grok_video_model(raw_model) or is_veo_video_model(raw_model) or is_sora_video_model(raw_model):
        return {"image"}
    return {"image"}


def resolve_grok_video_model(model: str, duration: str) -> str:
    if model in {"grok-videos-10s", "grok-videos-15s", "grok-video-3-10s", "grok-video-3-15s"}:
        return model
    base_model = "grok-videos" if str(model or "").startswith("grok-videos") else "grok-video-3"
    if str(duration) == "10":
        return f"{base_model}-10s"
    if str(duration) == "15":
        return f"{base_model}-15s"
    return base_model


def normalize_veo_video_base(model: str) -> str:
    raw = str(model or "")
    if raw.startswith("veo3.1-lite"):
        return "veo3.1-lite"
    return "veo3.1-fast" if raw.startswith("veo3.1-fast") else "veo3.1"


def normalize_veo_video_resolution(resolution: str) -> str:
    return "4K" if str(resolution or "").upper() == "4K" else "1080P"


def normalize_veo_video_mode(mode: str) -> str:
    return "多参" if str(mode or "") == "多参" else "首尾帧"


def resolve_veo_video_model(model: str, resolution: str = "1080P", mode: str = "首尾帧") -> str:
    raw = str(model or "")
    if raw == "omni-flash-components":
        return raw
    if raw == "omni-flash":
        return "omni-flash-components" if str(mode or "") == "多参" else "omni-flash"
    if raw in PRIVATE_VEO_VIDEO_REAL_MODELS and ("-4K" in raw or "components" in raw):
        return raw

    base = normalize_veo_video_base(raw)
    safe_resolution = normalize_veo_video_resolution(resolution)
    safe_mode = normalize_veo_video_mode(mode)
    if base == "veo3.1-lite":
        return "veo3.1-lite-4K" if safe_resolution == "4K" else "veo3.1-lite"
    if base == "veo3.1-fast":
        if safe_mode == "多参":
            return "veo3.1-fast-components-4K"
        return "veo3.1-fast-4K" if safe_resolution == "4K" else "veo3.1-fast"
    # veo3.1 (非 fast、非 lite)
    if safe_mode == "多参":
        return "veo3.1-components-4k"
    return "veo3.1-4k" if safe_resolution == "4K" else "veo3.1"


def vidu_mode_choices_for_model(model: str) -> list[str]:
    raw = str(model or "").lower()
    if raw in {"viduq3", "viduq3-mix"}:
        return ["参考"]
    if raw == "viduq3-pro":
        return ["图生", "首尾帧", "文生"]
    if raw == "viduq3-turbo":
        return ["参考", "图生", "首尾帧", "文生"]
    if raw == "viduq2-pro":
        return ["参考", "图生", "首尾帧"]
    if raw == "viduq2-turbo":
        return ["图生", "首尾帧"]
    if raw == "viduq2":
        return ["参考", "图生", "首尾帧", "文生"]
    return list(PRIVATE_VIDU_VIDEO_MODES)


def normalize_vidu_video_mode(model: str, mode: str) -> str:
    choices = vidu_mode_choices_for_model(model)
    return str(mode or "") if str(mode or "") in choices else choices[0]


def vidu_supports_audio_output(model: str, mode: str) -> bool:
    raw = str(model or "").lower()
    safe_mode = normalize_vidu_video_mode(raw, mode)
    q3_reference_models = {"viduq3", "viduq3-turbo", "viduq3-mix"}
    q3_generation_models = {"viduq3-pro", "viduq3-pro-fast", "viduq3-turbo"}
    image_audio_models = q3_generation_models | {
        "viduq2",
        "viduq2-pro",
        "viduq2-pro-fast",
        "viduq2-turbo",
        "viduq1",
        "viduq1-classic",
        "vidu2.0",
    }

    if safe_mode == "参考":
        return raw in q3_reference_models
    if safe_mode == "图生":
        return raw in image_audio_models
    if safe_mode in {"文生", "首尾帧"}:
        return raw in q3_generation_models
    return False


def vidu_supports_audio_type(model: str, mode: str) -> bool:
    raw = str(model or "").lower()
    safe_mode = normalize_vidu_video_mode(raw, mode)
    return safe_mode == "图生" and raw in {
        "viduq2",
        "viduq2-pro",
        "viduq2-pro-fast",
        "viduq2-turbo",
        "viduq1",
        "viduq1-classic",
        "vidu2.0",
    }


def vidu_duration_choices(model: str, mode: str) -> list[int]:
    raw = str(model or "").lower()
    safe_mode = normalize_vidu_video_mode(raw, mode)
    if raw == "viduq3" or (raw == "viduq3-turbo" and safe_mode == "参考"):
        return list(range(3, 17))
    if raw.startswith("viduq3"):
        return list(range(1, 17))
    if safe_mode == "首尾帧" and raw in {"viduq2-pro", "viduq2-turbo"}:
        return list(range(1, 9))
    if raw.startswith("viduq2"):
        return list(range(1, 11))
    return [5]


def normalize_vidu_duration(model: str, mode: str, duration: str) -> int:
    choices = vidu_duration_choices(model, mode)
    try:
        value = int(float(str(duration)))
    except Exception:
        value = 5
    if value in choices:
        return value
    return 5 if 5 in choices else choices[0]


def vidu_resolution_choices(model: str, mode: str) -> list[str]:
    raw = str(model or "").lower()
    if raw == "viduq3-mix":
        return ["720p", "1080p"]
    return list(PRIVATE_VIDU_VIDEO_RESOLUTIONS)


def normalize_vidu_resolution(model: str, mode: str, resolution: str) -> str:
    choices = vidu_resolution_choices(model, mode)
    raw = str(resolution or "").strip()
    lowered = raw.lower()
    for choice in choices:
        if lowered == choice.lower():
            return choice
    return "720p" if "720p" in choices else choices[0]


def normalize_vidu_aspect_ratio(model: str, mode: str, aspect_ratio: str) -> str | None:
    safe_mode = normalize_vidu_video_mode(model, mode)
    if safe_mode in {"图生", "首尾帧"}:
        return None
    choices = list(PRIVATE_VIDU_VIDEO_ASPECT_RATIOS)
    if safe_mode == "文生":
        choices = [value for value in choices if value != "auto"]
    raw = str(aspect_ratio or "").strip()
    return raw if raw in choices else ("auto" if "auto" in choices else "16:9")


def vidu_endpoint_for_mode(mode: str) -> str:
    return {
        "文生": "/ent/v2/text2video",
        "图生": "/ent/v2/img2video",
        "首尾帧": "/ent/v2/start-end2video",
        "参考": "/ent/v2/reference2video",
    }.get(str(mode or ""), "/ent/v2/reference2video")


def vidu_required_images_for_mode(mode: str) -> int:
    return {"文生": 0, "图生": 1, "首尾帧": 2, "参考": 1}.get(str(mode or ""), 1)


def vidu_max_images_for_mode(mode: str) -> int:
    return {"文生": 0, "图生": 1, "首尾帧": 2, "参考": MAX_VIDEO_IMAGE_INPUTS}.get(str(mode or ""), MAX_VIDEO_IMAGE_INPUTS)


def normalize_hailuo_video_mode(mode: str) -> str:
    raw = str(mode or "")
    return raw if raw in PRIVATE_HAILUO_VIDEO_MODES else "文生"


def normalize_hailuo_duration(duration: str) -> str:
    raw = str(duration or "").strip()
    return raw if raw in PRIVATE_HAILUO_VIDEO_DURATIONS else "6"


def normalize_hailuo_resolution(duration: str, resolution: str) -> str:
    if normalize_hailuo_duration(duration) == "10":
        return "768P"
    raw = str(resolution or "").strip().upper()
    return raw if raw in PRIVATE_HAILUO_VIDEO_RESOLUTIONS else "768P"


def normalize_kling_video_mode(model: str, mode: str) -> str:
    choices = PRIVATE_KLING_OMNI_VIDEO_MODES if is_kling_omni_video_model(model) else PRIVATE_KLING_VIDEO_MODES
    raw = str(mode or "")
    return raw if raw in choices else choices[0]


def normalize_kling_quality_mode(value: str) -> str:
    return "pro" if str(value or "").lower() == "pro" else "std"


def kling_duration_choices(model: str) -> list[str]:
    raw = str(model or "").lower()
    if raw in {"kling-v3", "kling-v3-omni"}:
        return [str(value) for value in range(3, 16)]
    if raw == "kling-video-o1":
        return [str(value) for value in range(3, 11)]
    return ["5", "10"]


def normalize_kling_duration(model: str, duration: str) -> str:
    choices = kling_duration_choices(model)
    raw = str(duration or "").strip()
    return raw if raw in choices else ("5" if "5" in choices else choices[0])


def normalize_kling_aspect_ratio(aspect_ratio: str) -> str:
    raw = str(aspect_ratio or "").strip()
    return raw if raw in PRIVATE_KLING_VIDEO_ASPECT_RATIOS else "16:9"


def private_kling_supports_generated_sound(model: str) -> bool:
    return str(model or "").lower() in {"kling-v3", "kling-v3-omni"}


def modelverse_kling_supports_generated_sound(model: str) -> bool:
    return str(model or "").lower() in {"kling-v3", "kling-v3-omni"}


def kling_action_for_mode(mode: str) -> str:
    return {
        "文生": "text2video",
        "图生": "image2video",
        "首尾帧": "image2video",
        "多图": "multi-image2video",
        "Omni": "omni-video",
    }.get(str(mode or ""), "text2video")


def canonical_modelverse_video_model(model: str) -> str:
    raw = str(model or "").strip()
    if not raw:
        return raw
    normalized = raw.lower()
    for model_id in MODELVERSE_VIDEO_MODELS:
        if normalized == model_id.lower():
            return model_id
    return MODELVERSE_VIDEO_MODEL_ALIAS_TO_ID.get(normalized, raw)


def modelverse_video_mode_choices(model: str) -> list[str]:
    raw = canonical_modelverse_video_model(model).lower()
    if raw in MODELVERSE_SEEDANCE_MODELS:
        return list(MODELVERSE_SEEDANCE_MODES)
    if raw in MODELVERSE_HAPPYHORSE_MODELS:
        return list(MODELVERSE_HAPPYHORSE_MODES)
    if raw in MODELVERSE_SORA_VIDEO_MODELS:
        return list(MODELVERSE_SORA_VIDEO_MODES)
    if raw in MODELVERSE_KLING_OMNI_VIDEO_MODELS:
        return list(MODELVERSE_KLING_OMNI_VIDEO_MODES)
    if raw in MODELVERSE_KLING_VIDEO_MODELS:
        return list(MODELVERSE_KLING_VIDEO_MODES)
    return ["文生"]


def normalize_modelverse_video_mode(model: str, mode: str) -> str:
    choices = modelverse_video_mode_choices(model)
    raw = str(mode or "")
    return raw if raw in choices else choices[0]


def modelverse_video_max_images(model: str) -> int:
    raw = canonical_modelverse_video_model(model).lower()
    if raw in MODELVERSE_SORA_VIDEO_MODELS:
        return 1
    if raw == "kling-v2-6":
        return 2
    if raw == "kling-v3":
        return 2
    return MODELVERSE_VIDEO_MAX_IMAGES


def normalize_modelverse_duration(model: str, duration: str, mode: str = "") -> str:
    raw = canonical_modelverse_video_model(model).lower()
    safe_mode = normalize_modelverse_video_mode(raw, mode)
    if raw in MODELVERSE_SEEDANCE_MODELS:
        choices = MODELVERSE_SEEDANCE_DURATIONS
    elif raw in MODELVERSE_HAPPYHORSE_MODELS:
        choices = MODELVERSE_HAPPYHORSE_DURATIONS
    elif raw in MODELVERSE_SORA_VIDEO_MODELS:
        choices = MODELVERSE_SORA_VIDEO_DURATIONS
    elif raw == "kling-v2-6":
        choices = MODELVERSE_KLING_V26_DURATIONS
    elif raw == "kling-video-o1":
        choices = MODELVERSE_KLING_O1_DURATIONS
    elif raw in {"kling-v3", "kling-v3-omni"}:
        choices = ["5", "10"] if safe_mode == "运动控制" else MODELVERSE_KLING_V3_DURATIONS
    else:
        choices = ["5"]
    raw_duration = str(duration or "").strip()
    return raw_duration if raw_duration in choices else ("5" if "5" in choices else choices[0])


def normalize_modelverse_resolution(model: str, resolution: str) -> str:
    raw = canonical_modelverse_video_model(model).lower()
    value = str(resolution or "").strip()
    if raw in MODELVERSE_SEEDANCE_MODELS:
        lowered = value.lower()
        return lowered if lowered in MODELVERSE_SEEDANCE_RESOLUTIONS else "720p"
    if raw in MODELVERSE_HAPPYHORSE_MODELS:
        upper = value.upper()
        return upper if upper in MODELVERSE_HAPPYHORSE_RESOLUTIONS else "1080P"
    if raw in {"kling-v2-6", "kling-video-o1"}:
        return "pro"
    if raw in MODELVERSE_KLING_VIDEO_MODELS:
        lowered = value.lower()
        return "pro" if lowered == "pro" else "std"
    return value


def normalize_modelverse_aspect_ratio(model: str, mode: str, aspect_ratio: str) -> str:
    raw = canonical_modelverse_video_model(model).lower()
    safe_mode = normalize_modelverse_video_mode(raw, mode)
    if raw in MODELVERSE_SEEDANCE_MODELS:
        choices = MODELVERSE_SEEDANCE_ASPECT_RATIOS
        value = str(aspect_ratio or "").strip()
        return value if value in choices else "adaptive"
    if raw in MODELVERSE_SORA_VIDEO_MODELS:
        choices = ["16:9", "9:16"]
        value = str(aspect_ratio or "").strip()
        return value if value in choices else "16:9"
    choices = MODELVERSE_VIDEO_ASPECT_RATIOS
    value = str(aspect_ratio or "").strip()
    if raw in MODELVERSE_HAPPYHORSE_MODELS and safe_mode in {"图生", "视频编辑"}:
        return "16:9"
    return value if value in choices else "16:9"


class PrivateVideoAPI:
    def __init__(self, api_key: str, api_url: str = "", media_upload_api_key: str = "", grsai_media_upload_api_key: str = ""):
        if not api_key:
            raise CometAPIError("缺少私有渠道 API Key，请先在设置中心填写。")
        self.api_key = api_key
        self.host = normalize_api_base_url(api_url or "", "")
        self.vidu_host = self.host
        self.happyhorse_host = self.host
        self.media_upload_api_key = str(media_upload_api_key or "").strip()
        self.grsai_media_upload_api_key = str(grsai_media_upload_api_key or "").strip()
        self._media_uploaders = None

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }

    def _extract_task_id_from_partial_text(self, text: str) -> str:
        value = r"([A-Za-z0-9][A-Za-z0-9._:-]{5,})"
        patterns = [
            rf'"task_id"\s*:\s*"{value}"',
            rf'"taskId"\s*:\s*"{value}"',
            rf'"id"\s*:\s*"{value}"',
            r'"task_id"\s*:\s*([0-9]{6,})',
            r'"taskId"\s*:\s*([0-9]{6,})',
            r'"id"\s*:\s*([0-9]{6,})',
            rf'"data"\s*:\s*"{value}"',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _partial_task_response(self, text: str) -> dict | None:
        task_id = self._extract_task_id_from_partial_text(text)
        if not task_id:
            return None
        return {
            "task_id": task_id,
            "data": {"task_id": task_id},
            "_partial_response": text[:300],
        }

    def _read_json_response(self, response, label: str, allow_partial_task_id: bool) -> dict:
        chunks = []
        partial = False
        chunk_size = 16 if allow_partial_task_id else 8192
        try:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    chunks.append(chunk)
        except requests.exceptions.ChunkedEncodingError:
            partial = True
            if not allow_partial_task_id or not chunks:
                raise

        content = b"".join(chunks)
        if not content:
            return {}

        text = content.decode(response.encoding or "utf-8", errors="replace").strip()
        try:
            return json.loads(text)
        except ValueError as exc:
            if partial and allow_partial_task_id:
                recovered = self._partial_task_response(text)
                if recovered:
                    print(f"[CometAPI Video] {label} response was cut off after submit; recovered task_id={recovered['task_id']}.")
                    return recovered
                raise SubmitResponseLostError(
                    f"{label} 提交请求的响应中断；服务端可能已经创建任务。为避免重复扣费，插件未自动重试。"
                    f"请到 API 后台查看最近任务。半截响应：{text[:300]}"
                ) from exc
            raise CometAPIError(f"{label} 返回了非 JSON 内容：{text[:300]}") from exc

    def _request_json(
        self,
        method: str,
        url: str,
        label: str,
        timeout: int = 60,
        attempts: int = 3,
        allow_partial_task_id: bool = False,
        **kwargs,
    ) -> dict:
        last_error = None
        for attempt in range(attempts):
            response = None
            try:
                response = requests.request(method, url, headers=self._headers(), timeout=(15, timeout), stream=True, **kwargs)
                response.raise_for_status()
                return self._read_json_response(response, label, allow_partial_task_id)
            except requests.exceptions.HTTPError as exc:
                # 打印 API 返回的错误详情
                body_text = ""
                try:
                    body_text = response.text[:500] if response is not None else ""
                except Exception:
                    pass
                logger.error(f"[{label}] HTTP {response.status_code if response else '?'}: {body_text}")
                raise CometAPIError(f"{label} HTTP {response.status_code if response else '?'}: {body_text or exc}") from exc
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if attempts > 1:
                    raise CometAPIError(f"{label} 连接中断，自动重试 {attempts} 次后仍失败：{exc}") from exc
                if allow_partial_task_id:
                    raise SubmitResponseLostError(
                        f"{label} 提交请求连接中断；服务端可能已经创建任务。为避免重复扣费，插件未自动重试。"
                        f"请到 API 后台查看最近任务。原始错误：{exc}"
                    ) from exc
                raise CometAPIError(f"{label} 连接中断：{exc}") from exc
            finally:
                if response is not None:
                    response.close()
        raise last_error

    def _post_json(self, path: str, payload: dict, timeout: int = 60) -> dict:
        return self._request_json(
            "POST",
            f"{self.host}{path}",
            "视频 API",
            timeout=timeout,
            attempts=1,
            allow_partial_task_id=True,
            json=payload,
        )

    def _get_json(self, path: str, timeout: int = 30) -> dict:
        return self._request_json("GET", f"{self.host}{path}", "视频 API", timeout=timeout)

    def _post_vidu_json(self, path: str, payload: dict, timeout: int = 60) -> dict:
        return self._request_json(
            "POST",
            f"{self.vidu_host}{path}",
            "Vidu API",
            timeout=timeout,
            attempts=1,
            allow_partial_task_id=True,
            json=payload,
        )

    def _get_vidu_json(self, path: str, timeout: int = 30) -> dict:
        return self._request_json("GET", f"{self.vidu_host}{path}", "Vidu API", timeout=timeout)

    def _submit_video_create(self, payload: dict) -> str:
        data = self._post_json("/v1/video/create", payload)
        task_id = data.get("id") or data.get("task_id")
        if not task_id and isinstance(data.get("data"), dict):
            task_id = data["data"].get("id") or data["data"].get("task_id")
        if not task_id:
            raise CometAPIError(f"API 没有返回任务 ID：{str(data)[:300]}")
        return str(task_id)

    def _query_video(self, task_id: str) -> tuple[str, str, dict]:
        data = self._get_json(f"/v1/video/query?id={task_id}")
        body = data.get("data") if isinstance(data.get("data"), dict) else data
        status = str(body.get("status") or data.get("status") or "running").lower()
        video_url = (
            body.get("video_url")
            or body.get("url")
            or data.get("video_url")
            or data.get("url")
        )
        return status, video_url or "", data

    def _wait_for_video_url(self, task_id: str, timeout: int = 900, interval: int = 5) -> str:
        started = time.time()
        last_status = "running"
        while time.time() - started < timeout:
            time.sleep(interval)
            status, video_url, data = self._query_video(task_id)
            last_status = status
            if status in {"success", "succeeded", "completed"}:
                if video_url:
                    return video_url
                raise CometAPIError(f"视频任务已成功，但 API 没有返回下载地址：{str(data)[:300]}")
            if any(token in status for token in ("failed", "failure", "error")):
                raise CometAPIError(f"视频任务失败：{str(data)[:300]}")
        raise CometAPIError(f"视频任务等待超时（{timeout} 秒），最后状态：{last_status}")

    def _submit_vidu_task(self, path: str, payload: dict) -> str:
        try:
            data = self._post_vidu_json(path, payload)
        except SubmitResponseLostError as exc:
            recovered_task_id = self._recover_vidu_task_from_recent_tasks(path, payload)
            if recovered_task_id:
                return recovered_task_id
            raise exc
        body = data.get("data") if isinstance(data.get("data"), dict) else data
        task_id = (
            body.get("task_id")
            or body.get("id")
            or data.get("task_id")
            or data.get("id")
            or self._first_nested_value(data, {"task_id", "taskid"})
        )
        if not task_id:
            raise CometAPIError(f"Vidu API 没有返回任务 ID：{str(data)[:300]}")
        return str(task_id)

    def _normalize_vidu_status(self, data: dict) -> str:
        candidates = []
        if isinstance(data, dict):
            candidates.extend([data.get("state"), data.get("status")])
            body = data.get("data")
            if isinstance(body, dict):
                candidates.extend([body.get("state"), body.get("status")])
            response = data.get("Response")
            if isinstance(response, dict):
                candidates.extend([response.get("Status"), response.get("status"), response.get("state")])
            candidates.append(self._first_nested_value(data, {"status", "state"}))
        raw = next((str(item).strip().lower() for item in candidates if item), "processing")
        if raw in {"success", "succeeded", "completed", "complete", "finish", "finished", "done"}:
            return "success"
        if raw in {"failed", "failure", "fail", "error", "cancelled", "canceled"}:
            return "failed"
        return raw or "processing"

    def _collect_vidu_urls(self, value, context: str = "", urls: list[tuple[int, str]] | None = None) -> list[tuple[int, str]]:
        if urls is None:
            urls = []
        if isinstance(value, dict):
            for key, child in value.items():
                self._collect_vidu_urls(child, f"{context}.{key}", urls)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._collect_vidu_urls(child, f"{context}[{index}]", urls)
        elif isinstance(value, str) and value.startswith(("http://", "https://")):
            lower_context = context.lower()
            lower_url = value.lower()
            score = 0
            if any(token in lower_url for token in (".mp4", ".mov", ".m3u8", ".webm")):
                score += 100
            if any(token in lower_context for token in ("creation", "output", "fileinfos", "result")):
                score += 30
            if any(token in lower_context for token in ("video", "url", "file")):
                score += 15
            if any(token in lower_context for token in ("input", "image", "cover")):
                score -= 30
            if "watermark" in lower_context:
                score -= 10
            urls.append((score, value))
        return urls

    def _extract_vidu_video_url(self, data: dict) -> str:
        urls = self._collect_vidu_urls(data)
        if not urls:
            return ""
        urls.sort(key=lambda item: item[0], reverse=True)
        if urls[0][0] <= 0:
            return ""
        return urls[0][1]

    def _first_nested_value(self, value, keys: set[str]):
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in keys and child not in (None, ""):
                    return child
            for child in value.values():
                found = self._first_nested_value(child, keys)
                if found not in (None, ""):
                    return found
        elif isinstance(value, list):
            for child in value:
                found = self._first_nested_value(child, keys)
                if found not in (None, ""):
                    return found
        return None

    def _clean_vidu_match_text(self, value) -> str:
        return "".join(ch for ch in str(value or "") if ch not in ZERO_WIDTH_CHARS).strip()

    def _extract_vidu_task_list(self, data: dict) -> list[dict]:
        if not isinstance(data, dict):
            return []
        containers = [data]
        for key in ("data", "Response", "response"):
            child = data.get(key)
            if isinstance(child, dict):
                containers.append(child)
        for container in containers:
            for key in ("tasks", "Tasks", "task_list", "TaskList", "items", "Items"):
                value = container.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        value = self._first_nested_value(data, {"tasks", "tasklist"})
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    def _vidu_recent_task_score(self, task: dict, payload: dict, path: str) -> int:
        task_id = task.get("id") or task.get("task_id") or self._first_nested_value(task, {"task_id", "taskid"})
        if not task_id:
            return -1

        score = 0
        task_prompt = self._clean_vidu_match_text(self._first_nested_value(task, {"prompt"}))
        payload_prompt = self._clean_vidu_match_text(payload.get("prompt"))
        if payload_prompt:
            if task_prompt != payload_prompt:
                return -1
            score += 10

        task_model = self._clean_vidu_match_text(self._first_nested_value(task, {"model", "modelname"})).lower()
        payload_model = self._clean_vidu_match_text(payload.get("model")).lower()
        if task_model and payload_model:
            if task_model != payload_model:
                return -1
            score += 4

        for key in ("duration", "resolution", "aspect_ratio"):
            task_value = task.get(key)
            payload_value = payload.get(key)
            if task_value not in (None, "") and payload_value not in (None, ""):
                if str(task_value).strip().lower() == str(payload_value).strip().lower():
                    score += 1

        task_images = task.get("images")
        payload_images = payload.get("images")
        if isinstance(task_images, list) and isinstance(payload_images, list) and payload_images:
            if set(map(str, payload_images)).issubset(set(map(str, task_images))):
                score += 4

        task_type = self._clean_vidu_match_text(task.get("type") or task.get("template")).lower()
        endpoint_kind = path.rsplit("/", 1)[-1].lower()
        if task_type and endpoint_kind and endpoint_kind in task_type:
            score += 2
        return score

    def _recover_vidu_task_from_recent_tasks(self, path: str, payload: dict, timeout: int = 30) -> str:
        deadline = time.time() + max(1, timeout)
        last_error = None
        while time.time() < deadline:
            try:
                data = self._get_vidu_json("/ent/v2/tasks?paper.page=0&paper.pagesz=20", timeout=30)
                tasks = self._extract_vidu_task_list(data)
                scored = [
                    (self._vidu_recent_task_score(task, payload, path), index, task)
                    for index, task in enumerate(tasks)
                ]
                scored = [item for item in scored if item[0] >= 10]
                if scored:
                    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                    task = scored[0][2]
                    task_id = task.get("id") or task.get("task_id") or self._first_nested_value(task, {"task_id", "taskid"})
                    if task_id:
                        print(f"[CometAPI Video] Vidu submit response was lost; recovered task_id={task_id} from recent tasks.")
                        return str(task_id)
            except Exception as exc:
                last_error = exc
            time.sleep(3)

        if last_error:
            print(f"[CometAPI Video] Vidu recent-task recovery failed: {last_error}")
        return ""

    def _extract_nested_video_url(self, data: dict) -> str:
        direct = self._first_nested_value(data, {"video_url", "download_url"})
        if isinstance(direct, str) and direct.startswith(("http://", "https://")):
            return direct
        urls = self._collect_vidu_urls(data)
        urls.sort(key=lambda item: item[0], reverse=True)
        return urls[0][1] if urls and urls[0][0] > 0 else ""

    def _query_vidu_video(self, task_id: str) -> tuple[str, str, dict]:
        data = self._get_vidu_json(f"/ent/v2/tasks/{task_id}/creations")
        return self._normalize_vidu_status(data), self._extract_vidu_video_url(data), data

    def _wait_for_vidu_video_url(self, task_id: str, timeout: int = 900, interval: int = 5) -> str:
        started = time.time()
        last_status = "processing"
        while time.time() - started < timeout:
            time.sleep(interval)
            status, video_url, data = self._query_vidu_video(task_id)
            last_status = status
            if status == "success":
                if video_url:
                    return video_url
                raise CometAPIError(f"Vidu 任务已成功，但 API 没有返回下载地址：{str(data)[:300]}")
            if status == "failed":
                raise CometAPIError(f"Vidu 任务失败：{str(data)[:300]}")
        raise CometAPIError(f"Vidu 任务等待超时（{timeout} 秒），最后状态：{last_status}")

    def _build_sora_payload(self, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str, duration: str, size: str) -> dict:
        """已废弃，保留兼容签名但不再使用"""
        return {}

    def _generate_sora_video(self, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str, duration: str) -> tuple[str, str, str]:
        """使用私有渠道的 OpenAI 兼容视频格式 POST /v1/videos 生成 sora-2 视频"""
        SORA_DURATIONS = {"4", "8", "12"}
        safe_duration = duration if duration in SORA_DURATIONS else "4"
        safe_size = "720x1280" if str(aspect_ratio or "").strip() == "9:16" else "1280x720"
        clean_prompt = add_prompt_variation(prompt.strip(), 0)

        fields = {
            "model": (None, "sora-2"),
            "prompt": (None, clean_prompt),
            "seconds": (None, safe_duration),
            "size": (None, safe_size),
        }

        # 如果有参考图，resize 到目标尺寸后作为 input_reference 上传
        if pil_images:
            target_w, target_h = (720, 1280) if safe_size == "720x1280" else (1280, 720)
            source = safe_pil_to_rgb(pil_images[0])
            # 等比缩放 + 居中裁剪到目标尺寸
            src_w, src_h = source.size
            scale = max(target_w / src_w, target_h / src_h)
            new_w, new_h = int(src_w * scale + 0.5), int(src_h * scale + 0.5)
            resized = source.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            cropped = resized.crop((left, top, left + target_w, top + target_h))
            image_buffer = BytesIO()
            cropped.save(image_buffer, format="JPEG", quality=95)
            image_buffer.seek(0)
            fields["input_reference"] = ("input_reference.jpeg", image_buffer, "image/jpeg")

        url = f"{self.host}/v1/videos"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(url, headers=headers, files=fields, timeout=(15, 120))
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as exc:
            body_text = ""
            try:
                body_text = response.text[:500] if response is not None else ""
            except Exception:
                pass
            raise CometAPIError(f"Sora API 请求失败: {body_text or exc}") from exc
        except Exception as exc:
            raise CometAPIError(f"Sora API 连接失败: {exc}") from exc

        task_id = data.get("id") or data.get("task_id")
        if not task_id:
            raise CometAPIError(f"Sora API 没有返回任务 ID：{str(data)[:300]}")

        # 轮询等待视频完成
        video_url = self._wait_for_sora_video(str(task_id))
        path = download_video_asset(video_url, prefix="CometAPISora")
        return path, video_url, str(task_id)

    def _wait_for_sora_video(self, task_id: str, timeout: int = 900, interval: int = 5) -> str:
        """轮询 GET /v1/videos/{id} 等待 sora 视频完成"""
        started = time.time()
        last_status = "queued"
        url = f"{self.host}/v1/videos/{requests.utils.quote(task_id, safe='')}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        while time.time() - started < timeout:
            time.sleep(interval)
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
            except Exception:
                continue
            status = str(data.get("status") or "queued").lower()
            last_status = status
            if status in {"completed", "succeeded", "success"}:
                video_url = data.get("video_url") or data.get("url") or data.get("output", {}).get("video_url", "")
                # 也可能在 output.videos[0].url
                if not video_url:
                    outputs = data.get("output") or data.get("outputs") or {}
                    if isinstance(outputs, dict):
                        videos = outputs.get("videos") or outputs.get("video") or []
                        if isinstance(videos, list) and videos:
                            video_url = videos[0].get("url") or videos[0].get("video_url") or ""
                if video_url:
                    return video_url
                raise CometAPIError(f"Sora 任务已完成但没有返回视频地址：{str(data)[:300]}")
            if status in {"failed", "failure", "error"}:
                raise CometAPIError(f"Sora 视频生成失败：{str(data)[:300]}")
        raise CometAPIError(f"Sora 视频等待超时（{timeout} 秒），最后状态：{last_status}")

    def _build_grok_payload(self, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str, duration: str) -> dict:
        safe_aspect = aspect_ratio if aspect_ratio in PRIVATE_GROK_VIDEO_ASPECT_RATIOS else "16:9"
        image_data = [pil_to_data_url(image) for image in pil_images[:MAX_VIDEO_IMAGE_INPUTS]]
        clean_prompt = convert_prompt_asset_mentions(prompt.strip(), image_count=len(image_data))
        payload = {
            "model": resolve_grok_video_model(model, duration),
            "prompt": add_prompt_variation(clean_prompt, 0),
            "aspect_ratio": safe_aspect,
            "size": "720P",
        }
        if image_data:
            payload["images"] = image_data
        return payload

    def _build_veo_payload(self, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str) -> dict:
        image_urls = []
        for pil_image in pil_images[:MAX_VIDEO_IMAGE_INPUTS]:
            url = upload_image_private(self.api_key, pil_image)
            if not url:
                raise CometAPIError("Veo 参考图上传失败，请检查输入图片或网络。")
            image_urls.append(url)

        payload = {
            "model": model,
            "prompt": add_prompt_variation(prompt.strip(), 0),
            "aspect_ratio": aspect_ratio if aspect_ratio in PRIVATE_VIDEO_ASPECT_RATIOS else "16:9",
            "enable_upsample": True,
            "enhance_prompt": True,
        }
        if image_urls:
            payload["images"] = image_urls
        return payload

    def _build_vidu_payload(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        duration: str,
        resolution: str,
        mode: str,
    ) -> tuple[str, dict]:
        safe_mode = normalize_vidu_video_mode(model, mode)
        required_images = vidu_required_images_for_mode(safe_mode)
        max_images = vidu_max_images_for_mode(safe_mode)
        selected_images = pil_images[:max_images]
        if len(selected_images) < required_images:
            raise CometAPIError(f"Vidu「{safe_mode}」模式至少需要 {required_images} 张参考图。")

        image_urls = []
        for pil_image in selected_images:
            url = upload_image_private(self.api_key, pil_image)
            if not url:
                raise CometAPIError("Vidu 参考图上传失败，请检查输入图片或网络。")
            image_urls.append(url)

        safe_duration = normalize_vidu_duration(model, safe_mode, duration)
        safe_resolution = normalize_vidu_resolution(model, safe_mode, resolution)
        safe_aspect = normalize_vidu_aspect_ratio(model, safe_mode, aspect_ratio)
        audio_enabled = vidu_supports_audio_output(model, safe_mode)
        payload = {
            "model": model,
            "prompt": add_prompt_variation(prompt.strip(), 0),
            "duration": safe_duration,
            "resolution": safe_resolution,
            "movement_amplitude": "auto",
            "audio": audio_enabled,
            "bgm": False,
            "off_peak": False,
            "watermark": False,
            "is_rec": False,
        }
        if audio_enabled and vidu_supports_audio_type(model, safe_mode):
            payload["audio_type"] = "all"
        if safe_aspect and safe_aspect != "auto":
            payload["aspect_ratio"] = safe_aspect
        if safe_mode == "文生":
            payload["style"] = "general"
        else:
            payload["images"] = image_urls
        return vidu_endpoint_for_mode(safe_mode), payload

    def _generate_vidu_video(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        duration: str,
        resolution: str,
        mode: str,
    ) -> tuple[str, str, str]:
        endpoint, payload = self._build_vidu_payload(prompt, model, pil_images, aspect_ratio, duration, resolution, mode)
        logger.info(f"[Vidu Debug] endpoint={endpoint} payload={json.dumps(payload, ensure_ascii=False, default=str)[:2000]}")
        task_id = self._submit_vidu_task(endpoint, payload)
        video_url = self._wait_for_vidu_video_url(task_id)
        path = download_video_asset(video_url, prefix="CometAPIVidu")
        return path, video_url, task_id

    def _submit_hailuo_task(self, payload: dict) -> str:
        data = self._post_json("/minimax/v1/video_generation", payload)
        base_resp = data.get("base_resp") if isinstance(data.get("base_resp"), dict) else {}
        if base_resp.get("status_code") not in (None, 0):
            raise CometAPIError(f"海螺 API 错误：{str(data)[:300]}")
        task_id = data.get("task_id") or self._first_nested_value(data, {"task_id", "id"})
        if not task_id:
            raise CometAPIError(f"海螺 API 没有返回任务 ID：{str(data)[:300]}")
        return str(task_id)

    def _normalize_hailuo_status(self, data: dict) -> str:
        if self._first_nested_value(data, {"file_id"}) or self._extract_nested_video_url(data):
            return "success"
        raw = self._first_nested_value(data, {"status", "state", "task_status"})
        raw = str(raw or "processing").strip().lower()
        if raw in {"success", "succeed", "succeeded", "completed", "complete", "finish", "finished", "done"}:
            return "success"
        if raw in {"fail", "failed", "failure", "error", "cancelled", "canceled"}:
            return "failed"
        base_resp = data.get("base_resp") if isinstance(data, dict) and isinstance(data.get("base_resp"), dict) else {}
        if base_resp.get("status_code") not in (None, 0):
            return "failed"
        return raw or "processing"

    def _retrieve_hailuo_file_url(self, file_id: str) -> str:
        data = self._get_json(f"/minimax/v1/files/retrieve?file_id={requests.utils.quote(str(file_id), safe='')}")
        video_url = self._extract_nested_video_url(data)
        if not video_url:
            raise CometAPIError(f"海螺文件查询没有返回下载地址：{str(data)[:300]}")
        return video_url

    def _query_hailuo_video(self, task_id: str) -> tuple[str, str, dict]:
        data = self._get_json(f"/minimax/v1/query/video_generation?task_id={requests.utils.quote(str(task_id), safe='')}")
        status = self._normalize_hailuo_status(data)
        video_url = self._extract_nested_video_url(data)
        if not video_url:
            file_id = self._first_nested_value(data, {"file_id"})
            if file_id:
                video_url = self._retrieve_hailuo_file_url(str(file_id))
        return status, video_url or "", data

    def _wait_for_hailuo_video_url(self, task_id: str, timeout: int = 900, interval: int = 5) -> str:
        started = time.time()
        last_status = "processing"
        while time.time() - started < timeout:
            time.sleep(interval)
            status, video_url, data = self._query_hailuo_video(task_id)
            last_status = status
            if status == "success":
                if video_url:
                    return video_url
                raise CometAPIError(f"海螺任务已成功，但 API 没有返回下载地址：{str(data)[:300]}")
            if status == "failed":
                raise CometAPIError(f"海螺任务失败：{str(data)[:300]}")
        raise CometAPIError(f"海螺任务等待超时（{timeout} 秒），最后状态：{last_status}")

    def _build_hailuo_payload(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        duration: str,
        resolution: str,
        mode: str,
    ) -> dict:
        safe_mode = normalize_hailuo_video_mode(mode)
        required_images = {"文生": 0, "图生": 1, "首尾帧": 2}.get(safe_mode, 0)
        if len(pil_images) < required_images:
            raise CometAPIError(f"海螺「{safe_mode}」模式至少需要 {required_images} 张参考图。")
        if safe_mode == "首尾帧" and model != "MiniMax-Hailuo-02":
            raise CometAPIError("海螺首尾帧模式目前需要使用 MiniMax-Hailuo-02。")

        safe_duration = normalize_hailuo_duration(duration)
        payload = {
            "model": model,
            "prompt": add_prompt_variation(prompt.strip(), 0),
            "duration": int(safe_duration),
            "resolution": normalize_hailuo_resolution(safe_duration, resolution),
            "prompt_optimizer": True,
            "fast_pretreatment": True,
            "aigc_watermark": False,
        }
        if safe_mode in {"图生", "首尾帧"}:
            payload["first_frame_image"] = pil_to_data_url(pil_images[0])
        if safe_mode == "首尾帧":
            payload["last_frame_image"] = pil_to_data_url(pil_images[1])
        return payload

    def _generate_hailuo_video(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        duration: str,
        resolution: str,
        mode: str,
    ) -> tuple[str, str, str]:
        payload = self._build_hailuo_payload(prompt, model, pil_images, duration, resolution, mode)
        task_id = self._submit_hailuo_task(payload)
        video_url = self._wait_for_hailuo_video_url(task_id)
        path = download_video_asset(video_url, prefix="CometAPIHailuo")
        return path, video_url, task_id

    def _submit_kling_task(self, action: str, payload: dict) -> str:
        data = self._post_json(f"/kling/v1/videos/{action}", payload)
        code = data.get("code")
        if code not in (None, 0):
            raise CometAPIError(f"可灵 API 错误：{str(data)[:300]}")
        body = data.get("data") if isinstance(data.get("data"), dict) else data
        task_id = body.get("task_id") or body.get("id") or data.get("task_id") or data.get("id")
        if not task_id:
            raise CometAPIError(f"可灵 API 没有返回任务 ID：{str(data)[:300]}")
        return str(task_id)

    def _normalize_kling_status(self, data: dict) -> str:
        raw = self._first_nested_value(data, {"task_status", "status", "state"})
        raw = str(raw or "processing").strip().lower()
        if raw in {"succeed", "success", "succeeded", "completed", "complete", "finish", "finished", "done"}:
            return "success"
        if raw in {"failed", "fail", "failure", "error", "cancelled", "canceled"}:
            return "failed"
        code = data.get("code") if isinstance(data, dict) else None
        if code not in (None, 0):
            return "failed"
        return raw or "processing"

    def _query_kling_video(self, action: str, task_id: str) -> tuple[str, str, dict]:
        data = self._get_json(f"/kling/v1/videos/{action}/{requests.utils.quote(str(task_id), safe='')}")
        return self._normalize_kling_status(data), self._extract_nested_video_url(data), data

    def _wait_for_kling_video_url(self, action: str, task_id: str, timeout: int = 900, interval: int = 5) -> str:
        started = time.time()
        last_status = "processing"
        while time.time() - started < timeout:
            time.sleep(interval)
            status, video_url, data = self._query_kling_video(action, task_id)
            last_status = status
            if status == "success":
                if video_url:
                    return video_url
                raise CometAPIError(f"可灵任务已成功，但 API 没有返回下载地址：{str(data)[:300]}")
            if status == "failed":
                raise CometAPIError(f"可灵任务失败：{str(data)[:300]}")
        raise CometAPIError(f"可灵任务等待超时（{timeout} 秒），最后状态：{last_status}")

    def _build_kling_payload(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        duration: str,
        quality_mode: str,
        mode: str,
    ) -> tuple[str, dict]:
        safe_mode = normalize_kling_video_mode(model, mode)
        action = kling_action_for_mode(safe_mode)
        safe_quality = normalize_kling_quality_mode(quality_mode)
        safe_duration = normalize_kling_duration(model, duration)
        clean_prompt = add_prompt_variation(prompt.strip(), 0)
        payload = {
            "model_name": model,
            "prompt": clean_prompt,
            "mode": safe_quality,
            "duration": safe_duration,
            "watermark_info": {"enabled": False},
        }

        if safe_mode == "文生":
            payload["aspect_ratio"] = normalize_kling_aspect_ratio(aspect_ratio)
            if private_kling_supports_generated_sound(model):
                payload["sound"] = "on"
            return action, payload

        if safe_mode == "图生":
            if not pil_images:
                raise CometAPIError("可灵图生模式至少需要 1 张参考图。")
            payload["image"] = pil_to_base64(pil_images[0])
            if private_kling_supports_generated_sound(model):
                payload["sound"] = "on"
            return action, payload

        if safe_mode == "首尾帧":
            if len(pil_images) < 2:
                raise CometAPIError("可灵首尾帧模式至少需要 2 张参考图。")
            payload["image"] = pil_to_base64(pil_images[0])
            payload["image_tail"] = pil_to_base64(pil_images[1])
            if private_kling_supports_generated_sound(model):
                payload["sound"] = "on"
            return action, payload

        if safe_mode == "多图":
            selected_images = pil_images[:4]
            if not selected_images:
                raise CometAPIError("可灵多图模式至少需要 1 张参考图。")
            payload["image_list"] = [{"image": pil_to_base64(image)} for image in selected_images]
            payload["aspect_ratio"] = normalize_kling_aspect_ratio(aspect_ratio)
            if private_kling_supports_generated_sound(model):
                payload["sound"] = "on"
            return action, payload

        selected_images = pil_images[:MAX_VIDEO_IMAGE_INPUTS]
        payload["aspect_ratio"] = normalize_kling_aspect_ratio(aspect_ratio)
        payload["sound"] = "on" if private_kling_supports_generated_sound(model) else "off"
        if selected_images:
            image_list = []
            for image in selected_images:
                image_url = upload_image_private(self.api_key, image)
                if not image_url:
                    raise CometAPIError("可灵 Omni 参考图上传失败，请检查输入图片或网络。")
                image_list.append({"image_url": image_url})
            payload["image_list"] = image_list
        return action, payload

    def _generate_kling_video(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        duration: str,
        quality_mode: str,
        mode: str,
    ) -> tuple[str, str, str]:
        action, payload = self._build_kling_payload(prompt, model, pil_images, aspect_ratio, duration, quality_mode, mode)
        task_id = self._submit_kling_task(action, payload)
        video_url = self._wait_for_kling_video_url(action, task_id)
        path = download_video_asset(video_url, prefix="CometAPIKling")
        return path, video_url, task_id

    # ===================== HappyHorse =====================
    def _happyhorse_get_media_uploaders(self) -> list[tuple[str, object]]:
        """Borrow RunningHub/grsai upload services so happyhorse can publish reference video URLs."""
        if self._media_uploaders is not None:
            return self._media_uploaders

        uploaders = []
        runninghub_key = self.media_upload_api_key or get_channel_api_key("", "runninghub", "", "video")
        if runninghub_key:
            uploaders.append(("RunningHub", RunningHubVideoAPI(runninghub_key)))

        grsai_key = self.grsai_media_upload_api_key or get_channel_api_key("", "grsai", "", "image")
        if grsai_key:
            uploaders.append(("grsai", GrsaiMediaUploadAPI(grsai_key)))

        if not uploaders:
            raise CometAPIError("无极 HappyHorse 视频编辑需要参考视频公网URL，请在设置中心填写 RunningHub 或 grsai 的 API Key 以借用上传服务。")

        self._media_uploaders = uploaders
        return uploaders

    def _happyhorse_upload_media(self, media_input, label: str, suffix: str) -> str:
        errors = []
        for uploader_label, uploader in self._happyhorse_get_media_uploaders():
            try:
                return uploader._upload_media_input(media_input, suffix)
            except CometAPIError as exc:
                errors.append(f"{uploader_label}: {exc}")
            except Exception as exc:
                errors.append(f"{uploader_label}: {exc}")
        detail = "；".join(errors) if errors else "没有可用上传渠道"
        raise CometAPIError(f"无极 HappyHorse {label}素材上传失败：{detail}")

    def _happyhorse_media_url(self, media_input, label: str, suffix: str) -> str:
        if media_input is None:
            return ""
        if isinstance(media_input, str) and media_input.strip().lower().startswith(("http://", "https://")):
            return media_input.strip()
        if isinstance(media_input, dict):
            for key in ("url", "video_url", "audio_url", "src"):
                value = str(media_input.get(key) or "").strip()
                if value.lower().startswith(("http://", "https://")):
                    return value
        for attr in ("video_url", "audio_url", "url", "src"):
            value = str(getattr(media_input, attr, "") or "").strip()
            if value.lower().startswith(("http://", "https://")):
                return value
        return self._happyhorse_upload_media(media_input, label, suffix)

    def _happyhorse_image_urls(self, pil_images: list[Image.Image], limit: int) -> list[str]:
        urls = []
        for pil_image in pil_images[:limit]:
            url = upload_image_private(self.api_key, pil_image)
            if not url:
                raise CometAPIError("无极 HappyHorse 参考图上传失败，请检查输入图片或网络。")
            urls.append(url)
        return urls

    def _build_happyhorse_payload(
        self,
        prompt: str,
        pil_images: list[Image.Image],
        video_inputs: list,
        aspect_ratio: str,
        duration: str,
        resolution: str,
        mode: str,
    ) -> dict:
        safe_mode = normalize_private_happyhorse_mode(mode)
        safe_resolution = normalize_private_happyhorse_resolution(resolution)
        safe_duration = normalize_private_happyhorse_duration(duration)
        safe_ratio = normalize_private_happyhorse_aspect(aspect_ratio, safe_mode)
        clean_prompt = add_prompt_variation(str(prompt or "").strip(), 0)

        if safe_mode == "文生":
            payload = {
                "model": "happyhorse-1.0-t2v",
                "input": {"prompt": clean_prompt},
                "parameters": {
                    "resolution": safe_resolution,
                    "duration": safe_duration,
                    "watermark": False,
                },
            }
            if safe_ratio:
                payload["parameters"]["ratio"] = safe_ratio
            return payload

        if safe_mode == "图生":
            if not pil_images:
                raise CometAPIError("无极 HappyHorse 图生模式至少需要 1 张参考图。")
            image_urls = self._happyhorse_image_urls(pil_images[:1], 1)
            return {
                "model": "happyhorse-1.0-i2v",
                "input": {
                    "prompt": clean_prompt,
                    "media": [{"type": "first_frame", "url": image_urls[0]}],
                },
                "parameters": {
                    "resolution": safe_resolution,
                    "duration": safe_duration,
                    "watermark": False,
                },
            }

        if safe_mode == "多图参考":
            if not pil_images:
                raise CometAPIError("无极 HappyHorse 多图参考模式至少需要 1 张参考图。")
            image_urls = self._happyhorse_image_urls(pil_images, PRIVATE_HAPPYHORSE_MAX_REFERENCE_IMAGES)
            payload = {
                "model": "happyhorse-1.0-r2v",
                "input": {
                    "prompt": clean_prompt,
                    "media": [{"type": "reference_image", "url": url} for url in image_urls],
                },
                "parameters": {
                    "resolution": safe_resolution,
                    "duration": safe_duration,
                    "watermark": False,
                },
            }
            if safe_ratio:
                payload["parameters"]["ratio"] = safe_ratio
            return payload

        # 视频编辑
        if not video_inputs:
            raise CometAPIError("无极 HappyHorse 视频编辑模式需要 1 个参考视频。")
        video_url = self._happyhorse_media_url(video_inputs[0], "视频", ".mp4")
        media = [{"type": "video", "url": video_url}]
        # 视频编辑模式最多支持 5 张参考图（可选）
        reference_image_urls = self._happyhorse_image_urls(pil_images, PRIVATE_HAPPYHORSE_MAX_REFERENCE_IMAGES) if pil_images else []
        for url in reference_image_urls:
            media.append({"type": "reference_image", "url": url})
        return {
            "model": "happyhorse-1.0-video-edit",
            "input": {
                "prompt": clean_prompt,
                "media": media,
            },
            "parameters": {
                "resolution": safe_resolution,
                "watermark": False,
                "audio_setting": "auto",
            },
        }

    def _post_happyhorse_json(self, path: str, payload: dict, timeout: int = 90) -> dict:
        return self._request_json(
            "POST",
            f"{self.happyhorse_host}{path}",
            "HappyHorse API",
            timeout=timeout,
            attempts=1,
            allow_partial_task_id=True,
            json=payload,
        )

    def _get_happyhorse_json(self, path: str, timeout: int = 30) -> dict:
        return self._request_json("GET", f"{self.happyhorse_host}{path}", "HappyHorse API", timeout=timeout)

    def _submit_happyhorse_task(self, payload: dict) -> str:
        data = self._post_happyhorse_json(
            "/alibailian/api/v1/services/aigc/video-generation/video-synthesis",
            payload,
        )
        # 兼容多种返回结构：output.task_id / data.task_id / 顶层 task_id
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        body = data.get("data") if isinstance(data.get("data"), dict) else {}
        task_id = (
            output.get("task_id")
            or body.get("task_id")
            or data.get("task_id")
            or output.get("id")
            or body.get("id")
            or data.get("id")
            or self._first_nested_value(data, {"task_id", "taskid"})
        )
        if not task_id:
            raise CometAPIError(f"HappyHorse API 没有返回任务 ID：{str(data)[:300]}")
        return str(task_id)

    def _normalize_happyhorse_status(self, data: dict) -> str:
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        raw = output.get("task_status") or data.get("task_status") or data.get("status")
        if not raw:
            raw = self._first_nested_value(data, {"task_status", "status", "state"})
        text = str(raw or "running").strip().lower()
        if text in {"success", "succeeded", "completed", "complete", "finished", "finish", "done"}:
            return "success"
        if text in {"failure", "failed", "fail", "error", "cancelled", "canceled", "expired"}:
            return "failed"
        return text or "running"

    def _extract_happyhorse_video_url(self, data: dict) -> str:
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        # 优先取 output.video_url / output.results 里的视频地址
        for key in ("video_url", "url", "download_url"):
            value = output.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
        results = output.get("results") if isinstance(output.get("results"), list) else []
        for item in results:
            if isinstance(item, dict):
                for key in ("video_url", "url", "download_url"):
                    value = item.get(key)
                    if isinstance(value, str) and value.startswith(("http://", "https://")):
                        return value
        # 兜底：从整个响应里找带视频后缀的 URL
        urls = self._collect_vidu_urls(data)
        urls.sort(key=lambda item: item[0], reverse=True)
        if urls and urls[0][0] > 0:
            return urls[0][1]
        return ""

    def _query_happyhorse_video(self, task_id: str) -> tuple[str, str, dict]:
        data = self._get_happyhorse_json(f"/alibailian/api/v1/tasks/{requests.utils.quote(str(task_id), safe='')}")
        return self._normalize_happyhorse_status(data), self._extract_happyhorse_video_url(data), data

    def _wait_for_happyhorse_video_url(self, task_id: str, timeout: int = 1200, interval: int = 5) -> str:
        started = time.time()
        last_status = "running"
        while time.time() - started < timeout:
            time.sleep(interval)
            status, video_url, data = self._query_happyhorse_video(task_id)
            last_status = status
            if status == "success":
                if video_url:
                    return video_url
                raise CometAPIError(f"HappyHorse 任务已成功，但 API 没有返回视频地址：{str(data)[:300]}")
            if status == "failed":
                output = data.get("output") if isinstance(data.get("output"), dict) else {}
                message = output.get("error_message") or output.get("message") or data.get("error") or data
                raise CometAPIError(f"HappyHorse 任务失败：{str(message)[:300]}")
        raise CometAPIError(f"HappyHorse 任务等待超时（{timeout} 秒），最后状态：{last_status}")

    def _generate_happyhorse_video(
        self,
        prompt: str,
        pil_images: list[Image.Image],
        video_inputs: list,
        aspect_ratio: str,
        duration: str,
        resolution: str,
        mode: str,
    ) -> tuple[str, str, str]:
        payload = self._build_happyhorse_payload(prompt, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
        logger.info(f"[HappyHorse Debug] payload={json.dumps(payload, ensure_ascii=False, default=str)[:2000]}")
        task_id = self._submit_happyhorse_task(payload)
        video_url = self._wait_for_happyhorse_video_url(task_id)
        path = download_video_asset(video_url, prefix="CometAPIHappyHorse")
        return path, video_url, task_id

    def generate_video(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        duration: str,
        size: str,
        resolution: str = "720p",
        mode: str = "参考",
        video_inputs: list | None = None,
    ) -> tuple[str, str, str]:
        video_inputs = list(video_inputs or [])
        if is_vidu_video_model(model):
            return self._generate_vidu_video(prompt, model, pil_images, aspect_ratio, duration, resolution, mode)
        if is_hailuo_video_model(model):
            return self._generate_hailuo_video(prompt, model, pil_images, duration, resolution, mode)
        if is_kling_video_model(model):
            return self._generate_kling_video(prompt, model, pil_images, aspect_ratio, duration, resolution, mode)
        if is_private_happyhorse_video_model(model):
            return self._generate_happyhorse_video(prompt, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
        if is_sora_video_model(model):
            return self._generate_sora_video(prompt, model, pil_images, aspect_ratio, duration)
        elif is_grok_video_model(model):
            payload = self._build_grok_payload(prompt, model, pil_images, aspect_ratio, duration)
        elif is_veo_video_model(model):
            payload = self._build_veo_payload(prompt, model, pil_images, aspect_ratio)
        else:
            raise CometAPIError(f"不支持的私有视频模型：{model}")

        task_id = self._submit_video_create(payload)
        video_url = self._wait_for_video_url(task_id)
        path = download_video_asset(video_url, prefix="CometAPIVideo")
        return path, video_url, task_id


class ModelVerseVideoAPI:
    host = "https://api.modelverse.cn"

    def __init__(self, api_key: str, media_upload_api_key: str = "", grsai_media_upload_api_key: str = ""):
        if not api_key:
            raise CometAPIError("缺少优云智算 API Key，请先在设置中心填写。")
        self.api_key = api_key
        self.media_upload_api_key = str(media_upload_api_key or "").strip()
        self.grsai_media_upload_api_key = str(grsai_media_upload_api_key or "").strip()
        self._media_uploaders = None

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post_json(self, path: str, payload: dict, timeout: int = 90) -> dict:
        response = requests.post(f"{self.host}{path}", headers=self._headers(), json=payload, timeout=(15, timeout))
        self._raise_for_status(response, "优云智算")
        return response.json()

    def _get_json(self, path: str, timeout: int = 30) -> dict:
        response = requests.get(f"{self.host}{path}", headers=self._headers(), timeout=(15, timeout))
        self._raise_for_status(response, "优云智算")
        return response.json()

    def _raise_for_status(self, response: requests.Response, label: str) -> None:
        if response.status_code < 400:
            return
        text = (response.text or "").strip()
        detail = ""
        if text:
            try:
                parsed = response.json()
                text = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                pass
            detail = f"：{redact_sensitive_text(text[:500])}"
        raise CometAPIError(f"{label} 请求失败：HTTP {response.status_code} {response.reason}{detail}")

    def _submit_task(self, payload: dict) -> str:
        data = self._post_json("/v1/tasks/submit", payload)
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        task_id = output.get("task_id") or data.get("task_id") or data.get("id") or data.get("taskId")
        if not task_id:
            raise CometAPIError(f"优云智算 API 没有返回任务 ID：{str(data)[:300]}")
        return str(task_id)

    def _normalize_status(self, data: dict) -> str:
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        raw = str(output.get("task_status") or data.get("task_status") or data.get("status") or "Running").strip().lower()
        if raw in {"success", "succeeded", "completed", "complete", "finished", "done"}:
            return "success"
        if raw in {"failure", "failed", "fail", "error", "cancelled", "canceled", "expired"}:
            return "failed"
        return raw or "running"

    def _collect_urls(self, value, urls: list[str] | None = None) -> list[str]:
        if urls is None:
            urls = []
        if isinstance(value, dict):
            for child in value.values():
                self._collect_urls(child, urls)
        elif isinstance(value, list):
            for child in value:
                self._collect_urls(child, urls)
        elif isinstance(value, str) and value.startswith(("http://", "https://")):
            urls.append(value)
        return urls

    def _query_task(self, task_id: str) -> tuple[str, str, dict]:
        data = self._get_json(f"/v1/tasks/status?task_id={requests.utils.quote(str(task_id), safe='')}")
        status = self._normalize_status(data)
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        urls = output.get("urls") if isinstance(output.get("urls"), list) else []
        if not urls:
            urls = self._collect_urls(output)
        video_urls = [url for url in urls if any(token in url.lower() for token in (".mp4", ".mov", ".webm", ".m3u8"))]
        return status, (video_urls[0] if video_urls else (urls[0] if urls else "")), data

    def _wait_for_video_url(self, task_id: str, timeout: int = 1200, interval: int = 5) -> str:
        started = time.time()
        last_status = "running"
        while time.time() - started < timeout:
            time.sleep(interval)
            status, video_url, data = self._query_task(task_id)
            last_status = status
            if status == "success":
                if video_url:
                    return video_url
                raise CometAPIError(f"优云智算任务已成功，但 API 没有返回视频地址：{str(data)[:300]}")
            if status == "failed":
                output = data.get("output") if isinstance(data.get("output"), dict) else {}
                message = output.get("error_message") or data.get("error") or data
                raise CometAPIError(f"优云智算任务失败：{str(message)[:300]}")
        raise CometAPIError(f"优云智算任务等待超时（{timeout} 秒），最后状态：{last_status}")

    def _get_media_uploaders(self) -> list[tuple[str, object]]:
        if self._media_uploaders is not None:
            return self._media_uploaders

        uploaders = []
        runninghub_key = self.media_upload_api_key or get_channel_api_key("", "runninghub", "", "video")
        if runninghub_key:
            uploaders.append(("RunningHub", RunningHubVideoAPI(runninghub_key)))

        grsai_key = self.grsai_media_upload_api_key or get_channel_api_key("", "grsai", "", "image")
        if grsai_key:
            uploaders.append(("grsai", GrsaiMediaUploadAPI(grsai_key)))

        if not uploaders:
            raise CometAPIError("视频和音频素材需要公网URL，需要借用 RunningHub 或 grsai 的临时媒体上传服务，请在设置中心填写这两个渠道中任意一个 API Key。")

        self._media_uploaders = uploaders
        return uploaders

    def _upload_media_input(self, media_input, label: str, suffix: str) -> str:
        errors = []
        for uploader_label, uploader in self._get_media_uploaders():
            try:
                return uploader._upload_media_input(media_input, suffix)
            except CometAPIError as exc:
                errors.append(f"{uploader_label}: {exc}")
            except Exception as exc:
                errors.append(f"{uploader_label}: {exc}")
        detail = "；".join(errors) if errors else "没有可用上传渠道"
        raise CometAPIError(f"优云智算 {label}素材上传失败：{detail}")

    def _media_url(self, media_input, label: str, suffix: str) -> str:
        if media_input is None:
            return ""
        if isinstance(media_input, str) and media_input.strip().lower().startswith(("http://", "https://")):
            return media_input.strip()
        if isinstance(media_input, dict):
            for key in ("url", "video_url", "audio_url", "src"):
                value = str(media_input.get(key) or "").strip()
                if value.lower().startswith(("http://", "https://")):
                    return value
        for attr in ("video_url", "audio_url", "url", "src"):
            value = str(getattr(media_input, attr, "") or "").strip()
            if value.lower().startswith(("http://", "https://")):
                return value
        return self._upload_media_input(media_input, label, suffix)

    def _image_data_urls(self, pil_images: list[Image.Image], limit: int) -> list[str]:
        return [pil_to_data_url(image) for image in pil_images[:limit]]

    def _image_b64(self, pil_images: list[Image.Image], limit: int) -> list[str]:
        return [pil_to_base64(image) for image in pil_images[:limit]]

    def _build_seedance_payload(self, model: str, prompt: str, pil_images: list[Image.Image], video_inputs: list, audio_inputs: list, aspect_ratio: str, duration: str, resolution: str, mode: str) -> dict:
        model = canonical_modelverse_video_model(model)
        safe_mode = normalize_modelverse_video_mode(model, mode)
        safe_duration = int(normalize_modelverse_duration(model, duration, safe_mode))
        safe_resolution = normalize_modelverse_resolution(model, resolution)
        safe_ratio = normalize_modelverse_aspect_ratio(model, safe_mode, aspect_ratio)
        content = []
        clean_prompt = str(prompt or "").strip()
        has_reference_video = False
        has_reference_audio = False
        if clean_prompt:
            content.append({"type": "text", "text": clean_prompt})

        if safe_mode == "首尾帧":
            image_urls = self._image_data_urls(pil_images, 2)
            if not image_urls:
                raise CometAPIError("优云 Seedance 首尾帧模式至少需要 1 张参考图。")
            for index, url in enumerate(image_urls):
                content.append({"type": "image_url", "image_url": {"url": url}, "role": "first_frame" if index == 0 else "last_frame"})
        elif safe_mode == "全能参考":
            image_urls = self._image_data_urls(pil_images, MODELVERSE_VIDEO_MAX_IMAGES)
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"})
            for video_input in video_inputs[:MODELVERSE_VIDEO_MAX_VIDEOS]:
                url = self._media_url(video_input, "视频", ".mp4")
                if url:
                    has_reference_video = True
                    content.append({"type": "video_url", "video_url": {"url": url}, "role": "reference_video"})
            for audio_input in audio_inputs[:MODELVERSE_VIDEO_MAX_AUDIOS]:
                url = self._media_url(audio_input, "音频", ".wav")
                if url:
                    has_reference_audio = True
                    content.append({"type": "audio_url", "audio_url": {"url": url}, "role": "reference_audio"})

        if not content:
            raise CometAPIError("优云 Seedance 需要提示词或参考素材。")
        parameters = {
            "resolution": safe_resolution,
            "ratio": safe_ratio,
            "duration": safe_duration,
            "generate_audio": True,
            "camera_fixed": False,
            "watermark": False,
        }
        if safe_mode == "全能参考" and (has_reference_video or has_reference_audio):
            parameters.update({"execution_expires_after": 3600, "draft": False})
        return {
            "model": model,
            "input": {"content": content},
            "parameters": parameters,
        }

    def _build_happyhorse_payload(self, prompt: str, pil_images: list[Image.Image], video_inputs: list, aspect_ratio: str, duration: str, resolution: str, mode: str) -> dict:
        base_model = "happyhorse-1.0"
        safe_mode = normalize_modelverse_video_mode(base_model, mode)
        safe_duration = int(normalize_modelverse_duration(base_model, duration, safe_mode))
        safe_resolution = normalize_modelverse_resolution(base_model, resolution)
        safe_ratio = normalize_modelverse_aspect_ratio(base_model, safe_mode, aspect_ratio)
        clean_prompt = str(prompt or "").strip()
        parameters = {"resolution": safe_resolution, "duration": safe_duration, "watermark": False}

        if safe_mode == "文生":
            return {"model": "happyhorse-1.0-t2v", "input": {"prompt": clean_prompt}, "parameters": {**parameters, "ratio": safe_ratio}}
        if safe_mode == "图生":
            if not pil_images:
                raise CometAPIError("优云 HappyHorse 图生模式至少需要 1 张参考图。")
            return {
                "model": "happyhorse-1.0-i2v",
                "input": {"prompt": clean_prompt, "img_url": pil_to_data_url(pil_images[0])},
                "parameters": parameters,
            }
        if safe_mode == "多图参考":
            image_urls = self._image_data_urls(pil_images, MODELVERSE_VIDEO_MAX_IMAGES)
            if not image_urls:
                raise CometAPIError("优云 HappyHorse 多图参考模式至少需要 1 张参考图。")
            return {
                "model": "happyhorse-1.0-r2v",
                "input": {"prompt": clean_prompt, "images": image_urls},
                "parameters": {**parameters, "ratio": safe_ratio},
            }

        if not video_inputs:
            raise CometAPIError("优云 HappyHorse 视频编辑模式需要 1 个公网视频 URL。")
        video_url = self._media_url(video_inputs[0], "视频", ".mp4")
        payload = {
            "model": "happyhorse-1.0-video-edit",
            "input": {
                "prompt": clean_prompt,
                "video_url": video_url,
            },
            "parameters": {"resolution": safe_resolution, "watermark": False, "audio_setting": "origin"},
        }
        image_urls = self._image_data_urls(pil_images, 5)
        if image_urls:
            payload["input"]["images"] = image_urls
        return payload

    def _sora_video_size(self, aspect_ratio: str) -> str:
        return "720x1280" if str(aspect_ratio or "").strip() == "9:16" else "1280x720"

    def _sora_video_dimensions(self, aspect_ratio: str) -> tuple[int, int]:
        return (720, 1280) if str(aspect_ratio or "").strip() == "9:16" else (1280, 720)

    def _sora_aspect_for_reference(self, image: Image.Image, fallback: str) -> str:
        width, height = getattr(image, "size", (0, 0))
        if width > height:
            return "16:9"
        if height > width:
            return "9:16"
        return fallback if fallback in {"16:9", "9:16"} else "16:9"

    def _sora_reference_image(self, image: Image.Image, aspect_ratio: str) -> Image.Image:
        target_size = self._sora_video_dimensions(aspect_ratio)
        source = safe_pil_to_rgb(image)
        if source.size == target_size:
            return source
        resampling = getattr(Image, "Resampling", Image)
        lanczos = getattr(resampling, "LANCZOS", Image.LANCZOS)
        background = ImageOps.fit(source, target_size, method=lanczos, centering=(0.5, 0.5))
        background = background.filter(ImageFilter.GaussianBlur(radius=28))
        foreground = ImageOps.contain(source, target_size, method=lanczos)
        offset = ((target_size[0] - foreground.size[0]) // 2, (target_size[1] - foreground.size[1]) // 2)
        background.paste(foreground, offset)
        return background

    def _submit_sora_video(self, prompt: str, pil_images: list[Image.Image], aspect_ratio: str, duration: str, mode: str) -> str:
        base_model = "sora-2"
        safe_mode = normalize_modelverse_video_mode(base_model, mode)
        safe_duration = normalize_modelverse_duration(base_model, duration, safe_mode)
        safe_aspect = normalize_modelverse_aspect_ratio(base_model, safe_mode, aspect_ratio)
        clean_prompt = str(prompt or "").strip()
        if not clean_prompt:
            raise CometAPIError("优云 Sora 需要提示词。")

        fields = {
            "model": (None, "sora-2"),
            "prompt": (None, clean_prompt),
            "seconds": (None, str(safe_duration)),
        }
        image_buffer = None
        if safe_mode == "图生":
            if not pil_images:
                raise CometAPIError("优云 Sora 图生模式至少需要 1 张首帧图。")
            safe_aspect = self._sora_aspect_for_reference(pil_images[0], safe_aspect)
            fields["size"] = (None, self._sora_video_size(safe_aspect))
            image_buffer = BytesIO()
            self._sora_reference_image(pil_images[0], safe_aspect).save(image_buffer, format="JPEG", quality=95)
            image_buffer.seek(0)
            fields["input_reference"] = ("input_reference.jpeg", image_buffer, "image/jpeg")
        else:
            fields["size"] = (None, self._sora_video_size(safe_aspect))

        try:
            def post_fields(active_fields: dict) -> requests.Response:
                if image_buffer is not None:
                    image_buffer.seek(0)
                return requests.post(
                    f"{self.host}/v1/videos",
                    headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
                    files=active_fields,
                    timeout=(15, 90),
                )

            response = post_fields(fields)
            if response.status_code >= 400 and image_buffer is not None:
                error_text = (response.text or "").lower()
                if "slot channel not found" in error_text:
                    fallback_fields = dict(fields)
                    fallback_fields.pop("size", None)
                    response = post_fields(fallback_fields)
            self._raise_for_status(response, "优云智算 Sora")
            data = response.json()
        finally:
            if image_buffer is not None:
                image_buffer.close()

        task_id = data.get("id") or data.get("task_id") or data.get("taskId")
        if not task_id and isinstance(data.get("output"), dict):
            output = data["output"]
            task_id = output.get("id") or output.get("task_id") or output.get("taskId")
        if not task_id:
            raise CometAPIError(f"优云智算 Sora API 没有返回任务 ID：{str(data)[:300]}")
        return str(task_id)

    def _query_sora_video(self, task_id: str) -> tuple[str, dict]:
        data = self._get_json(f"/v1/videos/{requests.utils.quote(str(task_id), safe='')}")
        return self._normalize_status(data), data

    def _wait_for_sora_video(self, task_id: str, timeout: int = 1200, interval: int = 5) -> dict:
        started = time.time()
        last_status = "running"
        while time.time() - started < timeout:
            time.sleep(interval)
            status, data = self._query_sora_video(task_id)
            last_status = status
            if status == "success":
                return data
            if status == "failed":
                message = data.get("error") or data.get("error_message") or data
                raise CometAPIError(f"优云智算 Sora 任务失败：{str(message)[:300]}")
        raise CometAPIError(f"优云智算 Sora 任务等待超时（{timeout} 秒），最后状态：{last_status}")

    def _download_sora_video_content(self, task_id: str, timeout: int = 600) -> tuple[str, str]:
        folder_paths = get_folder_paths()
        output_dir = folder_paths.get_output_directory()
        target_dir = os.path.join(output_dir, ASSET_SUBFOLDER)
        os.makedirs(target_dir, exist_ok=True)
        filename = f"{_safe_asset_prefix('CometAPIModelVerseSora', 'CometAPIVideo')}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
        path = os.path.join(target_dir, filename)
        content_url = f"{self.host}/v1/videos/{requests.utils.quote(str(task_id), safe='')}/content"
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "video/mp4,*/*"}
        last_error = None
        for _ in range(3):
            try:
                with requests.get(content_url, headers=headers, stream=True, timeout=(20, timeout)) as response:
                    self._raise_for_status(response, "优云智算 Sora 视频下载")
                    with open(path, "wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                if os.path.exists(path) and os.path.getsize(path) > 1024:
                    return path, content_url
            except Exception as exc:
                last_error = exc
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception:
                    pass
                time.sleep(2)
        raise CometAPIError(f"优云智算 Sora 视频下载失败：{last_error}")

    def _generate_sora_video(self, prompt: str, pil_images: list[Image.Image], aspect_ratio: str, duration: str, mode: str) -> tuple[str, str, str]:
        task_id = self._submit_sora_video(prompt, pil_images, aspect_ratio, duration, mode)
        self._wait_for_sora_video(task_id)
        local_path, content_url = self._download_sora_video_content(task_id)
        return local_path, content_url, task_id

    def _build_kling_payload(self, model: str, prompt: str, pil_images: list[Image.Image], video_inputs: list, aspect_ratio: str, duration: str, resolution: str, mode: str) -> dict:
        model = canonical_modelverse_video_model(model)
        safe_mode = normalize_modelverse_video_mode(model, mode)
        safe_duration = int(normalize_modelverse_duration(model, duration, safe_mode))
        safe_quality = normalize_modelverse_resolution(model, resolution) or "std"
        safe_aspect = normalize_modelverse_aspect_ratio(model, safe_mode, aspect_ratio)
        clean_prompt = str(prompt or "").strip()
        payload = {
            "model": model,
            "input": {"prompt": clean_prompt},
            "parameters": {
                "mode": safe_quality,
                "duration": safe_duration,
                "watermark_enabled": False,
            },
        }

        if model == "kling-v3":
            if safe_mode == "文生":
                sound = "on" if modelverse_kling_supports_generated_sound(model) else "off"
                payload["parameters"].update({"kling_v3_type": "t2v", "aspect_ratio": safe_aspect, "sound": sound})
                return payload
            if safe_mode in {"图生", "首尾帧"}:
                if not pil_images:
                    raise CometAPIError("优云 Kling 图生/首尾帧模式至少需要 1 张参考图。")
                sound = "on" if modelverse_kling_supports_generated_sound(model) else "off"
                payload["parameters"].update({"kling_v3_type": "i2v", "image": pil_to_base64(pil_images[0]), "sound": sound})
                if safe_mode == "首尾帧" and len(pil_images) > 1:
                    payload["parameters"]["image_tail"] = pil_to_base64(pil_images[1])
                return payload
            if len(pil_images) < 1 or not video_inputs:
                raise CometAPIError("优云 Kling 运动控制模式需要 1 张参考图和 1 个公网视频 URL。")
            payload["input"].update({"img_url": pil_to_base64(pil_images[0]), "video_url": self._media_url(video_inputs[0], "视频", ".mp4")})
            payload["parameters"].update({
                "kling_v3_type": "motion_control",
                "aspect_ratio": safe_aspect,
                "character_orientation": "image",
                "keep_original_sound": "yes",
            })
            return payload

        if model == "kling-v2-6":
            payload["parameters"].update({"aspect_ratio": safe_aspect, "mode": "pro" if safe_quality == "pro" else "pro"})
            image_urls = self._image_b64(pil_images, 2)
            if safe_mode == "文生":
                return payload
            if not image_urls:
                raise CometAPIError("优云 Kling v2.6 图生/首尾帧模式至少需要 1 张参考图。")
            payload["parameters"]["image"] = image_urls[0]
            if safe_mode == "首尾帧" and len(image_urls) > 1:
                payload["parameters"]["image_tail"] = image_urls[1]
            return payload

        has_reference_video = bool(video_inputs)
        sound = "on" if modelverse_kling_supports_generated_sound(model) and not has_reference_video else "off"
        payload["parameters"].update({"aspect_ratio": safe_aspect, "sound": sound})
        image_urls = self._image_b64(pil_images, MODELVERSE_VIDEO_MAX_IMAGES)
        if image_urls:
            payload["parameters"]["image_list"] = [{"image_url": url} for url in image_urls]
            if len(payload["parameters"]["image_list"]) >= 1:
                payload["parameters"]["image_list"][0]["type"] = "first_frame"
            if len(payload["parameters"]["image_list"]) >= 2:
                payload["parameters"]["image_list"][1]["type"] = "end_frame"
        if video_inputs:
            payload["parameters"]["sound"] = "off"
            payload["parameters"]["video_list"] = [
                {"video_url": self._media_url(video_inputs[0], "视频", ".mp4"), "refer_type": "base", "keep_original_sound": "yes"}
            ]
            payload["parameters"].pop("aspect_ratio", None)
        return payload

    def generate_video(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        video_inputs: list,
        audio_inputs: list,
        aspect_ratio: str,
        duration: str,
        resolution: str,
        mode: str,
    ) -> tuple[str, str, str]:
        raw_model = canonical_modelverse_video_model(model).lower()
        if raw_model in MODELVERSE_SEEDANCE_MODELS:
            payload = self._build_seedance_payload(raw_model, prompt, pil_images, video_inputs, audio_inputs, aspect_ratio, duration, resolution, mode)
        elif raw_model in MODELVERSE_HAPPYHORSE_MODELS:
            payload = self._build_happyhorse_payload(prompt, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
        elif raw_model in MODELVERSE_SORA_VIDEO_MODELS:
            return self._generate_sora_video(prompt, pil_images, aspect_ratio, duration, mode)
        elif raw_model in MODELVERSE_KLING_VIDEO_MODELS:
            payload = self._build_kling_payload(raw_model, prompt, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
        else:
            raise CometAPIError(f"不支持的优云智算视频模型：{model}")

        task_id = self._submit_task(payload)
        video_url = self._wait_for_video_url(task_id)
        local_path = download_video_asset(video_url, prefix="CometAPIModelVerse")
        return local_path, video_url, task_id


def canonical_runninghub_video_model(model: str) -> str:
    raw = str(model or "").strip()
    if not raw:
        return raw
    normalized = raw.lower()
    for model_id in RUNNINGHUB_VIDEO_MODELS:
        if normalized == model_id.lower():
            return model_id
    return RUNNINGHUB_VIDEO_MODEL_ALIAS_TO_ID.get(normalized, raw)


def is_runninghub_seedance_model(model: str) -> bool:
    return canonical_runninghub_video_model(model).lower() in RUNNINGHUB_SEEDANCE_MODELS


def is_runninghub_google_model(model: str) -> bool:
    return canonical_runninghub_video_model(model).lower() in RUNNINGHUB_GOOGLE_MODELS


def is_runninghub_grok_model(model: str) -> bool:
    return canonical_runninghub_video_model(model).lower() in RUNNINGHUB_GROK_MODELS


def is_runninghub_happyhorse_model(model: str) -> bool:
    return canonical_runninghub_video_model(model).lower() in RUNNINGHUB_HAPPYHORSE_MODELS


def runninghub_resolution_choices(model: str) -> list[str]:
    raw = canonical_runninghub_video_model(model).lower()
    if raw in RUNNINGHUB_GOOGLE_MODELS:
        return RUNNINGHUB_GOOGLE_RESOLUTIONS
    if raw in RUNNINGHUB_GROK_MODELS:
        return RUNNINGHUB_GROK_RESOLUTIONS
    if raw == "sparkvideo-2.0-fast":
        return RUNNINGHUB_SEEDANCE_FAST_RESOLUTIONS
    if raw == "sparkvideo-2.0":
        return RUNNINGHUB_SEEDANCE_RESOLUTIONS
    return RUNNINGHUB_HAPPYHORSE_RESOLUTIONS


def normalize_runninghub_mode(model: str, mode: str) -> str:
    if is_runninghub_google_model(model):
        choices = RUNNINGHUB_GOOGLE_MODES
    elif is_runninghub_grok_model(model):
        choices = RUNNINGHUB_GROK_MODES
    elif is_runninghub_seedance_model(model):
        choices = RUNNINGHUB_SEEDANCE_MODES
    else:
        choices = RUNNINGHUB_HAPPYHORSE_MODES
    raw = str(mode or "").strip()
    if is_runninghub_google_model(model) and raw == "图生":
        return "多参"
    return raw if raw in choices else choices[0]


def normalize_runninghub_duration(model: str, duration: str) -> str:
    if is_runninghub_google_model(model):
        choices = RUNNINGHUB_GOOGLE_DURATIONS
    elif is_runninghub_grok_model(model):
        choices = RUNNINGHUB_GROK_DURATIONS
    elif is_runninghub_seedance_model(model):
        choices = RUNNINGHUB_SEEDANCE_DURATIONS
    else:
        choices = RUNNINGHUB_HAPPYHORSE_DURATIONS
    raw = str(duration or "").strip()
    return raw if raw in choices else ("5" if "5" in choices else choices[0])


def normalize_runninghub_resolution(model: str, resolution: str) -> str:
    choices = runninghub_resolution_choices(model)
    raw = str(resolution or "").strip()
    return raw if raw in choices else ("720p" if "720p" in choices else choices[0])


def normalize_runninghub_aspect_ratio(model: str, mode: str, aspect_ratio: str) -> str:
    if is_runninghub_google_model(model):
        choices = RUNNINGHUB_GOOGLE_ASPECT_RATIOS
        fallback = "16:9"
    elif is_runninghub_grok_model(model):
        choices = RUNNINGHUB_GROK_ASPECT_RATIOS
        fallback = "16:9"
    elif is_runninghub_seedance_model(model):
        choices = RUNNINGHUB_SEEDANCE_ASPECT_RATIOS
        fallback = "adaptive"
    else:
        choices = RUNNINGHUB_HAPPYHORSE_ASPECT_RATIOS
        fallback = "16:9"
    raw = str(aspect_ratio or "").strip()
    return raw if raw in choices else fallback


def _extract_runninghub_url(data: dict) -> str:
    candidates = [
        data.get("download_url"),
        data.get("url"),
        data.get("fileUrl"),
        data.get("file_url"),
    ]
    body = data.get("data") if isinstance(data.get("data"), dict) else {}
    candidates.extend([body.get("download_url"), body.get("url"), body.get("fileUrl"), body.get("file_url")])
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _coerce_audio_to_wav(audio_input, output_path: str) -> bool:
    if not isinstance(audio_input, dict) or "waveform" not in audio_input:
        return False
    waveform = audio_input.get("waveform")
    sample_rate = int(audio_input.get("sample_rate") or 44100)
    if not isinstance(waveform, torch.Tensor):
        return False
    waveform = waveform.detach().cpu().float()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        return False
    waveform = waveform.clamp(-1.0, 1.0)
    samples = (waveform.numpy().T * 32767.0).astype(np.int16)
    channels = samples.shape[1] if samples.ndim > 1 else 1
    with wave.open(output_path, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return True


def _media_input_to_path(media_input, suffix: str) -> tuple[str, bool]:
    if media_input is None:
        return "", False
    existing_path = _resolve_existing_media_path(media_input) or _extract_object_media_path(media_input)
    if existing_path:
        return existing_path, False
    if isinstance(media_input, dict):
        for key in ("path", "file", "filename", "video_path", "audio_path"):
            value_path = _resolve_existing_media_path(media_input.get(key))
            if value_path:
                return value_path, False
        if suffix == ".wav":
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.close()
            if _coerce_audio_to_wav(media_input, tmp.name):
                return tmp.name, True
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
    source_getter = getattr(media_input, "get_stream_source", None)
    if callable(source_getter):
        source = source_getter()
        source_path = _resolve_existing_media_path(source)
        if source_path:
            return source_path, False
        if isinstance(source, BytesIO) or all(hasattr(source, attr) for attr in ("read", "seek")):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                source.seek(0)
                tmp.write(source.read())
                tmp.close()
                if os.path.getsize(tmp.name) > 0:
                    return tmp.name, True
            except Exception:
                tmp.close()
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
    save_to = getattr(media_input, "save_to", None)
    if callable(save_to):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        try:
            saved = save_to(tmp.name)
            saved_path = _resolve_existing_media_path(saved)
            if saved_path:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
                return saved_path, False
            if saved is False or not os.path.exists(tmp.name) or os.path.getsize(tmp.name) <= 0:
                raise CometAPIError("媒体素材保存失败，没有得到可用文件。")
            return tmp.name, True
        except Exception:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise
    return "", False


def canonical_runninghub_image_model(model: str) -> str:
    raw = str(model or "").strip()
    if not raw:
        return raw
    normalized = raw.lower()
    for model_id in RUNNINGHUB_IMAGE_MODELS:
        if normalized == model_id.lower():
            return model_id
    return RUNNINGHUB_IMAGE_MODEL_ALIAS_TO_ID.get(normalized, raw)


def runninghub_image_max_images(model: str) -> int:
    raw = canonical_runninghub_image_model(model)
    return RUNNINGHUB_IMAGE_V1_MAX_IMAGES if raw in RUNNINGHUB_IMAGE_V1_MODELS else RUNNINGHUB_IMAGE_MAX_IMAGES


def runninghub_image_spec(model: str) -> dict:
    raw = canonical_runninghub_image_model(model)
    if raw not in RUNNINGHUB_IMAGE_MODELS:
        raise CometAPIError(f"不支持的 RunningHub 图片模型：{model}")

    gpt_aspects = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
    if raw == "rhart-image-g-2":
        gpt_aspects = ["3:2", "1:1", "2:3", "5:4", "4:5", "16:9", "9:16", "21:9", "3:4", "4:3", "9:21"]

    banana_v1_aspects = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"]
    banana_2_aspects = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9", "1:4", "4:1", "1:8", "8:1"]
    banana_pro_aspects = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"]
    ultra_aspects = ["1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]

    if raw in {"rhart-image-g-2", "rhart-image-g-2-official"}:
        return {
            "base": raw,
            "text_endpoint": "text-to-image",
            "image_endpoint": "image-to-image",
            "aspect_ratios": gpt_aspects,
            "fallback_aspect": "1:1",
            "resolutions": RUNNINGHUB_IMAGE_RESOLUTIONS,
            "quality": raw in RUNNINGHUB_IMAGE_QUALITY_MODELS,
            "max_images": RUNNINGHUB_IMAGE_MAX_IMAGES,
        }
    if raw in RUNNINGHUB_IMAGE_V1_MODELS:
        return {
            "base": raw,
            "text_endpoint": "text-to-image",
            "image_endpoint": "edit",
            "aspect_ratios": banana_v1_aspects,
            "fallback_aspect": "auto",
            "resolutions": [],
            "quality": False,
            "max_images": RUNNINGHUB_IMAGE_V1_MAX_IMAGES,
        }
    if raw in {"rhart-image-n-g31-flash", "rhart-image-n-g31-flash-official"}:
        return {
            "base": raw,
            "text_endpoint": "text-to-image",
            "image_endpoint": "image-to-image",
            "aspect_ratios": banana_2_aspects,
            "fallback_aspect": "1:1",
            "resolutions": RUNNINGHUB_IMAGE_RESOLUTIONS,
            "quality": False,
            "max_images": RUNNINGHUB_IMAGE_MAX_IMAGES,
        }
    if raw in {"rhart-image-n-pro", "rhart-image-n-pro-official"}:
        return {
            "base": raw,
            "text_endpoint": "text-to-image",
            "image_endpoint": "image-to-image",
            "aspect_ratios": banana_pro_aspects,
            "fallback_aspect": "1:1",
            "resolutions": RUNNINGHUB_IMAGE_RESOLUTIONS,
            "quality": False,
            "max_images": RUNNINGHUB_IMAGE_MAX_IMAGES,
        }
    return {
        "base": "rhart-image-n-pro-official",
        "text_endpoint": "text-to-image-ultra",
        "image_endpoint": "edit-ultra",
        "aspect_ratios": ultra_aspects,
        "fallback_aspect": "1:1",
        "resolutions": RUNNINGHUB_IMAGE_ULTRA_RESOLUTIONS,
        "quality": False,
        "max_images": RUNNINGHUB_IMAGE_MAX_IMAGES,
    }


def _extract_runninghub_result_urls(data: dict, preferred_exts: set[str]) -> list[str]:
    containers = [data]
    body = data.get("data") if isinstance(data.get("data"), dict) else None
    if body:
        containers.append(body)

    scored = []
    for container in containers:
        results = container.get("results") if isinstance(container, dict) else None
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or item.get("download_url") or item.get("fileUrl") or item.get("file_url") or "").strip()
            if not url:
                continue
            output_type = str(item.get("outputType") or item.get("type") or "").lower().lstrip(".")
            url_lower = url.lower()
            score = 100 if output_type in preferred_exts else 0
            if any(token in url_lower for token in preferred_exts):
                score += 50
            scored.append((score, url))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [url for _score, url in scored]


class RunningHubImageAPI:
    host = "https://www.runninghub.cn"

    def __init__(self, api_key: str):
        if not api_key:
            raise CometAPIError("缺少 runninghub API Key，请先在设置中心填写。")
        self.api_key = api_key

    def _headers(self, json_content: bool = True) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers

    def _post_json(self, path: str, payload: dict, timeout: int = 60) -> dict:
        response = requests.post(f"{self.host}{path}", headers=self._headers(True), json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _upload_file(self, path: str) -> str:
        if not path or not os.path.exists(path):
            raise CometAPIError("RunningHub 图片文件不存在或无法读取。")
        with open(path, "rb") as handle:
            response = requests.post(
                f"{self.host}/openapi/v2/media/upload/binary",
                headers=self._headers(False),
                files={"file": (os.path.basename(path), handle)},
                timeout=180,
            )
        response.raise_for_status()
        data = response.json()
        url = _extract_runninghub_url(data)
        if not url:
            raise CometAPIError(f"RunningHub 上传后没有返回素材地址：{str(data)[:300]}")
        return url

    def _upload_image(self, pil_image: Image.Image) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.close()
        try:
            safe_pil_to_rgb(pil_image).save(tmp.name, format="PNG")
            return self._upload_file(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    def _submit_task(self, path: str, payload: dict) -> str:
        data = self._post_json(path, payload)
        task_id = data.get("taskId") or data.get("task_id") or data.get("id")
        body = data.get("data") if isinstance(data.get("data"), dict) else {}
        task_id = task_id or body.get("taskId") or body.get("task_id") or body.get("id")
        if not task_id:
            raise CometAPIError(f"RunningHub API 没有返回任务 ID：{str(data)[:300]}")
        return str(task_id)

    def _query_task(self, task_id: str) -> tuple[str, list[str], dict]:
        data = self._post_json("/openapi/v2/query", {"taskId": task_id}, timeout=30)
        status = str(data.get("status") or data.get("state") or "RUNNING").upper()
        urls = _extract_runninghub_result_urls(data, {"png", "jpg", "jpeg", "webp"})
        return status, urls, data

    def _wait_for_image_urls(self, task_id: str, timeout: int = 900, interval: int = 4) -> list[str]:
        started = time.time()
        last_status = "RUNNING"
        while time.time() - started < timeout:
            time.sleep(interval)
            status, image_urls, data = self._query_task(task_id)
            last_status = status
            if status in {"SUCCESS", "SUCCEEDED", "COMPLETED"}:
                if image_urls:
                    return image_urls
                raise CometAPIError(f"RunningHub 任务已成功，但 API 没有返回图片地址：{str(data)[:300]}")
            if status in {"FAILED", "FAILURE", "ERROR", "CANCELED", "CANCELLED"}:
                message = data.get("errorMessage") or data.get("message") or data.get("failedReason") or data
                raise CometAPIError(f"RunningHub 任务失败：{str(message)[:300]}")
        raise CometAPIError(f"RunningHub 图片任务等待超时（{timeout} 秒），最后状态：{last_status}")

    def generate_image(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        image_size: str,
        quality: str,
        subtask_idx: int,
    ) -> tuple[list[Image.Image], list[str]]:
        model = canonical_runninghub_image_model(model)
        spec = runninghub_image_spec(model)
        image_urls = [self._upload_image(image) for image in pil_images[: spec["max_images"]]]
        endpoint = spec["image_endpoint"] if image_urls else spec["text_endpoint"]

        resolved_ratio = nearest_aspect_ratio(
            aspect_ratio,
            pil_images,
            spec["aspect_ratios"],
            fallback=spec["fallback_aspect"],
        )
        payload = {
            "prompt": add_prompt_variation(prompt, subtask_idx),
            "aspectRatio": resolved_ratio,
        }
        if image_urls:
            payload["imageUrls"] = image_urls

        resolutions = spec["resolutions"]
        if resolutions:
            raw_size = str(image_size or "").strip().lower()
            payload["resolution"] = raw_size if raw_size in resolutions else ("4k" if "4k" in resolutions else resolutions[0])
        if spec["quality"]:
            payload["quality"] = quality if quality in GPT_IMAGE_QUALITY_VALUES else "medium"

        task_id = self._submit_task(f"/openapi/v2/{spec['base']}/{endpoint}", payload)
        image_urls = self._wait_for_image_urls(task_id)

        pil_results = []
        errors = []
        for url in image_urls:
            image = download_image(url, timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS)
            if image:
                pil_results.append(safe_pil_to_rgb(image))
            else:
                errors.append(f"图片下载失败：{url}")
        return pil_results, errors


class RunningHubVideoAPI:
    host = "https://www.runninghub.cn"

    def __init__(self, api_key: str):
        if not api_key:
            raise CometAPIError("缺少 runninghub API Key，请先在设置中心填写。")
        self.api_key = api_key

    def _headers(self, json_content: bool = True) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers

    def _post_json(self, path: str, payload: dict, timeout: int = 60) -> dict:
        response = requests.post(f"{self.host}{path}", headers=self._headers(True), json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _upload_file(self, path: str) -> str:
        if not path or not os.path.exists(path):
            raise CometAPIError("RunningHub 媒体文件不存在或无法读取。")
        with open(path, "rb") as handle:
            response = requests.post(
                f"{self.host}/openapi/v2/media/upload/binary",
                headers=self._headers(False),
                files={"file": (os.path.basename(path), handle)},
                timeout=180,
            )
        response.raise_for_status()
        data = response.json()
        url = _extract_runninghub_url(data)
        if not url:
            raise CometAPIError(f"RunningHub 上传后没有返回素材地址：{str(data)[:300]}")
        return url

    def _upload_image(self, pil_image: Image.Image) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.close()
        try:
            safe_pil_to_rgb(pil_image).save(tmp.name, format="PNG")
            return self._upload_file(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    def _upload_media_input(self, media_input, suffix: str) -> str:
        path, should_delete = _media_input_to_path(media_input, suffix)
        if not path:
            raise CometAPIError("不支持这个 RunningHub 媒体输入，请检查连接的素材类型。")
        try:
            return self._upload_file(path)
        finally:
            if should_delete:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    def _submit_task(self, path: str, payload: dict) -> str:
        data = self._post_json(path, payload)
        task_id = data.get("taskId") or data.get("task_id") or data.get("id")
        body = data.get("data") if isinstance(data.get("data"), dict) else {}
        task_id = task_id or body.get("taskId") or body.get("task_id") or body.get("id")
        if not task_id:
            raise CometAPIError(f"RunningHub API 没有返回任务 ID：{str(data)[:300]}")
        return str(task_id)

    def _query_task(self, task_id: str) -> tuple[str, str, dict]:
        data = self._post_json("/openapi/v2/query", {"taskId": task_id}, timeout=30)
        status = str(data.get("status") or data.get("state") or "RUNNING").upper()
        video_url = ""
        results = data.get("results")
        if isinstance(results, list):
            scored = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                output_type = str(item.get("outputType") or item.get("type") or "").lower()
                score = 100 if output_type in {"mp4", "mov", "webm", "m3u8"} else 0
                if any(token in url.lower() for token in (".mp4", ".mov", ".webm", ".m3u8")):
                    score += 50
                scored.append((score, url))
            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                video_url = scored[0][1]
        return status, video_url, data

    def _wait_for_video_url(self, task_id: str, timeout: int = 1200, interval: int = 5) -> str:
        started = time.time()
        last_status = "RUNNING"
        while time.time() - started < timeout:
            time.sleep(interval)
            status, video_url, data = self._query_task(task_id)
            last_status = status
            if status in {"SUCCESS", "SUCCEEDED", "COMPLETED"}:
                if video_url:
                    return video_url
                raise CometAPIError(f"RunningHub 任务已成功，但 API 没有返回视频地址：{str(data)[:300]}")
            if status in {"FAILED", "FAILURE", "ERROR", "CANCELED", "CANCELLED"}:
                message = data.get("errorMessage") or data.get("message") or data.get("failedReason") or data
                raise CometAPIError(f"RunningHub 任务失败：{str(message)[:300]}")
        raise CometAPIError(f"RunningHub 任务等待超时（{timeout} 秒），最后状态：{last_status}")

    def _build_seedance_payload(
        self,
        prompt: str,
        model: str,
        mode: str,
        pil_images: list[Image.Image],
        video_inputs: list,
        audio_inputs: list,
        aspect_ratio: str,
        duration: str,
        resolution: str,
    ) -> tuple[str, dict]:
        safe_mode = normalize_runninghub_mode(model, mode)
        safe_duration = normalize_runninghub_duration(model, duration)
        safe_resolution = normalize_runninghub_resolution(model, resolution)
        safe_ratio = normalize_runninghub_aspect_ratio(model, safe_mode, aspect_ratio)
        base = f"/openapi/v2/rhart-video/{model}"
        clean_prompt = str(prompt or "").strip()

        if safe_mode == "文生":
            return f"{base}/text-to-video", {
                "prompt": clean_prompt,
                "resolution": safe_resolution,
                "duration": safe_duration,
                "generateAudio": True,
                "ratio": safe_ratio,
                "webSearch": False,
                "returnLastFrame": False,
            }

        if safe_mode == "首尾帧":
            image_urls = [self._upload_image(image) for image in pil_images[:2]]
            if not image_urls:
                raise CometAPIError("RunningHub 首尾帧模式至少需要 1 张参考图")
            payload = {
                "prompt": clean_prompt or None,
                "resolution": safe_resolution,
                "duration": safe_duration,
                "firstFrameUrl": image_urls[0],
                "generateAudio": True,
                "ratio": safe_ratio,
                "realPersonMode": True,
                "conversionSlots": ["all"],
                "returnLastFrame": False,
            }
            if len(image_urls) > 1:
                payload["lastFrameUrl"] = image_urls[1]
            return f"{base}/image-to-video", payload

        image_urls = [self._upload_image(image) for image in pil_images[:RUNNINGHUB_MAX_IMAGES]]
        video_urls = [self._upload_media_input(item, ".mp4") for item in video_inputs[:RUNNINGHUB_MAX_VIDEOS]]
        audio_urls = [self._upload_media_input(item, ".wav") for item in audio_inputs[:RUNNINGHUB_MAX_AUDIOS]]
        clean_prompt = convert_prompt_asset_mentions(
            clean_prompt,
            image_count=len(image_urls),
            video_count=len(video_urls),
            audio_count=len(audio_urls),
        )
        return f"{base}/multimodal-video", {
            "prompt": clean_prompt,
            "resolution": safe_resolution,
            "duration": safe_duration,
            "imageUrls": image_urls,
            "videoUrls": video_urls,
            "audioUrls": audio_urls,
            "generateAudio": True,
            "ratio": safe_ratio,
            "realPersonMode": True,
            "conversionSlots": ["all"],
            "returnLastFrame": False,
        }

    def _build_google_payload(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        video_inputs: list,
        aspect_ratio: str,
        duration: str,
        resolution: str,
        mode: str,
    ) -> tuple[str, dict]:
        safe_mode = normalize_runninghub_mode(model, mode)
        safe_duration = normalize_runninghub_duration(model, duration)
        safe_resolution = normalize_runninghub_resolution(model, resolution)
        safe_ratio = normalize_runninghub_aspect_ratio(model, safe_mode, aspect_ratio)
        clean_prompt = str(prompt or "").strip()
        base = "/openapi/v2/gemini-omni-flash"

        if safe_mode == "文生":
            if not clean_prompt:
                raise CometAPIError("RunningHub Gemini Omni Flash 文生模式需要提示词。")
            return f"{base}/text-to-video", {
                "prompt": clean_prompt,
                "duration": safe_duration,
                "resolution": safe_resolution,
                "aspectRatio": safe_ratio,
            }

        if safe_mode == "视频编辑":
            if not video_inputs:
                raise CometAPIError("RunningHub Gemini Omni Flash 视频编辑模式需要 1 个参考视频。")
            image_urls = [self._upload_image(image) for image in pil_images[:3]]
            if len(image_urls) not in {0, 1, 3}:
                raise CometAPIError("RunningHub Gemini Omni Flash 视频编辑模式只支持 0、1 或 3 张参考图。")
            if not clean_prompt:
                raise CometAPIError("RunningHub Gemini Omni Flash 视频编辑模式需要提示词。")
            payload = {
                "prompt": clean_prompt,
                "resolution": safe_resolution,
                "aspectRatio": safe_ratio,
                "videoUrl": self._upload_media_input(video_inputs[0], ".mp4"),
            }
            if image_urls:
                payload["imageUrls"] = image_urls
            return f"{base}/video-edit", payload

        image_urls = [self._upload_image(image) for image in pil_images[:3]]
        if len(image_urls) not in {1, 3}:
            raise CometAPIError("RunningHub Gemini Omni Flash 多参模式只支持 1 张或 3 张参考图。")
        if not clean_prompt:
            raise CometAPIError("RunningHub Gemini Omni Flash 多参模式需要提示词。")
        return f"{base}/image-to-video", {
            "prompt": clean_prompt,
            "imageUrls": image_urls,
            "duration": safe_duration,
            "resolution": safe_resolution,
            "aspectRatio": safe_ratio,
        }

    def _build_grok_payload(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        duration: str,
        resolution: str,
        mode: str,
    ) -> tuple[str, dict]:
        safe_mode = normalize_runninghub_mode(model, mode)
        safe_duration = int(normalize_runninghub_duration(model, duration))
        safe_resolution = normalize_runninghub_resolution(model, resolution)
        safe_ratio = normalize_runninghub_aspect_ratio(model, safe_mode, aspect_ratio)
        clean_prompt = str(prompt or "").strip()
        if len(clean_prompt) < 5:
            raise CometAPIError("RunningHub Grok Video 1.5 提示词至少需要 5 个字符。")
        base = "/openapi/v2/rhart-video-g"
        payload = {
            "prompt": clean_prompt,
            "aspectRatio": safe_ratio,
            "resolution": safe_resolution,
            "duration": safe_duration,
        }
        if safe_mode == "文生":
            return f"{base}/text-to-video", payload

        if not pil_images:
            raise CometAPIError("RunningHub Grok Video 1.5 图生模式至少需要 1 张参考图。")
        payload["imageUrls"] = [self._upload_image(image) for image in pil_images[:RUNNINGHUB_GROK_MAX_IMAGES]]
        return f"{base}/image-to-video", payload

    def _build_happyhorse_payload(
        self,
        prompt: str,
        mode: str,
        pil_images: list[Image.Image],
        video_inputs: list,
        aspect_ratio: str,
        duration: str,
        resolution: str,
    ) -> tuple[str, dict]:
        safe_mode = normalize_runninghub_mode("happyhorse-1.0", mode)
        safe_duration = normalize_runninghub_duration("happyhorse-1.0", duration)
        safe_resolution = normalize_runninghub_resolution("happyhorse-1.0", resolution)
        safe_aspect = normalize_runninghub_aspect_ratio("happyhorse-1.0", safe_mode, aspect_ratio)
        clean_prompt = str(prompt or "").strip()
        base = "/openapi/v2/alibaba/happyhorse-1.0"

        if safe_mode == "文生":
            return f"{base}/text-to-video", {
                "prompt": clean_prompt,
                "resolution": safe_resolution,
                "duration": safe_duration,
                "aspectRatio": safe_aspect,
                "seed": None,
            }
        if safe_mode == "图生":
            if not pil_images:
                raise CometAPIError("RunningHub 图生模式至少需要 1 张参考图")
            return f"{base}/image-to-video", {
                "imageUrl": self._upload_image(pil_images[0]),
                "prompt": clean_prompt or None,
                "resolution": safe_resolution,
                "duration": safe_duration,
                "seed": None,
            }
        if safe_mode == "多图参考":
            image_urls = [self._upload_image(image) for image in pil_images[:RUNNINGHUB_MAX_IMAGES]]
            if not image_urls:
                raise CometAPIError("RunningHub 多图参考模式至少需要 1 张参考图")
            return f"{base}/reference-to-video", {
                "prompt": clean_prompt,
                "imageUrls": image_urls,
                "resolution": safe_resolution,
                "aspectRatio": safe_aspect,
                "duration": safe_duration,
                "seed": None,
            }

        if not video_inputs:
            raise CometAPIError("RunningHub 视频编辑模式需要 1 个参考视频")
        payload = {
            "videoUrl": self._upload_media_input(video_inputs[0], ".mp4"),
            "imageUrls": [self._upload_image(image) for image in pil_images[:5]],
            "prompt": clean_prompt,
            "resolution": safe_resolution,
            "audioSetting": "origin",
            "seed": None,
        }
        return f"{base}/video-edit", payload

    def generate_video(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        video_inputs: list,
        audio_inputs: list,
        aspect_ratio: str,
        duration: str,
        resolution: str,
        mode: str,
    ) -> tuple[str, str, str]:
        raw_model = canonical_runninghub_video_model(model).lower()
        if is_runninghub_google_model(raw_model):
            path, payload = self._build_google_payload(
                prompt, raw_model, pil_images, video_inputs, aspect_ratio, duration, resolution, mode
            )
        elif is_runninghub_grok_model(raw_model):
            path, payload = self._build_grok_payload(
                prompt, raw_model, pil_images, aspect_ratio, duration, resolution, mode
            )
        elif is_runninghub_seedance_model(raw_model):
            path, payload = self._build_seedance_payload(
                prompt, raw_model, mode, pil_images, video_inputs, audio_inputs, aspect_ratio, duration, resolution
            )
        elif is_runninghub_happyhorse_model(raw_model):
            path, payload = self._build_happyhorse_payload(prompt, mode, pil_images, video_inputs, aspect_ratio, duration, resolution)
        else:
            raise CometAPIError(f"不支持的 RunningHub 视频模型：{model}")

        task_id = self._submit_task(path, payload)
        video_url = self._wait_for_video_url(task_id)
        local_path = download_video_asset(video_url, prefix="CometAPIRunningHub")
        return local_path, video_url, task_id


def apimart_video_mode_choices(model: str) -> list[str]:
    raw = canonical_apimart_video_model(model)
    raw_lower = raw.lower()
    if raw_lower in APIMART_SEEDANCE_VIDEO_MODELS:
        return list(APIMART_SEEDANCE_VIDEO_MODES)
    if raw_lower in APIMART_GROK_VIDEO_MODELS:
        return list(APIMART_GROK_VIDEO_MODES)
    if raw_lower in APIMART_SORA_VIDEO_MODELS:
        return list(APIMART_SORA_VIDEO_MODES)
    if raw == "Omni-Flash-Ext":
        return list(APIMART_OMNI_FLASH_VIDEO_MODES)
    if raw_lower == "veo3.1-lite":
        return ["文生"]
    if raw_lower == "veo3.1-quality":
        return ["文生", "首尾帧"]
    if raw_lower in {item.lower() for item in APIMART_VEO_VIDEO_MODELS}:
        return list(APIMART_VEO_VIDEO_MODES)
    if raw in APIMART_HAILUO_VIDEO_MODELS:
        return list(APIMART_HAILUO_VIDEO_MODES)
    if raw_lower in APIMART_HAPPYHORSE_VIDEO_MODELS:
        return list(APIMART_HAPPYHORSE_VIDEO_MODES)
    if raw_lower in APIMART_KLING_OMNI_VIDEO_MODELS:
        return list(APIMART_KLING_OMNI_VIDEO_MODES)
    if raw_lower in APIMART_KLING_VIDEO_MODELS:
        return ["文生", "图生", "首尾帧"]
    if raw_lower in {"viduq3", "viduq3-mix"}:
        return ["参考"]
    if raw_lower in APIMART_VIDU_VIDEO_MODELS:
        return ["文生", "图生", "首尾帧"]
    if raw_lower == "wan2.7":
        return ["文生", "图生", "首尾帧", "视频参考"]
    if raw_lower == "wan2.7-r2v":
        return ["多图参考", "视频参考"]
    if raw_lower == "wan2.7-videoedit":
        return ["视频编辑"]
    return ["文生"]


def normalize_apimart_video_mode(model: str, mode: str) -> str:
    choices = apimart_video_mode_choices(model)
    raw = str(mode or "").strip()
    return raw if raw in choices else choices[0]


def apimart_duration_choices(model: str, mode: str = "", resolution: str = "") -> list[str]:
    raw = canonical_apimart_video_model(model)
    raw_lower = raw.lower()
    safe_mode = normalize_apimart_video_mode(raw, mode)
    if raw_lower in APIMART_SEEDANCE_VIDEO_MODELS:
        return list(APIMART_SEEDANCE_DURATIONS)
    if raw_lower in APIMART_GROK_VIDEO_MODELS:
        return list(APIMART_GROK_DURATIONS)
    if raw_lower in APIMART_SORA_VIDEO_MODELS:
        return list(APIMART_SORA_DURATIONS)
    if raw == "Omni-Flash-Ext":
        return [] if safe_mode == "视频参考" else list(APIMART_OMNI_FLASH_DURATIONS)
    if raw_lower in {item.lower() for item in APIMART_VEO_VIDEO_MODELS}:
        return list(APIMART_VEO_DURATIONS)
    if raw in APIMART_HAILUO_VIDEO_MODELS:
        if str(resolution or "").strip().lower() == "1080p":
            return ["6"]
        return list(APIMART_HAILUO_DURATIONS)
    if raw_lower in APIMART_HAPPYHORSE_VIDEO_MODELS:
        return [] if safe_mode == "视频编辑" else list(APIMART_HAPPYHORSE_DURATIONS)
    if raw_lower == "kling-video-o1":
        return list(APIMART_KLING_O1_DURATIONS)
    if raw_lower in APIMART_KLING_VIDEO_MODELS:
        return list(APIMART_KLING_V3_DURATIONS)
    if raw_lower == "viduq3":
        return [str(value) for value in range(3, 17)]
    if raw_lower in APIMART_VIDU_VIDEO_MODELS:
        return list(APIMART_VIDU_DURATIONS)
    if raw_lower == "wan2.7-videoedit":
        return list(APIMART_WAN_VIDEO_EDIT_DURATIONS)
    if raw_lower == "wan2.7-r2v" and safe_mode == "视频参考":
        return [str(value) for value in range(2, 11)]
    if raw_lower in APIMART_WAN_VIDEO_MODELS:
        return list(APIMART_WAN_DURATIONS)
    return ["5"]


def normalize_apimart_duration(model: str, mode: str, duration: str, resolution: str = "") -> int:
    choices = apimart_duration_choices(model, mode, resolution)
    if not choices:
        return 5
    raw = str(duration or "").strip()
    if raw in choices:
        return int(raw)
    fallback = "8" if "8" in choices else "6" if "6" in choices else "5" if "5" in choices else choices[0]
    return int(fallback)


def apimart_resolution_choices(model: str, mode: str = "", duration: str = "") -> list[str]:
    raw = canonical_apimart_video_model(model)
    raw_lower = raw.lower()
    if raw_lower == "doubao-seedance-2.0-fast":
        return list(APIMART_SEEDANCE_FAST_RESOLUTIONS)
    if raw_lower in APIMART_SEEDANCE_VIDEO_MODELS:
        return list(APIMART_SEEDANCE_RESOLUTIONS)
    if raw_lower in APIMART_GROK_VIDEO_MODELS:
        return list(APIMART_GROK_RESOLUTIONS)
    if raw_lower == "sora-2":
        return ["720p"]
    if raw_lower == "sora-2-pro":
        return list(APIMART_SORA_RESOLUTIONS)
    if raw == "Omni-Flash-Ext":
        return list(APIMART_OMNI_FLASH_RESOLUTIONS)
    if raw_lower == "veo3.1-lite":
        return []
    if raw_lower in {item.lower() for item in APIMART_VEO_VIDEO_MODELS}:
        return list(APIMART_VEO_RESOLUTIONS)
    if raw in APIMART_HAILUO_VIDEO_MODELS:
        if str(duration or "").strip() == "10":
            return ["768p"]
        return list(APIMART_HAILUO_RESOLUTIONS)
    if raw_lower in APIMART_HAPPYHORSE_VIDEO_MODELS:
        return list(APIMART_HAPPYHORSE_RESOLUTIONS)
    if raw_lower == "kling-video-o1":
        return ["std", "pro"]
    if raw_lower in APIMART_KLING_VIDEO_MODELS:
        return list(APIMART_KLING_QUALITY_MODES)
    if raw_lower == "viduq3-mix":
        return ["720p", "1080p"]
    if raw_lower in APIMART_VIDU_VIDEO_MODELS:
        return list(APIMART_VIDU_RESOLUTIONS)
    if raw_lower in APIMART_WAN_VIDEO_MODELS:
        return list(APIMART_WAN_RESOLUTIONS)
    return []


def normalize_apimart_resolution(model: str, mode: str, duration: str, resolution: str) -> str:
    choices = apimart_resolution_choices(model, mode, duration)
    if not choices:
        return ""
    raw = str(resolution or "").strip()
    raw_lower = raw.lower()
    for choice in choices:
        if raw_lower == str(choice).lower():
            return choice
    for preferred in ("1080P", "1080p", "720p", "720P", "std"):
        for choice in choices:
            if preferred.lower() == str(choice).lower():
                return choice
    return choices[0]


def apimart_aspect_choices(model: str, mode: str = "") -> list[str]:
    raw = canonical_apimart_video_model(model)
    raw_lower = raw.lower()
    safe_mode = normalize_apimart_video_mode(raw, mode)
    if raw_lower in APIMART_SEEDANCE_VIDEO_MODELS:
        return list(APIMART_SEEDANCE_ASPECT_RATIOS)
    if raw_lower in APIMART_GROK_VIDEO_MODELS:
        return list(APIMART_GROK_ASPECT_RATIOS)
    if raw_lower in APIMART_SORA_VIDEO_MODELS:
        return [] if safe_mode == "图生" else ["16:9", "9:16"]
    if raw == "Omni-Flash-Ext":
        return ["16:9", "9:16"]
    if raw_lower in {item.lower() for item in APIMART_VEO_VIDEO_MODELS}:
        return ["16:9", "9:16"]
    if raw in APIMART_HAILUO_VIDEO_MODELS:
        return []
    if raw_lower in APIMART_HAPPYHORSE_VIDEO_MODELS:
        return [] if safe_mode in {"图生", "视频编辑"} else list(APIMART_VIDEO_ASPECT_RATIOS)
    if raw_lower in APIMART_KLING_VIDEO_MODELS:
        return list(APIMART_KLING_ASPECT_RATIOS)
    if raw_lower in {"viduq3-pro", "viduq3-turbo"} and safe_mode in {"图生", "首尾帧"}:
        return []
    if raw_lower in APIMART_VIDU_VIDEO_MODELS:
        return list(APIMART_VIDEO_ASPECT_RATIOS)
    if raw_lower == "wan2.7" and safe_mode in {"图生", "首尾帧", "视频参考"}:
        return []
    if raw_lower == "wan2.7-r2v" and safe_mode == "视频参考":
        return []
    if raw_lower in APIMART_WAN_VIDEO_MODELS:
        return list(APIMART_VIDEO_ASPECT_RATIOS)
    return ["16:9", "9:16"]


def normalize_apimart_aspect_ratio(model: str, mode: str, aspect_ratio: str) -> str:
    choices = apimart_aspect_choices(model, mode)
    if not choices:
        return ""
    raw = str(aspect_ratio or "").strip()
    if raw in choices:
        return raw
    return "adaptive" if "adaptive" in choices else "16:9" if "16:9" in choices else choices[0]


def apimart_video_max_images(model: str, mode: str = "") -> int:
    raw = canonical_apimart_video_model(model)
    raw_lower = raw.lower()
    safe_mode = normalize_apimart_video_mode(raw, mode)
    if safe_mode == "文生":
        return 0
    if raw_lower in APIMART_SEEDANCE_VIDEO_MODELS:
        return 2 if safe_mode == "首尾帧" else APIMART_MAX_IMAGES
    if raw_lower in APIMART_GROK_VIDEO_MODELS:
        return 7
    if raw_lower in APIMART_SORA_VIDEO_MODELS:
        return 1
    if raw == "Omni-Flash-Ext":
        return 3 if safe_mode in {"多参", "视频参考"} else 0
    if raw_lower == "veo3.1-lite":
        return 0
    if raw_lower in {item.lower() for item in APIMART_VEO_VIDEO_MODELS}:
        return 2 if safe_mode == "首尾帧" else 3
    if raw in APIMART_HAILUO_VIDEO_MODELS:
        return 1
    if raw_lower in APIMART_HAPPYHORSE_VIDEO_MODELS:
        return 5 if safe_mode == "视频编辑" else 1 if safe_mode == "图生" else APIMART_MAX_IMAGES
    if raw_lower == "kling-v3":
        return 2 if safe_mode == "首尾帧" else 1
    if raw_lower == "kling-video-o1":
        return 1 if safe_mode == "视频参考" else 2
    if raw_lower == "kling-v3-omni":
        return 1 if safe_mode == "视频参考" else APIMART_MAX_IMAGES
    if raw_lower in {"viduq3", "viduq3-mix"}:
        return 7
    if raw_lower in APIMART_VIDU_VIDEO_MODELS:
        return 2 if safe_mode == "首尾帧" else 1 if safe_mode == "图生" else 0
    if raw_lower == "wan2.7-videoedit":
        return 4
    if raw_lower == "wan2.7-r2v":
        return 5
    if raw_lower == "wan2.7":
        return 2 if safe_mode == "首尾帧" else 1 if safe_mode in {"图生", "视频参考"} else 0
    return APIMART_MAX_IMAGES


def apimart_video_max_videos(model: str, mode: str = "") -> int:
    raw = canonical_apimart_video_model(model)
    raw_lower = raw.lower()
    safe_mode = normalize_apimart_video_mode(raw, mode)
    if raw_lower in APIMART_SEEDANCE_VIDEO_MODELS and safe_mode == "全能参考":
        return APIMART_MAX_VIDEOS
    if raw_lower in APIMART_HAPPYHORSE_VIDEO_MODELS and safe_mode == "视频编辑":
        return 1
    if raw_lower in APIMART_KLING_OMNI_VIDEO_MODELS and safe_mode == "视频参考":
        return 1
    if raw == "Omni-Flash-Ext" and safe_mode == "视频参考":
        return 1
    if raw_lower == "wan2.7" and safe_mode == "视频参考":
        return 1
    if raw_lower == "wan2.7-r2v" and safe_mode == "视频参考":
        return 5
    if raw_lower == "wan2.7-videoedit":
        return 1
    return 0


def apimart_video_max_audios(model: str, mode: str = "") -> int:
    raw = canonical_apimart_video_model(model)
    raw_lower = raw.lower()
    safe_mode = normalize_apimart_video_mode(raw, mode)
    if raw_lower in APIMART_SEEDANCE_VIDEO_MODELS and safe_mode == "全能参考":
        return APIMART_MAX_AUDIOS
    if raw_lower == "wan2.7" and safe_mode in {"文生", "图生"}:
        return 1
    if raw_lower == "wan2.7-r2v" and safe_mode == "多图参考":
        return 1
    return 0


def apimart_video_media_limits(model: str, mode: str = "") -> dict[str, int]:
    return {
        "image": apimart_video_max_images(model, mode),
        "video": apimart_video_max_videos(model, mode),
        "audio": apimart_video_max_audios(model, mode),
    }


def apimart_allowed_media_types(model: str, mode: str = "") -> set[str]:
    return {media_type for media_type, limit in apimart_video_media_limits(model, mode).items() if limit > 0}


class APIMartVideoAPI:
    host = "https://api.apimart.ai"
    upload_host = "https://apimart.ai"

    def __init__(self, api_key: str, media_upload_api_key: str = "", grsai_media_upload_api_key: str = ""):
        if not api_key:
            raise CometAPIError("缺少 Apimart API Key，请先在设置中心填写。")
        self.api_key = api_key
        self.media_upload_api_key = str(media_upload_api_key or "").strip()
        self.grsai_media_upload_api_key = str(grsai_media_upload_api_key or "").strip()
        self._media_uploaders = None

    def _headers(self, json_content: bool = True) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers

    def _raise_for_status(self, response: requests.Response, label: str = "Apimart") -> None:
        if response.status_code < 400:
            return
        text = (response.text or "").strip()
        detail = ""
        try:
            data = response.json()
            error = data.get("error") if isinstance(data, dict) else None
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("code") or "")
            elif isinstance(error, str):
                detail = error
            if not detail and isinstance(data, dict):
                detail = str(data.get("message") or "")
        except Exception:
            pass
        if not detail:
            detail = text[:500]
        suffix = f"：{redact_sensitive_text(detail)}" if detail else ""
        raise CometAPIError(f"{label} 请求失败：HTTP {response.status_code} {response.reason}{suffix}")

    def _post_json(self, path: str, payload: dict, timeout: int = 90) -> dict:
        response = requests.post(
            f"{self.host}{path}",
            headers=self._headers(True),
            json=payload,
            timeout=(15, timeout),
        )
        self._raise_for_status(response)
        return response.json()

    def _get_json(self, path: str, timeout: int = 30) -> dict:
        response = requests.get(
            f"{self.host}{path}",
            headers=self._headers(False),
            timeout=(15, timeout),
        )
        self._raise_for_status(response)
        return response.json()

    def _upload_image(self, pil_image: Image.Image) -> str:
        buffer = BytesIO()
        safe_pil_to_rgb(pil_image).save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        response = requests.post(
            f"{self.host}/v1/uploads/images",
            headers=self._headers(False),
            files={"file": ("comet_reference.jpg", buffer, "image/jpeg")},
            timeout=(15, 180),
        )
        self._raise_for_status(response, "Apimart 图片上传")
        data = response.json()
        url = str(data.get("url") or "").strip()
        if not url:
            raise CometAPIError(f"Apimart 图片上传没有返回 URL：{str(data)[:300]}")
        return url

    def _image_urls(self, pil_images: list[Image.Image], limit: int) -> list[str]:
        return [self._upload_image(image) for image in pil_images[:limit]]

    def _presign_upload_file(self, path: str, label: str, suffix: str = ".mp4") -> str:
        if not path or not os.path.exists(path):
            raise CometAPIError(f"Apimart {label}文件不存在或无法读取。")

        path_ext = os.path.splitext(path)[1].lower().lstrip(".")
        fallback_ext = str(suffix or ".mp4").lower().lstrip(".")
        ext = path_ext or fallback_ext or "mp4"
        filename = os.path.basename(path) or f"comet-reference.{ext}"
        preferred_content_types = {
            "mp4": "video/mp4",
            "mov": "video/quicktime",
            "webm": "video/webm",
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "m4a": "audio/mp4",
            "aac": "audio/aac",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
        }
        content_type = preferred_content_types.get(ext) or mimetypes.guess_type(filename)[0] or mimetypes.guess_type(f"file.{ext}")[0] or "application/octet-stream"

        try:
            token_res = requests.post(
                f"{self.upload_host}/api/upload/presign",
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                json={"contentType": content_type, "fileExtension": ext, "permanent": False},
                timeout=(15, 60),
            )
            self._raise_for_status(token_res, f"Apimart {label}上传凭证")
            token_data = token_res.json()
        except CometAPIError:
            raise
        except Exception as exc:
            raise CometAPIError(f"Apimart {label}上传凭证获取失败：{redact_sensitive_text(exc)}") from exc

        presigned_url = str(token_data.get("presignedUrl") or token_data.get("uploadUrl") or token_data.get("url") or "").strip()
        cdn_url = str(token_data.get("cdnUrl") or token_data.get("fileUrl") or token_data.get("publicUrl") or "").strip()
        if not presigned_url or not cdn_url:
            raise CometAPIError(f"Apimart {label}上传凭证缺少 URL：{redact_sensitive_text(str(token_data)[:300])}")

        try:
            with open(path, "rb") as media_file:
                upload_res = requests.put(
                    presigned_url,
                    headers={"Content-Type": content_type},
                    data=media_file,
                    timeout=(20, 600),
            )
            upload_res.raise_for_status()
        except Exception as exc:
            raise CometAPIError(f"Apimart {label}上传失败：{redact_sensitive_text(exc)}") from exc

        if not cdn_url.lower().startswith(("http://", "https://")):
            raise CometAPIError(f"Apimart {label}上传返回了无效 URL：{redact_sensitive_text(cdn_url)}")
        return cdn_url

    def _upload_presign_media_input(self, media_input, label: str, suffix: str) -> str:
        path, should_delete = _media_input_to_path(media_input, suffix)
        if not path:
            raise CometAPIError(f"不支持这个 Apimart {label}输入，请检查连接的素材类型。")
        try:
            return self._presign_upload_file(path, label, suffix)
        finally:
            if should_delete:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    def _get_media_uploaders(self) -> list[tuple[str, object]]:
        if self._media_uploaders is not None:
            return self._media_uploaders
        uploaders = []
        runninghub_key = self.media_upload_api_key or get_channel_api_key("", "runninghub", "", "video")
        if runninghub_key:
            uploaders.append(("RunningHub", RunningHubVideoAPI(runninghub_key)))
        grsai_key = self.grsai_media_upload_api_key or get_channel_api_key("", "grsai", "", "image")
        if grsai_key:
            uploaders.append(("grsai", GrsaiMediaUploadAPI(grsai_key)))
        if not uploaders:
            raise CometAPIError("Apimart 素材需要公网 URL，请在设置中心填写 RunningHub 或 grsai 的 API Key 以借用上传服务。")
        self._media_uploaders = uploaders
        return uploaders

    def _upload_media_input(self, media_input, label: str, suffix: str) -> str:
        if str(label or "") in {"视频", "音频"} or str(suffix or "").lower() in {".mp4", ".mov", ".webm", ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}:
            return self._upload_presign_media_input(media_input, label, suffix)

        errors = []
        for uploader_label, uploader in self._get_media_uploaders():
            try:
                return uploader._upload_media_input(media_input, suffix)
            except CometAPIError as exc:
                errors.append(f"{uploader_label}: {exc}")
            except Exception as exc:
                errors.append(f"{uploader_label}: {exc}")
        detail = "；".join(errors) if errors else "没有可用上传渠道"
        raise CometAPIError(f"Apimart {label}素材上传失败：{detail}")

    def _media_url(self, media_input, label: str, suffix: str) -> str:
        if media_input is None:
            return ""
        if isinstance(media_input, str) and media_input.strip().lower().startswith(("http://", "https://")):
            return media_input.strip()
        if isinstance(media_input, dict):
            for key in ("url", "video_url", "audio_url", "src"):
                value = str(media_input.get(key) or "").strip()
                if value.lower().startswith(("http://", "https://")):
                    return value
        for attr in ("video_url", "audio_url", "url", "src"):
            value = str(getattr(media_input, attr, "") or "").strip()
            if value.lower().startswith(("http://", "https://")):
                return value
        return self._upload_media_input(media_input, label, suffix)

    def _submit_task(self, payload: dict) -> str:
        data = self._post_json("/v1/videos/generations", payload)
        candidates = [data]
        body = data.get("data") if isinstance(data, dict) else None
        if isinstance(body, dict):
            candidates.append(body)
        elif isinstance(body, list):
            candidates.extend(item for item in body if isinstance(item, dict))
        for item in candidates:
            if not isinstance(item, dict):
                continue
            task_id = item.get("task_id") or item.get("taskId") or item.get("id")
            if task_id:
                return str(task_id)
        raise CometAPIError(f"Apimart API 没有返回视频任务 ID：{str(data)[:300]}")

    def _normalize_status(self, data: dict) -> str:
        body = data.get("data") if isinstance(data.get("data"), dict) else data
        raw = str(body.get("status") or data.get("status") or "processing").strip().lower()
        if raw in {"completed", "success", "succeeded", "complete", "finished", "done"}:
            return "success"
        if raw in {"failed", "failure", "fail", "error", "cancelled", "canceled"}:
            return "failed"
        return raw or "processing"

    def _collect_video_urls(self, value, context: str = "", urls: list[tuple[int, str]] | None = None) -> list[tuple[int, str]]:
        if urls is None:
            urls = []
        if isinstance(value, dict):
            for key, child in value.items():
                self._collect_video_urls(child, f"{context}.{key}", urls)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._collect_video_urls(child, f"{context}[{index}]", urls)
        elif isinstance(value, str) and value.startswith(("http://", "https://")):
            lower_context = context.lower()
            lower_url = value.lower()
            score = 0
            if any(token in lower_url for token in (".mp4", ".mov", ".webm", ".m3u8")):
                score += 100
            if any(token in lower_context for token in ("video", "videos", "result", "url")):
                score += 30
            if any(token in lower_context for token in ("thumbnail", "image", "cover")):
                score -= 40
            urls.append((score, value))
        return urls

    def _extract_video_url(self, data: dict) -> str:
        urls = self._collect_video_urls(data)
        urls.sort(key=lambda item: item[0], reverse=True)
        return urls[0][1] if urls and urls[0][0] > 0 else ""

    def _query_task(self, task_id: str) -> tuple[str, str, dict]:
        data = self._get_json(f"/v1/tasks/{requests.utils.quote(str(task_id), safe='')}?language=zh")
        return self._normalize_status(data), self._extract_video_url(data), data

    def _wait_for_video_url(self, task_id: str, timeout: int = 1800, interval: int = 5) -> str:
        started = time.time()
        last_status = "processing"
        transient_failures = 0
        max_transient_failures = 12
        while time.time() - started < timeout:
            time.sleep(interval)
            try:
                status, video_url, data = self._query_task(task_id)
                transient_failures = 0
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                transient_failures += 1
                if transient_failures >= max_transient_failures:
                    raise CometAPIError(f"Apimart 视频任务状态查询连续失败：{exc}") from exc
                print(f"[CometAPI] Apimart 视频任务状态查询失败，继续重试（{transient_failures}/{max_transient_failures}）：{exc}")
                continue
            last_status = status
            if status == "success":
                if video_url:
                    return video_url
                raise CometAPIError(f"Apimart 视频任务已完成，但没有返回视频地址：{str(data)[:300]}")
            if status == "failed":
                body = data.get("data") if isinstance(data.get("data"), dict) else data
                error = body.get("error") if isinstance(body, dict) else None
                if isinstance(error, dict):
                    error = error.get("message") or error.get("code") or error
                raise CometAPIError(f"Apimart 视频任务失败：{str(error or data)[:300]}")
        raise CometAPIError(f"Apimart 视频任务等待超时（{timeout} 秒），最后状态：{last_status}")

    def _build_seedance_payload(self, model: str, prompt: str, pil_images: list[Image.Image], video_inputs: list, audio_inputs: list, aspect_ratio: str, duration: str, resolution: str, mode: str) -> dict:
        safe_mode = normalize_apimart_video_mode(model, mode)
        safe_duration = normalize_apimart_duration(model, safe_mode, duration, resolution)
        safe_resolution = normalize_apimart_resolution(model, safe_mode, str(safe_duration), resolution)
        safe_ratio = normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio)
        clean_prompt = str(prompt or "").strip()
        payload = {
            "model": model,
            "prompt": add_prompt_variation(clean_prompt, 0) if clean_prompt else "",
            "resolution": safe_resolution,
            "size": safe_ratio,
            "duration": safe_duration,
            "generate_audio": True,
            "return_last_frame": False,
        }
        if safe_mode == "首尾帧":
            image_urls = self._image_urls(pil_images, 2)
            if not image_urls:
                raise CometAPIError("Apimart Seedance 首尾帧模式至少需要 1 张参考图。")
            payload["image_with_roles"] = [
                {"url": url, "role": "first_frame" if index == 0 else "last_frame"}
                for index, url in enumerate(image_urls)
            ]
        elif safe_mode == "全能参考":
            image_urls = self._image_urls(pil_images, APIMART_MAX_IMAGES)
            video_urls = [self._media_url(item, "视频", ".mp4") for item in video_inputs[:APIMART_MAX_VIDEOS]]
            audio_urls = [self._media_url(item, "音频", ".wav") for item in audio_inputs[:APIMART_MAX_AUDIOS]]
            if image_urls:
                payload["image_urls"] = image_urls
            if video_urls:
                payload["video_urls"] = video_urls
            if audio_urls:
                payload["audio_urls"] = audio_urls
            if clean_prompt or image_urls or video_urls or audio_urls:
                payload["prompt"] = convert_prompt_asset_mentions(
                    payload["prompt"],
                    image_count=len(image_urls),
                    video_count=len(video_urls),
                    audio_count=len(audio_urls),
                )
        if not payload.get("prompt") and not any(payload.get(key) for key in ("image_urls", "image_with_roles", "video_urls", "audio_urls")):
            raise CometAPIError("Apimart Seedance 需要提示词或参考素材。")
        return {key: value for key, value in payload.items() if value not in ("", None, [])}

    def _build_grok_payload(self, model: str, prompt: str, pil_images: list[Image.Image], aspect_ratio: str, duration: str, resolution: str, mode: str) -> dict:
        safe_mode = normalize_apimart_video_mode(model, mode)
        clean_prompt = str(prompt or "").strip()
        if not clean_prompt:
            raise CometAPIError("Apimart Grok 需要提示词。")
        payload = {
            "model": model,
            "prompt": add_prompt_variation(clean_prompt, 0),
            "size": normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio),
            "duration": normalize_apimart_duration(model, safe_mode, duration, resolution),
            "quality": normalize_apimart_resolution(model, safe_mode, duration, resolution),
        }
        if safe_mode == "多参":
            image_urls = self._image_urls(pil_images, 7)
            if not image_urls:
                raise CometAPIError("Apimart Grok 多参模式至少需要 1 张参考图。")
            payload["image_urls"] = image_urls
        return payload

    def _build_sora_payload(self, model: str, prompt: str, pil_images: list[Image.Image], aspect_ratio: str, duration: str, resolution: str, mode: str) -> dict:
        safe_mode = normalize_apimart_video_mode(model, mode)
        clean_prompt = str(prompt or "").strip()
        if not clean_prompt:
            raise CometAPIError("Apimart Sora 需要提示词。")
        payload = {
            "model": model,
            "prompt": add_prompt_variation(clean_prompt, 0),
            "duration": normalize_apimart_duration(model, safe_mode, duration, resolution),
            "resolution": normalize_apimart_resolution(model, safe_mode, duration, resolution),
        }
        if safe_mode == "图生":
            image_urls = self._image_urls(pil_images, 1)
            if not image_urls:
                raise CometAPIError("Apimart Sora 图生模式需要 1 张参考图。")
            payload["image_urls"] = image_urls
        else:
            payload["aspect_ratio"] = normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio)
        return payload

    def _build_veo_payload(
        self,
        model: str,
        prompt: str,
        pil_images: list[Image.Image],
        video_inputs: list,
        aspect_ratio: str,
        duration: str,
        resolution: str,
        mode: str,
    ) -> dict:
        safe_mode = normalize_apimart_video_mode(model, mode)
        clean_prompt = str(prompt or "").strip()
        if not clean_prompt:
            raise CometAPIError("Apimart VEO/Omni-Flash 需要提示词。")
        if model == "Omni-Flash-Ext":
            image_urls = self._image_urls(pil_images, 3) if safe_mode in {"多参", "视频参考"} else []
            if safe_mode in {"多参", "视频参考"}:
                if len(image_urls) == 2:
                    raise CometAPIError("Omni-Flash-Ext 不支持 2 张参考图，请传 1 张或 3 张。")
            if safe_mode == "视频参考":
                if not video_inputs:
                    raise CometAPIError("Omni-Flash-Ext 视频参考模式需要 1 个参考视频。")
                payload = {
                    "model": model,
                    "prompt": add_prompt_variation(clean_prompt, 0),
                    "resolution": normalize_apimart_resolution(model, safe_mode, duration, resolution),
                    "aspect_ratio": normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio),
                    "video_urls": [self._media_url(video_inputs[0], "视频", ".mp4")],
                }
                payload["size"] = payload["aspect_ratio"]
                if image_urls:
                    payload["image_urls"] = image_urls
                return {key: value for key, value in payload.items() if value not in ("", None, [])}
            if safe_mode == "多参":
                if not image_urls:
                    raise CometAPIError("Omni-Flash-Ext 多参模式需要 1 张或 3 张参考图。")
            payload = {
                "model": model,
                "prompt": add_prompt_variation(clean_prompt, 0),
                "duration": normalize_apimart_duration(model, safe_mode, duration, resolution),
                "resolution": normalize_apimart_resolution(model, safe_mode, duration, resolution),
                "aspect_ratio": normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio),
            }
            payload["size"] = payload["aspect_ratio"]
            if image_urls:
                payload["image_urls"] = image_urls
            return payload

        payload = {
            "model": model,
            "prompt": add_prompt_variation(clean_prompt, 0),
            "duration": 8,
            "aspect_ratio": normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio),
        }
        if model != "veo3.1-lite":
            payload["resolution"] = normalize_apimart_resolution(model, safe_mode, duration, resolution)
        image_limit = apimart_video_max_images(model, safe_mode)
        image_urls = self._image_urls(pil_images, image_limit) if image_limit else []
        if image_urls:
            if model == "veo3.1-lite":
                raise CometAPIError("Apimart veo3.1-lite 不支持参考图。")
            if safe_mode == "多参":
                if model == "veo3.1-quality":
                    raise CometAPIError("Apimart veo3.1-quality 不支持多参参考图模式。")
                payload["generation_type"] = "reference"
            elif safe_mode == "首尾帧" and len(image_urls) >= 2:
                payload["generation_type"] = "frame"
            payload["image_urls"] = image_urls
        return {key: value for key, value in payload.items() if value not in ("", None, [])}

    def _build_hailuo_payload(self, model: str, prompt: str, pil_images: list[Image.Image], duration: str, resolution: str, mode: str) -> dict:
        safe_mode = normalize_apimart_video_mode(model, mode)
        safe_duration = normalize_apimart_duration(model, safe_mode, duration, resolution)
        clean_prompt = str(prompt or "").strip()
        if not clean_prompt:
            raise CometAPIError("Apimart Hailuo 需要提示词。")
        payload = {
            "model": model,
            "prompt": add_prompt_variation(clean_prompt, 0),
            "duration": safe_duration,
            "resolution": normalize_apimart_resolution(model, safe_mode, str(safe_duration), resolution),
            "prompt_optimizer": True,
            "fast_pretreatment": False,
            "watermark": False,
        }
        if safe_mode == "图生":
            image_urls = self._image_urls(pil_images, 1)
            if not image_urls:
                raise CometAPIError("Apimart Hailuo 图生模式需要 1 张参考图。")
            payload["first_frame_image"] = image_urls[0]
        return payload

    def _build_happyhorse_payload(self, prompt: str, pil_images: list[Image.Image], video_inputs: list, aspect_ratio: str, duration: str, resolution: str, mode: str) -> dict:
        model = "happyhorse-1.0"
        safe_mode = normalize_apimart_video_mode(model, mode)
        clean_prompt = str(prompt or "").strip()
        safe_resolution = normalize_apimart_resolution(model, safe_mode, duration, resolution)
        safe_duration = normalize_apimart_duration(model, safe_mode, duration, resolution)
        payload = {
            "model": model,
            "prompt": add_prompt_variation(clean_prompt, 0) if clean_prompt else "",
            "resolution": safe_resolution,
            "watermark": False,
        }
        if safe_mode == "文生":
            if not clean_prompt:
                raise CometAPIError("Apimart HappyHorse 文生模式需要提示词。")
            payload.update({"duration": safe_duration, "size": normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio)})
        elif safe_mode == "图生":
            image_urls = self._image_urls(pil_images, 1)
            if not image_urls:
                raise CometAPIError("Apimart HappyHorse 图生模式需要 1 张参考图。")
            payload.update({"duration": safe_duration, "first_frame_image": image_urls[0]})
        elif safe_mode == "多图参考":
            image_urls = self._image_urls(pil_images, APIMART_MAX_IMAGES)
            if not image_urls:
                raise CometAPIError("Apimart HappyHorse 多图参考模式至少需要 1 张参考图。")
            payload.update({"duration": safe_duration, "size": normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio), "image_urls": image_urls})
        else:
            if not video_inputs:
                raise CometAPIError("Apimart HappyHorse 视频编辑模式需要 1 个参考视频。")
            payload["video_url"] = self._media_url(video_inputs[0], "视频", ".mp4")
            payload["audio_setting"] = "origin"
            image_urls = self._image_urls(pil_images, 5)
            if image_urls:
                payload["image_urls"] = image_urls
        return {key: value for key, value in payload.items() if value not in ("", None, [])}

    def _build_kling_payload(self, model: str, prompt: str, pil_images: list[Image.Image], video_inputs: list, aspect_ratio: str, duration: str, resolution: str, mode: str) -> dict:
        safe_mode = normalize_apimart_video_mode(model, mode)
        clean_prompt = str(prompt or "").strip()
        if not clean_prompt:
            raise CometAPIError("Apimart Kling 需要提示词。")
        payload = {
            "model": model,
            "prompt": add_prompt_variation(clean_prompt, 0),
            "mode": normalize_apimart_resolution(model, safe_mode, duration, resolution),
            "duration": normalize_apimart_duration(model, safe_mode, duration, resolution),
            "aspect_ratio": normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio),
            "watermark": False,
        }
        if safe_mode == "视频参考":
            if not video_inputs:
                raise CometAPIError("Apimart Kling 视频参考模式需要 1 个参考视频。")
            payload["video_list"] = [
                {
                    "video_url": self._media_url(video_inputs[0], "视频", ".mp4"),
                    "refer_type": "base",
                    "keep_original_sound": "yes",
                }
            ]
            image_urls = self._image_urls(pil_images, 1)
            if image_urls:
                payload["image_urls"] = image_urls
            payload.pop("audio", None)
            return payload
        if safe_mode == "图生":
            image_urls = self._image_urls(pil_images, 1)
            if not image_urls:
                raise CometAPIError("Apimart Kling 图生模式需要 1 张参考图。")
            payload["image_urls"] = image_urls
        elif safe_mode == "首尾帧":
            image_urls = self._image_urls(pil_images, 2)
            if len(image_urls) < 2:
                raise CometAPIError("Apimart Kling 首尾帧模式需要 2 张参考图。")
            payload["image_urls"] = image_urls
        elif safe_mode == "多参":
            image_urls = self._image_urls(pil_images, apimart_video_max_images(model, safe_mode))
            if not image_urls:
                raise CometAPIError("Apimart Kling 多参模式至少需要 1 张参考图。")
            payload["image_urls"] = image_urls
        if model in {"kling-v3", "kling-v3-omni"}:
            payload["audio"] = True
        return payload

    def _build_vidu_payload(self, model: str, prompt: str, pil_images: list[Image.Image], aspect_ratio: str, duration: str, resolution: str, mode: str) -> dict:
        safe_mode = normalize_apimart_video_mode(model, mode)
        clean_prompt = str(prompt or "").strip()
        payload = {
            "model": model,
            "prompt": add_prompt_variation(clean_prompt, 0) if clean_prompt else "",
            "duration": normalize_apimart_duration(model, safe_mode, duration, resolution),
            "resolution": normalize_apimart_resolution(model, safe_mode, duration, resolution),
            "audio": True,
        }
        if model in {"viduq3", "viduq3-mix"}:
            image_urls = self._image_urls(pil_images, 7)
            if not image_urls:
                raise CometAPIError("Apimart Vidu Q3 参考模式至少需要 1 张参考图。")
            payload["image_urls"] = image_urls
            payload["aspect_ratio"] = normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio)
        elif safe_mode == "文生":
            if not clean_prompt:
                raise CometAPIError("Apimart Vidu 文生模式需要提示词。")
            payload["aspect_ratio"] = normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio)
        elif safe_mode == "图生":
            image_urls = self._image_urls(pil_images, 1)
            if not image_urls:
                raise CometAPIError("Apimart Vidu 图生模式需要 1 张参考图。")
            payload["image_urls"] = image_urls
        else:
            image_urls = self._image_urls(pil_images, 2)
            if len(image_urls) < 2:
                raise CometAPIError("Apimart Vidu 首尾帧模式需要 2 张参考图。")
            payload["image_urls"] = image_urls
        return {key: value for key, value in payload.items() if value not in ("", None, [])}

    def _build_wan_payload(self, model: str, prompt: str, pil_images: list[Image.Image], video_inputs: list, audio_inputs: list, aspect_ratio: str, duration: str, resolution: str, mode: str) -> dict:
        safe_mode = normalize_apimart_video_mode(model, mode)
        clean_prompt = str(prompt or "").strip()
        payload = {
            "model": model,
            "prompt": add_prompt_variation(clean_prompt, 0) if clean_prompt else "",
            "resolution": normalize_apimart_resolution(model, safe_mode, duration, resolution),
            "duration": normalize_apimart_duration(model, safe_mode, duration, resolution),
            "prompt_extend": True,
            "watermark": False,
        }
        if model == "wan2.7-videoedit":
            if not video_inputs:
                raise CometAPIError("Apimart Wan 视频编辑模式需要 1 个参考视频。")
            payload["video_urls"] = [self._media_url(video_inputs[0], "视频", ".mp4")]
            image_urls = self._image_urls(pil_images, 4)
            if image_urls:
                payload["image_urls"] = image_urls
            safe_ratio = normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio)
            if safe_ratio:
                payload["size"] = safe_ratio
            payload["metadata"] = {"audio_setting": "origin"}
            return {key: value for key, value in payload.items() if value not in ("", None, [])}

        if model == "wan2.7-r2v":
            image_urls = self._image_urls(pil_images, 5)
            video_urls = [self._media_url(item, "视频", ".mp4") for item in video_inputs[:5]]
            if safe_mode == "多图参考" and not image_urls:
                raise CometAPIError("Apimart Wan R2V 多图参考模式至少需要 1 张参考图。")
            if safe_mode == "视频参考" and not video_urls:
                raise CometAPIError("Apimart Wan R2V 视频参考模式至少需要 1 个参考视频。")
            image_with_roles = [{"url": url, "role": "reference_image"} for url in image_urls]
            if image_with_roles and audio_inputs:
                image_with_roles[0]["reference_voice"] = self._media_url(audio_inputs[0], "音频", ".wav")
            if image_with_roles:
                payload["image_with_roles"] = image_with_roles
            if video_urls:
                payload["video_urls"] = video_urls
                payload["duration"] = normalize_apimart_duration(model, "视频参考", duration, resolution)
            safe_ratio = normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio)
            if safe_ratio:
                payload["size"] = safe_ratio
            return {key: value for key, value in payload.items() if value not in ("", None, [])}

        if safe_mode == "文生":
            if not clean_prompt:
                raise CometAPIError("Apimart Wan 文生模式需要提示词。")
            payload["size"] = normalize_apimart_aspect_ratio(model, safe_mode, aspect_ratio)
            if audio_inputs:
                payload["audio_url"] = self._media_url(audio_inputs[0], "音频", ".wav")
        elif safe_mode == "图生":
            image_urls = self._image_urls(pil_images, 1)
            if not image_urls:
                raise CometAPIError("Apimart Wan 图生模式需要 1 张参考图。")
            payload["image_urls"] = image_urls
            if audio_inputs:
                payload["audio_url"] = self._media_url(audio_inputs[0], "音频", ".wav")
        elif safe_mode == "首尾帧":
            image_urls = self._image_urls(pil_images, 2)
            if len(image_urls) < 2:
                raise CometAPIError("Apimart Wan 首尾帧模式需要 2 张参考图。")
            payload["image_urls"] = image_urls
        else:
            if not video_inputs:
                raise CometAPIError("Apimart Wan 视频参考模式需要 1 个参考视频。")
            payload["video_urls"] = [self._media_url(video_inputs[0], "视频", ".mp4")]
            image_urls = self._image_urls(pil_images, 1)
            if image_urls:
                payload["image_with_roles"] = [{"url": image_urls[0], "role": "last_frame"}]
        return {key: value for key, value in payload.items() if value not in ("", None, [])}

    def generate_video(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        video_inputs: list,
        audio_inputs: list,
        aspect_ratio: str,
        duration: str,
        resolution: str,
        mode: str,
    ) -> tuple[str, str, str]:
        model = canonical_apimart_video_model(model)
        raw_lower = model.lower()
        if raw_lower in APIMART_SEEDANCE_VIDEO_MODELS:
            payload = self._build_seedance_payload(model, prompt, pil_images, video_inputs, audio_inputs, aspect_ratio, duration, resolution, mode)
        elif raw_lower in APIMART_GROK_VIDEO_MODELS:
            payload = self._build_grok_payload(model, prompt, pil_images, aspect_ratio, duration, resolution, mode)
        elif raw_lower in APIMART_SORA_VIDEO_MODELS:
            payload = self._build_sora_payload(model, prompt, pil_images, aspect_ratio, duration, resolution, mode)
        elif model == "Omni-Flash-Ext" or raw_lower in {item.lower() for item in APIMART_VEO_VIDEO_MODELS}:
            payload = self._build_veo_payload(model, prompt, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
        elif model in APIMART_HAILUO_VIDEO_MODELS:
            payload = self._build_hailuo_payload(model, prompt, pil_images, duration, resolution, mode)
        elif raw_lower in APIMART_HAPPYHORSE_VIDEO_MODELS:
            payload = self._build_happyhorse_payload(prompt, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
        elif raw_lower in APIMART_KLING_VIDEO_MODELS:
            payload = self._build_kling_payload(model, prompt, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
        elif raw_lower in APIMART_VIDU_VIDEO_MODELS:
            payload = self._build_vidu_payload(model, prompt, pil_images, aspect_ratio, duration, resolution, mode)
        elif raw_lower in APIMART_WAN_VIDEO_MODELS:
            payload = self._build_wan_payload(model, prompt, pil_images, video_inputs, audio_inputs, aspect_ratio, duration, resolution, mode)
        else:
            raise CometAPIError(f"不支持的 Apimart 视频模型：{model}")

        logger.debug(f"[Apimart Video Debug] payload={json.dumps(payload, ensure_ascii=False, default=str)[:2000]}")
        task_id = self._submit_task(payload)
        video_url = self._wait_for_video_url(task_id)
        local_path = download_video_asset(video_url, prefix="CometAPIAPIMart")
        return local_path, video_url, task_id


LLM_REQUEST_TIMEOUT_SECONDS = 600


def call_llm_api(api_key: str, model: str, messages: list, api_format: str = "gemini", api_url: str = "") -> tuple[str, str]:
    """调用LLM API，返回(response_text, error_msg)"""
    try:
        if api_format == "gemini":
            return _call_gemini_api(api_key, model, messages, api_url)
        elif api_format == "openai":
            return _call_openai_api(api_key, model, messages, api_url)
        elif api_format == "claude":
            return _call_claude_api(api_key, model, messages, api_url)
        else:
            return "", f"不支持的接口格式：{api_format}"
    except Exception as exc:
        return "", format_error_message(exc)


def _call_gemini_api(api_key: str, model: str, messages: list, api_url: str = "") -> tuple[str, str]:
    """调用Gemini格式API"""
    url = build_api_url(api_url, f"/v1beta/models/{model}:generateContent", "https://grsai.dakka.com.cn") if api_url else f"https://grsai.dakka.com.cn/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload = {
        "contents": messages,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": DEFAULT_LLM_MAX_OUTPUT_TOKENS,
        },
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if e.response else str(e)
        try:
            error_data = json.loads(error_text)
            error_message = error_data.get("error", {}).get("message", error_text)
        except:
            error_message = error_text
        
        # 提供更友好的错误信息
        if "does not support image input" in error_message.lower():
            return "", f"模型 {model} 不支持图片输入，请移除参考图或更换模型"
        return "", f"API错误: {error_message}"
    except Exception as e:
        return "", f"请求失败: {str(e)}"
    
    candidates = data.get("candidates", [])
    if not candidates:
        return "", "no response from API"
    
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if p.get("text"))
    return text.strip(), ""


def _call_openai_api(api_key: str, model: str, messages: list, api_url: str = "") -> tuple[str, str]:
    """调用OpenAI格式API"""
    url = build_api_url(api_url, "/v1/chat/completions", "https://grsai.dakka.com.cn") if api_url else "https://grsai.dakka.com.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    
    choices = data.get("choices", [])
    if not choices:
        return "", "no response from API"
    
    content = choices[0].get("message", {}).get("content", "")
    return content.strip(), ""


def _call_claude_api(api_key: str, model: str, messages: list, api_url: str = "") -> tuple[str, str]:
    """调用Claude格式API"""
    url = build_api_url(api_url, "/v1/messages", "https://grsai.dakka.com.cn") if api_url else "https://grsai.dakka.com.cn/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": DEFAULT_LLM_MAX_OUTPUT_TOKENS,
        "temperature": 0.7,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    
    content = data.get("content", [])
    if not content:
        return "", "no response from API"
    
    text = "".join(block.get("text", "") for block in content if block.get("type") == "text")
    return text.strip(), ""


def _openai_image_size(aspect_ratio: str, image_size: str) -> str:
    size_key = str(image_size or "").upper()
    base = 1024
    if size_key == "2K":
        base = 1536
    elif size_key == "3K":
        base = 2048
    elif size_key in {"4K", "8K"}:
        base = 4096
    ratio = str(aspect_ratio or "1:1").strip()
    if ratio == "16:9":
        return f"{base}x{max(64, int(base * 9 / 16))}"
    if ratio == "9:16":
        return f"{max(64, int(base * 9 / 16))}x{base}"
    if ratio == "4:3":
        return f"{base}x{max(64, int(base * 3 / 4))}"
    if ratio == "3:4":
        return f"{max(64, int(base * 3 / 4))}x{base}"
    return f"{base}x{base}"


def _normalize_custom_image_pixel_size(value: str) -> str:
    text = str(value or "").strip().lower().replace("×", "x").replace("乘", "x")
    match = re.fullmatch(r"(\d{2,5})\s*x\s*(\d{2,5})", text)
    if not match:
        return ""
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 64 or height < 64 or width > 8192 or height > 8192:
        return ""
    return f"{width}x{height}"


def _download_pil_image(url: str) -> Image.Image:
    response = requests.get(str(url), timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).copy()


def _custom_image_pick_gpt_size(aspect_ratio: str, image_size: str, pixel_size: str = "") -> str:
    """gpt_image 路径优先用 GPT_IMAGE_VIP_SIZE_MAP，命中不到再退回到通用计算。"""
    explicit_size = _normalize_custom_image_pixel_size(pixel_size)
    if explicit_size:
        return explicit_size
    ratio = str(aspect_ratio or "1:1").strip() or "1:1"
    size_key = str(image_size or "1K").strip().upper()
    if size_key not in {"1K", "2K", "4K"}:
        size_key = "1K"
    bucket = GPT_IMAGE_VIP_SIZE_MAP.get(ratio)
    if bucket and bucket.get(size_key):
        return bucket[size_key]
    return _openai_image_size(ratio, size_key)


def _custom_image_gemini_image_size(image_size: str) -> str:
    """Gemini 系列只支持 1K/2K/4K，默认 1K。"""
    size_key = str(image_size or "1K").strip().upper()
    return size_key if size_key in {"1K", "2K", "4K"} else "1K"


def _custom_image_decode_b64(value: str) -> Image.Image | None:
    """从 b64_json / data URL 字符串解码 PIL，失败返回 None。"""
    if not value:
        return None
    raw = value
    if isinstance(raw, str) and raw.startswith("data:"):
        raw = raw.split(",", 1)[-1]
    try:
        return safe_pil_to_rgb(Image.open(BytesIO(base64.b64decode(raw))).copy())
    except Exception:
        return None


class CustomOpenAIImageAPI:
    def __init__(self, api_key: str, api_url: str):
        if not api_key:
            raise CometAPIError("Missing custom channel API Key.")
        self.api_key = api_key
        self.api_url = normalize_api_base_url(api_url or "https://api.openai.com")

    # ============= HTTP 工具 =============

    @staticmethod
    def _raise_for_status_with_body(response, url: str) -> None:
        """raise_for_status，但把上游响应体的可读信息附在异常里。"""
        if response.status_code < 400:
            return
        snippet = ""
        try:
            text = response.text or ""
        except Exception:
            text = ""
        if text:
            # 优先尝试解析 JSON 取 message 字段
            try:
                data = response.json()
                if isinstance(data, dict):
                    err = data.get("error")
                    if isinstance(err, dict):
                        snippet = str(err.get("message") or err.get("code") or "")[:300]
                    elif isinstance(err, str):
                        snippet = err[:300]
                    if not snippet:
                        snippet = str(data.get("message") or "")[:300]
            except Exception:
                pass
            if not snippet:
                snippet = text.strip()[:300]
        detail = f" - {snippet}" if snippet else ""
        raise CometAPIError(f"HTTP {response.status_code} {response.reason} for {url}{detail}")

    def _post_json(self, url: str, payload: dict, timeout: int = 720) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(url, headers=headers, json=payload, timeout=(15, timeout))
        self._raise_for_status_with_body(response, url)
        return response.json()

    def _post_files(self, url: str, payload: dict, files: list, timeout: int = 720) -> dict:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(url, headers=headers, data=payload, files=files, timeout=(15, timeout))
        self._raise_for_status_with_body(response, url)
        return response.json()

    def _get_json(self, url: str, timeout: int = 60) -> tuple[int, dict | None]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.get(url, headers=headers, timeout=(15, timeout))
        if response.status_code == 404:
            return 404, None
        self._raise_for_status_with_body(response, url)
        try:
            return response.status_code, response.json()
        except Exception:
            return response.status_code, None

    # ============= 响应解析 =============

    def _images_from_data_array(self, data_array, errors: list[str]) -> list[Image.Image]:
        """从 OpenAI Images 兼容 data:[{url|b64_json}] 数组中提取 PIL 图。"""
        pil_images: list[Image.Image] = []
        if not isinstance(data_array, list):
            return pil_images
        for item in data_array:
            if not isinstance(item, dict):
                continue
            if item.get("b64_json"):
                decoded = _custom_image_decode_b64(item["b64_json"])
                if decoded is not None:
                    pil_images.append(decoded)
                else:
                    errors.append("b64_json 图片解码失败")
            elif item.get("url"):
                try:
                    pil_images.append(safe_pil_to_rgb(_download_pil_image(item["url"])))
                except Exception as exc:
                    errors.append(f"图片下载失败：{format_error_message(exc)}")
        return pil_images

    def _images_from_gemini_candidates(self, candidates, errors: list[str]) -> list[Image.Image]:
        """从 Gemini :generateContent 的 candidates 数组里提取 PIL 图。"""
        pil_images: list[Image.Image] = []
        if not isinstance(candidates, list):
            return pil_images
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                inline = part.get("inline_data") or part.get("inlineData")
                if isinstance(inline, dict) and inline.get("data"):
                    decoded = _custom_image_decode_b64(inline["data"])
                    if decoded is not None:
                        pil_images.append(decoded)
                    else:
                        errors.append("Gemini inline_data 解码失败")
        return pil_images

    def _looks_like_async_task(self, data: dict) -> str | None:
        """如果响应是异步任务形态，返回 task_id；否则返回 None。"""
        if not isinstance(data, dict):
            return None
        # 顶层 task_id（zhenzhen 项目里的官方版形态）
        task_id = data.get("task_id")
        if isinstance(task_id, str) and task_id:
            return task_id
        # data 是一个对象且含 id/status（apimart 形态）
        inner = data.get("data")
        if isinstance(inner, dict):
            inner_id = inner.get("id") or inner.get("task_id")
            if isinstance(inner_id, str) and inner_id:
                return inner_id
        return None

    def _poll_async_task(self, task_id: str, provider_name: str) -> dict:
        """轮询异步任务直到 SUCCESS/completed 或超时。返回最终响应数据，失败抛 CometAPIError。"""
        primary = build_api_url(self.api_url, f"/v1/images/tasks/{task_id}", "https://api.openai.com")
        fallback = build_api_url(self.api_url, f"/v1/tasks/{task_id}", "https://api.openai.com")
        deadline = time.time() + CUSTOM_IMAGE_ASYNC_POLL_TIMEOUT_SEC
        url = primary
        last_status_text = ""
        while time.time() < deadline:
            try:
                code, data = self._get_json(url)
            except Exception as exc:
                # 单次轮询失败容忍：等下一轮再试
                last_status_text = format_error_message(exc)
                time.sleep(CUSTOM_IMAGE_ASYNC_POLL_INTERVAL_SEC)
                continue
            if code == 404 and url == primary:
                url = fallback
                continue
            if not isinstance(data, dict):
                time.sleep(CUSTOM_IMAGE_ASYNC_POLL_INTERVAL_SEC)
                continue
            inner = data.get("data") if isinstance(data.get("data"), dict) else {}
            status_raw = (
                inner.get("status")
                or data.get("status")
                or ""
            )
            status = str(status_raw).strip().lower()
            last_status_text = status or last_status_text
            if status in {"success", "completed", "succeeded", "ok"}:
                return data
            if status in {"failure", "failed", "error"}:
                fail_reason = (
                    inner.get("fail_reason")
                    or inner.get("error")
                    or data.get("error")
                    or "未知错误"
                )
                if isinstance(fail_reason, dict):
                    fail_reason = fail_reason.get("message") or str(fail_reason)
                raise CometAPIError(f"{provider_name} 异步任务失败：{fail_reason}")
            time.sleep(CUSTOM_IMAGE_ASYNC_POLL_INTERVAL_SEC)
        raise CometAPIError(f"{provider_name} 异步任务轮询超时（{CUSTOM_IMAGE_ASYNC_POLL_TIMEOUT_SEC} 秒），最后状态：{last_status_text or 'unknown'}")

    def _parse_image_response(self, data: dict, provider_name: str) -> tuple[list[Image.Image], list[str]]:
        """统一解析响应。同步直接取图，异步形态进入轮询后再解析最终结果。"""
        errors: list[str] = []
        # 异步嗅探优先
        task_id = self._looks_like_async_task(data)
        if task_id is not None:
            data = self._poll_async_task(task_id, provider_name)

        pil_images: list[Image.Image] = []

        # 形态 1：data:[{url|b64_json}]（标准 OpenAI Images）
        items = data.get("data") if isinstance(data, dict) else None
        pil_images.extend(self._images_from_data_array(items, errors))

        # 形态 2：data 是一个对象，里面 result.images[].url 数组（异步任务最终结果）
        if isinstance(items, dict):
            result = items.get("result") if isinstance(items.get("result"), dict) else {}
            for image_block in result.get("images") or []:
                if not isinstance(image_block, dict):
                    continue
                urls = image_block.get("url")
                if isinstance(urls, str):
                    urls = [urls]
                if isinstance(urls, list):
                    for url in urls:
                        if not url:
                            continue
                        try:
                            pil_images.append(safe_pil_to_rgb(_download_pil_image(url)))
                        except Exception as exc:
                            errors.append(f"图片下载失败：{format_error_message(exc)}")
                if image_block.get("b64_json"):
                    decoded = _custom_image_decode_b64(image_block["b64_json"])
                    if decoded is not None:
                        pil_images.append(decoded)

        # 形态 3：Gemini :generateContent
        if not pil_images:
            pil_images.extend(self._images_from_gemini_candidates(data.get("candidates") if isinstance(data, dict) else None, errors))

        if not pil_images and not errors:
            message = ""
            if isinstance(data, dict):
                error_block = data.get("error")
                if isinstance(error_block, dict):
                    message = str(error_block.get("message") or "")
                elif isinstance(error_block, str):
                    message = error_block
                if not message:
                    message = str(data.get("message") or "")
            errors.append(f"{provider_name} 响应里没有图片数据" + (f"：{message}" if message else ""))
        return pil_images, errors

    # ============= Gemini · 原生 =============

    def _gemini_native_generate(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        image_size: str,
        sub_family: str,
        subtask_idx: int,
    ) -> tuple[list[Image.Image], list[str]]:
        parts: list[dict] = [{"text": add_prompt_variation(prompt, subtask_idx)}]
        for pil_image in (pil_images or [])[:PRIVATE_MAX_IMAGES]:
            buffered = BytesIO()
            safe_pil_to_rgb(pil_image).save(buffered, format="JPEG", quality=90)
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(buffered.getvalue()).decode("utf-8"),
                    }
                }
            )
        ratio = str(aspect_ratio or "auto").strip() or "auto"
        image_config: dict = {"aspectRatio": ratio}
        if sub_family != GEMINI_SUB_FAMILY_2_5:
            image_config["imageSize"] = _custom_image_gemini_image_size(image_size)
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": image_config},
        }
        url = build_api_url(self.api_url, f"/v1beta/models/{model}:generateContent", "https://api.openai.com")
        data = self._post_json(url, payload)
        return self._parse_image_response(data, f"Custom Gemini ({model})")

    # ============= Gemini · OpenAI 兼容 =============

    def _gemini_openai_compat_generate(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        image_size: str,
        sub_family: str,
        subtask_idx: int,
    ) -> tuple[list[Image.Image], list[str]]:
        final_prompt = add_prompt_variation(prompt, subtask_idx)
        ratio = str(aspect_ratio or "auto").strip() or "auto"
        size = _custom_image_gemini_image_size(image_size)

        common_data = {
            "model": model,
            "prompt": final_prompt,
            "aspect_ratio": ratio,
            "response_format": "b64_json",
        }
        # 2.5 系列不带 image_size
        if sub_family != GEMINI_SUB_FAMILY_2_5:
            common_data["image_size"] = size

        if pil_images:
            files: list[tuple[str, tuple[str, bytes, str]]] = []
            for index, pil_image in enumerate(pil_images[:NANO_BANANA_MAX_IMAGES], start=1):
                buffered = BytesIO()
                safe_pil_to_rgb(pil_image).save(buffered, format="PNG")
                files.append(("image", (f"image_{index}.png", buffered.getvalue(), "image/png")))
            url = build_api_url(self.api_url, "/v1/images/edits", "https://api.openai.com")
            data = self._post_files(url, common_data, files)
        else:
            payload = {**common_data, "n": 1}
            url = build_api_url(self.api_url, "/v1/images/generations", "https://api.openai.com")
            data = self._post_json(url, payload)
        return self._parse_image_response(data, f"Custom Gemini ({model})")

    # ============= GPT-Image · 统一接口 =============

    def _gpt_image_unified_generate(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        image_size: str,
        quality: str,
        pixel_size: str,
        subtask_idx: int,
    ) -> tuple[list[Image.Image], list[str]]:
        final_prompt = add_prompt_variation(prompt, subtask_idx)
        size = _custom_image_pick_gpt_size(aspect_ratio, image_size, pixel_size)
        safe_quality = quality if str(quality or "").strip() in GPT_IMAGE_QUALITY_VALUES else ""
        payload = {
            "model": model,
            "prompt": final_prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
        }
        if safe_quality:
            payload["quality"] = safe_quality
        if pil_images:
            image_data = [pil_to_data_url(image) for image in pil_images[:GPT_IMAGE_MAX_IMAGES]]
            payload["image"] = image_data if len(image_data) > 1 else image_data[0]
        url = build_api_url(self.api_url, "/v1/images/generations", "https://api.openai.com")
        data = self._post_json(url, payload)
        return self._parse_image_response(data, f"Custom GPT-Image ({model})")

    # ============= GPT-Image · 分离接口 =============

    def _gpt_image_split_generate(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image],
        aspect_ratio: str,
        image_size: str,
        quality: str,
        pixel_size: str,
        subtask_idx: int,
    ) -> tuple[list[Image.Image], list[str]]:
        final_prompt = add_prompt_variation(prompt, subtask_idx)
        size = _custom_image_pick_gpt_size(aspect_ratio, image_size, pixel_size)
        safe_quality = quality if str(quality or "").strip() in GPT_IMAGE_QUALITY_VALUES else ""

        if pil_images:
            payload = {
                "model": model,
                "prompt": final_prompt,
                "n": "1",
                "size": size,
            }
            if safe_quality:
                payload["quality"] = safe_quality
            files: list[tuple[str, tuple[str, bytes, str]]] = []
            for index, pil_image in enumerate(pil_images[:GPT_IMAGE_MAX_IMAGES], start=1):
                buffered = BytesIO()
                safe_pil_to_rgb(pil_image).save(buffered, format="PNG")
                files.append(("image", (f"image_{index}.png", buffered.getvalue(), "image/png")))
            url = build_api_url(self.api_url, "/v1/images/edits", "https://api.openai.com")
            data = self._post_files(url, payload, files)
        else:
            payload = {
                "model": model,
                "prompt": final_prompt,
                "n": 1,
                "size": size,
                "response_format": "b64_json",
            }
            if safe_quality:
                payload["quality"] = safe_quality
            url = build_api_url(self.api_url, "/v1/images/generations", "https://api.openai.com")
            data = self._post_json(url, payload)
        return self._parse_image_response(data, f"Custom GPT-Image ({model})")

    # ============= 入口 =============

    def generate_image(
        self,
        prompt: str,
        model: str,
        pil_images: list[Image.Image] | None = None,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        quality: str = "medium",
        api_format: str = "gpt_image",
        interface_mode: str = "",
        pixel_size: str = "",
        subtask_idx: int = 0,
    ) -> tuple[list[Image.Image], list[str]]:
        try:
            fmt = api_format if api_format in IMAGE_API_FORMATS else "gpt_image"
            mode = normalize_image_interface_mode(interface_mode, fmt)
            pils = pil_images or []
            if fmt == "gemini_image":
                sub_family = detect_gemini_sub_family(model) or GEMINI_SUB_FAMILY_3
                if mode == "openai_compat":
                    return self._gemini_openai_compat_generate(
                        prompt=prompt,
                        model=model,
                        pil_images=pils,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                        sub_family=sub_family,
                        subtask_idx=subtask_idx,
                    )
                return self._gemini_native_generate(
                    prompt=prompt,
                    model=model,
                    pil_images=pils,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    sub_family=sub_family,
                    subtask_idx=subtask_idx,
                )
            # gpt_image
            if mode == "split":
                return self._gpt_image_split_generate(
                    prompt=prompt,
                    model=model,
                    pil_images=pils,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    quality=quality,
                    pixel_size=pixel_size,
                    subtask_idx=subtask_idx,
                )
            return self._gpt_image_unified_generate(
                prompt=prompt,
                model=model,
                pil_images=pils,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                quality=quality,
                pixel_size=pixel_size,
                subtask_idx=subtask_idx,
            )
        except Exception as exc:
            label = f"custom image task {subtask_idx}" if subtask_idx else "custom image"
            return [], [f"{label}: {format_error_message(exc)}"]


def _guess_media_mime(path: str, default: str) -> str:
    ext = os.path.splitext(str(path or "").lower())[1]
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }.get(ext, default)


def build_llm_messages(
    prompt: str,
    pil_images: list[Image.Image],
    video_paths: list[str] | None = None,
    audio_paths: list[str] | None = None,
    api_format: str = "gemini",
) -> list:
    """构建LLM消息格式"""
    video_paths = [path for path in (video_paths or []) if path and os.path.exists(path)]
    audio_paths = [path for path in (audio_paths or []) if path and os.path.exists(path)]
    if not pil_images and not video_paths and not audio_paths:
        if api_format == "gemini":
            return [{"role": "user", "parts": [{"text": prompt}]}]
        else:
            return [{"role": "user", "content": prompt}]
    
    # 多模态消息
    if api_format == "gemini":
        parts = [{"text": prompt}]
        for pil_image in pil_images:
            buffered = BytesIO()
            safe_pil_to_rgb(pil_image).save(buffered, format="JPEG", quality=90)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img_base64,
                }
            })
        for video_path in video_paths:
            with open(video_path, "rb") as handle:
                video_base64 = base64.b64encode(handle.read()).decode("utf-8")
            parts.append({
                "inlineData": {
                    "mimeType": _guess_media_mime(video_path, "video/mp4"),
                    "data": video_base64,
                }
            })
        for audio_path in audio_paths:
            with open(audio_path, "rb") as handle:
                audio_base64 = base64.b64encode(handle.read()).decode("utf-8")
            parts.append({
                "inlineData": {
                    "mimeType": _guess_media_mime(audio_path, "audio/wav"),
                    "data": audio_base64,
                }
            })
        return [{"role": "user", "parts": parts}]
    elif api_format == "openai":
        content = [{"type": "text", "text": prompt}]
        for pil_image in pil_images:
            buffered = BytesIO()
            safe_pil_to_rgb(pil_image).save(buffered, format="JPEG", quality=90)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            })
        return [{"role": "user", "content": content}]
    elif api_format == "claude":
        content = []
        for pil_image in pil_images:
            buffered = BytesIO()
            safe_pil_to_rgb(pil_image).save(buffered, format="JPEG", quality=90)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64},
            })
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]
    else:
        return [{"role": "user", "content": prompt}]


class CometAPIUnifiedLLMNode:
    CATEGORY = "COMET"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "channel": (get_llm_channel_choices(), {"default": "grsai"}),
                "model": (get_llm_model_choices(), {"default": "gemini-3-flash"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "media": (COMET_ANY,),
            },
        }
        for i in range(1, MAX_LLM_IMAGE_INPUTS + 1):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        for i in range(1, MAX_LLM_VIDEO_INPUTS + 1):
            inputs["optional"][f"video_{i}"] = (IO.VIDEO,)
        for i in range(1, MAX_LLM_AUDIO_INPUTS + 1):
            inputs["optional"][f"audio_{i}"] = (getattr(IO, "AUDIO", "AUDIO"),)
        return inputs

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def _error(self, message: str):
        message = redact_sensitive_text(message)
        print(f"[CometAPI LLM] {message}")
        return {"ui": {"comet_error": [message]}, "result": ("",)}

    def _collect_input_pils(self, kwargs: dict) -> list[Image.Image]:
        pil_images = []
        for i in range(1, MAX_LLM_IMAGE_INPUTS + 1):
            image_tensor = kwargs.get(f"image_{i}")
            if image_tensor is None:
                continue
            pil_images.extend(tensor_to_pil(image_tensor))
        return [safe_pil_to_rgb(image) for image in pil_images[:MAX_LLM_IMAGE_INPUTS]]

    def _collect_input_videos(self, kwargs: dict) -> list[str]:
        video_paths = []
        for i in range(1, MAX_LLM_VIDEO_INPUTS + 1):
            video = kwargs.get(f"video_{i}")
            if video is None:
                continue
            video_paths.append(video_input_to_path(video))
        return video_paths[:MAX_LLM_VIDEO_INPUTS]

    def _collect_input_audios(self, kwargs: dict) -> list[str]:
        audio_paths = []
        for i in range(1, MAX_LLM_AUDIO_INPUTS + 1):
            audio = kwargs.get(f"audio_{i}")
            if audio is None:
                continue
            path, _is_temp = _media_input_to_path(audio, ".wav")
            if path:
                audio_paths.append(path)
        return audio_paths[:MAX_LLM_AUDIO_INPUTS]

    def execute(
        self,
        channel: str,
        model: str,
        prompt: str,
        api_key: str = "",
        **kwargs,
    ):
        channel = str(channel or "grsai").lower()
        if channel not in get_llm_channel_choices():
            return self._error(f"不支持的渠道：{channel}")

        model = resolve_model_id(channel, model, "llm")
        final_api_key = get_channel_api_key(api_key, channel, model, "llm")
        if not final_api_key:
            return self._error(f"缺少 {channel} API Key，请先在设置中心填写。")

        # 从设置中获取api_format
        api_format = get_model_api_format(channel, model)
        api_url = get_channel_api_url(channel)

        try:
            pil_images = self._collect_input_pils(kwargs)
            video_paths = self._collect_input_videos(kwargs)
            audio_paths = self._collect_input_audios(kwargs)
            if (video_paths or audio_paths) and api_format != "gemini":
                return self._error("当前 LLM 的视频/音频输入只支持 Gemini 接口格式模型，请在设置中心或横条里切换模型。")
            
            messages = build_llm_messages(prompt, pil_images, video_paths, audio_paths, api_format)
            
            if get_private_channel_spec(channel):
                response_text, error_msg = run_private_llm_channel(
                    channel=channel,
                    api_key=final_api_key,
                    model=model,
                    prompt=prompt,
                    pil_images=pil_images,
                    video_paths=video_paths,
                    audio_paths=audio_paths,
                    messages=messages,
                    api_format=api_format,
                )
            else:
                response_text, error_msg = call_llm_api(final_api_key, model, messages, api_format, api_url)
            
            if error_msg:
                return self._error(error_msg)
            
            # 确保返回值不为空
            if not response_text:
                return self._error("API 返回了空响应。")
            
            return {"ui": {}, "result": (response_text,)}
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._error(format_error_message(exc))


class CometAPIUnifiedImage:
    CATEGORY = "COMET"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "channel": (get_image_channel_choices(), {"default": "grsai"}),
                "model": (get_image_model_choices(), {"default": "nano-banana-pro"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "concurrency": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (["1K", "2K", "3K", "4K", "8K"], {"default": "2K"}),
                "quality": (GPT_IMAGE_QUALITY_VALUES, {"default": "medium"}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }
        # These real backend inputs are hidden by the frontend. The virtual dot
        # compiles its links into image_1...image_14 before ComfyUI validates.
        for i in range(1, MAX_IMAGE_INPUTS + 1):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def _error(self, message: str):
        message = redact_sensitive_text(message)
        print(f"[CometAPI] {message}")
        image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return {"ui": {"comet_error": [message]}, "result": (image,)}

    def _credits_balance(self, api_key: str) -> str:
        try:
            res = requests.get(
                f"https://grsai.dakka.com.cn/client/common/getCredits?apikey={api_key}",
                timeout=10,
            )
            if res.status_code == 200:
                data = res.json()
                if data.get("code") == 0 and "data" in data:
                    return str(int(data["data"]["credits"]))
        except Exception:
            pass
        return "N/A"

    def _upload_inputs(self, api_key: str, kwargs: dict, max_images: int = 14) -> list[str] | dict:
        uploaded_urls = []
        image_tensors = [kwargs.get(f"image_{i}") for i in range(1, max_images + 1)]
        for image_tensor in image_tensors:
            if image_tensor is None:
                continue
            url = upload_image_grsai(api_key, image_tensor)
            if not url:
                return {"error": "参考图上传失败，请检查输入图片或网络。"}
            uploaded_urls.append(url)
        return uploaded_urls

    def _collect_input_pils(self, kwargs: dict, max_images: int) -> list[Image.Image]:
        pil_images = []
        for i in range(1, max_images + 1):
            image_tensor = kwargs.get(f"image_{i}")
            if image_tensor is None:
                continue
            pil_images.extend(tensor_to_pil(image_tensor))
        return [safe_pil_to_rgb(image) for image in pil_images[:max_images]]

    def _resolve_auto_aspect_ratio(self, kwargs: dict, max_images: int) -> str:
        first_image = next((kwargs.get(f"image_{i}") for i in range(1, max_images + 1) if kwargs.get(f"image_{i}") is not None), None)
        if first_image is None:
            return "1:1"
        try:
            pil_images = tensor_to_pil(first_image)
            if not pil_images:
                return "1:1"
            width, height = pil_images[0].size
            if width <= 0 or height <= 0:
                return "1:1"
            image_ratio = width / height
            return min(
                GPT_IMAGE_VIP_SIZE_MAP.keys(),
                key=lambda ratio: abs((float(ratio.split(":")[0]) / float(ratio.split(":")[1])) - image_ratio),
            )
        except Exception:
            return "1:1"

    def execute(
        self,
        channel: str,
        model: str,
        prompt: str,
        concurrency: int,
        aspect_ratio: str,
        image_size: str,
        quality: str = "medium",
        api_key: str = "",
        **kwargs,
    ):
        channel = str(channel or "").lower()
        if channel not in get_image_channel_choices():
            return self._error(f"不支持的渠道：{channel}")
        model = resolve_model_id(channel, model, "image")

        try:
            concurrency = int(concurrency)
        except Exception:
            concurrency = 1
        concurrency = max(1, min(10, concurrency))

        final_api_key = get_channel_api_key(api_key, channel, model, "image")
        if not final_api_key:
            return self._error(f"缺少 {channel} API Key，请先在设置中心填写。")

        is_grsai = channel == "grsai"
        is_runninghub = channel == "runninghub"
        is_modelverse = channel == "modelverse"
        is_apimart = channel == "apimart"
        is_private_image = bool(get_private_channel_spec(channel))
        is_custom_image = channel not in IMAGE_CHANNELS
        is_gpt_image = is_grsai and model in GPT_IMAGE_MODELS
        if is_grsai:
            max_refs = GPT_IMAGE_MAX_IMAGES if is_gpt_image else NANO_BANANA_MAX_IMAGES
            auto_aspect_ratio = self._resolve_auto_aspect_ratio(kwargs, max_refs)
            uploaded_urls = self._upload_inputs(final_api_key, kwargs, max_images=max_refs)
            if isinstance(uploaded_urls, dict):
                return self._error(uploaded_urls["error"])
            pil_refs = []
        elif is_runninghub:
            max_refs = runninghub_image_max_images(model)
            auto_aspect_ratio = "1:1"
            uploaded_urls = []
            pil_refs = self._collect_input_pils(kwargs, max_images=max_refs)
        elif is_modelverse:
            max_refs = modelverse_image_max_images(model)
            auto_aspect_ratio = "1:1"
            uploaded_urls = []
            pil_refs = self._collect_input_pils(kwargs, max_images=max_refs)
        elif is_apimart:
            max_refs = apimart_image_max_images(model)
            auto_aspect_ratio = "1:1"
            uploaded_urls = []
            pil_refs = self._collect_input_pils(kwargs, max_images=max_refs)
        elif is_custom_image:
            private_limits = private_media_limits_for_model(channel, model, "image") if is_private_image else {}
            max_refs = int(private_limits.get("image") or MAX_IMAGE_INPUTS)
            auto_aspect_ratio = "1:1"
            uploaded_urls = []
            pil_refs = self._collect_input_pils(kwargs, max_images=max_refs)
        else:
            max_refs = PRIVATE_MAX_IMAGES
            auto_aspect_ratio = "1:1"
            uploaded_urls = []
            pil_refs = self._collect_input_pils(kwargs, max_images=max_refs)

        target_size = image_size if image_size in {"1K", "2K", "3K", "4K", "8K"} else "2K"
        if is_grsai and not is_gpt_image:
            if model == "nano-banana-fast":
                target_size = "1K"
            elif model not in PRO_SIZE_MODELS:
                target_size = "1K"

        actual_model = model
        if model == "nano-banana-2-cl" and target_size == "4K":
            actual_model = "nano-banana-2-4k-cl"

        try:
            client = GrsaiAPI(final_api_key) if is_grsai else None
            all_images = []
            all_errors = []

            def submit_once(_):
                try:
                    if is_private_image and private_adapter_supports(channel, "image"):
                        return run_private_image_channel(
                            channel=channel,
                            api_key=final_api_key,
                            model=model,
                            prompt=prompt,
                            pil_images=pil_refs,
                            aspect_ratio=aspect_ratio,
                            image_size=target_size,
                            quality=quality,
                            subtask_idx=_,
                        )
                    if channel == "runninghub":
                        return RunningHubImageAPI(final_api_key).generate_image(
                            prompt=prompt,
                            model=model,
                            pil_images=pil_refs,
                            aspect_ratio=aspect_ratio,
                            image_size=target_size,
                            quality=quality,
                            subtask_idx=_,
                        )
                    if channel == "modelverse":
                        return ModelVerseImageAPI(final_api_key).generate_image(
                            prompt=prompt,
                            model=model,
                            pil_images=pil_refs,
                            aspect_ratio=aspect_ratio,
                            image_size=target_size,
                            quality=quality,
                            subtask_idx=_,
                        )
                    if channel == "apimart":
                        return APIMartImageAPI(final_api_key).generate_image(
                            prompt=prompt,
                            model=model,
                            pil_images=pil_refs,
                            aspect_ratio=aspect_ratio,
                            image_size=target_size,
                            quality=quality,
                            subtask_idx=_,
                        )
                    if is_custom_image:
                        custom_settings = load_comet_settings()
                        custom_channel = get_custom_channel_settings(custom_settings, channel)
                        if custom_channel:
                            custom_model = get_custom_model_settings(custom_channel, model, "image")
                        else:
                            custom_model = (custom_settings.get("channels", {}).get(channel, {}).get("models", {}) or {}).get(model, {})
                        custom_format = normalize_api_format_value(
                            custom_model.get("api_format") or detect_image_api_format(model),
                            "image",
                        )
                        custom_mode = normalize_image_interface_mode(
                            custom_model.get("interface_mode") or custom_model.get("endpoint_mode"),
                            custom_format,
                        )
                        return CustomOpenAIImageAPI(final_api_key, get_channel_api_url(channel)).generate_image(
                            prompt=prompt,
                            model=model,
                            pil_images=pil_refs,
                            aspect_ratio=aspect_ratio,
                            image_size=target_size,
                            quality=quality,
                            api_format=custom_format,
                            interface_mode=custom_mode,
                            subtask_idx=_,
                        )
                    if is_gpt_image:
                        return client.gpt_image_generate(
                            prompt=prompt,
                            model=model,
                            urls=uploaded_urls,
                            aspect_ratio=aspect_ratio if aspect_ratio in GPT_IMAGE_ASPECT_RATIOS else "1:1",
                            image_size=target_size,
                            quality=quality,
                            auto_aspect_ratio=auto_aspect_ratio,
                        )
                    return client.nano_banana_generate_image(
                        prompt=prompt,
                        model=actual_model,
                        urls=uploaded_urls,
                        aspect_ratio=aspect_ratio,
                        image_size=target_size,
                    )
                except Exception as exc:
                    return [], [format_error_message(exc)]

            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
                for images, errors in executor.map(submit_once, range(max(1, concurrency))):
                    all_images.extend(images)
                    all_errors.extend(errors)

            if not all_images:
                return self._error("; ".join(all_errors) if all_errors else "没有生成任何图片。")

            credits = self._credits_balance(final_api_key) if is_grsai else "N/A"
            detail = (
                f"{channel}_refs: {len(pil_refs)}"
                if channel in {"runninghub", "modelverse"}
                else f"refs: {len(uploaded_urls)}"
                if is_gpt_image
                else f"virtual_refs: {len(uploaded_urls)}"
            )
            ui = {}
            if all_errors:
                warning = (
                    f"已生成 {len(all_images)}/{max(1, concurrency)} 个任务，"
                    f"失败 {len(all_errors)} 个：{'; '.join(all_errors[:3])}"
                )
                if len(all_errors) > 3:
                    warning += f"；另外还有 {len(all_errors) - 3} 个失败"
                ui["comet_warning"] = [warning]
            return {"ui": ui, "result": (pil_to_tensor(all_images),)}
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._error(format_error_message(exc))


def _batch_image_tensor(pil_images: list[Image.Image]) -> torch.Tensor:
    if not pil_images:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    safe_images = [safe_pil_to_rgb(image) for image in pil_images if isinstance(image, Image.Image)]
    if not safe_images:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    max_width = max(max(1, image.width) for image in safe_images)
    max_height = max(max(1, image.height) for image in safe_images)
    padded = []
    for image in safe_images:
        if image.width == max_width and image.height == max_height:
            padded.append(image)
            continue
        canvas = Image.new("RGB", (max_width, max_height), (0, 0, 0))
        canvas.paste(image, ((max_width - image.width) // 2, (max_height - image.height) // 2))
        padded.append(canvas)
    return pil_to_tensor(padded)


def _batch_image_slug(value: str, fallback: str = "item") -> str:
    slug = _safe_filename_stem(value, fallback)
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if len(slug) > 48:
        slug = slug[:48].rstrip("_")
    return slug or fallback


class CometAPIBatchImage:
    CATEGORY = "COMET"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "summary")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "channel": (get_image_channel_choices(), {"default": "grsai"}),
                "model": (get_image_model_choices(), {"default": "nano-banana-pro"}),
                "batch_mode": (COMET_BATCH_IMAGE_MODES, {"default": COMET_BATCH_IMAGE_MODES[0]}),
                "pairing_mode": (COMET_BATCH_IMAGE_PAIRING_MODES, {"default": COMET_BATCH_IMAGE_PAIRING_MODES[0]}),
                "folder_path": ("STRING", {"default": "", "placeholder": "批量图片文件夹；支持 Windows 复制路径自带引号；相对路径会从当前目录、input、output 下查找"}),
                "concurrency": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (["1K", "2K", "3K", "4K", "8K"], {"default": "2K"}),
                "quality": (GPT_IMAGE_QUALITY_VALUES, {"default": "medium"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "batch_text": (COMET_BATCH_TEXT,),
            },
            "hidden": {
                "_unique_id": "UNIQUE_ID",
            },
        }
        for i in range(1, MAX_IMAGE_INPUTS + 1):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def _error(self, message: str):
        message = redact_sensitive_text(message)
        print(f"[CometAPI] Batch Image: {message}")
        image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return {"ui": {"comet_error": [message], "comet_text": [message]}, "result": (image, message)}

    def _collect_prompts(self, batch_text=None) -> list[str]:
        prompts = []
        if isinstance(batch_text, dict):
            values = batch_text.get("prompts")
            if isinstance(values, list):
                prompts = [_clean_batch_prompt_line(item) for item in values]
            if not prompts:
                prompts = parse_batch_prompts(batch_text.get("text") or batch_text.get("content") or "")
        elif isinstance(batch_text, list):
            for item in batch_text:
                if isinstance(item, dict):
                    values = item.get("prompts")
                    if isinstance(values, list):
                        prompts.extend(_clean_batch_prompt_line(value) for value in values)
                    else:
                        prompts.extend(parse_batch_prompts(item.get("text") or item.get("content") or ""))
                else:
                    prompts.extend(parse_batch_prompts(item))
        elif batch_text is not None and str(batch_text).strip():
            prompts = parse_batch_prompts(batch_text)

        prompts = [prompt for prompt in prompts if prompt]
        return prompts

    def _collect_fixed_refs(self, kwargs: dict, max_images: int) -> list[Image.Image]:
        refs: list[Image.Image] = []

        def extend_from(value):
            if value is None or len(refs) >= max_images:
                return
            for pil_image in tensor_to_pil(value):
                refs.append(safe_pil_to_rgb(pil_image))
                if len(refs) >= max_images:
                    break

        extend_from(kwargs.get("images"))
        for i in range(1, MAX_IMAGE_INPUTS + 1):
            extend_from(kwargs.get(f"image_{i}"))
            if len(refs) >= max_images:
                break
        return refs[:max_images]

    def _max_refs(self, channel: str, model: str) -> int:
        if channel == "grsai":
            return GPT_IMAGE_MAX_IMAGES if model in GPT_IMAGE_MODELS else NANO_BANANA_MAX_IMAGES
        if channel == "runninghub":
            return runninghub_image_max_images(model)
        if channel == "modelverse":
            return modelverse_image_max_images(model)
        if channel == "apimart":
            return apimart_image_max_images(model)
        if channel not in IMAGE_CHANNELS:
            limits = private_media_limits_for_model(channel, model, "image") if get_private_channel_spec(channel) else {}
            return int(limits.get("image") or MAX_IMAGE_INPUTS)
        return PRIVATE_MAX_IMAGES

    def _fixed_ref_collect_limit(self, channel: str, model: str, batch_mode: str, pairing_mode: str) -> int:
        batch_mode = self._normalize_batch_mode(batch_mode)
        pairing_mode = self._normalize_pairing_mode(pairing_mode, batch_mode)
        if batch_mode == COMET_BATCH_IMAGE_MODE_REGULAR and pairing_mode != COMET_BATCH_PAIRING_ALL_REFS:
            return MAX_IMAGE_INPUTS
        return self._max_refs(channel, model)

    def _normalize_batch_mode(self, batch_mode: str) -> str:
        raw = str(batch_mode or "").strip()
        lower = raw.lower()
        if "文件夹" in raw or lower in {"folder", "folder_batch"}:
            return COMET_BATCH_IMAGE_MODE_FOLDER
        return COMET_BATCH_IMAGE_MODE_REGULAR

    def _use_folder_mode(self, batch_mode: str) -> bool:
        return self._normalize_batch_mode(batch_mode) == COMET_BATCH_IMAGE_MODE_FOLDER

    def _normalize_pairing_mode(self, pairing_mode: str, batch_mode: str) -> str:
        if self._use_folder_mode(batch_mode):
            return COMET_BATCH_FOLDER_PAIRING_MODE

        raw = str(pairing_mode or "").strip()
        if raw == COMET_BATCH_PAIRING_ORDERED:
            return COMET_BATCH_PAIRING_ORDERED
        if raw == COMET_BATCH_PAIRING_EACH_REF:
            return COMET_BATCH_PAIRING_EACH_REF
        return COMET_BATCH_PAIRING_ALL_REFS

    def _credits_balance(self, api_key: str) -> str:
        try:
            res = requests.get(
                f"https://grsai.dakka.com.cn/client/common/getCredits?apikey={api_key}",
                timeout=10,
            )
            if res.status_code == 200:
                data = res.json()
                if data.get("code") == 0 and "data" in data:
                    return str(int(data["data"]["credits"]))
        except Exception:
            pass
        return "N/A"

    def _format_failure_summary(
        self,
        failure: dict,
        include_task_index: bool = False,
        include_folder_context: bool = False,
    ) -> str:
        error = str(failure.get("error") or "未知原因").strip() or "未知原因"
        if include_folder_context:
            parts = []
            try:
                task_number = int(failure.get("task_index") or 0)
            except Exception:
                task_number = 0
            if task_number > 0:
                parts.append(f"第 {task_number} 个任务")
            source_name = str(failure.get("source_name") or "").strip()
            source_index = failure.get("source_index")
            if source_name:
                try:
                    source_number = int(source_index or 0)
                except Exception:
                    source_number = 0
                parts.append(f"图 {source_number}：{source_name}" if source_number > 0 else source_name)
            prompt_index = failure.get("prompt_index")
            if prompt_index:
                parts.append(f"提示词 {prompt_index}")
            candidate_index = failure.get("candidate_index")
            if candidate_index:
                parts.append(f"第 {candidate_index} 次")
            return f"{' | '.join(parts)} -> {error}" if parts else error

        task_index = failure.get("task_index")
        if include_task_index:
            try:
                task_number = int(task_index)
            except Exception:
                task_number = 0
            if task_number > 0:
                return f"第 {task_number} 个任务失败：{error}"
        return error

    def _build_summary(
        self,
        channel: str,
        model: str,
        batch_mode: str,
        pairing_mode: str,
        prompts: list[str],
        fixed_refs: list[Image.Image],
        tasks: list[dict],
        results: list[dict],
        failures: list[dict],
        warnings: list[str],
        run_dir: str,
        api_key: str,
        aborted: bool = False,
        failure_image_policy: str = BATCH_IMAGE_FAILURE_POLICY_SKIP,
        folder_preview_limit: int = BATCH_IMAGE_FOLDER_PREVIEW_LIMIT,
    ) -> str:
        batch_mode = self._normalize_batch_mode(batch_mode)
        pairing_mode = self._normalize_pairing_mode(pairing_mode, batch_mode)
        use_folder = self._use_folder_mode(batch_mode)
        if use_folder:
            folder_path = next((task.get("folder_image") for task in tasks if task.get("folder_image")), "")
            folder_label = os.path.basename(os.path.dirname(folder_path)) if folder_path else "文件夹批量"
            source_count = len({task.get("source_index") for task in tasks if task.get("source_index")})
            lines = [
                "任务异常终止 (熔断)" if aborted else "批量任务完成",
                f"渠道：{channel}",
                f"模型：{model}",
                f"模式：{batch_mode} / {pairing_mode}",
                f"输入：{folder_label}",
                f"图片：{source_count} 张",
                f"Prompts：{len(prompts)} 条",
                f"固定参考图：{len(fixed_refs)} 张",
                f"总任务：{len(tasks)}",
                f"输出图片数：{len(results)}",
                f"记录报错数：{len(failures)}",
                f"失败处理：{failure_image_policy}",
            ]
            if channel == "grsai":
                lines.append(f"积分：{self._credits_balance(api_key)}")
            lines.append(f"路径：{os.path.abspath(run_dir)}")
            lines.append(f"⚠️ 预览仅显示前 {folder_preview_limit} 张图，全量图请查看文件夹。")
            if warnings:
                lines.append("提示：" + "；".join(warnings))
            if failures:
                lines.append("")
                lines.append("--- 报错记录 (前5个) ---")
                for failure in failures[:5]:
                    lines.append("- " + self._format_failure_summary(failure, include_folder_context=True))
                if len(failures) > 5:
                    lines.append(f"...以及其他 {len(failures) - 5} 个报错")
            else:
                lines.append("")
                lines.append("所有文件均处理成功。")
            return "\n".join(lines)

        lines = [
            f"批量生图完成：成功 {len(results)} 张，失败 {len(failures)} 个任务。",
            f"输出目录：{os.path.abspath(run_dir)}",
        ]
        if warnings:
            lines.append("提示：" + "；".join(warnings))
        if failures:
            is_grsai_multi_to_one = channel == "grsai"
            failure_prefix = "失败详情：" if is_grsai_multi_to_one else "失败示例："
            failure_limit = 5 if is_grsai_multi_to_one else 3
            lines.append(failure_prefix + "；".join(
                self._format_failure_summary(item, include_task_index=is_grsai_multi_to_one)
                for item in failures[:failure_limit]
            ))
        return "\n".join(lines)

    def _resolve_folder(self, folder_path: str) -> str:
        raw = _strip_wrapping_path_quotes(folder_path)
        if not raw:
            raise CometAPIError("文件夹批量模式需要填写图片文件夹路径。")
        if os.name != "nt" and _looks_like_windows_absolute_path(raw):
            raise CometAPIError("当前系统不是 Windows，不能直接读取 Windows 盘符路径；请改成当前系统可访问的路径。")

        normalized = _normalize_user_path(raw)
        if os.path.isabs(normalized):
            candidate = os.path.abspath(normalized)
            if os.path.isdir(candidate):
                return candidate
            raise CometAPIError(f"图片文件夹不存在或不可读取：{candidate}")

        folder_paths = get_folder_paths()
        candidates = [
            os.path.abspath(normalized),
            os.path.abspath(os.path.join(folder_paths.get_input_directory(), normalized)),
            os.path.abspath(os.path.join(folder_paths.get_output_directory(), normalized)),
            os.path.abspath(os.path.join(folder_paths.get_temp_directory(), normalized)),
        ]
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if os.path.isdir(candidate):
                return candidate
        raise CometAPIError(f"没有找到图片文件夹：{raw}；相对路径会依次从当前目录、input、output、temp 下查找。")

    def _folder_images(self, folder_path: str) -> list[str]:
        folder = self._resolve_folder(folder_path)
        items = []
        for name in sorted(os.listdir(folder), key=lambda value: value.lower()):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in COMET_BATCH_IMAGE_EXTS:
                items.append(path)
        if not items:
            raise CometAPIError(f"图片文件夹里没有可用图片：{folder}")
        return items

    def _load_folder_image(self, path: str) -> Image.Image:
        try:
            with Image.open(path) as image:
                pil_image = safe_pil_to_rgb(image.copy())
            pil_image.filename = os.path.abspath(path)
            return pil_image
        except Exception as exc:
            raise CometAPIError(f"读取文件夹图片失败：{path}；{exc}") from exc

    def _folder_failure_placeholder(self, task: dict, aspect_ratio: str, image_size: str, model: str) -> Image.Image:
        orig_width, orig_height = 512, 512
        folder_image = task.get("folder_image")
        if folder_image:
            try:
                with Image.open(folder_image) as image:
                    orig_width, orig_height = image.size
            except Exception:
                pass

        target_dim = {"1K": 1024, "2K": 2048, "3K": 3072, "4K": 4096, "8K": 8192}.get(str(image_size or ""), 2048)
        if str(model or "") == "nano-banana-fast":
            target_dim = 1024

        width, height = max(1, int(orig_width)), max(1, int(orig_height))
        ratio_text = str(aspect_ratio or "").strip()
        if ratio_text and ratio_text != "auto" and ":" in ratio_text:
            try:
                rw, rh = [max(1, float(part)) for part in ratio_text.split(":", 1)]
                if rw >= rh:
                    width = target_dim
                    height = int(target_dim * rh / rw)
                else:
                    height = target_dim
                    width = int(target_dim * rw / rh)
            except Exception:
                ratio_text = "auto"
        if not ratio_text or ratio_text == "auto":
            scale = target_dim / max(width, height, 1)
            width = int(width * scale)
            height = int(height * scale)

        width = max(64, (width // 8) * 8)
        height = max(64, (height // 8) * 8)
        return Image.new("RGB", (width, height), (255, 255, 255))

    def _build_tasks(
        self,
        prompts: list[str],
        fixed_refs: list[Image.Image],
        batch_mode: str,
        pairing_mode: str,
        folder_path: str,
        candidates_per_prompt: int,
        max_tasks: int,
    ) -> tuple[list[dict], list[str]]:
        warnings = []
        batch_mode = self._normalize_batch_mode(batch_mode)
        pairing_mode = self._normalize_pairing_mode(pairing_mode, batch_mode)
        use_folder = self._use_folder_mode(batch_mode)
        base_tasks = []

        if use_folder:
            image_paths = self._folder_images(folder_path)
            for source_index, path in enumerate(image_paths, start=1):
                for prompt_index, prompt in enumerate(prompts, start=1):
                    base_tasks.append(
                        {
                            "prompt": prompt,
                            "prompt_index": prompt_index,
                            "folder_image": path,
                            "source_index": source_index,
                            "source_name": os.path.basename(path),
                            "fixed_refs": fixed_refs,
                        }
                    )
        else:
            if fixed_refs and pairing_mode == COMET_BATCH_PAIRING_ALL_REFS:
                for prompt_index, prompt in enumerate(prompts, start=1):
                    base_tasks.append(
                        {
                            "prompt": prompt,
                            "prompt_index": prompt_index,
                            "folder_image": "",
                            "source_index": 0,
                            "source_name": "input_image_group",
                            "fixed_refs": fixed_refs,
                        }
                    )
            elif fixed_refs and pairing_mode == COMET_BATCH_PAIRING_ORDERED:
                pair_count = min(len(fixed_refs), len(prompts))
                if len(fixed_refs) != len(prompts):
                    warnings.append(f"一一配对数量不一致：图片 {len(fixed_refs)} 张，提示词 {len(prompts)} 条，本次只执行 {pair_count} 组。")
                for index in range(pair_count):
                    base_tasks.append(
                        {
                            "prompt": prompts[index],
                            "prompt_index": index + 1,
                            "folder_image": "",
                            "source_index": index + 1,
                            "source_name": f"input_image_{index + 1:03d}",
                            "fixed_refs": [fixed_refs[index]],
                        }
                    )
            elif fixed_refs:
                for source_index, ref in enumerate(fixed_refs, start=1):
                    for prompt_index, prompt in enumerate(prompts, start=1):
                        base_tasks.append(
                            {
                                "prompt": prompt,
                                "prompt_index": prompt_index,
                                "folder_image": "",
                                "source_index": source_index,
                                "source_name": f"input_image_{source_index:03d}",
                                "fixed_refs": [ref],
                            }
                        )
            else:
                for prompt_index, prompt in enumerate(prompts, start=1):
                    base_tasks.append(
                        {
                            "prompt": prompt,
                            "prompt_index": prompt_index,
                            "folder_image": "",
                            "source_index": 0,
                            "source_name": "",
                            "fixed_refs": [],
                        }
                    )

        expanded = []
        for base_index, task in enumerate(base_tasks, start=1):
            for candidate_index in range(1, candidates_per_prompt + 1):
                expanded.append({**task, "base_index": base_index, "candidate_index": candidate_index})

        if len(expanded) > max_tasks:
            warnings.append(f"任务数 {len(expanded)} 已超过 max_tasks={max_tasks}，本次只执行前 {max_tasks} 个任务。")
            expanded = expanded[:max_tasks]
        return expanded, warnings

    def _prepare_output_dir(self, filename_prefix: str) -> tuple[str, str, str, str | None]:
        target_dir, safe_prefix, subfolder, absolute_dir = _resolve_asset_output_target(filename_prefix, "BatchImage", ".png")
        run_name = f"{safe_prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_dir = os.path.join(target_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        if absolute_dir:
            return run_dir, safe_prefix, "", run_dir
        run_subfolder = "/".join(part for part in [subfolder, run_name] if part)
        return run_dir, safe_prefix, run_subfolder, None

    def _upload_refs_grsai(self, api_key: str, refs: list[Image.Image], cache: dict, lock: threading.Lock) -> list[str]:
        urls = []
        for pil_image in refs:
            path_key = getattr(pil_image, "filename", "")
            cache_key = f"path:{os.path.abspath(path_key)}" if path_key else f"pil:{id(pil_image)}"
            with lock:
                cached_url = cache.get(cache_key)
            if cached_url:
                urls.append(cached_url)
                continue
            url = upload_image_grsai(api_key, pil_image)
            if not url:
                raise CometAPIError("参考图上传失败，请检查输入图片或网络。")
            with lock:
                cache[cache_key] = url
            urls.append(url)
        return urls

    def _generate_once(
        self,
        channel: str,
        model: str,
        api_key: str,
        task: dict,
        aspect_ratio: str,
        image_size: str,
        quality: str,
        upload_cache: dict,
        upload_lock: threading.Lock,
        include_grsai_error_detail: bool = False,
    ) -> tuple[list[Image.Image], list[str], list[Image.Image]]:
        refs = []
        if task.get("folder_image"):
            refs.append(self._load_folder_image(task["folder_image"]))
        refs.extend(task.get("fixed_refs") or [])
        refs = refs[: self._max_refs(channel, model)]
        prompt_for_api = convert_prompt_asset_mentions(task["prompt"], image_count=len(refs))
        subtask_idx = int(task.get("base_index", 1)) * 1000 + int(task.get("candidate_index", 1))
        target_size = image_size if image_size in {"1K", "2K", "3K", "4K", "8K"} else "2K"

        try:
            if channel == "runninghub":
                images, errors = RunningHubImageAPI(api_key).generate_image(
                    prompt=prompt_for_api,
                    model=model,
                    pil_images=refs,
                    aspect_ratio=aspect_ratio,
                    image_size=target_size,
                    quality=quality,
                    subtask_idx=subtask_idx,
                )
                return images, errors, refs
            if channel == "modelverse":
                images, errors = ModelVerseImageAPI(api_key).generate_image(
                    prompt=prompt_for_api,
                    model=model,
                    pil_images=refs,
                    aspect_ratio=aspect_ratio,
                    image_size=target_size,
                    quality=quality,
                    subtask_idx=subtask_idx,
                )
                return images, errors, refs
            if channel == "apimart":
                images, errors = APIMartImageAPI(api_key).generate_image(
                    prompt=prompt_for_api,
                    model=model,
                    pil_images=refs,
                    aspect_ratio=aspect_ratio,
                    image_size=target_size,
                    quality=quality,
                    subtask_idx=subtask_idx,
                )
                return images, errors, refs
            if channel not in IMAGE_CHANNELS:
                if get_private_channel_spec(channel) and private_adapter_supports(channel, "image"):
                    images, errors = run_private_image_channel(
                        channel=channel,
                        api_key=api_key,
                        model=model,
                        prompt=prompt_for_api,
                        pil_images=refs,
                        aspect_ratio=aspect_ratio,
                        image_size=target_size,
                        quality=quality,
                        subtask_idx=subtask_idx,
                    )
                    return images, errors, refs
                custom_settings = load_comet_settings()
                custom_channel = get_custom_channel_settings(custom_settings, channel)
                if custom_channel:
                    custom_model = get_custom_model_settings(custom_channel, model, "image")
                else:
                    custom_model = (custom_settings.get("channels", {}).get(channel, {}).get("models", {}) or {}).get(model, {})
                custom_format = normalize_api_format_value(
                    custom_model.get("api_format") or detect_image_api_format(model),
                    "image",
                )
                custom_mode = normalize_image_interface_mode(
                    custom_model.get("interface_mode") or custom_model.get("endpoint_mode"),
                    custom_format,
                )
                images, errors = CustomOpenAIImageAPI(api_key, get_channel_api_url(channel)).generate_image(
                    prompt=prompt_for_api,
                    model=model,
                    pil_images=refs,
                    aspect_ratio=aspect_ratio,
                    image_size=target_size,
                    quality=quality,
                    api_format=custom_format,
                    interface_mode=custom_mode,
                    subtask_idx=subtask_idx,
                )
                return images, errors, refs

            is_gpt_image = model in GPT_IMAGE_MODELS
            if not is_gpt_image:
                if model == "nano-banana-fast" or model not in PRO_SIZE_MODELS:
                    target_size = "1K"
            actual_model = "nano-banana-2-4k-cl" if model == "nano-banana-2-cl" and target_size == "4K" else model
            uploaded_urls = self._upload_refs_grsai(api_key, refs, upload_cache, upload_lock)
            client = GrsaiAPI(api_key)
            if is_gpt_image:
                auto_aspect = nearest_aspect_ratio("auto", refs, list(GPT_IMAGE_VIP_SIZE_MAP.keys()), "1:1")
                return client.gpt_image_generate(
                    prompt=prompt_for_api,
                    model=model,
                    urls=uploaded_urls,
                    aspect_ratio=aspect_ratio if aspect_ratio in GPT_IMAGE_ASPECT_RATIOS else "1:1",
                    image_size=target_size,
                    quality=quality,
                    auto_aspect_ratio=auto_aspect,
                    include_error_detail=include_grsai_error_detail,
                ) + (refs,)
            return client.nano_banana_generate_image(
                prompt=prompt_for_api,
                model=actual_model,
                urls=uploaded_urls,
                aspect_ratio=aspect_ratio,
                image_size=target_size,
                include_error_detail=include_grsai_error_detail,
            ) + (refs,)
        except Exception as exc:
            return [], [format_error_message(exc)], refs

    def execute(
        self,
        channel: str,
        model: str,
        batch_mode: str,
        pairing_mode: str,
        folder_path: str,
        concurrency: int,
        aspect_ratio: str,
        image_size: str,
        quality: str = "medium",
        api_key: str = "",
        batch_text=None,
        **kwargs,
    ):
        channel = str(channel or "").lower()
        if channel not in get_image_channel_choices():
            return self._error(f"不支持的渠道：{channel}")
        model = resolve_model_id(channel, model, "image")
        batch_mode = self._normalize_batch_mode(batch_mode)
        pairing_mode = self._normalize_pairing_mode(pairing_mode, batch_mode)
        prompts = self._collect_prompts(batch_text)
        if not prompts:
            return self._error("批量生图需要连接批量文本卡片，并至少提供一条提示词。")

        try:
            advanced_settings = get_batch_image_advanced_settings()
        except Exception:
            advanced_settings = {
                "batch_concurrency": 20,
                "max_tasks": 200,
                "failure_image_policy": BATCH_IMAGE_FAILURE_POLICY_SKIP,
                "folder_preview_limit": BATCH_IMAGE_FOLDER_PREVIEW_LIMIT,
            }
        batch_concurrency = advanced_settings["batch_concurrency"]
        max_tasks = advanced_settings["max_tasks"]
        failure_image_policy = normalize_batch_image_failure_policy(advanced_settings.get("failure_image_policy"))
        folder_preview_limit = int(advanced_settings.get("folder_preview_limit") or BATCH_IMAGE_FOLDER_PREVIEW_LIMIT)
        try:
            candidates_per_prompt = max(1, min(10, int(concurrency)))
        except Exception:
            candidates_per_prompt = 1

        final_api_key = get_channel_api_key(api_key, channel, model, "image")
        if not final_api_key:
            return self._error(f"缺少 {channel} API Key，请先在设置中心填写。")

        try:
            max_refs = self._max_refs(channel, model)
            collect_ref_limit = self._fixed_ref_collect_limit(channel, model, batch_mode, pairing_mode)
            fixed_refs = self._collect_fixed_refs(kwargs, collect_ref_limit)
            tasks, warnings = self._build_tasks(
                prompts=prompts,
                fixed_refs=fixed_refs,
                batch_mode=batch_mode,
                pairing_mode=pairing_mode,
                folder_path=folder_path,
                candidates_per_prompt=candidates_per_prompt,
                max_tasks=max_tasks,
            )
            if not tasks:
                return self._error("没有可执行的批量任务。")

            use_folder_batch = self._use_folder_mode(batch_mode)
            is_grsai_multi_to_one = channel == "grsai" and not use_folder_batch
            run_dir, safe_prefix, run_subfolder, absolute_dir = self._prepare_output_dir("BatchImage")
            upload_cache: dict = {}
            upload_lock = threading.Lock()
            results = []
            failures = []
            consecutive_failures = 0
            abort_flag = False

            def run_task(index_and_task):
                task_index, task = index_and_task
                images, errors, refs = self._generate_once(
                    channel,
                    model,
                    final_api_key,
                    task,
                    aspect_ratio,
                    image_size,
                    quality,
                    upload_cache,
                    upload_lock,
                    include_grsai_error_detail=is_grsai_multi_to_one,
                )
                return task_index, task, images, errors, refs

            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_concurrency) as executor:
                futures = [executor.submit(run_task, (index, task)) for index, task in enumerate(tasks, start=1)]
                for future in concurrent.futures.as_completed(futures):
                    if use_folder_batch and abort_flag:
                        future.cancel()
                        continue

                    try:
                        task_index, task, images, errors, refs = future.result()
                    except Exception as exc:
                        failures.append({"task_index": -1, "error": format_error_message(exc)})
                        if use_folder_batch:
                            consecutive_failures += 1
                            if consecutive_failures >= BATCH_IMAGE_FOLDER_CIRCUIT_BREAKER_FAILURES and not abort_flag:
                                abort_flag = True
                                failures.append({"task_index": -1, "error": "BATCH ABORTED (Circuit Breaker)：连续发生硬性失败，停止后续任务。"})
                                for pending in futures:
                                    pending.cancel()
                        continue

                    if use_folder_batch and not images and failure_image_policy == BATCH_IMAGE_FAILURE_POLICY_WHITE:
                        original_error = "; ".join(errors) if errors else "0 output"
                        try:
                            images = [self._folder_failure_placeholder(task, aspect_ratio, image_size, model)]
                            errors = [f"白图替代(原错误: {original_error})"]
                        except Exception as placeholder_exc:
                            errors = [f"{original_error}; 生成白图失败: {format_error_message(placeholder_exc)}"]

                    if errors:
                        failures.append(
                            {
                                "task_index": task_index,
                                "prompt_index": task.get("prompt_index"),
                                "source_index": task.get("source_index"),
                                "candidate_index": task.get("candidate_index"),
                                "source_name": task.get("source_name") or "",
                                "error": "; ".join(errors),
                            }
                        )
                    elif not images:
                        failures.append(
                            {
                                "task_index": task_index,
                                "prompt_index": task.get("prompt_index"),
                                "source_index": task.get("source_index"),
                                "candidate_index": task.get("candidate_index"),
                                "source_name": task.get("source_name") or "",
                                "error": "0 output",
                            }
                        )
                    if use_folder_batch:
                        if images:
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                            if consecutive_failures >= BATCH_IMAGE_FOLDER_CIRCUIT_BREAKER_FAILURES and not abort_flag:
                                abort_flag = True
                                failures.append({"task_index": -1, "error": "BATCH ABORTED (Circuit Breaker)：连续发生硬性失败，停止后续任务。"})
                                for pending in futures:
                                    pending.cancel()
                                continue
                    for result_index, pil_image in enumerate(images, start=1):
                        safe_image = safe_pil_to_rgb(pil_image)
                        prompt_part = f"p{int(task.get('prompt_index') or 0):03d}"
                        source_part = f"s{int(task.get('source_index') or 0):03d}" if task.get("source_index") else "fixed"
                        candidate_part = f"c{int(task.get('candidate_index') or 1):02d}"
                        stem = _batch_image_slug(task.get("source_name") or safe_prefix, safe_prefix)
                        filename = f"{task_index:04d}_{prompt_part}_{source_part}_{candidate_part}_{result_index:02d}_{stem}_{uuid.uuid4().hex[:6]}.png"
                        save_path = os.path.join(run_dir, filename)
                        safe_image.save(save_path, "PNG")
                        ref = _make_asset_ref(filename, run_subfolder, absolute_dir)
                        results.append(
                            {
                                "task_index": task_index,
                                "prompt_index": task.get("prompt_index"),
                                "source_index": task.get("source_index"),
                                "candidate_index": task.get("candidate_index"),
                                "result_index": result_index,
                                "prompt": task.get("prompt"),
                                "source_name": task.get("source_name") or "",
                                "reference_count": len(refs),
                                "filename": filename,
                                "asset_ref": ref,
                            }
                        )

            results.sort(key=lambda item: (item["task_index"], item["result_index"]))
            summary = self._build_summary(
                channel=channel,
                model=model,
                batch_mode=batch_mode,
                pairing_mode=pairing_mode,
                prompts=prompts,
                fixed_refs=fixed_refs,
                tasks=tasks,
                results=results,
                failures=failures,
                warnings=warnings,
                run_dir=run_dir,
                api_key=final_api_key,
                aborted=abort_flag,
                failure_image_policy=failure_image_policy,
                folder_preview_limit=folder_preview_limit,
            )

            preview_results = results[:folder_preview_limit] if use_folder_batch else results
            output_images = [load_asset_image(item["asset_ref"]) for item in preview_results] if preview_results else []
            ui = {
                "comet_text": [summary],
            }
            if preview_results:
                ui["asset_ref"] = [asset_refs_to_json([item["asset_ref"] for item in preview_results])]
                ui["asset_index"] = [0]
            if warnings:
                ui["comet_warning"] = ["；".join(warnings)]
            if failures:
                failure_notice = f"有 {len(failures)} 个任务失败，详情见 summary 输出。"
                ui["comet_warning"] = [ui.get("comet_warning", [""])[0] + ("\n" if warnings else "") + failure_notice]
            if not results:
                ui["comet_error"] = [summary]
            # 把本次执行产生的 asset_refs 缓存起来，下游图像卡片可凭此免去对 padded tensor 落盘。
            if preview_results:
                remember_batch_asset_refs(
                    kwargs.get("_unique_id"),
                    [item["asset_ref"] for item in preview_results],
                )
            return {
                "ui": ui,
                "result": (_batch_image_tensor(output_images), summary),
            }
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._error(format_error_message(exc))


class PrivateMusicAPI:
    def __init__(self, api_key: str, api_url: str = ""):
        if not api_key:
            raise CometAPIError("缺少私有渠道 API Key，请先在设置中心填写。")
        self.api_key = api_key
        self.host = normalize_api_base_url(api_url or "", "")

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }

    def _parse_json_response(self, response) -> dict:
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            text = response.text.strip()
            if text.startswith(("http://", "https://")):
                return {"data": text}
            raise CometAPIError(f"音乐 API 返回了非 JSON 内容：{text[:300]}") from exc

    def _post_json(self, path: str, payload: dict, timeout: int = 60) -> dict:
        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(f"{self.host}{path}", headers=self._headers(), json=payload, timeout=timeout)
                response.raise_for_status()
                return self._parse_json_response(response)
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        raise last_error

    def _get_json(self, path: str, timeout: int = 30) -> dict:
        last_error = None
        for attempt in range(3):
            try:
                response = requests.get(f"{self.host}{path}", headers=self._headers(), timeout=timeout)
                response.raise_for_status()
                return self._parse_json_response(response)
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        raise last_error

    def _extract_task_id(self, data: dict) -> str:
        if isinstance(data.get("data"), str):
            return data["data"]
        if isinstance(data.get("data"), dict):
            body = data["data"]
            task_id = body.get("task_id") or body.get("id")
            if task_id:
                return str(task_id)
        task_id = data.get("task_id") or data.get("id")
        if task_id:
            return str(task_id)
        raise CometAPIError(f"\u97f3\u4e50 API \u6ca1\u6709\u8fd4\u56de\u4efb\u52a1 ID\uff1a{str(data)[:300]}")

    def _submit_task(self, path: str, payload: dict) -> str:
        data = self._post_json(path, payload, timeout=90)
        code = str(data.get("code") or "").lower()
        if code and code not in {"success", "ok", "0"}:
            raise CometAPIError(data.get("message") or str(data)[:300])
        return self._extract_task_id(data)

    def _task_body(self, data: dict):
        body = data.get("data")
        if isinstance(body, dict):
            return body
        if isinstance(body, list):
            return body
        return data

    def _nested_values(self, value, keys: set[str]) -> list:
        found = []
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in keys and child not in (None, ""):
                    found.append(child)
                found.extend(self._nested_values(child, keys))
        elif isinstance(value, list):
            for child in value:
                found.extend(self._nested_values(child, keys))
        return found

    def _first_nested_value(self, value, keys: set[str]):
        values = self._nested_values(value, keys)
        return values[0] if values else None

    def _status_of(self, data: dict, path_kind: str = "music") -> str:
        body = self._task_body(data)
        raw = self._first_nested_value(body, {"status", "state", "task_status"}) or data.get("status")
        status = str(raw or "").strip().lower()
        if status in {"failure", "failed", "fail", "error", "cancelled", "canceled"}:
            return "failed"
        if self._extract_audio_url(data):
            return "success"
        if path_kind == "lyrics":
            if self._extract_lyrics(data):
                return "success"
            if status in {"success", "succeeded", "completed", "complete", "finish", "finished", "done"}:
                return "success"
        elif status in {"success", "succeeded", "completed", "complete", "finish", "finished", "done"}:
            if self._resolve_ready_audio_urls(data, prefer_wav=False, quiet=True):
                return "success"
            return "processing"
        return status or "processing"

    def _failure_message(self, data: dict) -> str:
        body = self._task_body(data)
        message = (
            self._first_nested_value(body, {"failreason", "fail_reason", "error_message", "error", "message"})
            or data.get("message")
            or "\u672a\u77e5\u539f\u56e0"
        )
        return str(message)

    def _wait_for_task(self, task_id: str, timeout: int = 900, interval: int = 5, path_kind: str = "music") -> dict:
        started = time.time()
        last_data = {}
        while time.time() - started < timeout:
            data = self._get_json(f"/suno/fetch/{requests.utils.quote(str(task_id), safe='')}", timeout=30)
            last_data = data
            status = self._status_of(data, path_kind=path_kind)
            if status == "success":
                return data
            if status == "failed":
                raise CometAPIError(f"\u97f3\u4e50\u4efb\u52a1\u5931\u8d25\uff1a{self._failure_message(data)}")
            time.sleep(interval)
        raise CometAPIError(f"\u97f3\u4e50\u4efb\u52a1\u7b49\u5f85\u8d85\u65f6\uff08{timeout} \u79d2\uff09\uff0c\u6700\u540e\u72b6\u6001\uff1a{self._status_of(last_data, path_kind=path_kind)}")

    def _clip_candidates(self, data: dict) -> list:
        body = self._task_body(data)
        if isinstance(body, list):
            return [item for item in body if isinstance(item, dict)]
        source_keys = (
            "data",
            "clips",
            "clip",
            "songs",
            "song",
            "items",
            "results",
            "result",
            "audios",
            "audio",
            "feed",
        )
        if isinstance(body, dict):
            for key in source_keys:
                source = body.get(key)
                if isinstance(source, list):
                    clips = [item for item in source if isinstance(item, dict)]
                    if clips:
                        return clips
                if isinstance(source, dict):
                    return [source]
        return [body] if isinstance(body, dict) else []

    def _select_clip(self, data: dict, track_index: str = "1") -> dict:
        candidates = self._clip_candidates(data)
        if not candidates:
            body = self._task_body(data)
            return body if isinstance(body, dict) else data
        try:
            index = max(0, int(track_index) - 1)
        except Exception:
            index = 0
        return candidates[min(index, len(candidates) - 1)]

    def _collect_urls(self, value, urls: list[tuple[int, str]] | None = None, context: str = "") -> list[tuple[int, str]]:
        if urls is None:
            urls = []
        if isinstance(value, dict):
            for key, child in value.items():
                self._collect_urls(child, urls, f"{context}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._collect_urls(child, urls, f"{context}[{index}]")
        elif isinstance(value, str) and value.startswith(("http://", "https://")):
            lower_url = value.lower()
            lower_context = context.lower()
            score = 0
            if any(token in lower_url for token in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")):
                score += 100
            if any(token in lower_context for token in ("audio", "song", "music", "stream", "download", "url")):
                score += 30
            if any(token in lower_context for token in ("image", "cover", "avatar")):
                score -= 80
            urls.append((score, value))
        return urls

    def _extract_audio_url(self, data: dict, track_index: str = "1", prefer_wav: bool = False) -> str:
        clip = self._select_clip(data, track_index)
        direct_keys = {"wav_url", "audio_wav_url", "download_url"} if prefer_wav else set()
        direct_keys |= {"audio_url", "stream_audio_url", "source_audio_url", "url", "mp3_url", "song_url"}
        direct = self._first_nested_value(clip, direct_keys)
        if isinstance(direct, str) and direct.startswith(("http://", "https://")):
            return direct
        urls = self._collect_urls(clip)
        if not urls:
            urls = self._collect_urls(data)
        urls.sort(key=lambda item: item[0], reverse=True)
        return urls[0][1] if urls and urls[0][0] > 0 else ""

    def _extract_clip_id(self, data: dict, track_index: str = "1") -> str:
        clip = self._select_clip(data, track_index)
        if isinstance(clip, dict):
            for key in ("clip_id", "clipId", "clipID", "clipid"):
                value = clip.get(key)
                if value:
                    return str(value)
            value = clip.get("id")
            if value:
                return str(value)
        value = self._first_nested_value(clip, {"clip_id", "clipid", "id"})
        return str(value or "")

    def _extract_lyrics(self, data: dict) -> str:
        body = self._task_body(data)
        value = self._first_nested_value(body, {"lyrics", "lyric", "text", "prompt"})
        if isinstance(value, str):
            return value.strip()
        if isinstance(body, str):
            return body.strip()
        return ""

    def _lookup_audio_url_by_id(self, clip_id: str, prefer_wav: bool = False, quiet: bool = False) -> str:
        if not clip_id:
            return ""
        clip_id = str(clip_id).strip()
        if clip_id.startswith(("http://", "https://")):
            return clip_id
        encoded = requests.utils.quote(clip_id, safe="")
        lookups = (
            ("get", f"/suno/feed/{encoded}", None),
            ("get", f"/suno/fetch/{encoded}", None),
            ("post", "/suno/fetch", {"ids": [clip_id]}),
        )
        for method, path, payload in lookups:
            try:
                data = self._get_json(path, timeout=60) if method == "get" else self._post_json(path, payload or {}, timeout=60)
            except Exception as exc:
                if not quiet:
                    print(f"[CometAPI Music] 音频详情获取失败，继续尝试其他地址：{format_error_message(exc)}")
                continue
            value = data.get("data") if isinstance(data, dict) else None
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
            url = self._extract_audio_url(data, prefer_wav=prefer_wav)
            if url:
                return url
        return ""

    def _audio_url_for_clip(self, clip_id: str, prefer_wav: bool = False, quiet: bool = False) -> str:
        if not clip_id:
            return ""
        return self._lookup_audio_url_by_id(clip_id, prefer_wav=False, quiet=quiet)

    def _resolve_ready_audio_urls(self, data: dict, prefer_wav: bool = False, quiet: bool = False) -> dict:
        candidates = self._clip_candidates(data)
        track_total = max(1, min(2, len(candidates) or 1))
        resolved = {}
        for index in range(1, track_total + 1):
            track = str(index)
            url = self._extract_audio_url(data, track, prefer_wav=prefer_wav)
            if not url:
                clip_id = self._extract_clip_id(data, track)
                if clip_id:
                    url = self._audio_url_for_clip(clip_id, prefer_wav=prefer_wav, quiet=quiet)
            if not url:
                return {}
            resolved[track] = url
        if isinstance(data, dict):
            data["_comet_resolved_audio_urls"] = resolved
        return resolved

    def _submit_upload(self, audio_path: str) -> str:
        if not audio_path or not os.path.exists(audio_path):
            raise CometAPIError("\u4e0a\u4f20\u97f3\u9891\u6a21\u5f0f\u9700\u8981\u8fde\u63a5 AUDIO \u6216\u586b\u5199\u53ef\u8bbf\u95ee\u7684\u97f3\u9891\u8def\u5f84\u3002")
        ext = os.path.splitext(audio_path)[1].lower().lstrip(".") or "mp3"
        init = self._post_json("/suno/uploads/audio", {"extension": ext}, timeout=60)
        upload_id = str(init.get("id") or init.get("upload_id") or "")
        upload_url = str(init.get("url") or init.get("upload_url") or "")
        fields = init.get("fields") if isinstance(init.get("fields"), dict) else {}
        if not upload_id or not upload_url:
            raise CometAPIError(f"\u97f3\u9891\u4e0a\u4f20\u6388\u6743\u5931\u8d25\uff1a{str(init)[:300]}")

        content_type = fields.get("Content-Type") or _guess_media_mime(audio_path, "audio/mpeg")
        with open(audio_path, "rb") as handle:
            files = {"file": (os.path.basename(audio_path), handle, content_type)}
            response = requests.post(upload_url, data=fields, files=files, timeout=(20, 600))
            response.raise_for_status()

        self._post_json(
            f"/suno/uploads/audio/{requests.utils.quote(upload_id, safe='')}/upload-finish",
            {"upload_type": "file_upload", "upload_filename": os.path.basename(audio_path)},
            timeout=60,
        )
        started = time.time()
        while time.time() - started < 300:
            data = self._get_json(f"/suno/uploads/audio/{requests.utils.quote(upload_id, safe='')}", timeout=30)
            status = str(data.get("status") or "").strip().lower()
            if status in {"complete", "completed", "success", "succeeded"}:
                break
            if status in {"failed", "failure", "error"}:
                raise CometAPIError(data.get("error_message") or str(data)[:300])
            time.sleep(3)
        else:
            raise CometAPIError("\u97f3\u9891\u4e0a\u4f20\u5904\u7406\u8d85\u65f6\u3002")

        data = self._post_json(f"/suno/uploads/audio/{requests.utils.quote(upload_id, safe='')}/initialize-clip", {}, timeout=60)
        clip_id = str(data.get("clip_id") or data.get("id") or "")
        if not clip_id:
            raise CometAPIError(f"\u521d\u59cb\u5316\u97f3\u9891 clip \u5931\u8d25\uff1a{str(data)[:300]}")
        return clip_id

    def submit_music(self, payload: dict) -> str:
        return self._submit_task("/suno/submit/music", payload)

    def submit_lyrics(self, prompt: str) -> str:
        return self._submit_task("/suno/submit/lyrics", {"prompt": prompt})

    def submit_concat(self, clip_id: str, is_infill: bool = False) -> str:
        return self._submit_task("/suno/submit/concat", {"clip_id": clip_id, "is_infill": bool(is_infill)})

    def run_and_fetch(self, path_kind: str, payload: dict, timeout: int) -> tuple[str, dict]:
        if path_kind == "lyrics":
            task_id = self.submit_lyrics(str(payload.get("prompt") or ""))
        elif path_kind == "concat":
            task_id = self.submit_concat(str(payload.get("clip_id") or ""), bool(payload.get("is_infill")))
        else:
            task_id = self.submit_music(payload)
        return task_id, self._wait_for_task(task_id, timeout=timeout, path_kind=path_kind)


def canonical_runninghub_music_model(model: str) -> str:
    raw = str(model or "").strip()
    normalized = raw.lower()
    if normalized in {MUSIC_MODEL_V55.lower(), "v5.5", "suno v5.5"}:
        return RUNNINGHUB_MUSIC_MODEL_V55
    if normalized in {MUSIC_MODEL_V5.lower(), "v5", "suno v5"}:
        return RUNNINGHUB_MUSIC_MODEL_V5
    if normalized in {MUSIC_MODEL_V45.lower(), "v4.5", "suno v4.5"}:
        return RUNNINGHUB_MUSIC_MODEL_V45
    for model_id in RUNNINGHUB_MUSIC_MODELS:
        if raw == model_id or normalized == model_id.lower():
            return model_id
    return RUNNINGHUB_MUSIC_MODEL_ALIAS_TO_ID.get(normalized, raw or RUNNINGHUB_MUSIC_MODEL_V55)


class RunningHubMusicAPI:
    host = "https://www.runninghub.cn"

    def __init__(self, api_key: str):
        if not api_key:
            raise CometAPIError("\u7f3a\u5c11 runninghub API Key\uff0c\u8bf7\u5148\u5728\u8bbe\u7f6e\u4e2d\u5fc3\u586b\u5199\u3002")
        self.api_key = api_key

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _post_json(self, path: str, payload: dict, timeout: int = 60) -> dict:
        response = requests.post(f"{self.host}{path}", headers=self._headers(), json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json() if response.content else {}

    def _extract_task_id(self, data: dict) -> str:
        task_id = data.get("taskId") or data.get("task_id") or data.get("id")
        if task_id:
            return str(task_id)
        body = data.get("data")
        if isinstance(body, dict):
            task_id = body.get("taskId") or body.get("task_id") or body.get("id")
            if task_id:
                return str(task_id)
        raise CometAPIError(f"RunningHub 音乐 API 没有返回任务 ID：{str(data)[:300]}")

    def _submit_task(self, path: str, payload: dict) -> str:
        data = self._post_json(path, payload, timeout=120)
        message = data.get("errorMessage") or data.get("message") or ""
        status = str(data.get("status") or "").strip().upper()
        if status in {"FAILED", "FAILURE", "ERROR"}:
            raise CometAPIError(message or str(data)[:300])
        if data.get("errorCode"):
            raise CometAPIError(message or str(data)[:300])
        return self._extract_task_id(data)

    def _task_body(self, data: dict):
        return data.get("data") if isinstance(data.get("data"), dict) else data

    def _nested_values(self, value, keys: set[str]) -> list:
        found = []
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in keys and child not in (None, ""):
                    found.append(child)
                found.extend(self._nested_values(child, keys))
        elif isinstance(value, list):
            for child in value:
                found.extend(self._nested_values(child, keys))
        return found

    def _first_nested_value(self, value, keys: set[str]):
        values = self._nested_values(value, keys)
        return values[0] if values else None

    def _status_of(self, data: dict, path_kind: str = "music") -> str:
        raw = self._first_nested_value(data, {"status"}) or data.get("status")
        status = str(raw or "").strip().lower()
        if status in {"success", "succeeded", "completed", "complete", "finish", "finished", "done"}:
            return "success"
        if status in {"failed", "failure", "fail", "error", "cancelled", "canceled"}:
            return "failed"
        if self._extract_audio_url(data):
            return "success"
        if path_kind == "lyrics" and self._extract_lyrics(data):
            return "success"
        return status or "processing"

    def _failure_message(self, data: dict) -> str:
        failed_reason = data.get("failedReason")
        if isinstance(failed_reason, dict) and failed_reason:
            failed_reason = json.dumps(failed_reason, ensure_ascii=False)
        message = (
            data.get("errorMessage")
            or data.get("message")
            or failed_reason
            or self._first_nested_value(data, {"errormessage", "error_message", "error", "message"})
            or "\u672a\u77e5\u539f\u56e0"
        )
        return str(message)

    def _wait_for_task(self, task_id: str, timeout: int = 1200, interval: int = 5, path_kind: str = "music") -> dict:
        started = time.time()
        last_data = {}
        while time.time() - started < timeout:
            data = self._post_json("/openapi/v2/query", {"taskId": str(task_id)}, timeout=30)
            last_data = data
            status = self._status_of(data, path_kind=path_kind)
            if status == "success":
                return data
            if status == "failed":
                raise CometAPIError(f"RunningHub 音乐任务失败：{self._failure_message(data)}")
            time.sleep(interval)
        raise CometAPIError(f"RunningHub 音乐任务等待超时（{timeout} 秒），最后状态：{self._status_of(last_data)}")

    def _clip_candidates(self, data: dict) -> list:
        body = self._task_body(data)
        results = body.get("results") if isinstance(body, dict) else None
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
        if isinstance(results, dict):
            return [results]
        return [body] if isinstance(body, dict) else []

    def _select_clip(self, data: dict, track_index: str = "1") -> dict:
        candidates = self._clip_candidates(data)
        if not candidates:
            return data
        try:
            index = max(0, int(track_index) - 1)
        except Exception:
            index = 0
        return candidates[min(index, len(candidates) - 1)]

    def _collect_urls(self, value, urls: list[tuple[int, str]] | None = None, context: str = "") -> list[tuple[int, str]]:
        if urls is None:
            urls = []
        if isinstance(value, dict):
            output_type = str(value.get("outputType") or value.get("type") or "").lower()
            direct_url = value.get("url") or value.get("download_url")
            if isinstance(direct_url, str) and direct_url.startswith(("http://", "https://")):
                score = 40
                if output_type in {"mp3", "wav", "m4a", "aac", "flac", "ogg", "audio"}:
                    score += 120
                if output_type in {"png", "jpg", "jpeg", "webp", "mp4", "txt"}:
                    score -= 100
                urls.append((score, direct_url))
            for key, child in value.items():
                self._collect_urls(child, urls, f"{context}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._collect_urls(child, urls, f"{context}[{index}]")
        elif isinstance(value, str) and value.startswith(("http://", "https://")):
            lower_url = value.lower()
            lower_context = context.lower()
            score = 0
            if any(token in lower_url for token in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")):
                score += 100
            if any(token in lower_context for token in ("audio", "song", "music", "stream", "download", "url")):
                score += 30
            if any(token in lower_context for token in ("image", "cover", "avatar", "video", "text")):
                score -= 80
            urls.append((score, value))
        return urls

    def _extract_audio_url(self, data: dict, track_index: str = "1", prefer_wav: bool = False) -> str:
        clip = self._select_clip(data, track_index)
        output_type = str(clip.get("outputType") or clip.get("type") or "").lower() if isinstance(clip, dict) else ""
        direct = clip.get("url") if isinstance(clip, dict) else ""
        if isinstance(direct, str) and direct.startswith(("http://", "https://")):
            if output_type in {"mp3", "wav", "m4a", "aac", "flac", "ogg", "audio"}:
                return direct
            if output_type not in {"png", "jpg", "jpeg", "webp", "mp4", "txt"}:
                return direct
        urls = self._collect_urls(clip)
        if not urls:
            urls = self._collect_urls(data)
        urls.sort(key=lambda item: item[0], reverse=True)
        return urls[0][1] if urls and urls[0][0] > 0 else ""

    def _extract_clip_id(self, data: dict, track_index: str = "1") -> str:
        clip = self._select_clip(data, track_index)
        value = self._first_nested_value(clip, {"clipid", "clip_id", "nodeid", "node_id", "id"}) if isinstance(clip, dict) else ""
        return str(value or data.get("taskId") or "")

    def _extract_lyrics(self, data: dict) -> str:
        candidates = self._clip_candidates(data)
        texts = []
        for item in candidates:
            text = item.get("text") if isinstance(item, dict) else ""
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        if texts:
            return "\n\n".join(texts)
        value = self._first_nested_value(data, {"lyrics", "lyric", "text", "prompt"})
        return value.strip() if isinstance(value, str) else ""

    def run_and_fetch(self, path_kind: str, model: str, payload: dict, timeout: int) -> tuple[str, dict]:
        if path_kind == "lyrics":
            task_id = self._submit_task("/openapi/v2/rhart-audio/suno/lyrics", payload)
        else:
            safe_model = canonical_runninghub_music_model(model)
            action = "custom" if path_kind == "custom" else "single"
            task_id = self._submit_task(f"/openapi/v2/rhart-audio/{safe_model}/{action}", payload)
        return task_id, self._wait_for_task(task_id, timeout=timeout, path_kind=path_kind)


class CometAPIUnifiedMusic:
    CATEGORY = "COMET"
    FUNCTION = "execute"
    RETURN_TYPES = (IO.AUDIO, "STRING")
    RETURN_NAMES = ("audio", "summary")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "channel": (get_music_channel_choices(), {"default": get_default_music_channel()}),
                "music_mode": (MUSIC_MODES, {"default": MUSIC_MODE_GENERATE}),
                "music_submode": (MUSIC_SUBMODES, {"default": MUSIC_SUBMODE_INSPIRE}),
                "model": (get_music_model_choices(), {"default": MUSIC_MODEL_V55}),
                "prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "歌词、歌词主题或纯音乐描述"}),
                "title": ("STRING", {"default": "", "placeholder": "歌名"}),
                "tags": ("STRING", {"default": "", "placeholder": "风格标签，如 pop, electronic"}),
                "gpt_description_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "描述歌曲灵感、风格、情绪或场景"}),
                "vocal_gender": (MUSIC_VOCAL_GENDERS, {"default": MUSIC_VOCAL_GENDERS[0]}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def _error(self, message: str):
        message = redact_sensitive_text(message)
        print(f"[CometAPI Music] {message}")
        return {"ui": {"comet_error": [message], "comet_text": [message]}, "result": (silent_audio(), message)}

    def _normalize_mode(self, music_mode: str, music_submode: str) -> tuple[str, str]:
        mode = music_mode if music_mode in MUSIC_MODES else MUSIC_MODE_GENERATE
        submode = music_submode if music_submode in MUSIC_SUBMODES else MUSIC_SUBMODE_INSPIRE
        if mode == MUSIC_MODE_GENERATE and submode not in {MUSIC_SUBMODE_INSPIRE, MUSIC_SUBMODE_CUSTOM, MUSIC_SUBMODE_INSTRUMENTAL}:
            submode = MUSIC_SUBMODE_INSPIRE
        elif mode == MUSIC_MODE_LYRICS:
            submode = MUSIC_SUBMODE_LYRICS
        return mode, submode

    def _with_vocal_gender(self, payload: dict, vocal_gender: str) -> dict:
        if vocal_gender == MUSIC_VOCAL_GENDERS[1]:
            payload["metadata"] = {"create_mode": "custom", "vocal_gender": "m"}
        elif vocal_gender == MUSIC_VOCAL_GENDERS[2]:
            payload["metadata"] = {"create_mode": "custom", "vocal_gender": "f"}
        return payload

    def _input_audio_path(self, audio, audio_path: str) -> tuple[str, bool]:
        if audio is not None:
            return _media_input_to_path(audio, ".wav")
        path = _resolve_existing_media_path(audio_path)
        return path, False

    def _build_summary(
        self,
        *,
        mode: str,
        submode: str,
        model: str,
        task_id: str,
        clip_id: str = "",
        title: str = "",
        audio_url: str = "",
        audio_path: str = "",
        lyrics: str = "",
    ) -> str:
        lines = [
            "\u97f3\u4e50\u4efb\u52a1\u5b8c\u6210",
            f"\u6a21\u5f0f\uff1a{mode}",
            f"\u5b50\u6a21\u5f0f\uff1a{submode}",
            f"\u6a21\u578b\uff1a{model}",
            f"\u4efb\u52a1ID\uff1a{task_id}",
        ]
        if clip_id:
            lines.append(f"clip_id\uff1a{clip_id}")
        if title:
            lines.append(f"\u6807\u9898\uff1a{title}")
        if audio_path:
            lines.append(f"\u8f93\u51fa\u97f3\u9891\uff1a{audio_path}")
        if audio_url:
            lines.append(f"\u97f3\u9891URL\uff1a{audio_url}")
        if lyrics:
            lines.append("\u6b4c\u8bcd\uff1a")
            lines.append(lyrics)
        return "\n".join(lines)

    def _download_task_audio(
        self,
        api,
        task_data: dict,
        *,
        track: str,
        filename_prefix: str,
        timeout: int,
    ) -> tuple[dict, str, str, str]:
        clip_id = api._extract_clip_id(task_data, track)
        resolved_urls = task_data.get("_comet_resolved_audio_urls") if isinstance(task_data, dict) else None
        audio_url = resolved_urls.get(str(track), "") if isinstance(resolved_urls, dict) else ""
        if not audio_url:
            audio_url = api._extract_audio_url(task_data, track, prefer_wav=False)
        if not audio_url and clip_id and hasattr(api, "_audio_url_for_clip"):
            audio_url = api._audio_url_for_clip(clip_id, prefer_wav=False)
        if not audio_url:
            clip = api._select_clip(task_data, track) if hasattr(api, "_select_clip") else task_data
            detail = json.dumps(clip, ensure_ascii=False, default=str)[:600] if isinstance(clip, (dict, list)) else str(clip)[:600]
            raise CometAPIError(f"音乐任务已完成，但 API 没有返回可下载的音频地址。clip_id：{clip_id or '无'}；返回片段：{detail}")
        path = download_audio_asset(audio_url, prefix=filename_prefix, timeout=timeout)
        return load_audio_file(path), path, audio_url, clip_id

    def _download_all_task_audio(
        self,
        api,
        task_data: dict,
        *,
        filename_prefix: str,
        timeout: int,
    ) -> tuple[dict, list[str], list[str], list[str]]:
        candidates = api._clip_candidates(task_data)
        track_total = max(1, min(2, len(candidates) or 1))
        audios = []
        paths = []
        urls = []
        clip_ids = []
        last_error = None
        for index in range(1, track_total + 1):
            try:
                audio, path, url, clip_id = self._download_task_audio(
                    api,
                    task_data,
                    track=str(index),
                    filename_prefix=f"{filename_prefix}_{index}",
                    timeout=timeout,
                )
                audios.append(audio)
                paths.append(path)
                urls.append(url)
                if clip_id:
                    clip_ids.append(clip_id)
            except Exception as exc:
                last_error = exc
                if index == 1:
                    raise
        if not audios:
            raise CometAPIError(format_error_message(last_error) if last_error else "\u97f3\u4e50 API \u6ca1\u6709\u8fd4\u56de\u53ef\u7528\u97f3\u9891\u3002")
        audio_result = concat_audio_results(audios)
        audio_result["comet_audio_refs"] = [asset_ref_from_path(path) for path in paths if path]
        audio_result["comet_audio_paths"] = paths
        audio_result["comet_audio_urls"] = urls
        return audio_result, paths, urls, clip_ids

    def _upload_clip(self, api: PrivateMusicAPI, audio, audio_path: str) -> str:
        path, cleanup = self._input_audio_path(audio, audio_path)
        try:
            return api._submit_upload(path)
        finally:
            if cleanup and path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass

    def execute(
        self,
        channel: str = get_default_music_channel(),
        music_mode: str = MUSIC_MODE_GENERATE,
        music_submode: str = MUSIC_SUBMODE_INSPIRE,
        model: str = MUSIC_MODEL_V55,
        prompt: str = "",
        title: str = "",
        tags: str = "",
        gpt_description_prompt: str = "",
        vocal_gender: str = MUSIC_VOCAL_GENDERS[0],
        timeout: int = 1200,
        filename_prefix: str = "CometAPIMusic",
        api_key: str = "",
        **kwargs,
    ):
        channel_key = str(channel or get_default_music_channel()).lower()
        private_channel = get_private_channel_spec(channel_key)
        if channel_key not in get_music_channel_choices():
            return self._error(f"\u4e0d\u652f\u6301\u7684\u97f3\u4e50\u6e20\u9053\uff1a{channel}")
        mode, submode = self._normalize_mode(music_mode, music_submode)
        default_model = RUNNINGHUB_MUSIC_MODEL_V55 if channel_key == "runninghub" else MUSIC_MODEL_V55
        model = resolve_model_id(channel_key, model, "music") or default_model
        if channel_key == "runninghub":
            model = canonical_runninghub_music_model(model)
        model_label = (RUNNINGHUB_MUSIC_MODEL_ALIASES if channel_key == "runninghub" else MUSIC_MODEL_ALIASES).get(model, model)
        final_api_key = get_channel_api_key(api_key, channel_key, model, "music")
        if not final_api_key:
            return self._error(f"\u7f3a\u5c11 {channel_key} API Key\uff0c\u8bf7\u5148\u5728\u8bbe\u7f6e\u4e2d\u5fc3\u586b\u5199\u3002")

        try:
            if private_channel:
                timeout = max(60, int(timeout or 1200))
                audio_result, summary, ui = run_private_music_channel(
                    channel=channel_key,
                    api_key=final_api_key,
                    model=model,
                    prompt=prompt,
                    music_mode=mode,
                    music_submode=submode,
                    title=title,
                    tags=tags,
                    gpt_description_prompt=gpt_description_prompt,
                    vocal_gender=vocal_gender,
                    timeout=timeout,
                )
                ui = {"comet_text": [summary], **(ui or {})}
                return {"ui": ui, "result": (audio_result, summary)}
            api = RunningHubMusicAPI(final_api_key) if channel_key == "runninghub" else PrivateMusicAPI(final_api_key, get_channel_api_url(channel_key))
            prompt = str(prompt or "").strip()
            title = str(title or "").strip()
            tags = str(tags or "").strip()
            description = str(gpt_description_prompt or "").strip() or prompt or title
            timeout = max(60, int(timeout or 1200))

            if mode == MUSIC_MODE_LYRICS:
                lyrics_prompt = prompt or description
                if not lyrics_prompt:
                    return self._error("\u751f\u6210\u6b4c\u8bcd\u9700\u8981\u586b\u5199\u63d0\u793a\u8bcd\u3002")
                if channel_key == "runninghub":
                    task_id, task_data = api.run_and_fetch("lyrics", model, {"prompt": lyrics_prompt}, timeout)
                else:
                    task_id, task_data = api.run_and_fetch("lyrics", {"prompt": lyrics_prompt}, timeout)
                lyrics = api._extract_lyrics(task_data) or str(task_data)
                summary = self._build_summary(
                    mode=mode,
                    submode=submode,
                    model=model_label,
                    task_id=task_id,
                    title=title,
                    lyrics=lyrics,
                )
                return {"ui": {"comet_text": [summary], "comet_task_id": [task_id]}, "result": (silent_audio(), summary)}

            if channel_key == "runninghub":
                if submode == MUSIC_SUBMODE_CUSTOM:
                    if not prompt:
                        return self._error("\u81ea\u5b9a\u4e49\u6b4c\u8bcd\u6a21\u5f0f\u9700\u8981\u586b\u5199\u6b4c\u8bcd/\u63d0\u793a\u8bcd\u3002")
                    if not title or not tags:
                        return self._error("\u81ea\u5b9a\u4e49\u6b4c\u8bcd\u6a21\u5f0f\u9700\u8981\u586b\u5199\u6b4c\u540d\u548c\u98ce\u683c\u6807\u7b7e\u3002")
                    payload = {
                        "title": title,
                        "prompt": prompt,
                        "tags": tags,
                    }
                    task_kind = "custom"
                else:
                    single_description = (prompt or description or "").strip()
                    if not single_description:
                        return self._error("\u751f\u6210\u6b4c\u66f2\u9700\u8981\u586b\u5199\u97f3\u4e50\u63cf\u8ff0\u3002")
                    payload = {
                        "title": title or None,
                        "description": single_description[:400],
                        "make_instrumental": "true" if submode == MUSIC_SUBMODE_INSTRUMENTAL else "false",
                    }
                    task_kind = "single"
                task_id, task_data = api.run_and_fetch(task_kind, model, payload, timeout)
            else:
                if submode == MUSIC_SUBMODE_CUSTOM:
                    if not prompt:
                        return self._error("\u81ea\u5b9a\u4e49\u6b4c\u8bcd\u6a21\u5f0f\u9700\u8981\u586b\u5199\u6b4c\u8bcd/\u63d0\u793a\u8bcd\u3002")
                    payload = {
                        "prompt": prompt,
                        "mv": model,
                        "title": title,
                        "tags": tags,
                        "generation_type": "TEXT",
                    }
                    payload = self._with_vocal_gender(payload, vocal_gender)
                elif submode == MUSIC_SUBMODE_INSTRUMENTAL:
                    instrumental_prompt = (prompt or title).strip()
                    if not instrumental_prompt:
                        return self._error("\u7eaf\u97f3\u4e50\u6a21\u5f0f\u9700\u8981\u586b\u5199\u97f3\u4e50\u63cf\u8ff0\u3002")
                    payload = {
                        "prompt": instrumental_prompt,
                        "mv": model,
                        "title": title or None,
                        "tags": tags,
                        "make_instrumental": True,
                        "generation_type": "TEXT",
                    }
                else:
                    if not description:
                        return self._error("\u7075\u611f\u6a21\u5f0f\u9700\u8981\u586b\u5199\u97f3\u4e50\u63cf\u8ff0\u3002")
                    payload = {
                        "gpt_description_prompt": description,
                        "make_instrumental": False,
                        "mv": model,
                        "prompt": title or prompt or description,
                        "title": title or None,
                        "tags": tags,
                    }
                    payload = self._with_vocal_gender(payload, vocal_gender)
                task_id, task_data = api.run_and_fetch("music", payload, timeout)

            audio_result, saved_paths, audio_urls, result_clip_ids = self._download_all_task_audio(
                api,
                task_data,
                filename_prefix=filename_prefix,
                timeout=timeout,
            )
            lyrics = api._extract_lyrics(task_data)
            summary = self._build_summary(
                mode=mode,
                submode=submode,
                model=model_label,
                task_id=task_id,
                clip_id="\n".join(result_clip_ids),
                title=title,
                audio_url="\n".join(audio_urls),
                audio_path="\n".join(saved_paths),
                lyrics=lyrics,
            )
            return {"ui": {"comet_text": [summary], "comet_task_id": [task_id]}, "result": (audio_result, summary)}
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._error(format_error_message(exc))


def _load_private_channel_adapter(spec: dict):
    adapter_name = _sanitize_private_name(spec.get("adapter") or spec.get("id"))
    if not adapter_name:
        raise CometAPIError(f"私有渠道 {spec.get('id')} 没有配置 adapter。")
    cache_key = f"{spec.get('__dir')}::{adapter_name}"
    if cache_key in _PRIVATE_ADAPTER_CACHE:
        return _PRIVATE_ADAPTER_CACHE[cache_key]

    adapter_path = os.path.join(spec.get("__dir") or "", f"{adapter_name}.py")
    if not os.path.isfile(adapter_path):
        raise CometAPIError(f"私有渠道适配器不存在：{adapter_path}")

    module_name = f"cometapi_private_{spec.get('id')}_{adapter_name}_{abs(hash(adapter_path))}"
    module_spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    if module_spec is None or module_spec.loader is None:
        raise CometAPIError(f"无法加载私有渠道适配器：{adapter_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    _PRIVATE_ADAPTER_CACHE[cache_key] = module
    return module


def _private_adapter_function(spec: dict, category: str):
    adapter = _load_private_channel_adapter(spec)
    function_name = f"generate_{category}"
    generate = getattr(adapter, function_name, None)
    return generate if callable(generate) else None


def private_adapter_supports(channel: str, category: str) -> bool:
    spec = get_private_channel_spec(channel)
    if not spec:
        return False
    try:
        return callable(_private_adapter_function(spec, category))
    except Exception:
        return False


def _private_context(
    channel: str,
    api_key: str,
    model: str,
    category: str,
    prompt: str,
    pil_images: list[Image.Image] | None = None,
    video_inputs: list | None = None,
    audio_inputs: list | None = None,
    video_paths: list[str] | None = None,
    audio_paths: list[str] | None = None,
    messages: list | None = None,
    aspect_ratio: str = "",
    image_size: str = "",
    quality: str = "",
    concurrency: int = 1,
    duration: str = "",
    size: str = "",
    resolution: str = "",
    mode: str = "",
) -> dict:
    spec = get_private_channel_spec(channel)
    if not spec:
        raise CometAPIError(f"未找到私有渠道：{channel}")
    model_settings = get_private_channel_model_settings(channel, model, category)
    return {
        "channel": channel,
        "category": category,
        "api_key": api_key,
        "api_url": get_channel_api_url(channel),
        "model": model,
        "prompt": prompt,
        "pil_images": list(pil_images or []),
        "video_inputs": list(video_inputs or []),
        "audio_inputs": list(audio_inputs or []),
        "video_paths": list(video_paths or []),
        "audio_paths": list(audio_paths or []),
        "messages": list(messages or []),
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
        "quality": quality,
        "concurrency": concurrency,
        "duration": str(duration),
        "size": str(size),
        "resolution": resolution,
        "mode": mode,
        "channel_spec": spec,
        "model_settings": model_settings,
        "ui": private_model_ui(channel, model, category),
        "helpers": {
            "CometAPIError": CometAPIError,
            "pil_to_data_url": pil_to_data_url,
            "pil_to_base64": pil_to_base64,
            "tensor_to_pil": tensor_to_pil,
            "safe_pil_to_rgb": safe_pil_to_rgb,
            "pil_to_tensor": pil_to_tensor,
            "download_video_asset": download_video_asset,
            "download_audio_asset": download_audio_asset,
            "load_audio_file": load_audio_file,
            "format_error_message": format_error_message,
            "redact_sensitive_text": redact_sensitive_text,
        },
    }


def run_private_llm_channel(
    channel: str,
    api_key: str,
    model: str,
    prompt: str,
    pil_images: list[Image.Image],
    video_paths: list[str],
    audio_paths: list[str],
    messages: list,
    api_format: str,
) -> tuple[str, str]:
    spec = get_private_channel_spec(channel)
    if not spec:
        raise CometAPIError(f"未找到私有文本渠道：{channel}")
    generate = _private_adapter_function(spec, "llm")
    if not callable(generate):
        return call_llm_api(api_key, model, messages, api_format, get_channel_api_url(channel))
    context = _private_context(
        channel=channel,
        api_key=api_key,
        model=model,
        category="llm",
        prompt=prompt,
        pil_images=pil_images,
        video_paths=video_paths,
        audio_paths=audio_paths,
        messages=messages,
        aspect_ratio="",
        image_size="",
        quality="",
        concurrency=1,
        duration="",
        resolution="",
        mode="",
    )
    result = generate(context)
    if isinstance(result, dict):
        return str(result.get("text") or result.get("response") or result.get("content") or ""), str(result.get("error") or "")
    if isinstance(result, tuple) and len(result) >= 2:
        return str(result[0]), str(result[1] or "")
    return str(result or ""), ""


def run_private_image_channel(
    channel: str,
    api_key: str,
    model: str,
    prompt: str,
    pil_images: list[Image.Image],
    aspect_ratio: str,
    image_size: str,
    quality: str,
    subtask_idx: int = 0,
) -> tuple[list[Image.Image], list[str]]:
    spec = get_private_channel_spec(channel)
    if not spec:
        raise CometAPIError(f"未找到私有图像渠道：{channel}")
    generate = _private_adapter_function(spec, "image")
    if not callable(generate):
        settings = load_comet_settings()
        custom_model = (settings.get("channels", {}).get(channel, {}).get("models", {}) or {}).get(model, {})
        custom_format = normalize_api_format_value(custom_model.get("api_format") or detect_image_api_format(model), "image")
        custom_mode = normalize_image_interface_mode(
            custom_model.get("interface_mode") or custom_model.get("endpoint_mode"),
            custom_format,
        )
        return CustomOpenAIImageAPI(api_key, get_channel_api_url(channel)).generate_image(
            prompt=prompt,
            model=model,
            pil_images=pil_images,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            quality=quality,
            api_format=custom_format,
            interface_mode=custom_mode,
            subtask_idx=subtask_idx,
        )
    context = _private_context(
        channel=channel,
        api_key=api_key,
        model=model,
        category="image",
        prompt=prompt,
        pil_images=pil_images,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        quality=quality,
        concurrency=1,
        duration="",
        resolution="",
        mode="",
    )
    context["subtask_idx"] = subtask_idx
    result = generate(context)
    if isinstance(result, tuple) and len(result) >= 2:
        return list(result[0] or []), list(result[1] or [])
    if isinstance(result, dict):
        return list(result.get("images") or []), list(result.get("errors") or [])
    if isinstance(result, list):
        return result, []
    return [], [f"私有图像适配器返回格式无效：{type(result).__name__}"]


def run_private_video_channel(
    channel: str,
    api_key: str,
    model: str,
    prompt: str,
    pil_images: list[Image.Image],
    video_inputs: list,
    audio_inputs: list,
    aspect_ratio: str,
    duration: str,
    size: str,
    resolution: str,
    mode: str,
) -> tuple[str, str, str]:
    spec = get_private_channel_spec(channel)
    if not spec:
        raise CometAPIError(f"未找到私有视频渠道：{channel}")
    generate = _private_adapter_function(spec, "video")
    if not callable(generate):
        raise CometAPIError(f"私有渠道适配器 {spec.get('adapter')} 缺少 generate_video(context) 函数。")
    context = _private_context(
        channel=channel,
        api_key=api_key,
        model=model,
        category="video",
        prompt=prompt,
        pil_images=pil_images,
        video_inputs=video_inputs,
        audio_inputs=audio_inputs,
        aspect_ratio=aspect_ratio,
        duration=str(duration),
        size=size,
        resolution=resolution,
        mode=mode,
    )
    result = generate(context)
    if isinstance(result, dict):
        path = result.get("path") or result.get("local_path") or ""
        video_url = result.get("video_url") or result.get("url") or ""
        task_id = result.get("task_id") or result.get("id") or ""
        return str(path), str(video_url), str(task_id)
    if isinstance(result, tuple) and len(result) >= 3:
        return str(result[0]), str(result[1]), str(result[2])
    raise CometAPIError(f"私有渠道适配器返回格式无效：{type(result).__name__}")


def run_private_music_channel(
    channel: str,
    api_key: str,
    model: str,
    prompt: str,
    music_mode: str,
    music_submode: str,
    title: str,
    tags: str,
    gpt_description_prompt: str,
    vocal_gender: str,
    timeout: int,
) -> tuple[dict, str, dict]:
    spec = get_private_channel_spec(channel)
    if not spec:
        raise CometAPIError(f"未找到私有音乐渠道：{channel}")
    generate = _private_adapter_function(spec, "music")
    if not callable(generate):
        raise CometAPIError(f"私有渠道适配器 {spec.get('adapter')} 缺少 generate_music(context) 函数。")
    context = _private_context(
        channel=channel,
        api_key=api_key,
        model=model,
        category="music",
        prompt=prompt,
        aspect_ratio="",
        image_size="",
        quality="",
        concurrency=1,
        duration="",
        resolution="",
        mode=music_mode,
    )
    context.update(
        {
            "music_mode": music_mode,
            "music_submode": music_submode,
            "title": title,
            "tags": tags,
            "gpt_description_prompt": gpt_description_prompt,
            "vocal_gender": vocal_gender,
            "timeout": timeout,
        }
    )
    result = generate(context)
    if isinstance(result, dict):
        audio = result.get("audio") or silent_audio()
        summary = str(result.get("summary") or result.get("text") or "")
        ui = result.get("ui") if isinstance(result.get("ui"), dict) else {}
        return audio, summary, ui
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], str(result[1]), {}
    raise CometAPIError(f"私有音乐适配器返回格式无效：{type(result).__name__}")


class CometAPIUnifiedVideo:
    CATEGORY = "COMET"
    FUNCTION = "execute"
    RETURN_TYPES = (IO.VIDEO,)
    RETURN_NAMES = ("video",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "channel": (get_video_channel_choices(), {"default": get_default_video_channel()}),
                "model_family": (get_video_model_family_choices(), {"default": "grok"}),
                "model": (get_video_model_choices(), {"default": "grok-video-3"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (VIDEO_INPUT_ASPECT_RATIOS, {"default": "16:9"}),
                "duration": (VIDEO_INPUT_DURATIONS, {"default": "10"}),
                "resolution": (VIDEO_INPUT_RESOLUTIONS, {"default": "1080P"}),
                "mode": (VIDEO_INPUT_MODES, {"default": "首尾帧"}),
                "size": (PRIVATE_SORA_VIDEO_SIZES, {"default": "1280x720"}),
            },
            "optional": {
                "media": (COMET_ANY,),
            },
        }
        for i in range(1, MAX_VIDEO_MEDIA_INPUTS + 1):
            inputs["optional"][f"media_{i}"] = (COMET_ANY,)
            inputs["optional"][f"media_type_{i}"] = ("STRING", {"default": ""})
        return inputs

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def _error(self, message: str):
        message = redact_sensitive_text(message)
        print(f"[CometAPI Video] {message}")
        return {"ui": {"comet_error": [message]}, "result": (VideoAdapter("", comet_error=message),)}

    def _infer_media_type(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, torch.Tensor):
            return "image"
        if isinstance(value, dict) and "waveform" in value:
            return "audio"
        return "video"

    def _collect_unified_media(self, kwargs: dict, max_images: int) -> tuple[list[Image.Image], list, list, list[str]]:
        pil_images = []
        video_inputs = []
        audio_inputs = []
        used_media_types = []

        media_items = []
        direct_media = kwargs.get("media")
        if direct_media is not None:
            media_items.append((self._infer_media_type(direct_media), direct_media))
        for i in range(1, MAX_VIDEO_MEDIA_INPUTS + 1):
            value = kwargs.get(f"media_{i}")
            if value is None:
                continue
            media_type = str(kwargs.get(f"media_type_{i}") or "").strip().lower()
            if media_type not in {"image", "video", "audio"}:
                media_type = self._infer_media_type(value)
            media_items.append((media_type, value))

        for media_type, value in media_items:
            if value is not None:
                used_media_types.append(media_type)
            if media_type == "image":
                pil_images.extend(tensor_to_pil(value))
            elif media_type == "audio":
                audio_inputs.append(value)
            elif media_type == "video":
                video_inputs.append(value)

        return (
            [safe_pil_to_rgb(image) for image in pil_images[:max_images]],
            video_inputs[:MAX_VIDEO_FILE_INPUTS],
            audio_inputs[:MAX_AUDIO_INPUTS],
            used_media_types,
        )

    def execute(
        self,
        channel: str,
        model_family: str,
        model: str,
        prompt: str,
        aspect_ratio: str,
        duration: str,
        resolution: str = "1080P",
        mode: str = "首尾帧",
        size: str = "small",
        api_key: str = "",
        **kwargs,
    ):
        channel_key = str(channel or get_default_video_channel()).lower()
        private_channel = get_private_channel_spec(channel_key)
        if channel_key not in get_video_channel_choices():
            return self._error(f"不支持的视频渠道：{channel}")
        model = resolve_model_id(channel_key, model, "video")
        if channel_key == "apimart":
            actual_model = canonical_apimart_video_model(model)
        else:
            actual_model = model
        final_api_key = get_channel_api_key(api_key, channel_key, model, "video")
        if not final_api_key:
            return self._error(f"缺少 {channel_key} API Key，请先在设置中心填写。")

        try:
            if private_channel:
                private_limits = private_media_limits_for_model(channel_key, model, "video", mode)
                max_refs = int(private_limits.get("image") or 0)
                allowed_media_types = {media_type for media_type, limit in private_limits.items() if int(limit or 0) > 0}
            else:
                if channel_key == "modelverse":
                    max_refs = modelverse_video_max_images(actual_model)
                elif channel_key == "apimart":
                    max_refs = apimart_video_max_images(actual_model, mode)
                else:
                    max_refs = video_max_images_for_model(actual_model, mode)
                allowed_media_types = video_allowed_media_types(channel_key, actual_model, mode)
            pil_images, video_inputs, audio_inputs, used_media_types = self._collect_unified_media(kwargs, max_images=max_refs)
            unsupported_media_types = sorted({media_type for media_type in used_media_types if media_type not in allowed_media_types})
            if unsupported_media_types:
                unsupported_label = "、".join(media_type_label(media_type) for media_type in unsupported_media_types)
                allowed_label = "、".join(media_type_label(media_type) for media_type in sorted(allowed_media_types)) if allowed_media_types else "不接收素材"
                return self._error(
                    f"当前视频模式不支持接入{unsupported_label}素材；当前允许：{allowed_label}。"
                )
            if private_channel:
                path, video_url, task_id = run_private_video_channel(
                    channel=channel_key,
                    api_key=final_api_key,
                    model=model,
                    prompt=prompt,
                    pil_images=pil_images,
                    video_inputs=video_inputs,
                    audio_inputs=audio_inputs,
                    aspect_ratio=aspect_ratio,
                    duration=str(duration),
                    size=size,
                    resolution=resolution,
                    mode=mode,
                )
            elif channel_key == "runninghub":
                path, video_url, task_id = RunningHubVideoAPI(final_api_key).generate_video(
                    prompt=prompt,
                    model=actual_model,
                    pil_images=pil_images,
                    video_inputs=video_inputs,
                    audio_inputs=audio_inputs,
                    aspect_ratio=aspect_ratio,
                    duration=str(duration),
                    resolution=resolution,
                    mode=mode,
                )
            elif channel_key == "modelverse":
                runninghub_upload_key = get_channel_api_key("", "runninghub", "", "video")
                grsai_upload_key = get_channel_api_key("", "grsai", "", "image")
                path, video_url, task_id = ModelVerseVideoAPI(final_api_key, runninghub_upload_key, grsai_upload_key).generate_video(
                    prompt=prompt,
                    model=actual_model,
                    pil_images=pil_images,
                    video_inputs=video_inputs,
                    audio_inputs=audio_inputs,
                    aspect_ratio=aspect_ratio,
                    duration=str(duration),
                    resolution=resolution,
                    mode=mode,
                )
            elif channel_key == "apimart":
                runninghub_upload_key = get_channel_api_key("", "runninghub", "", "video")
                grsai_upload_key = get_channel_api_key("", "grsai", "", "image")
                path, video_url, task_id = APIMartVideoAPI(final_api_key, runninghub_upload_key, grsai_upload_key).generate_video(
                    prompt=prompt,
                    model=actual_model,
                    pil_images=pil_images,
                    video_inputs=video_inputs,
                    audio_inputs=audio_inputs,
                    aspect_ratio=aspect_ratio,
                    duration=str(duration),
                    resolution=resolution,
                    mode=mode,
                )
            else:
                runninghub_upload_key = get_channel_api_key("", "runninghub", "", "video")
                grsai_upload_key = get_channel_api_key("", "grsai", "", "image")
                path, video_url, task_id = PrivateVideoAPI(final_api_key, get_channel_api_url(channel_key), runninghub_upload_key, grsai_upload_key).generate_video(
                    prompt=prompt,
                    model=actual_model,
                    pil_images=pil_images,
                    aspect_ratio=aspect_ratio,
                    duration=str(duration),
                    size=size,
                    resolution=resolution,
                    mode=mode,
                    video_inputs=video_inputs,
                )
            ui = {
                "comet_video_url": [video_url],
                "comet_task_id": [task_id],
            }
            return {"ui": ui, "result": (VideoAdapter(path, video_url=video_url),)}
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._error(format_error_message(exc))


COMET_ASYNC_TASKS_FILE = os.path.join(DATA_DIR, "cometapi_async_tasks.json")
COMET_ASYNC_ACTIVE_STATUSES = {"queued", "submitted", "pending", "running", "processing", "in_progress"}
COMET_ASYNC_NO_RESULT_MARKER = "__comet_async_no_result__"
COMET_ASYNC_QUERY_INFO_DISPLAY_LIMIT = 20
COMET_ASYNC_LOCAL_CACHE_DIR = os.path.join(DATA_DIR, "cometapi_async_cache")
COMET_ASYNC_LOCAL_WORKERS = 4
COMET_ASYNC_LOCAL_IMAGE_PROVIDER = "local_wrapped_image"
COMET_ASYNC_LOCAL_VIDEO_PROVIDER = "local_wrapped_video"
_COMET_LOCAL_ASYNC_POOL = None
_COMET_LOCAL_ASYNC_POOL_LOCK = threading.Lock()
_COMET_LOCAL_ASYNC_RUNNING = set()


def make_comet_async_no_result(kind: str, message: str) -> dict:
    return {
        COMET_ASYNC_NO_RESULT_MARKER: True,
        "kind": str(kind or "").lower(),
        "message": redact_sensitive_text(message),
    }


def is_comet_async_no_result(value, kind: str = "") -> bool:
    if not isinstance(value, dict) or value.get(COMET_ASYNC_NO_RESULT_MARKER) is not True:
        return False
    if not kind:
        return True
    return str(value.get("kind") or "").lower() == str(kind or "").lower()


def comet_async_no_result_message(value, fallback: str) -> str:
    if isinstance(value, dict):
        message = str(value.get("message") or "").strip()
        if message:
            return redact_sensitive_text(message)
    return redact_sensitive_text(fallback)


COMET_ASYNC_SUCCESS_STATUSES = {"success", "succeeded", "completed", "complete", "finished", "done", "SUCCESS", "SUCCEEDED", "COMPLETED"}
COMET_ASYNC_FAILED_STATUSES = {"failed", "failure", "fail", "error", "cancelled", "canceled", "expired", "FAILED", "FAILURE", "ERROR", "CANCELLED", "CANCELED"}
_COMET_ASYNC_TASK_LOCK = threading.RLock()


def _async_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_async_tasks_unlocked() -> dict:
    if not os.path.exists(COMET_ASYNC_TASKS_FILE):
        return {}
    try:
        with open(COMET_ASYNC_TASKS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[CometAPI Async] 读取任务队列失败：{redact_sensitive_text(exc)}")
        return {}


def _trim_async_tasks(tasks: dict, max_history: int = 160) -> dict:
    active = {}
    inactive = []
    for task_id, info in (tasks or {}).items():
        if not isinstance(info, dict):
            continue
        if str(info.get("status") or "running").lower() in COMET_ASYNC_ACTIVE_STATUSES:
            active[task_id] = info
        else:
            inactive.append((task_id, info))
    inactive.sort(key=lambda item: item[1].get("updated_at") or item[1].get("submitted_at") or "", reverse=True)
    kept = {task_id: info for task_id, info in inactive[:max_history]}
    return {**active, **kept}


def _write_async_tasks_unlocked(tasks: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(COMET_ASYNC_TASKS_FILE, "w", encoding="utf-8") as handle:
        json.dump(_trim_async_tasks(tasks), handle, indent=2, ensure_ascii=False)


def read_comet_async_tasks() -> dict:
    with _COMET_ASYNC_TASK_LOCK:
        return copy.deepcopy(_read_async_tasks_unlocked())


def _next_async_task_number(tasks: dict, source: str) -> int:
    prefix_map = {
        "image": "图片任务",
        "batch_image": "批量图片任务",
        "video": "视频任务",
    }
    prefix = prefix_map.get(source, "任务")
    numbers = []
    for task_id in (tasks or {}).keys():
        text = str(task_id)
        if not text.startswith(prefix):
            continue
        match = re.search(r"(\d+)$", text)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def add_comet_async_task(record: dict) -> str:
    with _COMET_ASYNC_TASK_LOCK:
        tasks = _read_async_tasks_unlocked()
        source = str(record.get("source") or record.get("category") or "image")
        prefix_map = {
            "image": "图片任务",
            "batch_image": "批量图片任务",
            "video": "视频任务",
        }
        task_id = f"{prefix_map.get(source, '任务')}{_next_async_task_number(tasks, source)}"
        now = _async_now()
        clean_record = dict(record)
        clean_record.setdefault("submitted_at", now)
        clean_record["updated_at"] = now
        clean_record.setdefault("status", "running")
        clean_record.setdefault("subtasks", [])
        tasks[task_id] = clean_record
        _write_async_tasks_unlocked(tasks)
        return task_id


def update_comet_async_task(task_id: str, updates: dict) -> None:
    with _COMET_ASYNC_TASK_LOCK:
        tasks = _read_async_tasks_unlocked()
        info = dict(tasks.get(task_id) or {})
        info.update(updates or {})
        info["updated_at"] = _async_now()
        tasks[task_id] = info
        _write_async_tasks_unlocked(tasks)


def _get_local_async_pool():
    global _COMET_LOCAL_ASYNC_POOL
    with _COMET_LOCAL_ASYNC_POOL_LOCK:
        if _COMET_LOCAL_ASYNC_POOL is None:
            _COMET_LOCAL_ASYNC_POOL = concurrent.futures.ThreadPoolExecutor(
                max_workers=COMET_ASYNC_LOCAL_WORKERS,
                thread_name_prefix="CometLocalAsync",
            )
        return _COMET_LOCAL_ASYNC_POOL


def _local_async_key(kind: str) -> str:
    return f"local_{kind}_{uuid.uuid4().hex}"


def _mark_local_async_running(local_id: str, running: bool) -> None:
    with _COMET_LOCAL_ASYNC_POOL_LOCK:
        if running:
            _COMET_LOCAL_ASYNC_RUNNING.add(str(local_id))
        else:
            _COMET_LOCAL_ASYNC_RUNNING.discard(str(local_id))


def _is_local_async_running(local_id: str) -> bool:
    with _COMET_LOCAL_ASYNC_POOL_LOCK:
        return str(local_id) in _COMET_LOCAL_ASYNC_RUNNING


def _async_parent_status_from_subtasks(subtasks: list[dict]) -> tuple[str, str]:
    active = sum(1 for sub in subtasks if str(sub.get("status") or "").lower() in COMET_ASYNC_ACTIVE_STATUSES)
    ok = sum(1 for sub in subtasks if str(sub.get("status") or "").lower() == "succeeded")
    if active:
        status = "running"
    elif ok:
        status = "succeeded"
    else:
        status = "failed"
    last_error = "; ".join(
        str(sub.get("failure_reason") or sub.get("last_query_error") or "")
        for sub in subtasks
        if sub.get("failure_reason") or sub.get("last_query_error")
    )
    return status, last_error[:500]


def update_comet_async_subtask(task_id: str, local_id: str, updates: dict) -> None:
    with _COMET_ASYNC_TASK_LOCK:
        tasks = _read_async_tasks_unlocked()
        info = dict(tasks.get(task_id) or {})
        subtasks = copy.deepcopy(info.get("subtasks") if isinstance(info.get("subtasks"), list) else [])
        changed = False
        for index, subtask in enumerate(subtasks):
            if str(subtask.get("local_task_id") or subtask.get("api_task_id") or "") != str(local_id):
                continue
            updated = dict(subtask)
            updated.update(updates or {})
            subtasks[index] = updated
            changed = True
            break
        if not changed:
            return
        status, last_error = _async_parent_status_from_subtasks(subtasks)
        info["subtasks"] = subtasks
        info["status"] = status
        info["last_error"] = last_error
        info["updated_at"] = _async_now()
        tasks[task_id] = info
        _write_async_tasks_unlocked(tasks)


def _local_async_image_paths(task_id: str, local_id: str, images: list[Image.Image]) -> list[str]:
    target_dir = os.path.join(COMET_ASYNC_LOCAL_CACHE_DIR, "image")
    os.makedirs(target_dir, exist_ok=True)
    safe_task = re.sub(r"[^0-9A-Za-z_-]+", "_", str(task_id or "task")).strip("_") or "task"
    paths = []
    for index, image in enumerate(images, start=1):
        filename = f"{safe_task}_{local_id}_{index:02d}.png"
        path = os.path.abspath(os.path.join(target_dir, filename))
        safe_pil_to_rgb(image).save(path, "PNG")
        paths.append(path)
    return paths


def _private_adapter_has_functions(channel: str, *names: str) -> bool:
    spec = get_private_channel_spec(channel)
    if not spec:
        return False
    try:
        adapter = _load_private_channel_adapter(spec)
    except Exception:
        return False
    return all(callable(getattr(adapter, name, None)) for name in names)


def image_channel_uses_remote_async(channel: str) -> bool:
    channel = str(channel or "").lower()
    if channel in {"grsai", "runninghub", "apimart"}:
        return True
    return _private_adapter_has_functions(channel, "submit_image_task", "query_image_task")


def video_channel_uses_remote_async(channel: str, model: str) -> bool:
    channel = str(channel or "").lower()
    model = str(model or "")
    if channel in {"runninghub", "apimart", "modelverse"}:
        return True
    if get_private_channel_spec(channel):
        if _private_adapter_has_functions(channel, "submit_video_task", "query_video_task"):
            return True
        if private_adapter_supports(channel, "video"):
            return False
    return any(
        (
            is_vidu_video_model(model),
            is_hailuo_video_model(model),
            is_kling_video_model(model),
            is_private_happyhorse_video_model(model),
            is_sora_video_model(model),
            is_grok_video_model(model),
            is_veo_video_model(model),
        )
    )


def make_local_wrapped_image_subtask(prompt: str, subtask_idx: int = 0, extra: dict | None = None) -> dict:
    local_id = _local_async_key("image")
    subtask = {
        "provider": COMET_ASYNC_LOCAL_IMAGE_PROVIDER,
        "api_task_id": local_id,
        "local_task_id": local_id,
        "status": "running",
        "image_urls": [],
        "local_image_paths": [],
        "failure_reason": "",
        "original_prompt": str(prompt or ""),
        "subtask_idx": subtask_idx,
    }
    subtask.update(extra or {})
    return subtask


def make_local_wrapped_video_subtask(prompt: str) -> dict:
    local_id = _local_async_key("video")
    return {
        "provider": COMET_ASYNC_LOCAL_VIDEO_PROVIDER,
        "api_task_id": local_id,
        "local_task_id": local_id,
        "status": "running",
        "video_url": "",
        "local_path": "",
        "failure_reason": "",
        "original_prompt": str(prompt or ""),
    }


def _run_local_wrapped_image_subtask(task_id: str, local_id: str, payload: dict) -> None:
    _mark_local_async_running(local_id, True)
    try:
        task = dict(payload.get("task") or {})
        if isinstance(task.get("fixed_refs"), list):
            task["fixed_refs"] = list(task.get("fixed_refs") or [])
        channel = str(payload.get("channel") or "").lower()
        model = str(payload.get("model") or "")
        generator = CometAPIBatchImage()
        upload_lock = payload.get("upload_lock")
        if not all(hasattr(upload_lock, attr) for attr in ("acquire", "release")):
            upload_lock = threading.Lock()
        images, errors, _refs = generator._generate_once(
            channel=channel,
            model=model,
            api_key=str(payload.get("api_key") or ""),
            task=task,
            aspect_ratio=str(payload.get("aspect_ratio") or "auto"),
            image_size=str(payload.get("image_size") or "2K"),
            quality=str(payload.get("quality") or "medium"),
            upload_cache=payload.get("upload_cache") if isinstance(payload.get("upload_cache"), dict) else {},
            upload_lock=upload_lock,
        )
        if not images:
            reason = "；".join(errors[:3]) if errors else "本地异步生成没有返回图片"
            update_comet_async_subtask(task_id, local_id, {"status": "failed", "failure_reason": reason})
            return
        paths = _local_async_image_paths(task_id, local_id, images)
        update_comet_async_subtask(
            task_id,
            local_id,
            {
                "status": "succeeded",
                "local_image_paths": paths,
                "result_count": len(paths),
                "failure_reason": "；".join(errors[:3]) if errors else "",
                "completed_at": _async_now(),
            },
        )
    except Exception as exc:
        print_sanitized_exception(exc)
        update_comet_async_subtask(task_id, local_id, {"status": "failed", "failure_reason": format_error_message(exc)})
    finally:
        _mark_local_async_running(local_id, False)


def _run_local_wrapped_video_subtask(task_id: str, local_id: str, payload: dict) -> None:
    _mark_local_async_running(local_id, True)
    try:
        channel = str(payload.get("channel") or "").lower()
        model = str(payload.get("model") or "")
        api_key = str(payload.get("api_key") or "")
        prompt = str(payload.get("prompt") or "")
        pil_images = list(payload.get("pil_images") or [])
        video_inputs = list(payload.get("video_inputs") or [])
        audio_inputs = list(payload.get("audio_inputs") or [])
        aspect_ratio = str(payload.get("aspect_ratio") or "16:9")
        duration = str(payload.get("duration") or "4")
        resolution = str(payload.get("resolution") or "1080P")
        mode = str(payload.get("mode") or "")
        size = str(payload.get("size") or "small")
        if get_private_channel_spec(channel):
            path, video_url, remote_task_id = run_private_video_channel(
                channel=channel,
                api_key=api_key,
                model=model,
                prompt=prompt,
                pil_images=pil_images,
                video_inputs=video_inputs,
                audio_inputs=audio_inputs,
                aspect_ratio=aspect_ratio,
                duration=duration,
                size=size,
                resolution=resolution,
                mode=mode,
            )
        else:
            runninghub_upload_key = get_channel_api_key("", "runninghub", "", "video")
            grsai_upload_key = get_channel_api_key("", "grsai", "", "image")
            path, video_url, remote_task_id = PrivateVideoAPI(api_key, get_channel_api_url(channel), runninghub_upload_key, grsai_upload_key).generate_video(
                prompt=prompt,
                model=model,
                pil_images=pil_images,
                aspect_ratio=aspect_ratio,
                duration=duration,
                size=size,
                resolution=resolution,
                mode=mode,
                video_inputs=video_inputs,
            )
        if not path and not video_url:
            raise CometAPIError("本地异步视频生成没有返回视频文件或视频地址。")
        update_comet_async_subtask(
            task_id,
            local_id,
            {
                "status": "succeeded",
                "local_path": str(path or ""),
                "video_url": str(video_url or ""),
                "remote_task_id": str(remote_task_id or ""),
                "completed_at": _async_now(),
            },
        )
    except Exception as exc:
        print_sanitized_exception(exc)
        update_comet_async_subtask(task_id, local_id, {"status": "failed", "failure_reason": format_error_message(exc)})
    finally:
        _mark_local_async_running(local_id, False)


def schedule_local_wrapped_image_subtask(task_id: str, subtask: dict, payload: dict) -> None:
    local_id = str(subtask.get("local_task_id") or subtask.get("api_task_id") or "")
    _mark_local_async_running(local_id, True)
    _get_local_async_pool().submit(_run_local_wrapped_image_subtask, task_id, local_id, payload)


def schedule_local_wrapped_video_subtask(task_id: str, subtask: dict, payload: dict) -> None:
    local_id = str(subtask.get("local_task_id") or subtask.get("api_task_id") or "")
    _mark_local_async_running(local_id, True)
    _get_local_async_pool().submit(_run_local_wrapped_video_subtask, task_id, local_id, payload)


def _short_async_text(value: str, limit: int = 28) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return "无提示词"
    return text if len(text) <= limit else f"{text[:limit]}..."


def _async_status_label(status: str) -> str:
    raw = str(status or "running").lower()
    return {
        "running": "运行中",
        "processing": "运行中",
        "pending": "排队中",
        "queued": "排队中",
        "submitted": "已提交",
        "succeeded": "待收取",
        "success": "待收取",
        "failed": "失败",
        "downloaded": "已收取",
        "download_failed": "收取失败",
    }.get(raw, status or "运行中")


def _canonical_async_status(status: str, has_result: bool = False) -> str:
    raw = str(status or "").strip()
    lower = raw.lower()
    if raw in COMET_ASYNC_SUCCESS_STATUSES or lower in {item.lower() for item in COMET_ASYNC_SUCCESS_STATUSES}:
        return "succeeded"
    if raw in COMET_ASYNC_FAILED_STATUSES or lower in {item.lower() for item in COMET_ASYNC_FAILED_STATUSES}:
        return "failed"
    if has_result:
        return "succeeded"
    if lower in {"", "queued", "submitted", "pending"}:
        return "running"
    return "running" if lower in COMET_ASYNC_ACTIVE_STATUSES else lower


def _async_subtask_counts(info: dict) -> tuple[int, int, int]:
    subtasks = info.get("subtasks") if isinstance(info, dict) else []
    if not isinstance(subtasks, list):
        return 0, 0, 0
    total = len(subtasks)
    ok = sum(1 for sub in subtasks if str(sub.get("status") or "").lower() == "succeeded")
    fail = sum(1 for sub in subtasks if str(sub.get("status") or "").lower() == "failed")
    return total, ok, fail


def format_comet_async_task_panel(category: str) -> str:
    category = str(category or "image").lower()
    title = "Comet 异步图片任务队列" if category == "image" else "Comet 异步视频任务队列"
    tasks = read_comet_async_tasks()
    items = [
        (task_id, info)
        for task_id, info in tasks.items()
        if isinstance(info, dict) and str(info.get("category") or "").lower() == category
    ]
    lines = [f"--- {title} ---"]
    if not items:
        lines.append("当前没有异步任务。")
        return "\n".join(lines)

    items.sort(key=lambda item: item[1].get("submitted_at") or "", reverse=True)
    hidden_count = max(0, len(items) - COMET_ASYNC_QUERY_INFO_DISPLAY_LIMIT)
    for task_id, info in items[:COMET_ASYNC_QUERY_INFO_DISPLAY_LIMIT]:
        total, ok, fail = _async_subtask_counts(info)
        prompt = _short_async_text(info.get("prompt") or info.get("title") or "")
        channel = info.get("channel") or ""
        model = info.get("model") or ""
        status = _async_status_label(info.get("status"))
        line = f"[{status}] {task_id} ({prompt}) | {channel}/{model} | {total}个子任务 (成功: {ok} | 失败: {fail})"
        last_error = str(info.get("last_error") or "").strip()
        if last_error and str(info.get("status") or "").lower() in {"failed", "download_failed"}:
            line += f" | {redact_sensitive_text(last_error)[:120]}"
        lines.append(line)
    if hidden_count:
        lines.append(f"... 仅显示最近 {COMET_ASYNC_QUERY_INFO_DISPLAY_LIMIT} 条，另有 {hidden_count} 条未显示。")
    return "\n".join(lines)


def _extract_task_id_from_response(data: dict, label: str) -> str:
    if not isinstance(data, dict):
        raise CometAPIError(f"{label} 没有返回 JSON 任务响应。")
    candidates = [data.get("id"), data.get("task_id"), data.get("taskId")]
    for key in ("data", "output", "result"):
        body = data.get(key)
        if isinstance(body, dict):
            candidates.extend([body.get("id"), body.get("task_id"), body.get("taskId")])
    for value in candidates:
        if value:
            return str(value)
    raise CometAPIError(f"{label} 没有返回任务 ID：{str(data)[:300]}")


def _build_grsai_async_image_payload(
    prompt: str,
    model: str,
    uploaded_urls: list[str],
    aspect_ratio: str,
    image_size: str,
    quality: str,
    pil_refs: list[Image.Image],
    subtask_idx: int,
) -> tuple[str, dict, str]:
    prompt_text = add_prompt_variation(prompt, subtask_idx)
    target_size = image_size if image_size in {"1K", "2K", "3K", "4K", "8K"} else "2K"
    if model in GPT_IMAGE_MODELS:
        payload = {
            "model": model,
            "prompt": prompt_text,
            "urls": uploaded_urls,
            "shutProgress": True,
            "webHook": "-1",
        }
        if model == "gpt-image-2-vip":
            resolved_ratio = aspect_ratio
            if resolved_ratio == "auto":
                resolved_ratio = nearest_aspect_ratio("auto", pil_refs, list(GPT_IMAGE_VIP_SIZE_MAP.keys()), "1:1")
            resolved_ratio = resolved_ratio if resolved_ratio in GPT_IMAGE_VIP_SIZE_MAP else "1:1"
            safe_size = target_size if target_size in {"1K", "2K", "4K"} else "2K"
            mapped_size = GPT_IMAGE_VIP_SIZE_MAP.get(resolved_ratio, {}).get(safe_size)
            if not mapped_size:
                raise CometAPIError(f"gpt-image-2-vip 不支持这个尺寸：{resolved_ratio} / {safe_size}")
            payload["aspectRatio"] = mapped_size
            payload["quality"] = quality if quality in GPT_IMAGE_QUALITY_VALUES else "medium"
        else:
            payload["aspectRatio"] = aspect_ratio if aspect_ratio in GPT_IMAGE_ASPECT_RATIOS else "auto"
        return "/v1/draw/completions", payload, model

    if model == "nano-banana-fast" or model not in PRO_SIZE_MODELS:
        target_size = "1K"
    actual_model = "nano-banana-2-4k-cl" if model == "nano-banana-2-cl" and target_size == "4K" else model
    payload = {
        "model": actual_model,
        "prompt": prompt_text,
        "urls": uploaded_urls,
        "shutProgress": True,
        "webHook": "-1",
        "aspectRatio": aspect_ratio,
    }
    if actual_model in PRO_SIZE_MODELS or actual_model == "nano-banana-2-4k-cl":
        payload["imageSize"] = target_size
    return "/v1/draw/nano-banana", payload, actual_model


def submit_async_image_subtask(
    channel: str,
    model: str,
    api_key: str,
    prompt: str,
    pil_refs: list[Image.Image],
    aspect_ratio: str,
    image_size: str,
    quality: str,
    subtask_idx: int = 0,
    grsai_upload_cache: dict | None = None,
    grsai_upload_lock: threading.Lock | None = None,
) -> dict:
    channel = str(channel or "").lower()
    prompt_for_api = str(prompt or "")
    if channel == "grsai":
        max_refs = GPT_IMAGE_MAX_IMAGES if model in GPT_IMAGE_MODELS else NANO_BANANA_MAX_IMAGES
        selected_refs = [safe_pil_to_rgb(image) for image in pil_refs[:max_refs]]
        uploaded_urls = []
        for pil_image in selected_refs:
            cache_key = f"pil:{id(pil_image)}:{getattr(pil_image, 'filename', '')}"
            cached_url = None
            if grsai_upload_cache is not None and grsai_upload_lock is not None:
                with grsai_upload_lock:
                    cached_url = grsai_upload_cache.get(cache_key)
            if cached_url:
                uploaded_urls.append(cached_url)
                continue
            url = upload_image_grsai(api_key, pil_image)
            if not url:
                raise CometAPIError("grsai 参考图上传失败，请检查输入图片或网络。")
            if grsai_upload_cache is not None and grsai_upload_lock is not None:
                with grsai_upload_lock:
                    grsai_upload_cache[cache_key] = url
            uploaded_urls.append(url)
        endpoint, payload, actual_model = _build_grsai_async_image_payload(
            prompt_for_api, model, uploaded_urls, aspect_ratio, image_size, quality, selected_refs, subtask_idx
        )
        data = GrsaiAPI(api_key)._make_request("POST", endpoint, data=payload, timeout=120)
        return {
            "provider": "grsai_image",
            "api_task_id": _extract_task_id_from_response(data, "grsai 图片"),
            "status": "running",
            "image_urls": [],
            "progress": 0,
            "failure_reason": "",
            "original_prompt": prompt_for_api,
            "model": actual_model,
        }

    if channel == "runninghub":
        api = RunningHubImageAPI(api_key)
        raw_model = canonical_runninghub_image_model(model)
        spec = runninghub_image_spec(raw_model)
        image_urls = [api._upload_image(image) for image in pil_refs[: spec["max_images"]]]
        endpoint = spec["image_endpoint"] if image_urls else spec["text_endpoint"]
        resolved_ratio = nearest_aspect_ratio(aspect_ratio, pil_refs, spec["aspect_ratios"], fallback=spec["fallback_aspect"])
        payload = {
            "prompt": add_prompt_variation(prompt_for_api, subtask_idx),
            "aspectRatio": resolved_ratio,
        }
        if image_urls:
            payload["imageUrls"] = image_urls
        resolutions = spec["resolutions"]
        if resolutions:
            raw_size = str(image_size or "").strip().lower()
            payload["resolution"] = raw_size if raw_size in resolutions else ("4k" if "4k" in resolutions else resolutions[0])
        if spec["quality"]:
            payload["quality"] = quality if quality in GPT_IMAGE_QUALITY_VALUES else "medium"
        task_id = api._submit_task(f"/openapi/v2/{spec['base']}/{endpoint}", payload)
        return {"provider": "runninghub_image", "api_task_id": task_id, "status": "running", "image_urls": [], "progress": 0, "failure_reason": "", "original_prompt": prompt_for_api}

    if channel == "apimart":
        api = APIMartImageAPI(api_key)
        if model not in APIMART_IMAGE_MODELS:
            raise CometAPIError(f"不支持的 Apimart 图片模型：{model}")
        is_gpt = model in APIMART_GPT_IMAGE_MODELS
        if is_gpt:
            aspect_choices = APIMART_GPT_IMAGE_ASPECT_RATIOS
        elif model in APIMART_GEMINI_31_IMAGE_MODELS:
            aspect_choices = APIMART_GEMINI_31_ASPECT_RATIOS
        else:
            aspect_choices = APIMART_GEMINI_ASPECT_RATIOS
        resolved_ratio = nearest_aspect_ratio(aspect_ratio, pil_refs, aspect_choices, "1:1")
        safe_size = image_size if image_size in {"1K", "2K", "4K"} else "2K"
        if model in APIMART_GEMINI_25_IMAGE_MODELS:
            safe_size = "1K"
        payload = {
            "model": model,
            "prompt": add_prompt_variation(prompt_for_api, subtask_idx),
            "size": resolved_ratio,
            "resolution": safe_size.lower() if is_gpt else safe_size,
            "n": 1,
        }
        image_urls = [pil_to_data_url(image) for image in pil_refs[: apimart_image_max_images(model)]]
        if image_urls:
            payload["image_urls"] = image_urls
        if model == "gpt-image-2-official":
            payload["quality"] = quality if quality in GPT_IMAGE_QUALITY_VALUES else "medium"
        task_id = api._submit_task(payload)
        return {"provider": "apimart_image", "api_task_id": task_id, "status": "running", "image_urls": [], "progress": 0, "failure_reason": "", "original_prompt": prompt_for_api}

    spec = get_private_channel_spec(channel)
    if spec:
        submit = getattr(_load_private_channel_adapter(spec), "submit_image_task", None)
        if callable(submit):
            context = _private_context(
                channel=channel,
                api_key=api_key,
                model=model,
                category="image",
                prompt=prompt_for_api,
                pil_images=pil_refs,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                quality=quality,
            )
            context["subtask_idx"] = subtask_idx
            result = submit(context)
            if not isinstance(result, dict):
                raise CometAPIError("私有异步图片提交函数需要返回 dict。")
            task_id = result.get("task_id") or result.get("id")
            if not task_id:
                raise CometAPIError(f"私有异步图片提交没有返回 task_id：{str(result)[:300]}")
            return {
                "provider": "private_adapter_image",
                "api_task_id": str(task_id),
                "status": "running",
                "image_urls": list(result.get("image_urls") or []),
                "adapter_data": result,
                "failure_reason": "",
                "original_prompt": prompt_for_api,
            }
    raise CometAPIError(f"{channel}/{model} 暂不支持异步图片提交；当前只支持 grsai、RunningHub、Apimart，或带 submit_image_task/query_image_task 的私有适配器。")


def _extract_grsai_image_urls(data: dict) -> list[str]:
    body = data.get("data") if isinstance(data.get("data"), dict) else data
    urls = []
    if isinstance(body, dict):
        for item in body.get("results") or []:
            if isinstance(item, dict) and item.get("url"):
                urls.append(str(item["url"]))
        if body.get("url"):
            urls.append(str(body["url"]))
    return urls


def _query_async_image_subtask(record: dict, subtask: dict, api_key: str) -> dict:
    provider = str(subtask.get("provider") or "")
    task_id = str(subtask.get("api_task_id") or "")
    if provider == COMET_ASYNC_LOCAL_IMAGE_PROVIDER:
        if str(subtask.get("status") or "").lower() in COMET_ASYNC_ACTIVE_STATUSES and not _is_local_async_running(task_id):
            return {**subtask, "status": "failed", "failure_reason": "本地异步任务不在运行，可能 ComfyUI 已重启。"}
        return subtask
    if not task_id:
        return {**subtask, "status": "failed", "failure_reason": "缺少远端任务 ID"}

    if provider == "grsai_image":
        data = GrsaiAPI(api_key)._make_request("POST", "/v1/draw/result", data={"id": task_id}, timeout=60)
        body = data.get("data") if isinstance(data.get("data"), dict) else data
        urls = _extract_grsai_image_urls(data)
        raw_status = str(body.get("status") if isinstance(body, dict) else "running")
        status = _canonical_async_status(raw_status, has_result=bool(urls))
        failure = ""
        if status == "failed" and isinstance(body, dict):
            failure = str(body.get("failure_reason") or body.get("error") or body.get("message") or "unknown")
        return {**subtask, "status": status, "image_urls": urls or subtask.get("image_urls") or [], "progress": body.get("progress", subtask.get("progress", 0)) if isinstance(body, dict) else 0, "failure_reason": failure}

    if provider == "runninghub_image":
        status_raw, urls, data = RunningHubImageAPI(api_key)._query_task(task_id)
        status = _canonical_async_status(status_raw, has_result=bool(urls))
        failure = ""
        if status == "failed":
            failure = str(data.get("errorMessage") or data.get("message") or data.get("failedReason") or data)[:300]
        return {**subtask, "status": status, "image_urls": urls or subtask.get("image_urls") or [], "failure_reason": failure}

    if provider == "apimart_image":
        status_raw, urls, data = APIMartImageAPI(api_key)._query_task(task_id)
        status = _canonical_async_status(status_raw, has_result=bool(urls))
        failure = ""
        if status == "failed":
            error = data.get("error") if isinstance(data, dict) else ""
            failure = str(error or data)[:300]
        return {**subtask, "status": status, "image_urls": urls or subtask.get("image_urls") or [], "failure_reason": failure}

    if provider == "private_adapter_image":
        spec = get_private_channel_spec(str(record.get("channel") or ""))
        if not spec:
            return {**subtask, "status": "failed", "failure_reason": "私有渠道不存在"}
        query = getattr(_load_private_channel_adapter(spec), "query_image_task", None)
        if not callable(query):
            return {**subtask, "status": "failed", "failure_reason": "私有适配器缺少 query_image_task(context)"}
        context = _private_context(
            channel=str(record.get("channel") or ""),
            api_key=api_key,
            model=str(record.get("model") or ""),
            category="image",
            prompt=str(record.get("prompt") or ""),
        )
        context.update({"task_id": task_id, "adapter_data": subtask.get("adapter_data") or {}})
        result = query(context)
        if not isinstance(result, dict):
            return {**subtask, "status": "failed", "failure_reason": "私有异步图片查询返回格式无效"}
        urls = list(result.get("image_urls") or result.get("urls") or [])
        status = _canonical_async_status(str(result.get("status") or ""), has_result=bool(urls))
        return {**subtask, "status": status, "image_urls": urls or subtask.get("image_urls") or [], "failure_reason": str(result.get("error") or result.get("failure_reason") or "")}

    return {**subtask, "status": "failed", "failure_reason": f"未知异步图片 provider：{provider}"}


def _submit_private_sora_video_task(api: PrivateVideoAPI, prompt: str, model: str, pil_images: list[Image.Image], aspect_ratio: str, duration: str) -> str:
    safe_duration = duration if str(duration) in {"4", "8", "12"} else "4"
    safe_size = "720x1280" if str(aspect_ratio or "").strip() == "9:16" else "1280x720"
    clean_prompt = add_prompt_variation(str(prompt or "").strip(), 0)
    fields = {
        "model": (None, "sora-2"),
        "prompt": (None, clean_prompt),
        "seconds": (None, safe_duration),
        "size": (None, safe_size),
    }
    image_buffer = None
    try:
        if pil_images:
            target_w, target_h = (720, 1280) if safe_size == "720x1280" else (1280, 720)
            source = safe_pil_to_rgb(pil_images[0])
            src_w, src_h = source.size
            scale = max(target_w / max(1, src_w), target_h / max(1, src_h))
            new_w, new_h = int(src_w * scale + 0.5), int(src_h * scale + 0.5)
            resized = source.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            cropped = resized.crop((left, top, left + target_w, top + target_h))
            image_buffer = BytesIO()
            cropped.save(image_buffer, format="JPEG", quality=95)
            image_buffer.seek(0)
            fields["input_reference"] = ("input_reference.jpeg", image_buffer, "image/jpeg")
        response = requests.post(f"{api.host}/v1/videos", headers={"Authorization": f"Bearer {api.api_key}"}, files=fields, timeout=(15, 120))
        response.raise_for_status()
        data = response.json()
    finally:
        if image_buffer is not None:
            image_buffer.close()
    task_id = data.get("id") or data.get("task_id")
    if not task_id:
        raise CometAPIError(f"Sora API 没有返回任务 ID：{str(data)[:300]}")
    return str(task_id)


def _query_private_sora_video(api: PrivateVideoAPI, task_id: str) -> tuple[str, str, dict]:
    response = requests.get(f"{api.host}/v1/videos/{requests.utils.quote(str(task_id), safe='')}", headers={"Authorization": f"Bearer {api.api_key}"}, timeout=30)
    response.raise_for_status()
    data = response.json()
    status = _canonical_async_status(str(data.get("status") or "queued"))
    video_url = data.get("video_url") or data.get("url") or ""
    if not video_url:
        outputs = data.get("output") or data.get("outputs") or {}
        if isinstance(outputs, dict):
            videos = outputs.get("videos") or outputs.get("video") or []
            if isinstance(videos, list) and videos:
                video_url = videos[0].get("url") or videos[0].get("video_url") or ""
    return status, video_url or "", data


def submit_async_video_subtask(
    channel: str,
    model: str,
    api_key: str,
    prompt: str,
    pil_images: list[Image.Image],
    video_inputs: list,
    audio_inputs: list,
    aspect_ratio: str,
    duration: str,
    resolution: str,
    mode: str,
    size: str,
) -> dict:
    channel = str(channel or "").lower()
    prompt_text = str(prompt or "")
    spec = get_private_channel_spec(channel)
    if spec:
        submit = getattr(_load_private_channel_adapter(spec), "submit_video_task", None)
        if callable(submit):
            context = _private_context(
                channel=channel,
                api_key=api_key,
                model=model,
                category="video",
                prompt=prompt_text,
                pil_images=pil_images,
                video_inputs=video_inputs,
                audio_inputs=audio_inputs,
                aspect_ratio=aspect_ratio,
                duration=str(duration),
                size=size,
                resolution=resolution,
                mode=mode,
            )
            result = submit(context)
            if not isinstance(result, dict):
                raise CometAPIError("私有异步视频提交函数需要返回 dict。")
            task_id = result.get("task_id") or result.get("id")
            if not task_id:
                raise CometAPIError(f"私有异步视频提交没有返回 task_id：{str(result)[:300]}")
            return {"provider": "private_adapter_video", "api_task_id": str(task_id), "status": "running", "video_url": str(result.get("video_url") or ""), "adapter_data": result, "failure_reason": ""}

    if channel == "runninghub":
        api = RunningHubVideoAPI(api_key)
        raw_model = canonical_runninghub_video_model(model).lower()
        if is_runninghub_google_model(raw_model):
            path, payload = api._build_google_payload(prompt_text, raw_model, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
        elif is_runninghub_grok_model(raw_model):
            path, payload = api._build_grok_payload(prompt_text, raw_model, pil_images, aspect_ratio, duration, resolution, mode)
        elif is_runninghub_seedance_model(raw_model):
            path, payload = api._build_seedance_payload(prompt_text, raw_model, mode, pil_images, video_inputs, audio_inputs, aspect_ratio, duration, resolution)
        elif is_runninghub_happyhorse_model(raw_model):
            path, payload = api._build_happyhorse_payload(prompt_text, mode, pil_images, video_inputs, aspect_ratio, duration, resolution)
        else:
            raise CometAPIError(f"不支持的 RunningHub 视频模型：{model}")
        task_id = api._submit_task(path, payload)
        return {"provider": "runninghub_video", "api_task_id": task_id, "status": "running", "video_url": "", "failure_reason": ""}

    if channel == "apimart":
        api = APIMartVideoAPI(api_key, get_channel_api_key("", "runninghub", "", "video"), get_channel_api_key("", "grsai", "", "image"))
        raw_model = canonical_apimart_video_model(model)
        raw_lower = raw_model.lower()
        if raw_lower in APIMART_SEEDANCE_VIDEO_MODELS:
            payload = api._build_seedance_payload(raw_model, prompt_text, pil_images, video_inputs, audio_inputs, aspect_ratio, duration, resolution, mode)
        elif raw_lower in APIMART_GROK_VIDEO_MODELS:
            payload = api._build_grok_payload(raw_model, prompt_text, pil_images, aspect_ratio, duration, resolution, mode)
        elif raw_lower in APIMART_SORA_VIDEO_MODELS:
            payload = api._build_sora_payload(raw_model, prompt_text, pil_images, aspect_ratio, duration, resolution, mode)
        elif raw_model == "Omni-Flash-Ext" or raw_lower in {item.lower() for item in APIMART_VEO_VIDEO_MODELS}:
            payload = api._build_veo_payload(raw_model, prompt_text, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
        elif raw_model in APIMART_HAILUO_VIDEO_MODELS:
            payload = api._build_hailuo_payload(raw_model, prompt_text, pil_images, duration, resolution, mode)
        elif raw_lower in APIMART_HAPPYHORSE_VIDEO_MODELS:
            payload = api._build_happyhorse_payload(prompt_text, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
        elif raw_lower in APIMART_KLING_VIDEO_MODELS:
            payload = api._build_kling_payload(raw_model, prompt_text, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
        elif raw_lower in APIMART_VIDU_VIDEO_MODELS:
            payload = api._build_vidu_payload(raw_model, prompt_text, pil_images, aspect_ratio, duration, resolution, mode)
        elif raw_lower in APIMART_WAN_VIDEO_MODELS:
            payload = api._build_wan_payload(raw_model, prompt_text, pil_images, video_inputs, audio_inputs, aspect_ratio, duration, resolution, mode)
        else:
            raise CometAPIError(f"不支持的 Apimart 视频模型：{model}")
        task_id = api._submit_task(payload)
        return {"provider": "apimart_video", "api_task_id": task_id, "status": "running", "video_url": "", "failure_reason": ""}

    if channel == "modelverse":
        api = ModelVerseVideoAPI(api_key, get_channel_api_key("", "runninghub", "", "video"), get_channel_api_key("", "grsai", "", "image"))
        raw_model = canonical_modelverse_video_model(model).lower()
        if raw_model in MODELVERSE_SEEDANCE_MODELS:
            payload = api._build_seedance_payload(raw_model, prompt_text, pil_images, video_inputs, audio_inputs, aspect_ratio, duration, resolution, mode)
            task_id = api._submit_task(payload)
            provider = "modelverse_video"
        elif raw_model in MODELVERSE_HAPPYHORSE_MODELS:
            payload = api._build_happyhorse_payload(prompt_text, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
            task_id = api._submit_task(payload)
            provider = "modelverse_video"
        elif raw_model in MODELVERSE_SORA_VIDEO_MODELS:
            task_id = api._submit_sora_video(prompt_text, pil_images, aspect_ratio, duration, mode)
            provider = "modelverse_sora_video"
        elif raw_model in MODELVERSE_KLING_VIDEO_MODELS:
            payload = api._build_kling_payload(raw_model, prompt_text, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
            task_id = api._submit_task(payload)
            provider = "modelverse_video"
        else:
            raise CometAPIError(f"不支持的优云智算视频模型：{model}")
        return {"provider": provider, "api_task_id": task_id, "status": "running", "video_url": "", "failure_reason": ""}

    api = PrivateVideoAPI(api_key, get_channel_api_url(channel), get_channel_api_key("", "runninghub", "", "video"), get_channel_api_key("", "grsai", "", "image"))
    if is_vidu_video_model(model):
        endpoint, payload = api._build_vidu_payload(prompt_text, model, pil_images, aspect_ratio, duration, resolution, mode)
        task_id = api._submit_vidu_task(endpoint, payload)
        return {"provider": "private_vidu_video", "api_task_id": task_id, "status": "running", "video_url": "", "failure_reason": ""}
    if is_hailuo_video_model(model):
        payload = api._build_hailuo_payload(prompt_text, model, pil_images, duration, resolution, mode)
        task_id = api._submit_hailuo_task(payload)
        return {"provider": "private_hailuo_video", "api_task_id": task_id, "status": "running", "video_url": "", "failure_reason": ""}
    if is_kling_video_model(model):
        action, payload = api._build_kling_payload(prompt_text, model, pil_images, aspect_ratio, duration, resolution, mode)
        task_id = api._submit_kling_task(action, payload)
        return {"provider": "private_kling_video", "api_task_id": task_id, "status": "running", "video_url": "", "query_action": action, "failure_reason": ""}
    if is_private_happyhorse_video_model(model):
        payload = api._build_happyhorse_payload(prompt_text, pil_images, video_inputs, aspect_ratio, duration, resolution, mode)
        task_id = api._submit_happyhorse_task(payload)
        return {"provider": "private_happyhorse_video", "api_task_id": task_id, "status": "running", "video_url": "", "failure_reason": ""}
    if is_sora_video_model(model):
        task_id = _submit_private_sora_video_task(api, prompt_text, model, pil_images, aspect_ratio, duration)
        return {"provider": "private_sora_video", "api_task_id": task_id, "status": "running", "video_url": "", "failure_reason": ""}
    if is_grok_video_model(model):
        payload = api._build_grok_payload(prompt_text, model, pil_images, aspect_ratio, duration)
    elif is_veo_video_model(model):
        payload = api._build_veo_payload(prompt_text, model, pil_images, aspect_ratio)
    else:
        raise CometAPIError(f"不支持的异步视频模型：{model}")
    task_id = api._submit_video_create(payload)
    return {"provider": "private_generic_video", "api_task_id": task_id, "status": "running", "video_url": "", "failure_reason": ""}


def _query_async_video_subtask(record: dict, subtask: dict, api_key: str) -> dict:
    provider = str(subtask.get("provider") or "")
    task_id = str(subtask.get("api_task_id") or "")
    channel = str(record.get("channel") or "").lower()
    model = str(record.get("model") or "")
    if provider == COMET_ASYNC_LOCAL_VIDEO_PROVIDER:
        if str(subtask.get("status") or "").lower() in COMET_ASYNC_ACTIVE_STATUSES and not _is_local_async_running(task_id):
            return {**subtask, "status": "failed", "failure_reason": "本地异步任务不在运行，可能 ComfyUI 已重启。"}
        return subtask
    if not task_id:
        return {**subtask, "status": "failed", "failure_reason": "缺少远端任务 ID"}

    if provider == "runninghub_video":
        status_raw, video_url, data = RunningHubVideoAPI(api_key)._query_task(task_id)
    elif provider == "apimart_video":
        status_raw, video_url, data = APIMartVideoAPI(api_key)._query_task(task_id)
    elif provider == "modelverse_video":
        status_raw, video_url, data = ModelVerseVideoAPI(api_key)._query_task(task_id)
    elif provider == "modelverse_sora_video":
        api = ModelVerseVideoAPI(api_key)
        status_raw, data = api._query_sora_video(task_id)
        video_url = f"{api.host}/v1/videos/{requests.utils.quote(str(task_id), safe='')}/content" if _canonical_async_status(status_raw) == "succeeded" else ""
    elif provider == "private_adapter_video":
        spec = get_private_channel_spec(channel)
        if not spec:
            return {**subtask, "status": "failed", "failure_reason": "私有渠道不存在"}
        query = getattr(_load_private_channel_adapter(spec), "query_video_task", None)
        if not callable(query):
            return {**subtask, "status": "failed", "failure_reason": "私有适配器缺少 query_video_task(context)"}
        context = _private_context(channel=channel, api_key=api_key, model=model, category="video", prompt=str(record.get("prompt") or ""))
        context.update({"task_id": task_id, "adapter_data": subtask.get("adapter_data") or {}})
        result = query(context)
        if not isinstance(result, dict):
            return {**subtask, "status": "failed", "failure_reason": "私有异步视频查询返回格式无效"}
        status = _canonical_async_status(str(result.get("status") or ""), has_result=bool(result.get("video_url") or result.get("url") or result.get("path")))
        return {
            **subtask,
            "status": status,
            "video_url": str(result.get("video_url") or result.get("url") or subtask.get("video_url") or ""),
            "local_path": str(result.get("path") or result.get("local_path") or subtask.get("local_path") or ""),
            "failure_reason": str(result.get("error") or result.get("failure_reason") or ""),
        }
    elif provider.startswith("private_"):
        api = PrivateVideoAPI(api_key, get_channel_api_url(channel), get_channel_api_key("", "runninghub", "", "video"), get_channel_api_key("", "grsai", "", "image"))
        if provider == "private_vidu_video":
            status_raw, video_url, data = api._query_vidu_video(task_id)
        elif provider == "private_hailuo_video":
            status_raw, video_url, data = api._query_hailuo_video(task_id)
        elif provider == "private_kling_video":
            status_raw, video_url, data = api._query_kling_video(str(subtask.get("query_action") or kling_action_for_mode(normalize_kling_video_mode(model, record.get("mode") or ""))), task_id)
        elif provider == "private_happyhorse_video":
            status_raw, video_url, data = api._query_happyhorse_video(task_id)
        elif provider == "private_sora_video":
            status_raw, video_url, data = _query_private_sora_video(api, task_id)
        else:
            status_raw, video_url, data = api._query_video(task_id)
    else:
        return {**subtask, "status": "failed", "failure_reason": f"未知异步视频 provider：{provider}"}

    status = _canonical_async_status(status_raw, has_result=bool(video_url))
    failure = ""
    if status == "failed":
        failure = str(data.get("error") or data.get("message") or data.get("failedReason") or data)[:300] if isinstance(data, dict) else str(data)[:300]
    return {**subtask, "status": status, "video_url": video_url or subtask.get("video_url") or "", "failure_reason": failure}


def refresh_comet_async_tasks(category: str) -> None:
    category = str(category or "").lower()
    snapshot = read_comet_async_tasks()
    for task_id, info in snapshot.items():
        if not isinstance(info, dict) or str(info.get("category") or "").lower() != category:
            continue
        if str(info.get("status") or "").lower() not in COMET_ASYNC_ACTIVE_STATUSES:
            continue
        channel = str(info.get("channel") or "").lower()
        model = str(info.get("model") or "")
        subtasks = copy.deepcopy(info.get("subtasks") if isinstance(info.get("subtasks"), list) else [])
        active_subtasks = [
            subtask
            for subtask in subtasks
            if str(subtask.get("status") or "").lower() in COMET_ASYNC_ACTIVE_STATUSES
        ]
        local_only = bool(active_subtasks) and all(
            str(subtask.get("provider") or "") in {COMET_ASYNC_LOCAL_IMAGE_PROVIDER, COMET_ASYNC_LOCAL_VIDEO_PROVIDER}
            for subtask in active_subtasks
        )
        api_key = "" if local_only else get_channel_api_key("", channel, model, category)
        if not api_key and not local_only:
            update_comet_async_task(task_id, {"last_error": f"缺少 {channel} API Key，无法查询。"})
            continue
        changed = False
        for index, subtask in enumerate(subtasks):
            if str(subtask.get("status") or "").lower() not in COMET_ASYNC_ACTIVE_STATUSES:
                continue
            try:
                updated = _query_async_image_subtask(info, subtask, api_key) if category == "image" else _query_async_video_subtask(info, subtask, api_key)
            except Exception as exc:
                updated = {**subtask, "last_query_error": format_error_message(exc)}
            if updated != subtask:
                subtasks[index] = updated
                changed = True
        if not changed:
            continue
        ok = sum(1 for sub in subtasks if str(sub.get("status") or "").lower() == "succeeded")
        fail = sum(1 for sub in subtasks if str(sub.get("status") or "").lower() == "failed")
        active = sum(1 for sub in subtasks if str(sub.get("status") or "").lower() in COMET_ASYNC_ACTIVE_STATUSES)
        if active:
            status = "running"
        elif ok:
            status = "succeeded"
        else:
            status = "failed"
        last_error = "; ".join(str(sub.get("failure_reason") or sub.get("last_query_error") or "") for sub in subtasks if sub.get("failure_reason") or sub.get("last_query_error"))
        update_comet_async_task(task_id, {"subtasks": subtasks, "status": status, "last_error": last_error[:500]})


class CometAPIAsyncImage(CometAPIUnifiedImage):
    CATEGORY = "COMET/异步"
    FUNCTION = "submit"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("query_info",)
    OUTPUT_NODE = True

    def _text_result(self, message: str, is_error: bool = False):
        message = redact_sensitive_text(message)
        ui = {"comet_text": [message], "string": [message]}
        if is_error:
            ui["comet_error"] = [message]
        return {"ui": ui, "result": (message,)}

    def submit(self, channel: str, model: str, prompt: str, concurrency: int, aspect_ratio: str, image_size: str, quality: str = "medium", api_key: str = "", **kwargs):
        channel = str(channel or "").lower()
        if channel not in get_image_channel_choices():
            return self._text_result(f"不支持的渠道：{channel}", True)
        model = resolve_model_id(channel, model, "image")
        try:
            concurrency = max(1, min(10, int(concurrency)))
        except Exception:
            concurrency = 1
        final_api_key = get_channel_api_key(api_key, channel, model, "image")
        if not final_api_key:
            return self._text_result(f"缺少 {channel} API Key，请先在设置中心填写。", True)

        try:
            if channel == "grsai":
                max_refs = GPT_IMAGE_MAX_IMAGES if model in GPT_IMAGE_MODELS else NANO_BANANA_MAX_IMAGES
            elif channel == "runninghub":
                max_refs = runninghub_image_max_images(model)
            elif channel == "modelverse":
                max_refs = modelverse_image_max_images(model)
            elif channel == "apimart":
                max_refs = apimart_image_max_images(model)
            else:
                limits = private_media_limits_for_model(channel, model, "image") if get_private_channel_spec(channel) else {}
                max_refs = int(limits.get("image") or MAX_IMAGE_INPUTS)
            pil_refs = self._collect_input_pils(kwargs, max_refs)
            upload_cache = {}
            upload_lock = threading.Lock()
            local_jobs = []
            if image_channel_uses_remote_async(channel):
                subtasks = [
                    submit_async_image_subtask(
                        channel=channel,
                        model=model,
                        api_key=final_api_key,
                        prompt=prompt,
                        pil_refs=pil_refs,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                        quality=quality,
                        subtask_idx=index,
                        grsai_upload_cache=upload_cache,
                        grsai_upload_lock=upload_lock,
                    )
                    for index in range(concurrency)
                ]
                submit_mode = "remote_async"
            else:
                subtasks = []
                submit_mode = "local_async"
                for index in range(concurrency):
                    subtask = make_local_wrapped_image_subtask(prompt, index)
                    subtasks.append(subtask)
                    local_jobs.append(
                        (
                            subtask,
                            {
                                "channel": channel,
                                "model": model,
                                "api_key": final_api_key,
                                "task": {
                                    "prompt": prompt,
                                    "fixed_refs": pil_refs,
                                    "base_index": 0,
                                    "candidate_index": index + 1,
                                },
                                "aspect_ratio": aspect_ratio,
                                "image_size": image_size,
                                "quality": quality,
                                "upload_cache": upload_cache,
                                "upload_lock": upload_lock,
                            },
                        )
                    )
            actual_models = [sub.get("model") for sub in subtasks if sub.get("model")]
            task_id = add_comet_async_task(
                {
                    "category": "image",
                    "source": "image",
                    "channel": channel,
                    "model": actual_models[0] if actual_models else model,
                    "prompt": prompt,
                    "status": "running",
                    "mode": submit_mode,
                    "concurrency": concurrency,
                    "subtasks": subtasks,
                }
            )
            for subtask, payload in local_jobs:
                schedule_local_wrapped_image_subtask(task_id, subtask, payload)
            message = f"异步图片提交成功 | {task_id} | 模型: {actual_models[0] if actual_models else model} | 子任务数: {len(subtasks)}"
            return self._text_result(message)
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._text_result(f"异步图片提交失败：{format_error_message(exc)}", True)


class CometAPIAsyncBatchImage(CometAPIBatchImage):
    CATEGORY = "COMET/异步"
    FUNCTION = "submit"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("query_info",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].pop("batch_mode", None)
        inputs["required"].pop("folder_path", None)
        return inputs

    def _text_result(self, message: str, is_error: bool = False):
        message = redact_sensitive_text(message)
        ui = {"comet_text": [message], "string": [message]}
        if is_error:
            ui["comet_error"] = [message]
        return {"ui": ui, "result": (message,)}

    def submit(
        self,
        channel: str,
        model: str,
        pairing_mode: str,
        concurrency: int,
        aspect_ratio: str,
        image_size: str,
        quality: str = "medium",
        api_key: str = "",
        batch_text=None,
        **kwargs,
    ):
        channel = str(channel or "").lower()
        if channel not in get_image_channel_choices():
            return self._text_result(f"不支持的渠道：{channel}", True)
        model = resolve_model_id(channel, model, "image")
        prompts = self._collect_prompts(batch_text)
        if not prompts:
            return self._text_result("异步批量生图需要连接批量文本卡片，并至少提供一条提示词。", True)
        final_api_key = get_channel_api_key(api_key, channel, model, "image")
        if not final_api_key:
            return self._text_result(f"缺少 {channel} API Key，请先在设置中心填写。", True)

        try:
            advanced_settings = get_batch_image_advanced_settings()
            max_tasks = int(advanced_settings.get("max_tasks") or 200)
            batch_concurrency = int(advanced_settings.get("batch_concurrency") or 20)
        except Exception:
            max_tasks = 200
            batch_concurrency = 20
        try:
            candidates_per_prompt = max(1, min(10, int(concurrency)))
        except Exception:
            candidates_per_prompt = 1

        try:
            batch_mode = COMET_BATCH_IMAGE_MODE_REGULAR
            pairing_mode = self._normalize_pairing_mode(pairing_mode, batch_mode)
            collect_ref_limit = self._fixed_ref_collect_limit(channel, model, batch_mode, pairing_mode)
            fixed_refs = self._collect_fixed_refs(kwargs, collect_ref_limit)
            tasks, warnings = self._build_tasks(
                prompts=prompts,
                fixed_refs=fixed_refs,
                batch_mode=batch_mode,
                pairing_mode=pairing_mode,
                folder_path="",
                candidates_per_prompt=candidates_per_prompt,
                max_tasks=max_tasks,
            )
            if not tasks:
                return self._text_result("没有可提交的异步批量任务。", True)
            upload_cache = {}
            upload_lock = threading.Lock()

            if not image_channel_uses_remote_async(channel):
                subtasks = []
                local_jobs = []
                for index, task in enumerate(tasks, start=1):
                    subtask = make_local_wrapped_image_subtask(
                        task.get("prompt") or "",
                        index,
                        {
                            "batch_index": index,
                            "prompt_index": task.get("prompt_index"),
                            "source_index": task.get("source_index"),
                            "candidate_index": task.get("candidate_index"),
                            "source_name": task.get("source_name") or "",
                        },
                    )
                    subtasks.append(subtask)
                    local_jobs.append(
                        (
                            subtask,
                            {
                                "channel": channel,
                                "model": model,
                                "api_key": final_api_key,
                                "task": task,
                                "aspect_ratio": aspect_ratio,
                                "image_size": image_size,
                                "quality": quality,
                                "upload_cache": upload_cache,
                                "upload_lock": upload_lock,
                            },
                        )
                    )
                task_id = add_comet_async_task(
                    {
                        "category": "image",
                        "source": "batch_image",
                        "channel": channel,
                        "model": model,
                        "prompt": f"批量图像：{len(tasks)} 个任务",
                        "status": "running",
                        "mode": "local_async",
                        "batch_mode": batch_mode,
                        "pairing_mode": pairing_mode,
                        "subtasks": subtasks,
                        "submit_errors": [],
                    }
                )
                for subtask, payload in local_jobs:
                    schedule_local_wrapped_image_subtask(task_id, subtask, payload)
                message = f"异步批量图片提交完成 | {task_id} | 子任务数: {len(subtasks)}"
                if warnings:
                    message += "\n提示：" + "；".join(warnings)
                return self._text_result(message)

            def submit_one(index_and_task):
                index, task = index_and_task
                refs = list(task.get("fixed_refs") or [])
                refs = refs[: self._max_refs(channel, model)]
                prompt_for_api = convert_prompt_asset_mentions(task["prompt"], image_count=len(refs))
                subtask = submit_async_image_subtask(
                    channel=channel,
                    model=model,
                    api_key=final_api_key,
                    prompt=prompt_for_api,
                    pil_refs=refs,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    quality=quality,
                    subtask_idx=index,
                    grsai_upload_cache=upload_cache,
                    grsai_upload_lock=upload_lock,
                )
                subtask.update(
                    {
                        "batch_index": index,
                        "prompt_index": task.get("prompt_index"),
                        "source_index": task.get("source_index"),
                        "candidate_index": task.get("candidate_index"),
                        "source_name": task.get("source_name") or "",
                    }
                )
                return subtask

            subtasks = []
            errors = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, batch_concurrency)) as executor:
                futures = [executor.submit(submit_one, item) for item in enumerate(tasks, start=1)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        subtasks.append(future.result())
                    except Exception as exc:
                        errors.append(format_error_message(exc))

            if not subtasks:
                return self._text_result(f"异步批量提交全部失败：{'; '.join(errors[:3])}", True)
            task_id = add_comet_async_task(
                {
                    "category": "image",
                    "source": "batch_image",
                    "channel": channel,
                    "model": model,
                    "prompt": f"批量图像：{len(tasks)} 个任务",
                    "status": "running",
                    "mode": "remote_async",
                    "batch_mode": batch_mode,
                    "pairing_mode": pairing_mode,
                    "subtasks": subtasks,
                    "submit_errors": errors[:20],
                }
            )
            message = f"异步批量图片提交完成 | {task_id} | 成功: {len(subtasks)}/{len(tasks)} | 失败: {len(errors)}"
            if warnings:
                message += "\n提示：" + "；".join(warnings)
            if errors:
                message += "\n失败示例：" + "；".join(errors[:3])
            return self._text_result(message)
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._text_result(f"异步批量图片提交失败：{format_error_message(exc)}", True)


class CometAPIAsyncVideo(CometAPIUnifiedVideo):
    CATEGORY = "COMET/异步"
    FUNCTION = "submit"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("query_info",)
    OUTPUT_NODE = True

    def _text_result(self, message: str, is_error: bool = False):
        message = redact_sensitive_text(message)
        ui = {"comet_text": [message], "string": [message]}
        if is_error:
            ui["comet_error"] = [message]
        return {"ui": ui, "result": (message,)}

    def submit(
        self,
        channel: str,
        model_family: str,
        model: str,
        prompt: str,
        aspect_ratio: str,
        duration: str,
        resolution: str = "1080P",
        mode: str = "首尾帧",
        size: str = "small",
        api_key: str = "",
        **kwargs,
    ):
        channel_key = str(channel or get_default_video_channel()).lower()
        if channel_key not in get_video_channel_choices():
            return self._text_result(f"不支持的视频渠道：{channel}", True)
        model = resolve_model_id(channel_key, model, "video")
        actual_model = canonical_apimart_video_model(model) if channel_key == "apimart" else model
        final_api_key = get_channel_api_key(api_key, channel_key, model, "video")
        if not final_api_key:
            return self._text_result(f"缺少 {channel_key} API Key，请先在设置中心填写。", True)

        try:
            private_channel = get_private_channel_spec(channel_key)
            if private_channel:
                limits = private_media_limits_for_model(channel_key, model, "video", mode)
                max_refs = int(limits.get("image") or 0)
                allowed_media_types = {media_type for media_type, limit in limits.items() if int(limit or 0) > 0}
            else:
                if channel_key == "modelverse":
                    max_refs = modelverse_video_max_images(actual_model)
                elif channel_key == "apimart":
                    max_refs = apimart_video_max_images(actual_model, mode)
                else:
                    max_refs = video_max_images_for_model(actual_model, mode)
                allowed_media_types = video_allowed_media_types(channel_key, actual_model, mode)
            pil_images, video_inputs, audio_inputs, used_media_types = self._collect_unified_media(kwargs, max_images=max_refs)
            unsupported_media_types = sorted({media_type for media_type in used_media_types if media_type not in allowed_media_types})
            if unsupported_media_types:
                unsupported_label = "、".join(media_type_label(media_type) for media_type in unsupported_media_types)
                allowed_label = "、".join(media_type_label(media_type) for media_type in sorted(allowed_media_types)) if allowed_media_types else "不接收素材"
                return self._text_result(f"当前视频模式不支持接入{unsupported_label}素材；当前允许：{allowed_label}。", True)
            local_payload = None
            if video_channel_uses_remote_async(channel_key, actual_model):
                subtask = submit_async_video_subtask(
                    channel=channel_key,
                    model=actual_model,
                    api_key=final_api_key,
                    prompt=prompt,
                    pil_images=pil_images,
                    video_inputs=video_inputs,
                    audio_inputs=audio_inputs,
                    aspect_ratio=aspect_ratio,
                    duration=str(duration),
                    resolution=resolution,
                    mode=mode,
                    size=size,
                )
                submit_mode = "remote_async"
            else:
                subtask = make_local_wrapped_video_subtask(prompt)
                submit_mode = "local_async"
                local_payload = {
                    "channel": channel_key,
                    "model": actual_model,
                    "api_key": final_api_key,
                    "prompt": prompt,
                    "pil_images": pil_images,
                    "video_inputs": video_inputs,
                    "audio_inputs": audio_inputs,
                    "aspect_ratio": aspect_ratio,
                    "duration": str(duration),
                    "resolution": resolution,
                    "mode": mode,
                    "size": size,
                }
            task_id = add_comet_async_task(
                {
                    "category": "video",
                    "source": "video",
                    "channel": channel_key,
                    "model": actual_model,
                    "model_family": model_family,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "duration": str(duration),
                    "resolution": resolution,
                    "mode": mode,
                    "status": "running",
                    "mode_kind": submit_mode,
                    "subtasks": [subtask],
                }
            )
            if local_payload is not None:
                schedule_local_wrapped_video_subtask(task_id, subtask, local_payload)
            message = f"异步视频提交成功 | {task_id} | 模型: {actual_model}"
            return self._text_result(message)
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._text_result(f"异步视频提交失败：{format_error_message(exc)}", True)


def _ready_async_tasks(category: str, limit: int = 1) -> list[tuple[str, dict]]:
    tasks = read_comet_async_tasks()
    ready = [
        (task_id, info)
        for task_id, info in tasks.items()
        if isinstance(info, dict)
        and str(info.get("category") or "").lower() == category
        and str(info.get("status") or "").lower() == "succeeded"
    ]
    ready.sort(key=lambda item: item[1].get("submitted_at") or "2999-01-01")
    try:
        safe_limit = max(1, min(50, int(limit)))
    except Exception:
        safe_limit = 1
    return ready[:safe_limit]


def _oldest_ready_async_task(category: str) -> tuple[str, dict] | tuple[None, None]:
    ready = _ready_async_tasks(category, 1)
    return ready[0] if ready else (None, None)


def _normalize_async_receive_count(run_count: int) -> int:
    try:
        return max(1, min(50, int(run_count)))
    except Exception:
        return 1


class CometAPIAsyncImageReceiver:
    CATEGORY = "COMET/异步"
    FUNCTION = "receive"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "query_info")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_count": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1, "display_name": "运行次数"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def _result(self, image, text: str, is_error: bool = False, no_result: bool = False):
        text = redact_sensitive_text(text)
        ui = {"comet_text": [text], "string": [text]}
        if is_error:
            ui["comet_error"] = [text]
        if no_result:
            image = make_comet_async_no_result("image", text)
            ui["comet_async_no_result"] = ["image"]
        return {"ui": ui, "result": (image, text)}

    def _download_ready_image_task(self, task_id: str, info: dict) -> tuple[torch.Tensor, int, list[str]]:
        urls = []
        local_paths = []
        for subtask in info.get("subtasks") or []:
            if str(subtask.get("status") or "").lower() != "succeeded":
                continue
            urls.extend(str(url) for url in (subtask.get("image_urls") or []) if str(url or "").strip())
            local_paths.extend(str(path) for path in (subtask.get("local_image_paths") or []) if str(path or "").strip())
        if not urls and not local_paths:
            update_comet_async_task(task_id, {"status": "failed", "last_error": "任务成功但没有图片地址"})
            raise CometAPIError(f"{task_id} 状态成功但没有图片地址或本地缓存。")

        pil_images = []
        errors = []
        for path in local_paths:
            try:
                if not os.path.exists(path):
                    errors.append(f"本地缓存图片不存在：{path}")
                    continue
                with Image.open(path) as image:
                    pil_images.append(safe_pil_to_rgb(image.copy()))
            except Exception as exc:
                errors.append(f"读取本地缓存图片失败：{path}: {format_error_message(exc)}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(12, len(urls)))) as executor:
            future_to_url = {executor.submit(download_image, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    image = future.result()
                    if image:
                        pil_images.append(safe_pil_to_rgb(image))
                    else:
                        errors.append(f"图片下载失败：{url}")
                except Exception as exc:
                    errors.append(f"图片下载异常：{url}: {format_error_message(exc)}")
        if not pil_images:
            update_comet_async_task(task_id, {"status": "download_failed", "last_error": "；".join(errors[:5])})
            raise CometAPIError(f"{task_id} 图片下载失败：{'；'.join(errors[:5])}")

        update_comet_async_task(task_id, {"status": "downloaded", "downloaded_at": _async_now(), "result_count": len(pil_images)})
        return _batch_image_tensor(pil_images), len(pil_images), errors

    def receive(self, run_count: int = 1):
        try:
            refresh_comet_async_tasks("image")
            ready_tasks = _ready_async_tasks("image", _normalize_async_receive_count(run_count))
            if not ready_tasks:
                return self._result(None, f"暂无可用图片结果。\n{format_comet_async_task_panel('image')}", no_result=True)

            last_image = None
            received = []
            warnings = []
            failures = []
            for task_id, info in ready_tasks:
                try:
                    image, image_count, errors = self._download_ready_image_task(task_id, info)
                    last_image = image
                    received.append((task_id, image_count))
                    if errors:
                        warnings.append(f"{task_id}: {'；'.join(errors[:3])}")
                except Exception as exc:
                    failures.append(f"{task_id}: {format_error_message(exc)}")

            status_text = format_comet_async_task_panel("image")
            if received:
                details = "；".join(f"{task_id}: 图片 {image_count}" for task_id, image_count in received)
                final_text = f"收取成功: {len(received)} 个图片任务 | {details}\n{status_text}"
                if warnings:
                    final_text += "\n提示：" + "；".join(warnings[:3])
                if failures:
                    final_text += "\n未收取：" + "；".join(failures[:3])
                return self._result(last_image, final_text)
            failure_text = "；".join(failures[:5]) if failures else "没有可收取的图片结果。"
            return self._result(None, f"收取图片结果失败：{failure_text}\n{status_text}", True)
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._result(None, f"收取图片结果失败：{format_error_message(exc)}", True)


class CometAPIAsyncVideoReceiver:
    CATEGORY = "COMET/异步"
    FUNCTION = "receive"
    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "query_info")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_count": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1, "display_name": "运行次数"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def _result(self, video, text: str, is_error: bool = False, no_result: bool = False):
        text = redact_sensitive_text(text)
        ui = {"comet_text": [text], "string": [text]}
        if is_error:
            ui["comet_error"] = [text]
        if no_result:
            video = make_comet_async_no_result("video", text)
            ui["comet_async_no_result"] = ["video"]
        return {"ui": ui, "result": (video, text)}

    def _download_ready_video(self, info: dict, subtask: dict) -> tuple[str, str]:
        local_path = str(subtask.get("local_path") or "")
        if local_path and os.path.exists(local_path):
            return local_path, str(subtask.get("video_url") or "")
        provider = str(subtask.get("provider") or "")
        task_id = str(subtask.get("api_task_id") or "")
        channel = str(info.get("channel") or "").lower()
        model = str(info.get("model") or "")
        api_key = get_channel_api_key("", channel, model, "video")
        if provider == "modelverse_sora_video":
            api = ModelVerseVideoAPI(api_key)
            return api._download_sora_video_content(task_id)
        video_url = str(subtask.get("video_url") or "")
        if not video_url:
            raise CometAPIError("任务成功但没有视频地址")
        return download_video_asset(video_url, prefix="CometAPIAsyncVideo"), video_url

    def _download_ready_video_task(self, task_id: str, info: dict) -> tuple[VideoAdapter, str]:
        subtask = next(
            (sub for sub in info.get("subtasks") or [] if str(sub.get("status") or "").lower() == "succeeded"),
            None,
        )
        if not subtask:
            update_comet_async_task(task_id, {"status": "failed", "last_error": "任务成功但没有可收取的子任务"})
            raise CometAPIError(f"{task_id} 没有可收取的已完成子任务。")
        path, video_url = self._download_ready_video(info, subtask)
        update_comet_async_task(task_id, {"status": "downloaded", "downloaded_at": _async_now(), "video_url": video_url, "local_path": path})
        return VideoAdapter(path, video_url=video_url), os.path.basename(path)

    def receive(self, run_count: int = 1):
        try:
            refresh_comet_async_tasks("video")
            ready_tasks = _ready_async_tasks("video", _normalize_async_receive_count(run_count))
            if not ready_tasks:
                return self._result(None, f"暂无可用视频结果。\n{format_comet_async_task_panel('video')}", no_result=True)

            last_video = None
            received = []
            failures = []
            for task_id, info in ready_tasks:
                try:
                    video, filename = self._download_ready_video_task(task_id, info)
                    last_video = video
                    received.append((task_id, filename))
                except Exception as exc:
                    failures.append(f"{task_id}: {format_error_message(exc)}")

            status_text = format_comet_async_task_panel("video")
            if received:
                details = "；".join(f"{task_id}: {filename}" for task_id, filename in received)
                final_text = f"收取成功: {len(received)} 个视频任务 | {details}\n{status_text}"
                if failures:
                    final_text += "\n未收取：" + "；".join(failures[:3])
                return self._result(last_video, final_text)
            failure_text = "；".join(failures[:5]) if failures else "没有可收取的视频结果。"
            return self._result(None, f"收取视频结果失败：{failure_text}\n{status_text}", True)
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._result(None, f"收取视频结果失败：{format_error_message(exc)}", True)


class CometAPIImageCard:
    CATEGORY = "COMET"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "batch_image")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "CometAPIImageCard",
                        "display_name": "\u4fdd\u5b58\u8def\u5f84/\u6587\u4ef6\u540d\u524d\u7f00",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "asset_ref": ("STRING", {"multiline": False, "default": ""}),
                "asset_index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
                # 由前端在 graphToPrompt 阶段写入：batch_image 输出端口是否被连线消费。
                # 没人连就不构造大 tensor，避免几十张图把内存吃满。
                "_batch_image_required": ("BOOLEAN", {"default": False}),
                # 由前端在 graphToPrompt 阶段写入：image 输入端口溯源到的最近一个 batch 节点 id。
                # 多个 batch 节点同时跑时，可以精确把对应的原图列表挂回来，避免误命中其它 batch。
                "_batch_source_node_id": ("STRING", {"default": ""}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, filename_prefix="CometAPIImageCard", image=None, asset_ref="", asset_index=0):
        if image is not None:
            return float("NaN")
        return f"{asset_ref}:{asset_index}" or filename_prefix

    def _error(self, message: str):
        message = redact_sensitive_text(message)
        print(f"[CometAPI] Image Card: {message}")
        image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return {"ui": {"comet_error": [message]}, "result": (image, image)}

    def _copy_original_pngs(
        self,
        sniffed_refs: list[dict],
        prefix: str,
    ) -> tuple[list[dict], list[Image.Image]] | None:
        """根据 batch 缓存里的原图引用，把每张原 PNG 复制到目标目录，返回 (asset_refs, pil_images)。

        任何一步失败都返回 None，调用方退回旧逻辑。
        """
        try:
            target_dir, safe_prefix, subfolder, absolute_dir = _resolve_asset_output_target(
                prefix, "CometAPIImageCard", ".png"
            )
            os.makedirs(target_dir, exist_ok=True)

            saved_refs: list[dict] = []
            pil_images: list[Image.Image] = []
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for index, src_ref in enumerate(sniffed_refs, start=1):
                src_path = asset_abs_path(src_ref)
                if not src_path or not os.path.exists(src_path):
                    return None
                ext = os.path.splitext(src_path)[1].lower() or ".png"
                if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
                    ext = ".png"
                filename = f"{safe_prefix}_{timestamp}_{index:02d}_{uuid.uuid4().hex[:8]}{ext}"
                dest_path = os.path.join(target_dir, filename)
                # 直接二进制拷贝原文件，保留原尺寸/无任何 pad
                with open(src_path, "rb") as src_handle, open(dest_path, "wb") as dst_handle:
                    dst_handle.write(src_handle.read())
                saved_refs.append(_make_asset_ref(filename, subfolder, absolute_dir))
                # 同步加载 PIL 用于下游 image / batch_image 端口输出
                with Image.open(dest_path) as opened:
                    pil_images.append(safe_pil_to_rgb(opened.copy()))
            return saved_refs, pil_images
        except Exception as exc:
            print(f"[CometAPI] Image Card: 嗅探 batch 原图复制失败，退回 tensor 落盘：{redact_sensitive_text(exc)}")
            return None

    def execute(self, filename_prefix: str = "CometAPIImageCard", image=None, asset_ref: str = "", asset_index: int = 0, _batch_image_required: bool = False, _batch_source_node_id: str = ""):
        try:
            try:
                selected_index = max(0, int(asset_index))
            except Exception:
                selected_index = 0

            # 占位张量：用于 batch_image 端口没人连时的轻量返回，避免无谓的大 tensor 构造
            placeholder = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

            if is_comet_async_no_result(image, "image"):
                message = comet_async_no_result_message(image, "暂无可用图片结果。")
                return {
                    "ui": {"comet_text": [message], "comet_clear_message": ["1"]},
                    "result": (placeholder, placeholder),
                }

            if image is not None:
                prefix = filename_prefix.strip() or "CometAPIImageCard"
                tensor_count = int(image.shape[0]) if image.ndim == 4 else 1

                # 多图场景优先嗅探 batch 缓存：拿到的是无黑边的原图
                use_native = None
                if tensor_count >= 2:
                    sniffed = consume_batch_asset_refs_for_count(tensor_count, _batch_source_node_id)
                    if sniffed:
                        use_native = self._copy_original_pngs(sniffed, prefix)

                if use_native is not None:
                    saved_refs, pil_images = use_native
                else:
                    # 回退：单图、或者嗅探不命中时，仍走 tensor 落盘（可能继承上游的 pad 黑边）
                    saved_refs, pil_images = save_asset_images(image, prefix=prefix)

                if not saved_refs:
                    return self._error("没有可保存的图片。")

                selected_index = 0
                ref_json = asset_refs_to_json(saved_refs)
                # main 端口：当前选中那张原图（原尺寸，单张 tensor）
                main_tensor = pil_to_tensor(pil_images[selected_index])
                # batch_image 端口：按需构造。
                # - 单图：跟 main 一致；
                # - 多图但下游没连：返回 1x1 占位，省去对每张图做 pad+cat 的内存开销；
                # - 多图且下游已连：构造 padded 大 tensor（受 ComfyUI IMAGE 协议限制必须同尺寸）
                if len(pil_images) >= 2:
                    batch_tensor = _batch_image_tensor(pil_images) if _batch_image_required else placeholder
                else:
                    batch_tensor = main_tensor
                return {
                    "ui": {
                        "images": saved_refs,
                        "asset_ref": [ref_json],
                        "asset_index": [selected_index],
                        "asset_count": [len(saved_refs)],
                    },
                    "result": (main_tensor, batch_tensor),
                }

            refs = normalize_asset_refs(asset_ref)
            if not refs:
                return self._error("还没有已保存的图片素材。")

            selected_index = min(selected_index, len(refs) - 1)
            selected_ref = refs[selected_index]
            try:
                pil_image = load_asset_image(selected_ref)
            except FileNotFoundError as exc:
                missing_path = str(exc)
                return self._error(
                    f"找不到原始图片文件，可能已被手动删除或移动到其它位置。\n"
                    f"路径：{missing_path}\n"
                    f"建议：重新跑一次上游节点生成新的图片，或把右键菜单 → 选择文件 切到一个仍在硬盘上的素材。"
                )
            ref_json = asset_refs_to_json(refs)
            main_tensor = pil_to_tensor(pil_image)
            # 复用模式：batch_image 同样按需构造
            if len(refs) >= 2 and _batch_image_required:
                try:
                    all_pil = []
                    missing_count = 0
                    for ref in refs:
                        try:
                            all_pil.append(load_asset_image(ref))
                        except FileNotFoundError:
                            missing_count += 1
                    if missing_count == 0 and all_pil:
                        batch_tensor = _batch_image_tensor(all_pil)
                    elif all_pil:
                        # 部分缺失：尽力构造，附加提示
                        batch_tensor = _batch_image_tensor(all_pil)
                        print(f"[CometAPI] Image Card: batch_image 端口构造时有 {missing_count} 张原图缺失，已用现有 {len(all_pil)} 张拼接。")
                    else:
                        batch_tensor = placeholder
                except Exception:
                    batch_tensor = placeholder
            elif len(refs) >= 2:
                batch_tensor = placeholder
            else:
                batch_tensor = main_tensor
            return {
                "ui": {
                    "images": refs,
                    "asset_ref": [ref_json],
                    "asset_index": [selected_index],
                    "asset_count": [len(refs)],
                },
                "result": (main_tensor, batch_tensor),
            }
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._error(format_error_message(exc))


class CometAPIVideoCard:
    CATEGORY = "COMET"
    FUNCTION = "execute"
    RETURN_TYPES = (IO.VIDEO,)
    RETURN_NAMES = ("video",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "CometAPIVideoCard",
                        "display_name": "\u4fdd\u5b58\u8def\u5f84/\u6587\u4ef6\u540d\u524d\u7f00",
                    },
                ),
            },
            "optional": {
                "video": (IO.VIDEO,),
                "asset_ref": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, filename_prefix="CometAPIVideoCard", video=None, asset_ref=""):
        if video is not None:
            return float("NaN")
        return asset_ref or filename_prefix

    def _error(self, message: str):
        message = redact_sensitive_text(message)
        print(f"[CometAPI] Video Card: {message}")
        return {"ui": {"comet_error": [message]}, "result": (VideoAdapter("", comet_error=message),)}

    def execute(self, filename_prefix: str = "CometAPIVideoCard", video=None, asset_ref: str = ""):
        try:
            if is_comet_async_no_result(video, "video"):
                message = comet_async_no_result_message(video, "暂无可用视频结果。")
                return {
                    "ui": {"comet_text": [message], "comet_clear_message": ["1"]},
                    "result": (VideoAdapter(""),),
                }

            if video is not None:
                prefix = filename_prefix.strip() or "CometAPIVideoCard"
                saved_ref, adapter = save_asset_video(video, prefix=prefix)
                return {
                    "ui": video_card_ui(saved_ref),
                    "result": (adapter,),
                }

            refs = normalize_asset_refs(asset_ref)
            if not refs:
                return self._error("还没有已保存的视频素材。")

            selected_ref = refs[0]
            return {
                "ui": video_card_ui(selected_ref),
                "result": (load_asset_video(selected_ref),),
            }
        except CometAPIError as exc:
            return self._error(format_error_message(exc))
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._error(format_error_message(exc))


def audio_card_ui(asset_refs: list[dict]) -> dict:
    refs = [ref for ref in asset_refs if ref]
    native_refs = [ref for ref in (asset_ref_to_native_view_ref(ref) for ref in refs) if ref]
    ui = {
        "asset_ref": [asset_refs_to_json(refs) if refs else ""],
        "asset_count": [len(refs)],
        "comet_audio": refs,
    }
    if native_refs:
        ui["audio"] = native_refs
    return ui


class CometAPIAudioCard:
    CATEGORY = "COMET"
    FUNCTION = "execute"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "CometAPIAudioCard",
                        "display_name": "\u4fdd\u5b58\u8def\u5f84/\u6587\u4ef6\u540d\u524d\u7f00",
                    },
                ),
            },
            "optional": {
                "audio": (IO.AUDIO,),
                "asset_ref": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, filename_prefix="CometAPIAudioCard", audio=None, asset_ref=""):
        if audio is not None:
            return float("NaN")
        return asset_ref or filename_prefix

    def _error(self, message: str):
        message = redact_sensitive_text(message)
        print(f"[CometAPI] Audio Card: {message}")
        return {"ui": {"comet_error": [message]}, "result": ()}

    def execute(self, filename_prefix: str = "CometAPIAudioCard", audio=None, asset_ref: str = ""):
        try:
            if audio is not None:
                refs = normalize_asset_refs(audio.get("comet_audio_refs")) if isinstance(audio, dict) else []
                if not refs:
                    prefix = filename_prefix.strip() or "CometAPIAudioCard"
                    refs = save_asset_audio(audio, prefix=prefix)
                return {"ui": audio_card_ui(refs), "result": ()}

            refs = normalize_asset_refs(asset_ref)
            if not refs:
                return self._error("还没有可显示的音频素材。")
            return {"ui": audio_card_ui(refs), "result": ()}
        except Exception as exc:
            print_sanitized_exception(exc)
            return self._error(format_error_message(exc))


def _clean_batch_prompt_line(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"^\s*(?:[-*•]\s+|\d{1,4}[.)、）]\s*|[一二三四五六七八九十百]+[、.)）]\s*)", "", text).strip()
    text = text.strip(" \t\r\n\"'“”‘’`")
    return text.strip()


def _prompts_from_json_data(data) -> list[str]:
    prompts: list[str] = []

    if isinstance(data, str):
        cleaned = _clean_batch_prompt_line(data)
        return [cleaned] if cleaned else []

    if isinstance(data, list):
        for item in data:
            prompts.extend(_prompts_from_json_data(item))
        return prompts

    if isinstance(data, dict):
        for key in ("prompt", "text", "content", "description"):
            value = data.get(key)
            if isinstance(value, str):
                cleaned = _clean_batch_prompt_line(value)
                if cleaned:
                    return [cleaned]
        for key in ("prompts", "items", "list", "data", "result", "results"):
            value = data.get(key)
            if isinstance(value, (list, dict, str)):
                prompts.extend(_prompts_from_json_data(value))
                if prompts:
                    return prompts

    return prompts


def parse_batch_prompts(raw_text) -> list[str]:
    text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    json_candidates = [text]
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fence_match:
        json_candidates.insert(0, fence_match.group(1).strip())

    array_start, array_end = text.find("["), text.rfind("]")
    if 0 <= array_start < array_end:
        json_candidates.append(text[array_start : array_end + 1])
    object_start, object_end = text.find("{"), text.rfind("}")
    if 0 <= object_start < object_end:
        json_candidates.append(text[object_start : object_end + 1])

    for candidate in json_candidates:
        try:
            prompts = _prompts_from_json_data(json.loads(candidate))
        except Exception:
            continue
        prompts = [_clean_batch_prompt_line(prompt) for prompt in prompts]
        prompts = [prompt for prompt in prompts if prompt]
        if prompts:
            return prompts

    marker_re = re.compile(r"^\s*(?:[-*•]\s+|\d{1,4}[.)、）]\s*|[一二三四五六七八九十百]+[、.)）]\s*)")
    marked_lines: list[str] = []
    plain_lines: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line in {"---", "———", "```"}:
            continue
        if marker_re.match(line):
            cleaned = _clean_batch_prompt_line(line)
            if cleaned:
                marked_lines.append(cleaned)
        else:
            cleaned = _clean_batch_prompt_line(line)
            if cleaned:
                plain_lines.append(cleaned)

    return marked_lines or plain_lines


def batch_prompts_payload(prompts: list[str]) -> dict:
    normalized = "\n".join(prompts)
    return {
        "type": COMET_BATCH_TEXT,
        "prompts": prompts,
        "count": len(prompts),
        "text": normalized,
    }


class CometAPITextCard:
    CATEGORY = "COMET"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Text",
                    },
                ),
            },
            "optional": {
                "text": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, content="", text=None):
        return f"{text if text is not None else ''}\n{content or ''}"

    def execute(self, content: str = "", text: str | None = None):
        stored_value = str(content or "")
        incoming_value = None if text is None else str(text or "")
        value = incoming_value if incoming_value else stored_value
        return {"ui": {"comet_text": [value]}, "result": (value,)}


class CometAPIBatchTextCard:
    CATEGORY = "COMET"
    FUNCTION = "execute"
    RETURN_TYPES = (COMET_BATCH_TEXT,)
    RETURN_NAMES = ("batch_text",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "一行一条批量提示词；也可以接收 LLM 输出的编号列表或 JSON 数组",
                    },
                ),
            },
            "optional": {
                "text": (
                    "STRING",
                    {
                        "forceInput": True,
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, content="", text=None):
        return f"{text if text is not None else ''}\n{content or ''}"

    def execute(self, content: str = "", text: str | None = None):
        stored_prompts = parse_batch_prompts(content)
        incoming_prompts = parse_batch_prompts(text) if text is not None and str(text).strip() else []
        prompts = incoming_prompts if incoming_prompts else stored_prompts

        payload = batch_prompts_payload(prompts)
        return {
            "ui": {
                "comet_batch_text": [payload],
            },
            "result": (payload,),
        }


class CometAPITextAppend:
    CATEGORY = "COMET/Internal"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("STRING", {"forceInput": True}),
                "suffix": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    def execute(self, base: str = "", suffix: str = ""):
        base_text = str(base or "")
        suffix_text = str(suffix or "")
        if base_text and suffix_text:
            separator = "" if base_text.endswith(("\n", " ", "\t")) or suffix_text.startswith(("\n", " ", "\t")) else "\n"
            return (f"{base_text}{separator}{suffix_text}",)
        return (base_text or suffix_text,)


def _collect_ancestors(prompt: dict, target_ids: list[str]) -> set[str]:
    seen = set()
    stack = list(target_ids)
    while stack:
        node_id = str(stack.pop())
        if node_id in seen:
            continue
        node = prompt.get(node_id)
        if node is None:
            continue
        seen.add(node_id)
        for value in node.get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
                src_id = str(value[0])
                if src_id not in seen:
                    stack.append(src_id)
    return seen


def _get_output_class_names() -> list[str]:
    try:
        import nodes as nodes_module
    except Exception:
        return []
    return sorted(
        name
        for name, cls in nodes_module.NODE_CLASS_MAPPINGS.items()
        if getattr(cls, "OUTPUT_NODE", False) is True
    )


def _is_safe_class(class_type: str, cls) -> bool:
    safe_classes = {
        "CometAPIUnifiedImage",
        "CometAPIBatchImage",
        "CometAPIAsyncImage",
        "CometAPIAsyncBatchImage",
        "CometAPIAsyncImageReceiver",
        "CometAPIUnifiedLLMNode",
        "CometAPIUnifiedVideo",
        "CometAPIAsyncVideo",
        "CometAPIAsyncVideoReceiver",
        "CometAPIUnifiedMusic",
        "CometAPIImageCard",
        "CometAPIVideoCard",
        "CometAPIAudioCard",
        "CometAPITextCard",
        "CometAPIBatchTextCard",
        "CometAPITextAppend",
        "PreviewImage",
        "SaveImage",
        "LoadImage",
        "LoadImageOutput",
        "LoadImageMask",
        "LoadAudio",
        "LoadVideo",
        "VHS_LoadAudio",
        "VHS_LoadVideo",
        "VHS_LoadVideoPath",
        "VHS_LoadAudioUpload",
        "easy showAnything",
        "ShowAnything",
        "Show Any",
        "ShowText|pysssss",
        "ShowText",
        "PreviewAny",
        "PreviewText",
        "PreviewTextNode",
        "DisplayString",
        "Display Text",
        "Text Preview",
        "CometAPIParallelTextSink",
        "Note",
        "PrimitiveNode",
        "Reroute",
    }
    return class_type in safe_classes or getattr(cls, "OUTPUT_NODE", False) is True and class_type in {"CometAPIImageCard", "CometAPIVideoCard", "CometAPIAudioCard", "CometAPITextCard", "CometAPIBatchTextCard"}


def _validate_parallel_safe(prompt: dict, subset: set[str], target_ids: list[str]) -> tuple[bool, str]:
    try:
        import nodes as nodes_module
    except Exception as exc:
        return False, f"cannot import nodes: {exc}"

    target_set = {str(node_id) for node_id in target_ids}
    for node_id in subset:
        node = prompt.get(str(node_id))
        if not node:
            return False, f"node {node_id} not in prompt"
        class_type = node.get("class_type")
        cls = nodes_module.NODE_CLASS_MAPPINGS.get(class_type)
        if cls is None:
            return False, f"unknown class_type: {class_type}"
        if str(node_id) in target_set and getattr(cls, "OUTPUT_NODE", False) is True:
            continue
        if not _is_safe_class(class_type, cls):
            return False, f"class '{class_type}' is not allowed in prototype local run (node #{node_id})"
    return True, ""


def _targets_are_output_nodes(prompt: dict, target_ids: list[str]) -> tuple[bool, str | None]:
    try:
        import nodes as nodes_module
    except Exception as exc:
        return False, f"cannot import nodes: {exc}"

    for node_id in target_ids:
        node = prompt.get(str(node_id))
        if node is None:
            return False, f"target node '{node_id}' not in prompt"
        class_type = node.get("class_type")
        cls = nodes_module.NODE_CLASS_MAPPINGS.get(class_type)
        if cls is None:
            return False, f"unknown class_type: {class_type}"
        if getattr(cls, "OUTPUT_NODE", False) is not True:
            return False, class_type
    return True, None


def _get_pool():
    global _EXECUTOR_POOL
    if _EXECUTOR_POOL is None:
        _EXECUTOR_POOL = concurrent.futures.ThreadPoolExecutor(
            max_workers=RUN_NODE_WORKERS,
            thread_name_prefix="CometAPIVWireRun",
        )
    return _EXECUTOR_POOL


def _record_history(prompt_queue, prompt_id, prompt, extra_data, target_ids, executor, status) -> None:
    try:
        from execution import MAXIMUM_HISTORY_SIZE
    except Exception:
        MAXIMUM_HISTORY_SIZE = 10000

    with prompt_queue.mutex:
        if len(prompt_queue.history) >= MAXIMUM_HISTORY_SIZE:
            try:
                prompt_queue.history.pop(next(iter(prompt_queue.history)))
            except StopIteration:
                pass

        fake_item = (-1, prompt_id, prompt, extra_data, list(target_ids))
        status_dict = None
        if status is not None:
            status_dict = {
                "status_str": status.status_str,
                "completed": status.completed,
                "messages": status.messages,
            }
        history_extra_data = dict(extra_data or {})
        history_extra_data.setdefault("create_time", int(time.time() * 1000))
        prompt_queue.history[prompt_id] = {
            "prompt": (fake_item[0], fake_item[1], fake_item[2], history_extra_data, fake_item[4]),
            "outputs": {},
            "status": status_dict,
        }
        history_result = getattr(executor, "history_result", None)
        if history_result:
            prompt_queue.history[prompt_id].update(history_result)

    try:
        prompt_queue.server.queue_updated()
    except Exception:
        pass


def _make_external_queue_item(prompt_id: str, prompt: dict, extra_data: dict, target_ids: list[str], client_id: str | None):
    queue_extra_data = dict(extra_data or {})
    queue_extra_data.setdefault("create_time", int(time.time() * 1000))
    if client_id:
        queue_extra_data["client_id"] = client_id
    return (-1, prompt_id, prompt, queue_extra_data, list(target_ids), {})


def _start_external_history_task(prompt_queue, item):
    with prompt_queue.mutex:
        item_id = prompt_queue.task_counter
        prompt_queue.currently_running[item_id] = copy.deepcopy(item)
        prompt_queue.task_counter += 1
        prompt_queue.server.queue_updated()
        return item_id


def _finish_external_history_task(prompt_queue, item_id, history_result, status) -> bool:
    if item_id is None:
        return False
    try:
        prompt_queue.task_done(
            item_id,
            history_result or {},
            status=status,
            process_item=lambda prompt: prompt[:5] + prompt[6:] if len(prompt) > 5 else prompt,
        )
        return True
    except Exception as exc:
        logger.warning(f"External history task_done failed: {redact_sensitive_text(exc)}")
        try:
            with prompt_queue.mutex:
                prompt_queue.currently_running.pop(item_id, None)
                prompt_queue.server.queue_updated()
        except Exception:
            pass
        return False


def _validation_error_status(prompt_id: str, detail) -> object:
    try:
        from execution import PromptQueue
    except Exception:
        return None
    now = int(time.time() * 1000)
    return PromptQueue.ExecutionStatus(
        status_str="error",
        completed=False,
        messages=[
            ("execution_start", {"prompt_id": prompt_id, "timestamp": now}),
            (
                "execution_error",
                {
                    "prompt_id": prompt_id,
                    "timestamp": now,
                    "exception_type": "validation_error",
                    "exception_message": format_error_message(detail),
                },
            ),
        ],
    )


def _send_run_node_finished(srv, prompt_id: str, success: bool, target_ids: list[str], error: str = "") -> None:
    try:
        srv.send_sync(
            "cometapi_run_node_finished",
            {
                "prompt_id": prompt_id,
                "ok": bool(success),
                "target_node_ids": list(target_ids or []),
                "error": redact_sensitive_text(error),
                "timestamp": int(time.time() * 1000),
            },
            None,
        )
    except Exception as exc:
        logger.warning(f"[{prompt_id}] send run_node finished event failed: {redact_sensitive_text(exc)}")


def _run_partial_blocking(prompt: dict, prompt_id: str, extra_data: dict, target_ids: list[str], client_id: str | None) -> None:
    try:
        import execution
        import server
        from execution import PromptExecutor, PromptQueue
    except Exception as exc:
        logger.error(f"Cannot import execution/server: {redact_sensitive_text(exc)}")
        return

    srv = server.PromptServer.instance
    if srv is None:
        logger.error("PromptServer.instance is None; cannot run partial")
        return

    if client_id:
        extra_data.setdefault("client_id", client_id)

    try:
        cache_type = execution.CacheType.NONE
    except Exception:
        cache_type = False

    executor = PromptExecutor(srv, cache_type=cache_type, cache_args={"lru": 0, "ram": 0})
    external_item_id = None
    try:
        external_item = _make_external_queue_item(prompt_id, prompt, extra_data, target_ids, client_id)
        external_item_id = _start_external_history_task(srv.prompt_queue, external_item)
    except Exception as exc:
        logger.warning(f"[{prompt_id}] external queue mirror failed: {redact_sensitive_text(exc)}")

    try:
        validate_coro = execution.validate_prompt(prompt_id, prompt, target_ids)
        future = asyncio_run_coroutine_threadsafe(validate_coro, srv.loop)
        valid = future.result(timeout=120)
    except Exception as exc:
        logger.error(f"[{prompt_id}] validate_prompt failed: {redact_sensitive_text(exc)}")
        print_sanitized_exception(exc)
        status = _validation_error_status(prompt_id, exc)
        if not _finish_external_history_task(srv.prompt_queue, external_item_id, {}, status):
            _record_history(srv.prompt_queue, prompt_id, prompt, extra_data, target_ids, executor, status)
        _send_run_node_finished(srv, prompt_id, False, target_ids, format_error_message(exc))
        return

    if not valid[0]:
        status = _validation_error_status(prompt_id, valid[1])
        if not _finish_external_history_task(srv.prompt_queue, external_item_id, {}, status):
            _record_history(srv.prompt_queue, prompt_id, prompt, extra_data, target_ids, executor, status)
        _send_run_node_finished(srv, prompt_id, False, target_ids, format_error_message(valid[1]))
        return

    with _RUNNING_LOCK:
        _RUNNING_PROMPTS[prompt_id] = {
            "started": time.time(),
            "target_ids": list(target_ids),
            "client_id": client_id,
        }

    exec_error = None
    try:
        executor.execute(prompt, prompt_id, extra_data, valid[2])
    except Exception as exc:
        exec_error = exc
        logger.error(f"[{prompt_id}] execute error: {redact_sensitive_text(exc)}")
        print_sanitized_exception(exc)
    finally:
        with _RUNNING_LOCK:
            _RUNNING_PROMPTS.pop(prompt_id, None)

    try:
        status = PromptQueue.ExecutionStatus(
            status_str="success" if exec_error is None and getattr(executor, "success", False) else "error",
            completed=exec_error is None and getattr(executor, "success", False),
            messages=list(getattr(executor, "status_messages", [])),
        )
        history_result = getattr(executor, "history_result", None)
        if not _finish_external_history_task(srv.prompt_queue, external_item_id, history_result, status):
            _record_history(srv.prompt_queue, prompt_id, prompt, extra_data, target_ids, executor, status)
    except Exception as exc:
        logger.warning(f"[{prompt_id}] history write failed: {redact_sensitive_text(exc)}")
    finally:
        run_success = exec_error is None and getattr(executor, "success", False)
        _send_run_node_finished(
            srv,
            prompt_id,
            run_success,
            target_ids,
            "" if run_success else format_error_message(exec_error or "局部执行失败"),
        )


def asyncio_run_coroutine_threadsafe(coro, loop):
    import asyncio

    return asyncio.run_coroutine_threadsafe(coro, loop)


def _register_routes() -> bool:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True

    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as exc:
        logger.error(f"Cannot import server/aiohttp: {redact_sensitive_text(exc)}")
        return False

    inst = getattr(PromptServer, "instance", None)
    if inst is None:
        return False

    routes = inst.routes

    @routes.get(f"{ROUTE_PREFIX}/info")
    async def _info(request):
        return web.json_response(
            {
                "enabled": True,
                "parallel_classes": [
                    "CometAPIUnifiedImage",
                    "CometAPIBatchImage",
                    "CometAPIAsyncImage",
                    "CometAPIAsyncBatchImage",
                    "CometAPIAsyncImageReceiver",
                    "CometAPIUnifiedLLMNode",
                    "CometAPIUnifiedVideo",
                    "CometAPIAsyncVideo",
                    "CometAPIAsyncVideoReceiver",
                    "CometAPIUnifiedMusic",
                ],
                "asset_classes": ["CometAPIImageCard", "CometAPIVideoCard", "CometAPIAudioCard", "CometAPITextCard", "CometAPIBatchTextCard"],
                "output_node_classes": _get_output_class_names(),
                "running": sorted(_RUNNING_PROMPTS.keys()),
            }
        )

    @routes.get(f"{ROUTE_PREFIX}/settings")
    async def _settings_get(request):
        return web.json_response(
            {
                "settings": load_comet_settings(),
                "models": get_model_catalog(),
            }
        )

    @routes.get(f"{ROUTE_PREFIX}/announcement")
    async def _announcement_get(request):
        try:
            force = str(request.query.get("force") or "").lower() in {"1", "true", "yes", "on"}
            data = await asyncio.to_thread(get_comet_announcement, force)
            return web.json_response(data)
        except Exception as exc:
            return web.json_response({"ok": False, "error": format_error_message(exc), "announcement": _default_announcement()}, status=500)

    @routes.get(f"{ROUTE_PREFIX}/asset")
    async def _asset_view(request):
        try:
            ref = normalize_asset_ref(
                {
                    "filename": request.query.get("filename", ""),
                    "subfolder": request.query.get("subfolder", ""),
                    "type": request.query.get("type", "output"),
                    "absolute_path": request.query.get("absolute_path", "") or request.query.get("path", ""),
                }
            )
            if not ref:
                return web.json_response({"ok": False, "error": "缺少素材文件名"}, status=400)
            path = asset_abs_path(ref)
            ext = os.path.splitext(path)[1].lower()
            if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}:
                return web.json_response({"ok": False, "error": "不支持预览这个文件类型"}, status=403)
            if not os.path.exists(path):
                return web.json_response({"ok": False, "error": "素材文件不存在"}, status=404)
            return web.FileResponse(
                path,
                headers={
                    "Cache-Control": "no-store",
                    "Content-Type": mimetypes.guess_type(path)[0] or "application/octet-stream",
                },
            )
        except Exception as exc:
            return web.json_response({"ok": False, "error": format_error_message(exc)}, status=500)

    @routes.get(f"{ROUTE_PREFIX}/video_thumbnail")
    async def _video_thumbnail(request):
        try:
            ref = normalize_asset_ref(
                {
                    "filename": request.query.get("filename", ""),
                    "subfolder": request.query.get("subfolder", ""),
                    "type": request.query.get("type", "output"),
                    "absolute_path": request.query.get("absolute_path", "") or request.query.get("path", ""),
                }
            )
            if not ref:
                return web.json_response({"ok": False, "error": "missing filename"}, status=400)
            path = asset_abs_path(ref)
            if ref.get("type") != "absolute":
                folder_paths = get_folder_paths()
                allowed_roots = [
                    folder_paths.get_input_directory(),
                    folder_paths.get_output_directory(),
                    folder_paths.get_temp_directory(),
                ]
                real_path = os.path.realpath(path)
                if not any(os.path.commonpath([real_path, os.path.realpath(root)]) == os.path.realpath(root) for root in allowed_roots):
                    return web.json_response({"ok": False, "error": "invalid video path"}, status=403)
            if not os.path.exists(path):
                return web.json_response({"ok": False, "error": "video not found"}, status=404)
            image = video_thumbnail_image(path)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=82)
            return web.Response(
                body=buffer.getvalue(),
                content_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )
        except Exception as exc:
            return web.json_response({"ok": False, "error": format_error_message(exc)}, status=500)

    @routes.post(f"{ROUTE_PREFIX}/settings")
    async def _settings_post(request):
        try:
            data = await request.json()
            settings = save_comet_settings(data.get("settings") if isinstance(data, dict) else data)
            return web.json_response({"ok": True, "settings": settings, "models": get_model_catalog()})
        except Exception as exc:
            return web.json_response({"ok": False, "error": format_error_message(exc)}, status=400)

    @routes.post(f"{ROUTE_PREFIX}/run_node")
    async def _run_node(request):
        try:
            data = await request.json()
        except Exception as exc:
            return web.json_response({"error": f"invalid json: {exc}"}, status=400)

        prompt = data.get("prompt") or {}
        target_node_ids = [str(node_id) for node_id in (data.get("target_node_ids") or [])]
        client_id = data.get("client_id")
        extra_data = data.get("extra_data") or {}

        if not isinstance(prompt, dict) or not prompt:
            return web.json_response({"error": "missing 'prompt'"}, status=400)
        if not target_node_ids:
            return web.json_response({"error": "missing 'target_node_ids'"}, status=400)

        output_ok, output_detail = _targets_are_output_nodes(prompt, target_node_ids)
        if not output_ok:
            return web.json_response(
                {
                    "error": "target node is not an OUTPUT_NODE",
                    "detail": output_detail,
                    "hint": "Prototype local run needs an OUTPUT_NODE target. Connect Nano Banana to Comet 图像卡片, Preview Image, or Save Image.",
                },
                status=400,
            )

        subset = _collect_ancestors(prompt, target_node_ids)
        safe, reason = _validate_parallel_safe(prompt, subset, target_node_ids)
        if not safe:
            return web.json_response(
                {
                    "error": "subgraph not allowed in prototype local run",
                    "detail": reason,
                    "hint": "This prototype only allows CometAPI nodes, Comet 图像卡片, and simple image I/O nodes in local parallel runs.",
                    "touched_nodes": sorted(subset),
                },
                status=400,
            )

        prompt_id = str(data.get("prompt_id") or uuid.uuid4())
        _get_pool().submit(_run_partial_blocking, prompt, prompt_id, extra_data, target_node_ids, client_id)
        return web.json_response(
            {
                "prompt_id": prompt_id,
                "status": "queued_parallel",
                "target_node_ids": target_node_ids,
                "touched_nodes": sorted(subset),
            }
        )

    _ROUTES_REGISTERED = True
    logger.info(f"Registered routes: {ROUTE_PREFIX}/info, {ROUTE_PREFIX}/settings, {ROUTE_PREFIX}/announcement, {ROUTE_PREFIX}/video_thumbnail, {ROUTE_PREFIX}/run_node")
    return True


def _register_routes_when_ready() -> None:
    if _register_routes():
        return

    def _wait_for_server():
        deadline = time.time() + 120.0
        while time.time() < deadline:
            if _register_routes():
                return
            time.sleep(0.05)
        logger.warning("Route registration incomplete after 120s.")

    threading.Thread(target=_wait_for_server, daemon=True, name="CometAPIVWireRouteRegister").start()


class CometAPISettings:
    CATEGORY = "COMET"
    FUNCTION = "execute"
    RETURN_TYPES = ()
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def execute(self):
        return ()


NODE_CLASS_MAPPINGS = {
    "CometAPIUnifiedImage": CometAPIUnifiedImage,
    "CometAPIBatchImage": CometAPIBatchImage,
    "CometAPIAsyncImage": CometAPIAsyncImage,
    "CometAPIAsyncBatchImage": CometAPIAsyncBatchImage,
    "CometAPIAsyncImageReceiver": CometAPIAsyncImageReceiver,
    "CometAPIUnifiedLLMNode": CometAPIUnifiedLLMNode,
    "CometAPIUnifiedVideo": CometAPIUnifiedVideo,
    "CometAPIAsyncVideo": CometAPIAsyncVideo,
    "CometAPIAsyncVideoReceiver": CometAPIAsyncVideoReceiver,
    "CometAPIUnifiedMusic": CometAPIUnifiedMusic,
    "CometAPISettings": CometAPISettings,
    "CometAPIImageCard": CometAPIImageCard,
    "CometAPIVideoCard": CometAPIVideoCard,
    "CometAPIAudioCard": CometAPIAudioCard,
    "CometAPITextCard": CometAPITextCard,
    "CometAPIBatchTextCard": CometAPIBatchTextCard,
    "CometAPITextAppend": CometAPITextAppend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CometAPIUnifiedImage": "Comet 图像",
    "CometAPIBatchImage": "Comet 批量图像",
    "CometAPIAsyncImage": "Comet 异步图像提交",
    "CometAPIAsyncBatchImage": "Comet 异步批量图像提交",
    "CometAPIAsyncImageReceiver": "Comet 异步图片收取",
    "CometAPIUnifiedLLMNode": "Comet 文本",
    "CometAPIUnifiedVideo": "Comet 视频",
    "CometAPIAsyncVideo": "Comet 异步视频提交",
    "CometAPIAsyncVideoReceiver": "Comet 异步视频收取",
    "CometAPIUnifiedMusic": "Comet 音乐",
    "CometAPISettings": "CometAPI 设置中心",
    "CometAPIImageCard": "Comet 图像卡片",
    "CometAPIVideoCard": "Comet 视频卡片",
    "CometAPIAudioCard": "Comet 音频卡片",
    "CometAPITextCard": "Comet 文本卡片",
    "CometAPIBatchTextCard": "Comet 批量文本卡片",
    "CometAPITextAppend": "Comet 文本追加",
}


_register_routes_when_ready()
