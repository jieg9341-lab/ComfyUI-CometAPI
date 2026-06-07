from __future__ import annotations

import base64
import io
import json
import re
import sys
from functools import lru_cache

import requests
from PIL import Image


REQUEST_TIMEOUT = (20, 900)
SOURCEFUL_REQUEST_LIMIT_BYTES = 4_500_000


@lru_cache(maxsize=1)
def _public():
    for module in reversed(list(sys.modules.values())):
        if not module:
            continue
        source = str(getattr(module, "__file__", "") or "").replace("\\", "/")
        if not source.endswith("/nodes.py"):
            continue
        required = (
            "APIMartVideoAPI",
            "CometAPIError",
            "RunningHubImageAPI",
            "download_image",
            "format_error_message",
            "get_channel_api_key",
            "pil_to_data_url",
            "redact_sensitive_text",
            "safe_pil_to_rgb",
            "upload_image_grsai",
        )
        if all(hasattr(module, name) for name in required):
            return module
    raise RuntimeError("Unable to locate the public CometAPI module for private openrouter adapter.")


def _text(value, default=""):
    result = str(value if value is not None else "").strip()
    return result or default


def _chat_completions_url(api_url):
    base = _text(api_url, "https://openrouter.ai/api/v1").rstrip("/")
    lower = base.lower()
    if lower.endswith("/chat/completions"):
        return base
    if lower.endswith("/api/v1") or lower.endswith("/v1"):
        return f"{base}/chat/completions"
    if lower.endswith("/api"):
        return f"{base}/v1/chat/completions"
    if "openrouter.ai" in lower:
        return f"{base}/api/v1/chat/completions"
    return f"{base}/chat/completions"


def _field_choices(context, field):
    ui = context.get("ui") if isinstance(context.get("ui"), dict) else {}
    fields = ui.get("fields") if isinstance(ui.get("fields"), dict) else {}
    spec = fields.get(field) if isinstance(fields.get(field), dict) else ui.get(field)
    values = spec.get("values") if isinstance(spec, dict) else None
    if not isinstance(values, list):
        values = spec.get("choices") if isinstance(spec, dict) else None
    return [str(item).strip() for item in values or [] if str(item).strip()]


def _safe_choice(context, field, value, default):
    choices = _field_choices(context, field)
    raw = _text(value, default)
    if choices and raw not in choices:
        return default if default in choices else choices[0]
    return raw


def _runninghub_key(public):
    return (
        public.get_channel_api_key("", "runninghub", "", "image")
        or public.get_channel_api_key("", "runninghub", "", "video")
    )


def _upload_reference_url(public, image):
    safe_image = public.safe_pil_to_rgb(image)

    runninghub_key = _runninghub_key(public)
    if runninghub_key:
        try:
            return public.RunningHubImageAPI(runninghub_key)._upload_image(safe_image)
        except Exception as exc:
            print(f"[CometAPI] OpenRouter RunningHub upload skipped: {public.redact_sensitive_text(exc)}")

    grsai_key = public.get_channel_api_key("", "grsai", "", "image")
    if grsai_key:
        try:
            url = public.upload_image_grsai(grsai_key, safe_image)
            if url:
                return url
        except Exception as exc:
            print(f"[CometAPI] OpenRouter Grsai upload skipped: {public.redact_sensitive_text(exc)}")

    apimart_key = (
        public.get_channel_api_key("", "apimart", "", "image")
        or public.get_channel_api_key("", "apimart", "", "video")
    )
    if apimart_key:
        try:
            return public.APIMartVideoAPI(apimart_key)._upload_image(safe_image)
        except Exception as exc:
            print(f"[CometAPI] OpenRouter Apimart upload skipped: {public.redact_sensitive_text(exc)}")

    return public.pil_to_data_url(safe_image, "PNG")


def _model_max_images(context):
    settings = context.get("model_settings") if isinstance(context.get("model_settings"), dict) else {}
    runtime = settings.get("runtime") if isinstance(settings.get("runtime"), dict) else {}
    try:
        return max(0, int(runtime.get("max_images") or 0))
    except Exception:
        return 0


def _build_payload(context, public):
    model = _text(context.get("model"))
    prompt = _text(context.get("prompt"))
    if not prompt:
        raise public.CometAPIError("OpenRouter 生图提示词不能为空。")

    max_images = _model_max_images(context) or 4
    refs = list(context.get("pil_images") or [])[:max_images]
    content = [{"type": "text", "text": prompt}]
    for image in refs:
        content.append({"type": "image_url", "image_url": {"url": _upload_reference_url(public, image)}})

    image_config = {}
    aspect_ratio = _safe_choice(context, "aspect_ratio", context.get("aspect_ratio"), "auto")
    if aspect_ratio and aspect_ratio != "auto":
        image_config["aspect_ratio"] = aspect_ratio

    image_size = _safe_choice(context, "image_size", context.get("image_size"), "1K")
    if image_size:
        image_config["image_size"] = image_size

    background_mode = _safe_choice(context, "background_mode", context.get("background_mode"), "默认")
    if background_mode == "透明":
        image_config["background_mode"] = "transparent"

    reasoning_effort = _safe_choice(context, "reasoning_effort", context.get("reasoning_effort"), "medium")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image"],
        "reasoning": {"effort": reasoning_effort},
    }
    if image_config:
        payload["image_config"] = image_config

    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(encoded) > SOURCEFUL_REQUEST_LIMIT_BYTES:
        raise public.CometAPIError(
            "OpenRouter Sourceful 请求体超过 4.5MB。请在设置中心填 RunningHub 或 Grsai API Key，"
            "让参考图先上传成 URL 后再提交。"
        )
    return payload


def _error_detail(data):
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return _text(error.get("message") or error.get("detail") or error.get("code"))
        if error:
            return _text(error)
        for key in ("message", "detail"):
            if data.get(key):
                return _text(data.get(key))
    return _text(data)[:500]


def _post_openrouter(context, payload, public):
    api_key = _text(context.get("api_key"))
    if not api_key:
        raise public.CometAPIError("缺少 OpenRouter API Key，请先在设置中心填写。")
    response = requests.post(
        _chat_completions_url(context.get("api_url")),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    try:
        data = response.json()
    except Exception:
        data = {"message": response.text}
    if response.status_code >= 400:
        detail = _error_detail(data)
        raise public.CometAPIError(f"OpenRouter HTTP {response.status_code}: {detail or response.reason}")
    if not isinstance(data, dict):
        raise public.CometAPIError(f"OpenRouter 返回不是 JSON 对象：{str(data)[:300]}")
    return data


def _nested_values(data):
    if isinstance(data, dict):
        for value in data.values():
            yield value
            yield from _nested_values(value)
    elif isinstance(data, list):
        for value in data:
            yield value
            yield from _nested_values(value)


def _extract_image_refs(data):
    refs = []
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        for item in message.get("images") or []:
            if not isinstance(item, dict):
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and image_url.get("url"):
                refs.append(str(image_url["url"]))
            elif item.get("url"):
                refs.append(str(item["url"]))
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                image_url = item.get("image_url") or item.get("image")
                if isinstance(image_url, dict) and image_url.get("url"):
                    refs.append(str(image_url["url"]))
                elif isinstance(image_url, str):
                    refs.append(image_url)

    if not refs:
        for value in _nested_values(data):
            if isinstance(value, str) and value.startswith("data:image/"):
                refs.append(value)
    return list(dict.fromkeys(refs))


def _decode_data_url(value):
    match = re.match(r"^data:image/[^;]+;base64,(.+)$", value, flags=re.I | re.S)
    if not match:
        return None
    raw = base64.b64decode(match.group(1))
    return Image.open(io.BytesIO(raw))


def _image_from_ref(public, ref):
    value = _text(ref)
    if not value:
        return None
    if value.startswith("data:image/"):
        return _decode_data_url(value)
    if value.startswith("http://") or value.startswith("https://"):
        return public.download_image(value)
    return None


def _has_alpha(image):
    if not isinstance(image, Image.Image):
        return False
    if "A" in image.getbands():
        return True
    return image.mode == "P" and "transparency" in image.info


def _result_image(public, image, preserve_alpha):
    if preserve_alpha and _has_alpha(image):
        return image.convert("RGBA")
    return public.safe_pil_to_rgb(image)


def generate_image(context):
    public = _public()
    preserve_alpha = _text(context.get("background_mode")) == "透明"
    payload = _build_payload(context, public)
    data = _post_openrouter(context, payload, public)
    images = []
    errors = []
    for ref in _extract_image_refs(data):
        try:
            image = _image_from_ref(public, ref)
            if image:
                images.append(_result_image(public, image, preserve_alpha))
            else:
                errors.append(f"OpenRouter 图片解析失败：{ref[:120]}")
        except Exception as exc:
            errors.append(public.format_error_message(exc))
    if not images and not errors:
        errors.append(f"OpenRouter 没有返回图片：{str(data)[:500]}")
    return {"images": images, "errors": errors}
