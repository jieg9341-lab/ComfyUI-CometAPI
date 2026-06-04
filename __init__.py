from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    from . import cometapi_chat  # noqa: F401
except Exception as exc:
    print(f"[ComfyUI-CometAPI] Comet Chat integration failed to load: {exc}")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
