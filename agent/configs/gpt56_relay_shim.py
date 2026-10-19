"""



    "content": null      -> 200
    "content": ""        -> 200







"""

from __future__ import annotations

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

_TARGET_MODELS = {
    "gpt-5.6-sol",
    "gpt-5.6-sol-fast",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "g56-sol-relay",
}


def _model_matches(model: str | None) -> bool:
    if not isinstance(model, str):
        return False
    return model in _TARGET_MODELS or model.split("/")[-1] in _TARGET_MODELS


def _ensure_assistant_content(request: dict) -> dict:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return request
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if message.get("content") is None:
            message["content"] = ""
    return request


_ORIGINAL_TRANSFORM_REQUEST = OpenAIGPTConfig.transform_request


def _patched_transform_request(
    self, model, messages, optional_params, litellm_params, headers
):
    request = _ORIGINAL_TRANSFORM_REQUEST(
        self, model, messages, optional_params, litellm_params, headers
    )
    if isinstance(request, dict) and _model_matches(request.get("model") or model):
        request = _ensure_assistant_content(request)
    return request


if getattr(OpenAIGPTConfig.transform_request, "_gpt56_relay_shim", False) is False:
    _patched_transform_request._gpt56_relay_shim = True
    OpenAIGPTConfig.transform_request = _patched_transform_request
    print("[gpt56_relay_shim] patched OpenAIGPTConfig.transform_request")


class Gpt56RelayShim(CustomLogger):
    """

    """


gpt56_relay_shim = Gpt56RelayShim()
