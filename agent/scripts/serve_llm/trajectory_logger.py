"""
Trajectory Logger - LiteLLM Custom Callback

{
    "session_id": "default",
    "request_body": {
        "model": "...",
        "max_tokens": ...
    },
    "response_body": {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "...",
                    "reasoning_content": "...",
                    "tool_calls": [...]
                }
            }
        ]
    }
}
"""

import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from litellm.integrations.custom_logger import CustomLogger


# Default output directory (next to this script)
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "trajectories"
_TRAJECTORY_OUTPUT_PATH_HEADER = "x-trajectory-output-path"
_STICKY_ROUTING_KEY_HEADER = "x-litellm-routing-key"
_STICKY_ROUTING_ALIASES_ENV = "LITELLM_STICKY_ROUTING_ALIASES"
_STICKY_ROUTING_FAILURE_THRESHOLD_ENV = "LITELLM_STICKY_ROUTING_FAILURE_THRESHOLD"
_STICKY_ROUTING_COOLDOWN_SECONDS_ENV = "LITELLM_STICKY_ROUTING_COOLDOWN_SECONDS"
_STICKY_ROUTING_DEBUG_ENV = "LITELLM_STICKY_ROUTING_DEBUG"
_TEMPERATURE_HEADER = "x-sagent-temperature"
_DEFAULT_TOP_P = 0.95
_DEFAULT_STICKY_FAILURE_THRESHOLD = 2
_DEFAULT_STICKY_COOLDOWN_SECONDS = 300.0
_TRIAL_RANDOM_SUFFIX_RE = re.compile(r"__[A-Za-z0-9]{6,}$")


class TrajectoryLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.excluded_models = []
        self._sticky_state_lock = threading.Lock()
        self._sticky_alias_failures: dict[str, int] = {}
        self._sticky_alias_cooldown_until: dict[str, float] = {}
        aliases = self._parse_sticky_routing_aliases()
        print(
            "[TrajectoryLogger] initialized"
            f" sticky_models={','.join(sorted(aliases)) or '<none>'}"
        )

    def _apply_sampling_defaults(self, payload):
        """..."""
        if not isinstance(payload, dict):
            return payload

        payload.setdefault("top_p", _DEFAULT_TOP_P)

        optional_params = payload.get("optional_params")
        if isinstance(optional_params, dict):
            optional_params.setdefault("top_p", _DEFAULT_TOP_P)

        return payload

    def _apply_temperature_override(self, payload):
        """Apply a per-request temperature override from a custom header."""
        if not isinstance(payload, dict):
            return payload

        temperature = self._parse_temperature(
            self._extract_request_header(payload, _TEMPERATURE_HEADER)
        )
        if temperature is None:
            return payload

        payload.setdefault("temperature", temperature)

        optional_params = payload.get("optional_params")
        if isinstance(optional_params, dict):
            optional_params.setdefault("temperature", temperature)

        return payload

    def _apply_sticky_routing(self, payload):
        """Route repeated calls for the same task to the same backend alias."""
        if not isinstance(payload, dict):
            return payload

        model = payload.get("model")
        if not isinstance(model, str) or not model:
            return payload

        aliases_by_model = self._parse_sticky_routing_aliases()
        _, model_group = self._split_provider_model(model)
        aliases = aliases_by_model.get(model) or aliases_by_model.get(model_group)
        if not aliases:
            self._debug_sticky_routing(payload, reason="no_aliases")
            return payload

        routing_key = self._extract_sticky_routing_key(payload)
        if not routing_key:
            self._debug_sticky_routing(payload, reason="no_routing_key")
            return payload

        alias, primary_alias, fallback_applied = self._select_sticky_alias(
            routing_key,
            aliases,
        )
        if alias == model:
            return payload

        payload["model"] = alias
        metadata = payload.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.setdefault("original_model", model)
            metadata["sticky_routing_key"] = routing_key
            metadata["sticky_routed_model"] = payload["model"]
            metadata["sticky_primary_model"] = primary_alias
            metadata["sticky_routing_fallback"] = fallback_applied
        self._debug_sticky_routing(payload, reason="routed")
        return payload

    def _debug_sticky_routing(self, payload, reason: str) -> None:
        if os.environ.get(_STICKY_ROUTING_DEBUG_ENV) != "1":
            return
        headers = payload.get("proxy_server_request", {}).get("headers", {})
        if not isinstance(headers, dict):
            headers = {}
        print(
            "[TrajectoryLogger] sticky_debug"
            f" reason={reason}"
            f" model={payload.get('model')}"
            f" routing_key={self._extract_sticky_routing_key(payload)}"
            f" header_keys={sorted(headers)}"
        )

    @staticmethod
    def _split_provider_model(model: str) -> tuple[str, str]:
        provider, separator, model_group = model.partition("/")
        if separator and provider and model_group:
            return f"{provider}/", model_group
        return "", model

    def _select_sticky_alias(
        self,
        routing_key: str,
        aliases: list[str],
    ) -> tuple[str, str, bool]:
        """Pick the sticky alias, skipping aliases currently in local cooldown."""
        digest = hashlib.sha256(routing_key.encode("utf-8")).digest()
        primary_index = int.from_bytes(digest[:8], "big") % len(aliases)
        primary_alias = aliases[primary_index]

        with self._sticky_state_lock:
            now = time.monotonic()
            self._expire_sticky_cooldowns(now)
            if not self._is_sticky_alias_on_cooldown(primary_alias, now):
                return primary_alias, primary_alias, False

            for offset in range(1, len(aliases)):
                alias = aliases[(primary_index + offset) % len(aliases)]
                if not self._is_sticky_alias_on_cooldown(alias, now):
                    return alias, primary_alias, True

        return primary_alias, primary_alias, False

    def _expire_sticky_cooldowns(self, now: float) -> None:
        expired_aliases = [
            alias
            for alias, cooldown_until in self._sticky_alias_cooldown_until.items()
            if cooldown_until <= now
        ]
        for alias in expired_aliases:
            self._sticky_alias_cooldown_until.pop(alias, None)
            self._sticky_alias_failures.pop(alias, None)

    def _is_sticky_alias_on_cooldown(
        self, alias: str, now: float | None = None
    ) -> bool:
        if now is None:
            now = time.monotonic()
        cooldown_until = self._sticky_alias_cooldown_until.get(alias)
        return cooldown_until is not None and cooldown_until > now

    @staticmethod
    def _parse_sticky_routing_aliases() -> dict[str, list[str]]:
        """Parse MODEL=ALIAS_A,ALIAS_B;OTHER=... from the environment."""
        raw_value = os.environ.get(_STICKY_ROUTING_ALIASES_ENV, "").strip()
        if not raw_value:
            return {}

        aliases_by_model: dict[str, list[str]] = {}
        for group in raw_value.split(";"):
            model, separator, aliases = group.partition("=")
            model = model.strip()
            if not separator or not model:
                continue

            parsed_aliases = [
                alias.strip() for alias in aliases.split(",") if alias.strip()
            ]
            if parsed_aliases:
                aliases_by_model[model] = parsed_aliases
        return aliases_by_model

    @staticmethod
    def _parse_positive_int_env(name: str, default: int) -> int:
        raw_value = os.environ.get(name)
        if raw_value in (None, ""):
            return default
        try:
            value = int(raw_value)
        except ValueError:
            print(f"[TrajectoryLogger] Ignoring invalid {name}: {raw_value}")
            return default
        if value < 1:
            print(f"[TrajectoryLogger] Ignoring invalid {name}: {raw_value}")
            return default
        return value

    @staticmethod
    def _parse_nonnegative_float_env(name: str, default: float) -> float:
        raw_value = os.environ.get(name)
        if raw_value in (None, ""):
            return default
        try:
            value = float(raw_value)
        except ValueError:
            print(f"[TrajectoryLogger] Ignoring invalid {name}: {raw_value}")
            return default
        if value < 0:
            print(f"[TrajectoryLogger] Ignoring invalid {name}: {raw_value}")
            return default
        return value

    def _extract_sticky_routing_key(self, kwargs) -> str | None:
        explicit_key = self._extract_request_header(kwargs, _STICKY_ROUTING_KEY_HEADER)
        if explicit_key:
            return self._normalize_routing_key(explicit_key)

        metadata = kwargs.get("metadata", {})
        if isinstance(metadata, dict):
            sticky_routing_key = metadata.get("sticky_routing_key")
            if sticky_routing_key:
                return self._normalize_routing_key(str(sticky_routing_key))

        instance_id = self._extract_instance_id(kwargs)
        if instance_id:
            return self._normalize_routing_key(str(instance_id))

        trajectory_output_path = self._extract_trajectory_output_path(kwargs)
        if trajectory_output_path:
            return self._normalize_routing_key(
                self._routing_key_from_trajectory_path(trajectory_output_path)
            )

        return None

    @staticmethod
    def _routing_key_from_trajectory_path(trajectory_output_path: str) -> str:
        path = Path(trajectory_output_path)
        if path.name == "litellm-trajectory.jsonl" and path.parent.name == "agent":
            trial_dir = path.parent.parent.name
            if trial_dir:
                return trial_dir
        return path.stem or str(path)

    @staticmethod
    def _normalize_routing_key(value: str) -> str:
        return _TRIAL_RANDOM_SUFFIX_RE.sub("", value.strip())

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        data = self._apply_sampling_defaults(data)
        data = self._apply_temperature_override(data)
        data = self._apply_sticky_routing(data)

        messages = data.get("messages")
        if not messages:
            return data

        for msg in messages:
            content = msg.get("content")

            if content == "":
                msg["content"] = None
                continue

            if isinstance(content, list):
                filtered = [
                    block
                    for block in content
                    if not (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and not block.get("text")  # "" or None
                    )
                ]
                msg["content"] = filtered if filtered else None

        return data

    def _get_output_dir(self):
        """
        """
        return Path(os.environ.get("TRAJECTORY_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR))

    @staticmethod
    def _get_header_value(
        headers: dict[str, str] | None,
        header_name: str,
    ) -> str | None:
        if not isinstance(headers, dict):
            return None

        header_name = header_name.lower()
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == header_name and value:
                return str(value)

        return None

    @staticmethod
    def _parse_temperature(value: str | float | int | None) -> float | None:
        if value in (None, ""):
            return None
        if isinstance(value, bool):
            return None
        try:
            temperature = float(value)
        except (TypeError, ValueError):
            print(f"[TrajectoryLogger] Ignoring invalid {_TEMPERATURE_HEADER}: {value}")
            return None
        if not 0.0 <= temperature <= 2.0:
            print(
                f"[TrajectoryLogger] Ignoring out-of-range {_TEMPERATURE_HEADER}: {value}"
            )
            return None
        return temperature

    def _extract_request_header(self, kwargs, header_name: str) -> str | None:
        if not isinstance(kwargs, dict):
            return None

        header_sources = [
            kwargs.get("proxy_server_request", {}).get("headers"),
            kwargs.get("litellm_params", {})
            .get("proxy_server_request", {})
            .get("headers"),
            kwargs.get("metadata", {}).get("headers"),
            kwargs.get("metadata", {}).get("requester_metadata", {}).get("headers"),
        ]

        for headers in header_sources:
            value = self._get_header_value(headers, header_name)
            if value:
                return value

        return None

    def _extract_trajectory_output_path(self, kwargs) -> str | None:
        trajectory_output_path = self._extract_request_header(
            kwargs,
            _TRAJECTORY_OUTPUT_PATH_HEADER,
        )
        if trajectory_output_path:
            return trajectory_output_path

        metadata = kwargs.get("metadata", {})
        if isinstance(metadata, dict):
            trajectory_output_path = metadata.get("trajectory_output_path")
            if trajectory_output_path:
                return str(trajectory_output_path)

        return None

    def _extract_instance_id(self, kwargs):
        """

        """
        # 1. Try proxy request headers (set by agent tools like OpenCode)
        instance_id = self._extract_request_header(kwargs, "x-instance-id")
        if instance_id:
            return instance_id

        # 2. Try LiteLLM metadata field
        metadata = kwargs.get("metadata", {})
        if isinstance(metadata, dict):
            instance_id = metadata.get("instance_id")
            if instance_id:
                return instance_id

        return None

    def _extract_sticky_alias_from_kwargs(
        self,
        kwargs,
        failure_info: dict | None = None,
    ) -> str | None:
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            sticky_alias = metadata.get("sticky_routed_model")
            if isinstance(sticky_alias, str) and sticky_alias:
                return sticky_alias

        model = kwargs.get("model")
        if not isinstance(model, str) or not model:
            return None

        aliases_by_model = self._parse_sticky_routing_aliases()
        known_aliases = {
            alias for aliases in aliases_by_model.values() for alias in aliases
        }
        if model in known_aliases:
            return model

        api_base = None
        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, dict):
            api_base = litellm_params.get("api_base")

        if not api_base and isinstance(failure_info, dict):
            deployment = failure_info.get("deployment")
            if isinstance(deployment, dict):
                api_base = deployment.get("api_base")

        if isinstance(api_base, str) and api_base:
            return self._match_sticky_alias_from_api_base(api_base, known_aliases)

        return None

    @staticmethod
    def _match_sticky_alias_from_api_base(
        api_base: str,
        aliases: set[str],
    ) -> str | None:
        api_base_lower = api_base.lower()
        for alias in sorted(aliases, key=len, reverse=True):
            _, separator, suffix = alias.rpartition("@")
            if separator and suffix and suffix.lower() in api_base_lower:
                return alias
        return None

    def _mark_sticky_alias_success(self, kwargs) -> None:
        alias = self._extract_sticky_alias_from_kwargs(kwargs)
        if not alias:
            return

        with self._sticky_state_lock:
            self._sticky_alias_failures.pop(alias, None)
            self._sticky_alias_cooldown_until.pop(alias, None)

    def _mark_sticky_alias_failure(self, kwargs, failure_info: dict) -> None:
        alias = self._extract_sticky_alias_from_kwargs(kwargs, failure_info)
        if not alias or not self._should_cooldown_sticky_alias(failure_info):
            return

        threshold = self._parse_positive_int_env(
            _STICKY_ROUTING_FAILURE_THRESHOLD_ENV,
            _DEFAULT_STICKY_FAILURE_THRESHOLD,
        )
        cooldown_seconds = self._parse_nonnegative_float_env(
            _STICKY_ROUTING_COOLDOWN_SECONDS_ENV,
            _DEFAULT_STICKY_COOLDOWN_SECONDS,
        )

        with self._sticky_state_lock:
            failure_count = self._sticky_alias_failures.get(alias, 0) + 1
            self._sticky_alias_failures[alias] = failure_count
            if cooldown_seconds > 0 and failure_count >= threshold:
                self._sticky_alias_cooldown_until[alias] = (
                    time.monotonic() + cooldown_seconds
                )
                self._sticky_alias_failures[alias] = 0
                print(
                    "[TrajectoryLogger] Sticky routing cooldown: "
                    f"{alias} for {cooldown_seconds:g}s after {failure_count} failures"
                )

    @staticmethod
    def _should_cooldown_sticky_alias(failure_info: dict) -> bool:
        status_code = failure_info.get("status_code")
        if status_code is not None:
            try:
                return int(status_code) >= 500 or int(status_code) in {408, 429}
            except (TypeError, ValueError):
                pass

        message = str(failure_info.get("message") or "").lower()
        error_type = str(failure_info.get("error_type") or "").lower()
        transient_markers = (
            "connection error",
            "timeout",
            "bad gateway",
            "cloudflare",
            "apierror",
            "internalservererror",
            "service unavailable",
        )
        return any(
            marker in message or marker in error_type for marker in transient_markers
        )

    def _should_log(self, model):
        """..."""
        if not model:
            return True
        model_lower = model.lower()
        return not any(excluded in model_lower for excluded in self.excluded_models)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """..."""
        try:
            model = kwargs.get("model", "")
            if not self._should_log(model):
                return
            record = self._build_record(kwargs, response_obj, start_time, end_time)
            self._mark_sticky_alias_success(kwargs)
            self._write_record(record)
        except Exception as e:
            print(f"[TrajectoryLogger] logging failed: {e}")
            import traceback

            traceback.print_exc()

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """..."""
        try:
            record = self._build_record(
                kwargs, response_obj, start_time, end_time, success=False
            )
            self._mark_sticky_alias_failure(kwargs, record.get("failure", {}))
            self._write_record(record)
        except Exception as e:
            print(f"[TrajectoryLogger] logging failed: {e}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """..."""
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """..."""
        self.log_failure_event(kwargs, response_obj, start_time, end_time)

    def _get_field(self, value, field_name, default=None):
        """..."""
        if isinstance(value, dict):
            return value.get(field_name, default)
        return getattr(value, field_name, default)

    def _to_jsonable(self, value):
        """..."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return {
                key: self._to_jsonable(sub_value)
                for key, sub_value in value.items()
                if not callable(sub_value)
            }

        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]

        if hasattr(value, "model_dump"):
            try:
                return self._to_jsonable(value.model_dump(exclude_none=True))
            except TypeError:
                try:
                    return self._to_jsonable(value.model_dump())
                except Exception:
                    pass
            except Exception:
                pass

        if hasattr(value, "dict"):
            try:
                return self._to_jsonable(value.dict())
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            return {
                key: self._to_jsonable(sub_value)
                for key, sub_value in vars(value).items()
                if not key.startswith("_") and not callable(sub_value)
            }

        return str(value)

    def _redact_sensitive(self, value):
        """Redact obvious secrets from failure diagnostics before writing logs."""
        if isinstance(value, dict):
            redacted = {}
            for key, sub_value in value.items():
                if self._is_sensitive_key(key):
                    redacted[key] = "[REDACTED]"
                else:
                    redacted[key] = self._redact_sensitive(sub_value)
            return redacted

        if isinstance(value, list):
            return [self._redact_sensitive(item) for item in value]

        return value

    @staticmethod
    def _is_sensitive_key(key) -> bool:
        key_str = str(key).lower().replace("-", "_")
        if "api_key" in key_str or "apikey" in key_str:
            return True
        if "authorization" in key_str or "secret" in key_str:
            return True
        return key_str == "token" or key_str.endswith("_token")

    def _bounded_jsonable(self, value, *, max_chars=4000):
        """Convert diagnostics to JSON-safe data without letting records explode."""
        jsonable = self._redact_sensitive(self._to_jsonable(value))
        try:
            encoded = json.dumps(jsonable, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            encoded = str(jsonable)

        if len(encoded) <= max_chars:
            return jsonable
        return {
            "truncated": True,
            "preview": encoded[:max_chars],
            "original_length": len(encoded),
        }

    def _extract_status_code(self, value):
        for field in ("status_code", "status", "http_status", "code"):
            status_code = self._get_field(value, field)
            if status_code is not None:
                return status_code
        return None

    def _extract_message(self, value):
        for field in ("message", "error", "detail", "body", "response"):
            message = self._get_field(value, field)
            if message:
                return self._bounded_jsonable(message, max_chars=2000)
        if value is not None:
            return str(value)[:2000]
        return None

    def _build_failure_info(self, kwargs, response_obj):
        """Capture LiteLLM failure context for debugging failed attempts."""
        candidates = [
            kwargs.get("exception"),
            kwargs.get("error"),
            kwargs.get("original_exception"),
            kwargs.get("original_response"),
            response_obj,
        ]
        primary_error = next((candidate for candidate in candidates if candidate), None)

        failure_info = {
            "error_type": type(primary_error).__name__
            if primary_error is not None
            else None,
            "status_code": self._extract_status_code(primary_error),
            "message": self._extract_message(primary_error),
            "litellm_call_id": kwargs.get("litellm_call_id"),
            "log_event_type": kwargs.get("log_event_type"),
        }

        response_headers = kwargs.get("response_headers")
        if response_headers:
            failure_info["response_headers"] = self._bounded_jsonable(
                response_headers,
                max_chars=2000,
            )

        standard_logging_object = kwargs.get("standard_logging_object")
        if standard_logging_object:
            failure_info["standard_logging_object"] = self._bounded_jsonable(
                standard_logging_object,
                max_chars=4000,
            )

        hidden_params = self._get_field(response_obj, "_hidden_params")
        if hidden_params:
            failure_info["response_hidden_params"] = self._bounded_jsonable(
                hidden_params,
                max_chars=2000,
            )

        litellm_params = kwargs.get("litellm_params") or {}
        deployment = {
            "model": kwargs.get("model"),
            "api_base": litellm_params.get("api_base"),
            "custom_llm_provider": litellm_params.get("custom_llm_provider"),
        }
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            deployment["original_model"] = metadata.get("original_model")
            deployment["sticky_routed_model"] = metadata.get("sticky_routed_model")
            deployment["sticky_routing_key"] = metadata.get("sticky_routing_key")
        failure_info["deployment"] = self._redact_sensitive(deployment)

        if primary_error is not None:
            failure_info["raw_error"] = self._bounded_jsonable(
                primary_error, max_chars=4000
            )

        return {
            key: value
            for key, value in failure_info.items()
            if value not in (None, "", {}, [])
        }

    def _build_openai_messages(self, messages, system):
        """..."""
        normalized_messages = []
        has_explicit_system = any(
            self._get_field(message, "role") in {"system", "developer"}
            for message in (messages or [])
        )

        system_content = self._to_jsonable(system)
        if system_content not in (None, "", []) and not has_explicit_system:
            normalized_messages.append(
                {
                    "role": "system",
                    "content": system_content,
                }
            )

        for message in messages or []:
            message_dict = self._to_jsonable(message)
            if isinstance(message_dict, dict):
                normalized_messages.append(message_dict)

        return normalized_messages

    def _convert_tools_to_openai_format(self, tools):
        """..."""
        if not tools:
            return []

        openai_tools = []
        for tool in tools:
            tool_dict = self._to_jsonable(tool)
            if not isinstance(tool_dict, dict):
                continue

            if tool_dict.get("type") == "function" and isinstance(
                tool_dict.get("function"), dict
            ):
                openai_tools.append(tool_dict)
                continue

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_dict.get("name", ""),
                        "description": tool_dict.get("description", ""),
                        "parameters": tool_dict.get("input_schema")
                        or tool_dict.get("parameters")
                        or {},
                    },
                }
            )

        return openai_tools

    def _stringify_function_arguments(self, arguments):
        """..."""
        if arguments is None:
            return "{}"
        if isinstance(arguments, str):
            return arguments
        return json.dumps(self._to_jsonable(arguments), ensure_ascii=False, default=str)

    def _serialize_tool_calls(self, tool_calls):
        """..."""
        serialized_tool_calls = []

        for tool_call in tool_calls or []:
            function = self._get_field(tool_call, "function", {}) or {}
            tool_call_entry = {
                "type": self._get_field(tool_call, "type", "function") or "function",
                "function": {
                    "name": self._get_field(function, "name", ""),
                    "arguments": self._stringify_function_arguments(
                        self._get_field(function, "arguments")
                    ),
                },
            }

            tool_id = self._get_field(tool_call, "id")
            if tool_id is not None:
                tool_call_entry["id"] = tool_id

            serialized_tool_calls.append(tool_call_entry)

        return serialized_tool_calls

    def _extract_reasoning_content(self, message):
        """..."""
        reasoning_content = self._get_field(message, "reasoning_content")
        if reasoning_content is None:
            reasoning_content = self._get_field(message, "reasoning")
        if reasoning_content:
            return reasoning_content

        thinking_blocks = self._get_field(message, "thinking_blocks") or []
        reasoning_parts = []
        for block in thinking_blocks:
            block_dict = self._to_jsonable(block)
            if isinstance(block_dict, dict):
                thinking_text = block_dict.get("thinking") or block_dict.get("text")
            else:
                thinking_text = str(block_dict)
            if thinking_text:
                reasoning_parts.append(thinking_text)

        return "\n\n".join(reasoning_parts) if reasoning_parts else None

    def _build_openai_usage(self, usage):
        """..."""
        openai_usage = {}
        for key, value in usage.items():
            if key == "cost" or value is None:
                continue
            if isinstance(value, dict) and not value:
                continue
            openai_usage[key] = value
        return openai_usage

    def _build_openai_response_body(self, kwargs, response_obj, usage, success=True):
        """..."""
        raw_choices = self._get_field(response_obj, "choices") or []
        choices = []

        for index, choice in enumerate(raw_choices):
            message = self._get_field(choice, "message")
            if message is None:
                continue

            tool_calls = self._serialize_tool_calls(
                self._get_field(message, "tool_calls")
            )
            content = self._to_jsonable(self._get_field(message, "content"))
            if content == "" and tool_calls:
                content = None

            message_body = {
                "role": self._get_field(message, "role", "assistant") or "assistant",
                "content": content,
            }

            reasoning_content = self._extract_reasoning_content(message)
            if reasoning_content:
                message_body["reasoning_content"] = reasoning_content

            if tool_calls:
                message_body["tool_calls"] = tool_calls

            function_call = self._get_field(message, "function_call")
            if function_call is not None:
                message_body["function_call"] = {
                    "name": self._get_field(function_call, "name", ""),
                    "arguments": self._stringify_function_arguments(
                        self._get_field(function_call, "arguments")
                    ),
                }

            refusal = self._get_field(message, "refusal")
            if refusal is not None:
                message_body["refusal"] = self._to_jsonable(refusal)

            choice_body = {
                "index": self._get_field(choice, "index", index),
                "finish_reason": self._get_field(choice, "finish_reason"),
                "message": message_body,
            }

            logprobs = self._get_field(choice, "logprobs")
            if logprobs is not None:
                choice_body["logprobs"] = self._to_jsonable(logprobs)

            choices.append(choice_body)

        response_body = {
            "id": self._get_field(response_obj, "id"),
            "object": self._get_field(response_obj, "object", "chat.completion")
            or "chat.completion",
            "created": self._get_field(response_obj, "created") or int(time.time()),
            "model": self._get_field(response_obj, "model") or kwargs.get("model", ""),
            "choices": choices,
            "usage": self._build_openai_usage(usage),
        }

        for field in ("system_fingerprint", "service_tier"):
            field_value = self._get_field(response_obj, field)
            if field_value is not None:
                response_body[field] = field_value

        if not choices and response_obj is not None:
            fallback_key = "error" if not success else "raw_response"
            response_body[fallback_key] = self._to_jsonable(response_obj)

        if not success and not response_body.get("error"):
            response_body["error"] = {
                "type": "empty_or_unstructured_failure",
                "message": "LiteLLM marked this attempt as failed but did not expose a structured error.",
            }

        return {key: value for key, value in response_body.items() if value is not None}

    def _summarize_response_types(self, response_body):
        """..."""
        choices = response_body.get("choices") or []
        if not choices:
            return ["error"] if response_body.get("error") else []

        message = choices[0].get("message") or {}
        content_types = []
        if message.get("reasoning_content"):
            content_types.append("reasoning")
        if message.get("content") not in (None, "", []):
            content_types.append("text")
        if message.get("tool_calls"):
            content_types.append("tool_calls")
        if message.get("function_call"):
            content_types.append("function_call")
        return content_types

    def _build_usage(self, kwargs, response_obj):
        """..."""
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": {},
            "completion_tokens_details": {},
            "cost": None,
        }

        # --- token counts from response_obj.usage ---
        resp_usage = None
        if response_obj and hasattr(response_obj, "usage") and response_obj.usage:
            resp_usage = response_obj.usage

        if resp_usage:
            usage["prompt_tokens"] = getattr(resp_usage, "prompt_tokens", 0) or 0
            usage["completion_tokens"] = (
                getattr(resp_usage, "completion_tokens", 0) or 0
            )
            usage["total_tokens"] = getattr(resp_usage, "total_tokens", 0) or 0

            # detailed breakdowns (cache, thinking, etc.)
            prompt_details = getattr(resp_usage, "prompt_tokens_details", None)
            if prompt_details:
                usage["prompt_tokens_details"] = (
                    prompt_details
                    if isinstance(prompt_details, dict)
                    else {
                        k: v
                        for k, v in vars(prompt_details).items()
                        if not k.startswith("_") and v is not None
                    }
                )

            completion_details = getattr(resp_usage, "completion_tokens_details", None)
            if completion_details:
                usage["completion_tokens_details"] = (
                    completion_details
                    if isinstance(completion_details, dict)
                    else {
                        k: v
                        for k, v in vars(completion_details).items()
                        if not k.startswith("_") and v is not None
                    }
                )

            # cache-related fields surfaced by some providers
            for field in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                val = getattr(resp_usage, field, None)
                if val is not None:
                    usage[field] = val

        # --- cost ---
        # LiteLLM attaches cost in several places; try them all
        cost = kwargs.get("response_cost") or kwargs.get("cost")
        if cost is None and response_obj:
            cost = (
                getattr(response_obj, "_hidden_params", {}).get("response_cost")
                if hasattr(response_obj, "_hidden_params")
                else None
            )
        if cost is not None:
            try:
                usage["cost"] = float(cost)
            except (TypeError, ValueError):
                usage["cost"] = None

        return usage

    # Keys already captured explicitly in request_body or used internally
    _KNOWN_REQUEST_KEYS = frozenset(
        {
            "messages",
            "tools",
            "system",
            "model",
            "max_tokens",
            "temperature",
            "top_p",
        }
    )
    _INTERNAL_KWARGS_KEYS = frozenset(
        {
            "messages",
            "tools",
            "system",
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            # LiteLLM internal / already extracted elsewhere
            "optional_params",
            "litellm_params",
            "metadata",
            "acompletion",
            "complete_input_dict",
            "litellm_call_id",
            "litellm_logging_obj",
            "original_response",
            "response_cost",
            "cost",
            "standard_logging_object",
            "call_type",
            "start_time",
            "end_time",
            "api_key",
            "additional_args",
            "log_event_type",
            "cache_hit",
            "response_headers",
        }
    )

    def _collect_extra_params(self, kwargs):
        """Collect any request parameters not already captured explicitly.

        Merges optional_params and top-level kwargs, excluding known/internal
        keys, so that params like stop, seed, frequency_penalty, tool_choice,
        response_format, etc. are preserved automatically.
        """
        extra = {}

        # 1. From optional_params (provider-level overrides)
        optional_params = kwargs.get("optional_params") or {}
        for k, v in optional_params.items():
            if k not in self._KNOWN_REQUEST_KEYS and v is not None:
                extra[k] = v

        # 2. From top-level kwargs (caller-level overrides)
        for k, v in kwargs.items():
            if k in self._INTERNAL_KWARGS_KEYS or k in self._KNOWN_REQUEST_KEYS:
                continue
            if k.startswith("_"):
                continue
            if v is None:
                continue
            # Avoid duplicating what optional_params already provided
            if k not in extra:
                extra[k] = v

        # Drop values that are not JSON-serializable (e.g. callables, objects)
        sanitized = {}
        for k, v in extra.items():
            try:
                sanitized[k] = self._to_jsonable(v)
                json.dumps(sanitized[k], default=str)
            except (TypeError, ValueError, OverflowError):
                sanitized[k] = str(v)
        return sanitized

    def _build_record(self, kwargs, response_obj, start_time, end_time, success=True):
        """..."""
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools", [])

        optional_params = kwargs.get("optional_params", {})
        system = optional_params.get("system") or kwargs.get("system")
        # Fallback: LiteLLM proxy stores the original request body (including
        # the Claude-native "system" field) under litellm_params →
        # proxy_server_request → body.  When the above paths yield nothing,
        # grab it from there.
        if not system:
            proxy_body = (
                kwargs.get("litellm_params", {})
                .get("proxy_server_request", {})
                .get("body", {})
            )
            system = proxy_body.get("system")
        max_tokens = optional_params.get("max_tokens") or kwargs.get(
            "max_tokens", 131072
        )
        temperature = optional_params.get("temperature")
        if temperature is None:
            temperature = kwargs.get("temperature")
        if temperature is None:
            temperature = self._parse_temperature(
                self._extract_request_header(kwargs, _TEMPERATURE_HEADER)
            )
        top_p = optional_params.get("top_p")
        if top_p is None:
            top_p = kwargs.get("top_p", _DEFAULT_TOP_P)

        trajectory_output_path = self._extract_trajectory_output_path(kwargs)

        instance_id = self._extract_instance_id(kwargs)
        if not instance_id and trajectory_output_path:
            instance_id = Path(trajectory_output_path).stem
        if not instance_id:
            instance_id = "default"

        extra_params = self._collect_extra_params(kwargs)

        usage = self._build_usage(kwargs, response_obj)

        request_body = {
            "messages": self._build_openai_messages(messages, system),
            "tools": self._convert_tools_to_openai_format(tools),
            "model": kwargs.get("model", ""),
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if temperature is not None:
            request_body["temperature"] = temperature
        if extra_params:
            request_body.update(extra_params)

        response_body = self._build_openai_response_body(
            kwargs,
            response_obj,
            usage=usage,
            success=success,
        )

        request_time = int(time.time() * 1000)

        record = {
            "session_id": instance_id,
            "success": success,
            "request_time": request_time,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": int((end_time - start_time).total_seconds() * 1000),
            "request_body": request_body,
            "response_body": response_body,
            "usage": usage,
        }

        if not success:
            record["failure"] = self._build_failure_info(kwargs, response_obj)

        if trajectory_output_path:
            record["trajectory_output_path"] = trajectory_output_path

        return record

    def _write_record(self, record):
        """..."""
        session_id = record.get("session_id", "default")
        output_file = record.get("trajectory_output_path")
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = self._get_output_dir() / f"{session_id}.jsonl"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        model = record.get("request_body", {}).get("model", "unknown")
        content_types = self._summarize_response_types(record.get("response_body", {}))
        success = "✓" if record.get("success", True) else "✗"
        usage = record.get("usage", {})
        tokens_str = f"in={usage.get('prompt_tokens', 0)} out={usage.get('completion_tokens', 0)}"
        cost = usage.get("cost")
        cost_str = f" cost=${cost:.6f}" if cost is not None else ""
        print(
            f"[TrajectoryLogger] {success} model={model} | {tokens_str}{cost_str} | content_types={content_types}"
        )


trajectory_logger = TrajectoryLogger()
