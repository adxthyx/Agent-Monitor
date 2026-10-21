import httpx

from agentmon.adapters.base import Adapter, AdapterError, Conversation
from agentmon.adapters.sse import SSEAdapter
from agentmon.adapters.webhook import WebhookAdapter
from agentmon.models import AdapterSpec

ADAPTERS: dict[str, type[Adapter]] = {"json_post": Adapter, "openai_chat": Adapter, "sse": SSEAdapter, "webhook": WebhookAdapter}


def build_adapter(spec: AdapterSpec, client: httpx.AsyncClient, env: dict[str, str] | None = None) -> Adapter:
    return ADAPTERS[spec.kind](spec, client, env)


__all__ = ["Adapter", "AdapterError", "Conversation", "SSEAdapter", "WebhookAdapter", "build_adapter"]
