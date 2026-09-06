"""
STEP: it the file which Verifies the guardrails data and sends that to the llm model 
and if it is not working then fallback's to other model too
It is used for Caching also
"""
"""Step 6b: route the LLM through the Portkey gateway.

Instead of calling Groq directly, the main LLM call goes through Portkey.
Portkey stores the real Groq credentials behind a "slug" (set up once in
the Portkey dashboard) - our code never sees the raw Groq key.

NOTE on fallback: we used to send a "config" (strategy: fallback + a
list of targets) via the x-portkey-config header, either as inline JSON
or as a saved config's "pc-..." slug. This Portkey workspace has
"block_inline_config" enabled, and there's no saved config to reference
either, so ANY x-portkey-config header - inline or slug - gets rejected
with `inline_config_blocked`. Routing straight to one provider via
x-portkey-provider sidesteps the config mechanism entirely (that header
isn't validated the same way), which is why this version doesn't send a
config at all. The tradeoff: no more automatic Portkey-side fallback to
a second slug if @hrpolicy fails - see docs/05_portkey_gateway.md.
"""

from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

## my main Model
# we are trying to achieve model routing
PRIMARY_TARGET = {
    "provider": "@hrpolicy",
    "override_params": {"model": config.LLM_MODEL_NAME},
}

FALLBACK_TARGET = {
    "provider": "@hrpolicybackup",
    "override_params": {"model": "openai/gpt-oss-20b"},
}

# gateway features
# loadbalancing
# model routing
# caching
# fallback
#
# for all this features we need config
# we can keep our config public or private, it's our wish
# public -- anyone can edit
# private -- only you can edit and those who you gave access to edit

### CONFIG -- not usable right now (block_inline_config is enabled on this
### workspace and there's no saved "pc-..." config to reference), kept here
### for when a saved config slug becomes available.
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "targets": [PRIMARY_TARGET, FALLBACK_TARGET],
}
JUDGE_PROVIDER = "@hrpolicyjudge"


def get_gateway_llm() -> ChatOpenAI:
    """Return a chat model routed through Portkey (no config/fallback - see module docstring)."""
    logger.info("Routing LLM calls through Portkey (provider=%s)", PRIMARY_TARGET["provider"])

    headers = createHeaders(
        api_key=config.PORTKEY_API_KEY,
        provider=PRIMARY_TARGET["provider"],
    )
    return ChatOpenAI(
        api_key=config.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=PRIMARY_TARGET["override_params"]["model"],
        default_headers=headers,
    )

   

    def get_judge_llm() -> ChatOpenAI:
     """Return a chat model routed through Portkey (no config/fallback - see module docstring)."""
     logger.info("Routing LLM calls through Portkey (provider=%s)", JUDGE_PROVIDER)

     headers = createHeaders(
        api_key=config.PORTKEY_API_KEY,
        provider=JUDGE_PROVIDER,
     )
     return ChatOpenAI(
        api_key=config.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=PRIMARY_TARGET["override_params"]["model"],
        default_headers=headers,
     )