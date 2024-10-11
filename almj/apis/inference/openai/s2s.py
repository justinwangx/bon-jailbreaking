import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from almj.data_models import LLMResponse, Prompt

from ..model import InferenceAPIModel

LOGGER = logging.getLogger(__name__)


class S2SRateLimiter:
    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.tokens = calls_per_minute
        self.last_refill_time = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_refill_time
            self.tokens = min(self.calls_per_minute, self.tokens + time_passed * (self.calls_per_minute / 60))
            self.last_refill_time = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / (self.calls_per_minute / 60)
                LOGGER.info(f"Waiting {wait_time} seconds")
                await asyncio.sleep(wait_time)
                self.tokens = 1

            self.tokens -= 1


class OpenAIS2SModel(InferenceAPIModel):
    def __init__(self, api_key: str):
        self.api_key = api_key
        # prnt warning that this is not implemented
        LOGGER.warning("OpenAI S2S method is not implemented to obfuscate private code.")

    def log_retry(self, retry_state):
        raise NotImplementedError("This method is not implemented to obfuscate private code.")

    def process_responses(self, audio_output: List[bytes], text_output: List[str], audio_out_dir: Path | str):
        raise NotImplementedError("This method is not implemented to obfuscate private code.")

    async def connect(self):
        raise NotImplementedError("This method is not implemented to obfuscate private code.")

    async def send_message(self, websocket, message):
        raise NotImplementedError("This method is not implemented to obfuscate private code.")

    async def receive_message(self, websocket):
        raise NotImplementedError("This method is not implemented to obfuscate private code.")

    async def run_query(self, audio_data: List[bytes], audio_out_dir: str | Path, params: Dict[str, Any]) -> List[Any]:
        raise NotImplementedError("This method is not implemented to obfuscate private code.")

    async def __call__(
        self,
        model_ids: str | tuple[str, ...],
        prompt: Prompt,
        audio_out_dir: str | Path,
        print_prompt_and_response: bool = False,
        max_attempts: int = 10,
        **kwargs,
    ) -> List[LLMResponse]:
        raise NotImplementedError("This method is not implemented to obfuscate private code.")
