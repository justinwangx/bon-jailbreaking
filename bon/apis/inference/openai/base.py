import asyncio
import logging
import time
from itertools import cycle
from pathlib import Path
from traceback import format_exc
from typing import Optional

import openai
import tiktoken

from bon.data_models import LLMResponse

from ..model import InferenceAPIModel
from .utils import COMPLETION_MODELS, price_per_token

LOGGER = logging.getLogger(__name__)


class Resource:
    """
    A resource that is consumed over time and replenished at a constant rate.
    """

    def __init__(self, refresh_rate, total=0, throughput=0):
        self.refresh_rate = refresh_rate
        self.total = total
        self.throughput = throughput
        self.last_update_time = time.time()
        self.start_time = time.time()
        self.value = self.refresh_rate

    def _replenish(self):
        """
        Updates the value of the resource based on the time since the last update.
        """
        curr_time = time.time()
        self.value = min(
            self.refresh_rate,
            self.value + (curr_time - self.last_update_time) * self.refresh_rate / 60,
        )
        self.last_update_time = curr_time
        self.throughput = self.total / (curr_time - self.start_time) * 60

    def geq(self, amount: float) -> bool:
        self._replenish()
        return self.value >= amount

    def consume(self, amount: float):
        """
        Consumes the given amount of the resource.
        """
        assert self.geq(amount), f"Resource does not have enough capacity to consume {amount} units"
        self.value -= amount
        self.total += amount


class OpenAIModel(InferenceAPIModel):
    def __init__(
        self,
        frac_rate_limit: float,
        prompt_history_dir: Path = None,
        base_url: str | None = None,
    ):
        self.frac_rate_limit = frac_rate_limit
        self.prompt_history_dir = prompt_history_dir
        self.model_ids = set()
        self.is_local_model = base_url is not None
        
        # Initialize OpenAI client with optional base_url
        client_kwargs = {}
        if base_url is not None:
            client_kwargs['base_url'] = base_url
        self.aclient = openai.AsyncClient(**client_kwargs)
        
        # Only initialize rate limiting for non-local models
        if not self.is_local_model:
            self.token_capacity = dict()
            self.request_capacity = dict()
            self.lock_add = asyncio.Lock()
            self.lock_consume = asyncio.Lock()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def _assert_valid_id(model_id: str):
        raise NotImplementedError

    @staticmethod
    async def _get_dummy_response_header(model_id: str):
        raise NotImplementedError

    @staticmethod
    def _count_prompt_token_capacity(prompt, **kwargs) -> int:
        raise NotImplementedError

    async def _make_api_call(self, prompt, model_id, **params) -> list[LLMResponse]:
        raise NotImplementedError

    @staticmethod
    def _print_prompt_and_response(prompt, responses):
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    async def add_model_id(self, model_id: str):
        self._assert_valid_id(model_id)
        if model_id in self.model_ids:
            return

        self.model_ids.add(model_id)

        # make dummy request to get token and request capacity
        model_metadata = await self._get_dummy_response_header(model_id)
        token_capacity = int(model_metadata["x-ratelimit-limit-tokens"])
        request_capacity = int(model_metadata["x-ratelimit-limit-requests"])
        print(f"got capacities for model {model_id}: {token_capacity}, {request_capacity}")
        tokens_consumed = token_capacity - int(model_metadata["x-ratelimit-remaining-tokens"])
        requests_consumed = request_capacity - int(model_metadata["x-ratelimit-remaining-requests"])
        print(f"consumed capacities for model {model_id}: {tokens_consumed}, {requests_consumed}")
        token_cap = token_capacity * self.frac_rate_limit
        request_cap = request_capacity * self.frac_rate_limit
        if model_id in COMPLETION_MODELS:
            token_cap *= 10000  # openai does not track token limit so we can increase it

        print(f"setting cap for model {model_id}: {token_cap}, {request_cap}")
        token_capacity = Resource(token_cap)
        request_capacity = Resource(request_cap)
        token_capacity.consume(min(token_cap, tokens_consumed))
        request_capacity.consume(min(request_cap, requests_consumed))
        self.token_capacity[model_id] = token_capacity
        self.request_capacity[model_id] = request_capacity

    async def __call__(
        self,
        model_ids: tuple[str, ...],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        LOGGER.info(f"OpenAIModel.__call__ started with model_ids: {model_ids}")

        async def attempt_api_call():
            LOGGER.info("Entering attempt_api_call")
            try:
                # Skip rate limiting for local models
                if not self.is_local_model:
                    LOGGER.info("Attempting rate limiting")
                    for model_id in cycle(model_ids):
                        async with self.lock_consume:
                            request_capacity, token_capacity = (
                                self.request_capacity[model_id],
                                self.token_capacity[model_id],
                            )
                            if request_capacity.geq(1) and token_capacity.geq(token_count):
                                request_capacity.consume(1)
                                token_capacity.consume(token_count)
                                break
                            else:
                                await asyncio.sleep(0.01)
                                continue
                
                LOGGER.info("About to make API call")
                result = await self._make_api_call(prompt, model_ids[0], start, **kwargs)
                LOGGER.info("API call completed")
                return result
            except Exception as e:
                LOGGER.error(f"Error in attempt_api_call: {str(e)}")
                raise

        # For local models, skip rate limit initialization
        if not self.is_local_model:
            LOGGER.info("Initializing rate limits")
            model_ids = tuple(sorted(model_ids, key=lambda model_id: price_per_token(model_id)))
            async with self.lock_add:
                for model_id in model_ids:
                    LOGGER.info(f"Adding model_id: {model_id}")
                    await self.add_model_id(model_id)
        
        LOGGER.info("Counting tokens")
        token_count = self._count_prompt_token_capacity(prompt, **kwargs)
        
        if not self.is_local_model:
            assert (
                max(self.token_capacity[model_id].refresh_rate for model_id in model_ids) >= token_count
            ), "Prompt is too long for any model to handle."

        responses: Optional[list[LLMResponse]] = None
        for i in range(max_attempts):
            try:
                LOGGER.info(f"Attempt {i+1}/{max_attempts}")
                responses = await attempt_api_call()
                if responses is not None and not all(is_valid(response.completion) for response in responses):
                    raise RuntimeError(f"Invalid responses according to is_valid {responses}")
                break
            except Exception as e:
                LOGGER.warn(f"Encountered API error: {str(e)}. Retrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)

        if responses is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        if print_prompt_and_response:
            self._print_prompt_and_response(prompt, responses)

        LOGGER.info("OpenAIModel.__call__ completed")
        return responses
