from typing import Any, List, Generator, Optional, Sequence, cast, AsyncGenerator
from loguru import logger

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.schema import NodeWithScore, QueryType, QueryBundle
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.llms import LLM
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    Response,
    StreamingResponse,
    AsyncStreamingResponse,
)
from llama_index.core.instrumentation.events.synthesis import (
    SynthesizeStartEvent,
    SynthesizeEndEvent,
)
from llama_index.core.callbacks.schema import CBEventType, EventPayload
import llama_index.core.instrumentation as instrument

from pai_rag.integrations.data_analysis.nl2sql.nl2sql_prompts import (
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT,
)
from pai_rag.integrations.synthesizer.pai_synthesizer import DEFAULT_EMPTY_RESPONSE_GEN


dispatcher = instrument.get_dispatcher(__name__)


def empty_response_generator() -> Generator[str, None, None]:
    yield DEFAULT_EMPTY_RESPONSE_GEN


async def empty_response_agenerator() -> AsyncGenerator[str, None]:
    yield DEFAULT_EMPTY_RESPONSE_GEN


class DataAnalysisSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
        response_synthesis_prompt: Optional[BasePromptTemplate] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            streaming=streaming,
        )

        self._response_synthesis_prompt = (
            response_synthesis_prompt or DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        )
        logger.info("DataAnalysisSynthesizer initialized")

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"response_synthesis_prompt": self._response_synthesis_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "response_synthesis_prompt" in prompts:
            self._response_synthesis_prompt = prompts["response_synthesis_prompt"]

    async def aget_response(
        self,
        query_str: str,
        db_description_str: str,
        retrieved_nodes: List[NodeWithScore],
        streaming: bool = False,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        query_df_output = [n.node.get_content() for n in retrieved_nodes]
        logger.info(f"db_description_str: {db_description_str}")
        partial_prompt_tmpl = self._response_synthesis_prompt.partial_format(
            query_str=query_str,
            db_schema=db_description_str,
            query_code_instruction=[
                n.node.metadata["query_code_instruction"] for n in retrieved_nodes
            ],
        )
        truncated_df_output = self._prompt_helper.truncate(
            prompt=partial_prompt_tmpl,
            text_chunks=["\n".join(query_df_output)],
        )
        logger.info(f"truncated_df_output: {str(truncated_df_output)}")

        response: RESPONSE_TEXT_TYPE
        if not streaming:
            response = await self._llm.apredict(
                self._response_synthesis_prompt,
                query_str=query_str,
                db_schema=db_description_str,
                query_code_instruction=[
                    n.node.metadata["query_code_instruction"] for n in retrieved_nodes
                ],  # sql or pandas query
                query_output=truncated_df_output,  # query output
                **response_kwargs,
            )
        else:
            response = await self._llm.astream(
                self._response_synthesis_prompt,
                query_str=query_str,
                db_schema=db_description_str,
                query_code_instruction=[
                    n.node.metadata["query_code_instruction"] for n in retrieved_nodes
                ],
                query_output=truncated_df_output,
                **response_kwargs,
            )

        if isinstance(response, str):
            response = response or DEFAULT_EMPTY_RESPONSE_GEN
        else:
            response = cast(Generator, response)

        return response

    def get_response(
        self,
        query_str: str,
        db_description_str: str,
        retrieved_nodes: List[NodeWithScore],
        streaming: bool = False,
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        query_df_output = [n.node.get_content() for n in retrieved_nodes]
        logger.info(f"db_description_str: {db_description_str}")

        partial_prompt_tmpl = self._response_synthesis_prompt.partial_format(
            query_str=query_str,
            db_schema=db_description_str,
            query_code_instruction=[
                n.node.metadata["query_code_instruction"] for n in retrieved_nodes
            ],
        )
        truncated_df_output = self._prompt_helper.truncate(
            prompt=partial_prompt_tmpl,
            text_chunks=["\n".join(query_df_output)],
        )
        logger.info(f"truncated_df_output: {truncated_df_output}")

        response: RESPONSE_TEXT_TYPE
        if not streaming:
            response = self._llm.predict(
                self._response_synthesis_prompt,
                query_str=query_str,
                db_schema=db_description_str,
                query_code_instruction=[
                    n.node.metadata["query_code_instruction"] for n in retrieved_nodes
                ],  # sql or pandas query
                query_output=truncated_df_output,  # query output
                **kwargs,
            )
        else:
            response = self._llm.stream(
                self._response_synthesis_prompt,
                query_str=query_str,
                db_schema=db_description_str,
                query_code_instruction=[
                    n.node.metadata["query_code_instruction"] for n in retrieved_nodes
                ],
                query_output=truncated_df_output,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or DEFAULT_EMPTY_RESPONSE_GEN
        else:
            response = cast(Generator, response)

        return response

    @dispatcher.span
    def synthesize(
        self,
        query: QueryType,
        description: str,
        nodes: List[NodeWithScore],
        streaming: bool = False,
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )

        if len(nodes) == 0:
            if streaming:
                empty_response = StreamingResponse(
                    response_gen=empty_response_generator()
                )
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response
            else:
                empty_response = Response(DEFAULT_EMPTY_RESPONSE_GEN)
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query.query_str},
        ) as event:
            response_str = self.get_response(
                query_str=query.query_str,
                db_description_str=description,
                retrieved_nodes=nodes,
                streaming=streaming,
                **response_kwargs,
            )

            additional_source_nodes = additional_source_nodes or []
            source_nodes = list(nodes) + list(additional_source_nodes)

            response = self._prepare_response_output(response_str, source_nodes)

            event.on_end(payload={EventPayload.RESPONSE: response})

        dispatcher.event(
            SynthesizeEndEvent(
                query=query,
                response=response,
            )
        )
        return response

    @dispatcher.span
    async def asynthesize(
        self,
        query: QueryType,
        description: str,
        nodes: List[NodeWithScore],
        streaming: bool = False,
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )
        if len(nodes) == 0:
            if streaming:
                empty_response = AsyncStreamingResponse(
                    response_gen=empty_response_agenerator()
                )
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response
            else:
                empty_response = Response(DEFAULT_EMPTY_RESPONSE_GEN)
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query.query_str},
        ) as event:
            response_str = await self.aget_response(
                query_str=query.query_str,
                db_description_str=description,
                retrieved_nodes=nodes,
                streaming=streaming,
                **response_kwargs,
            )

            additional_source_nodes = additional_source_nodes or []
            source_nodes = list(nodes) + list(additional_source_nodes)

            response = self._prepare_response_output(response_str, source_nodes)

            event.on_end(payload={EventPayload.RESPONSE: response})

        dispatcher.event(
            SynthesizeEndEvent(
                query=query,
                response=response,
            )
        )
        return response
