from abc import ABC, abstractmethod
import json
from typing import List, Optional
from pydantic.v1 import BaseModel, Field
from loguru import logger

from llama_index.core.llms.llm import LLM
from llama_index.core import Settings
from llama_index.core import BasePromptTemplate
from llama_index.core.schema import QueryBundle

from pai_rag.integrations.data_analysis.text2sql.utils.prompts import (
    DEFAULT_KEYWORD_EXTRACTION_PROMPT,
)


class QueryProcessor(ABC):
    """自然语言查询处理接口"""

    @abstractmethod
    def process(self, nl_query: QueryBundle):
        pass

    @abstractmethod
    async def aprocess(self, nl_query: QueryBundle):
        pass


class KeywordExtractor(QueryProcessor):
    def __init__(
        self,
        llm: Optional[LLM] = None,
        keyword_extraction_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        self._llm = llm or Settings.llm
        self._keyword_extraction_prompt = (
            keyword_extraction_prompt or DEFAULT_KEYWORD_EXTRACTION_PROMPT
        )

    def process(self, nl_query: QueryBundle) -> List[str]:
        sllm = self._llm.as_structured_llm(output_cls=KeywordList)
        keyword_list = sllm.predict(
            prompt=self._keyword_extraction_prompt,
            query_str=nl_query.query_str,
            fewshot_examples="",
        )

        keywords = json.loads(keyword_list)["Keywords"]
        # later check if parser needed
        # keywords = parse(self, keywords)
        logger.info(
            f"keyword_list: {keywords} extracted. nl_query: {nl_query.query_str}."
        )

        return keywords

    async def aprocess(self, nl_query: QueryBundle) -> List[str]:
        sllm = self._llm.as_structured_llm(output_cls=KeywordList)
        keyword_list = await sllm.apredict(
            prompt=self._keyword_extraction_prompt,
            query_str=nl_query.query_str,
            fewshot_examples="",
        )
        keywords = json.loads(keyword_list)["Keywords"]
        # later check if parser needed
        # keywords = parse(self, keywords)
        logger.info(
            f"keyword_list: {keywords} extracted. nl_query: {nl_query.query_str}."
        )

        return keywords


class KeywordList(BaseModel):
    """Data model for KeywordList."""

    Keywords: List[str] = Field(description="从查询问题中提取的关键词、关键短语和命名实体列表")


# TODO
# class QueryTransformation(QueryProcessor):
#     """query改写，包括考虑上下文的改写等"""
