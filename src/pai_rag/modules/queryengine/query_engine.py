"""query engine factory based on config"""

import logging
from typing import Dict, List, Any

# from llama_index.core.query_engine import RetrieverQueryEngine
from pai_rag.integrations.query_engine.multi_modal_query_engine import (
    MySimpleMultiModalQueryEngine,
)
from pai_rag.modules.queryengine.pai_retriever_query_engine import (
    PaiRetrieverQueryEngine,
)
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.utils.prompt_template import (
    DEFAULT_MULTI_MODAL_TEXT_QA_PROMPT_TMPL,
    DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL,
)


logger = logging.getLogger(__name__)


class QueryEngineModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return [
            "QueryTransformModule",
            "RetrieverModule",
            "SynthesizerModule",
            "PostprocessorModule",
            "LlmModule",
            "MultiModalLlmModule",
        ]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        query_transform = new_params["QueryTransformModule"]
        retriever = new_params["RetrieverModule"]
        synthesizer = new_params["SynthesizerModule"]
        postprocessor = new_params["PostprocessorModule"]
        llm = new_params["LlmModule"]
        multi_modal_llm = new_params["MultiModalLlmModule"]

        if config["type"] == "RetrieverQueryEngine":
            if (not postprocessor) or (
                "NLSQLRetriever" or "PandasQueryRetriever" in retriever.__repr__()
            ):
                logger.info("Query_engine without postprocess created")
                my_query_engine = PaiRetrieverQueryEngine(
                    query_transform=query_transform,
                    retriever=retriever,
                    response_synthesizer=synthesizer,
                )
            elif isinstance(postprocessor, List):
                my_query_engine = PaiRetrieverQueryEngine(
                    query_transform=query_transform,
                    retriever=retriever,
                    response_synthesizer=synthesizer,
                    node_postprocessors=postprocessor,
                )
            else:
                my_query_engine = PaiRetrieverQueryEngine(
                    query_transform=query_transform,
                    retriever=retriever,
                    response_synthesizer=synthesizer,
                    node_postprocessors=[postprocessor],
                )
            logger.info("Query_engine instance created")
            return my_query_engine

        elif config["type"] == "SimpleMultiModalQueryEngine":
            if not postprocessor:
                multi_modal_query_engine = MySimpleMultiModalQueryEngine(
                    retriever=retriever,
                    multi_modal_llm=multi_modal_llm,
                    llm=llm,
                    text_qa_template=DEFAULT_MULTI_MODAL_TEXT_QA_PROMPT_TMPL,
                    image_qa_template=DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL,
                )
            elif isinstance(postprocessor, List):
                multi_modal_query_engine = MySimpleMultiModalQueryEngine(
                    retriever=retriever,
                    multi_modal_llm=multi_modal_llm,
                    llm=llm,
                    text_qa_template=DEFAULT_MULTI_MODAL_TEXT_QA_PROMPT_TMPL,
                    image_qa_template=DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL,
                    node_postprocessors=postprocessor,
                )
            else:
                multi_modal_query_engine = MySimpleMultiModalQueryEngine(
                    retriever=retriever,
                    multi_modal_llm=multi_modal_llm,
                    llm=llm,
                    text_qa_template=DEFAULT_MULTI_MODAL_TEXT_QA_PROMPT_TMPL,
                    image_qa_template=DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL,
                    node_postprocessors=[postprocessor],
                )
            logger.info("SimpleMultiModalQueryEngine instance created")
            return multi_modal_query_engine

        else:
            raise ValueError(
                "Supports RetrieverQueryEngine & TransformQueryEngine only."
            )
