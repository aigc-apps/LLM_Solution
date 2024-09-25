from typing import Dict, List, Any
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.evaluation.pai_evaluator import PaiEvaluator
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG


class EvaluationModule(ConfigurableModule):
    """Class for managing indices.

    RagIndex to manage vector indices for RagApplication.
    When initializing, the index is empty or load from existing index.
    User can add nodes to index when needed.
    """

    @staticmethod
    def get_dependencies() -> List[str]:
        return [
            "DataLoaderModule",
            "LlmModule",
            "IndexModule",
            "PostprocessorModule",
            "QueryEngineModule",
        ]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        data_loader = new_params["DataLoaderModule"]
        llm = new_params["LlmModule"]
        index = new_params["IndexModule"]
        node_postprocessors = new_params["PostprocessorModule"]
        query_engine = new_params["QueryEngineModule"]

        evaluation_dataset = config.get("dataset_path", None)
        retrieval_metrics = config.get("retrieval", None)
        response_metrics = config.get("response", None)

        return PaiEvaluator(
            data_loader=data_loader,
            llm=llm,
            index=index,
            query_engine=query_engine,
            node_postprocessors=node_postprocessors,
            evaluation_dataset=evaluation_dataset,
            retrieval_metrics=retrieval_metrics,
            response_metrics=response_metrics,
        )
