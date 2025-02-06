import asyncio
from pai_rag.evaluation.utils.create_components import (
    get_rag_components,
    get_rag_config_and_mode,
    get_multimodal_eval_components,
)
from pai_rag.evaluation.dataset.state_manager import DatasetState


def run_multimodal_evaluation_pipeline(
    config_file=None,
    oss_path=None,
    qca_dataset_path=None,
    data_path=None,
    pattern=None,
    exp_name="default",
    eval_model_llm_config=None,
    tested_multimodal_llm_config=None,
):
    config, mode, exist_flag = get_rag_config_and_mode(config_file, exp_name)
    assert mode == "image"
    data_loader, vector_index, query_engine = get_rag_components(config)
    multimodal_qca_generator, evaluator = get_multimodal_eval_components(
        config,
        exp_name,
        vector_index,
        query_engine,
        eval_model_llm_config,
        tested_multimodal_llm_config,
        qca_dataset_path,
    )
    if qca_dataset_path:
        multimodal_qca_generator.state_manager.mark_completed(DatasetState.QCA)
        _ = asyncio.run(
            multimodal_qca_generator.agenerate_predicted_multimodal_dataset_only_via_vlm()
        )
        asyncio.run(evaluator.aevaluation_for_response())
        return None

    assert (oss_path is not None) or (
        data_path is not None
    ), "Must provide either local path or oss path."
    assert (oss_path is None) or (
        data_path is None
    ), f"Can not provide both local path '{data_path}' and oss path '{oss_path}'."

    if not exist_flag:
        data_loader.load_data(
            file_path_or_directory=data_path,
            filter_pattern=pattern,
            oss_path=oss_path,
            from_oss=oss_path is not None,
            enable_raptor=False,
        )

    _ = asyncio.run(multimodal_qca_generator.agenerate_all_dataset())
    asyncio.run(evaluator.aevaluation_for_retrieval())
    asyncio.run(evaluator.aevaluation_for_response())
    return None
