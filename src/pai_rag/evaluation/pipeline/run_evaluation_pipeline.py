import asyncio
from pai_rag.evaluation.utils.create_components import (
    get_rag_components,
    get_rag_config_and_mode,
    get_eval_components,
)


def run_rag_evaluation_pipeline(
    config_file=None,
    oss_path=None,
    data_path=None,
    pattern=None,
    exp_name="default",
    eval_model_llm_config=None,
    dataset=None,
    use_pai_eval=False,
):
    assert (oss_path is not None) or (
        data_path is not None
    ), "Must provide either local path or oss path."
    assert (oss_path is None) or (
        data_path is None
    ), f"Can not provide both local path '{data_path}' and oss path '{oss_path}'."

    config, mode, exist_flag = get_rag_config_and_mode(config_file, exp_name)
    data_loader, vector_index, query_engine = get_rag_components(config, dataset)
    if not exist_flag:
        data_loader.load_data(
            file_path_or_directory=data_path,
            filter_pattern=pattern,
            oss_path=oss_path,
            from_oss=oss_path is not None,
            enable_raptor=False,
        )

    qca_generator, evaluator = get_eval_components(
        config,
        vector_index,
        query_engine,
        mode,
        eval_model_llm_config,
        use_pai_eval,
    )

    _ = asyncio.run(
        qca_generator.agenerate_all_dataset(dataset=dataset, dataset_path=data_path)
    )
    asyncio.run(evaluator.aevaluation_for_retrieval())
    asyncio.run(evaluator.aevaluation_for_response())
    asyncio.run(evaluator.aevaluation_for_all())

    # return {"retrieval": retrieval_result, "response": response_result}

    return None
