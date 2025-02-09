import yaml
import click
import time
import json
import hashlib
from pai_rag.utils.format_logging import format_logging
from pai_rag.evaluation.pipeline.run_evaluation_pipeline import (
    run_rag_evaluation_pipeline,
)
from pai_rag.evaluation.pipeline.run_multimodal_evaluation_pipeline import (
    run_multimodal_evaluation_pipeline,
)
from loguru import logger


def validate_json_file(ctx, param, value):
    """检查文件路径是否以 .json 结尾"""
    if value is not None and not value.endswith(".json"):
        raise click.BadParameter(
            "Output path must be a JSON file with a .json extension."
        )
    return value


def calculate_md5_from_json(data):
    """计算 JSON 内容的 MD5 值"""
    hasher = hashlib.md5()
    # 将 JSON 对象转换为字符串，并确保顺序一致
    json_str = json.dumps(data, sort_keys=True)
    hasher.update(json_str.encode("utf-8"))
    return hasher.hexdigest()


def run_experiment(exp_params):
    exp_name = exp_params["name"]
    dataset = exp_params.get("dataset", None)
    use_pai_eval = exp_params.get("use_pai_eval", False)
    logger.info(
        f"Running experiment with name={exp_name}, dataset={dataset}, exp_params={exp_params}"
    )
    try:
        run_rag_evaluation_pipeline(
            config_file=exp_params["rag_setting_file"],
            data_path=exp_params["eval_data_path"],
            exp_name=exp_name,
            eval_model_llm_config=exp_params["eval_model_llm"],
            dataset=dataset,
            use_pai_eval=use_pai_eval,
        )
        logger.info(f"Finished evaluation experiment with name={exp_name}")
    except Exception as e:
        logger.error(f"Error running experiment {exp_name}: {e}")

    return {"name": exp_params["name"], "parameters": exp_params}


def run_multimodal_experiment(exp_params):
    exp_name = exp_params["name"]
    logger.info(f"Running experiment with name={exp_name}, exp_params={exp_params}")
    try:
        run_multimodal_evaluation_pipeline(
            config_file=exp_params["rag_setting_file"],
            exp_name=exp_name,
            qca_dataset_path=exp_params.get("qca_dataset_path", None),
            data_path=exp_params.get("eval_data_path", None),
            eval_model_llm_config=exp_params["eval_model_llm"],
            tested_multimodal_llm_config=exp_params.get("tested_multimodal_llm", None),
        )
        logger.info(f"Finished experiment with name={exp_name}")
    except Exception as e:
        logger.error(f"Error running experiment {exp_name}: {e}")

    return {"name": exp_params["name"], "parameters": exp_params}


@click.command()
@click.option(
    "-i",
    "--input_exp_config",
    show_default=True,
    default="src/pai_rag/config/evaluation/config.yaml",
)
@click.option("-o", "--output_path", callback=validate_json_file, show_default=True)
def run(input_exp_config=None, output_path=None):
    format_logging()
    with open(input_exp_config) as file:
        configs = yaml.safe_load(file)

    if not output_path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_key = calculate_md5_from_json(configs)
        output_path = f"localdata/eval_exp_data/results_{file_key}_{timestamp}.json"
    results = []
    for exp in configs["experiment"]:
        if exp.get("tested_multimodal_llm", None):
            result = run_multimodal_experiment(exp)
        else:
            result = run_experiment(exp)
        results.append(result)

    with open(output_path, "w") as result_file:
        json.dump(results, result_file, ensure_ascii=False, indent=4)

    logger.info(f"Results saved to {output_path}")
