import argparse
import yaml
from loguru import logger
from typing import List
from pai_rag.tools.data_process.ops.base_op import OPERATORS
from pai_rag.tools.data_process.ray_executor import RayExecutor


def extract_parameters(yaml_dict, cfg):
    print("yaml_dict", yaml_dict)
    extracted_params = {key: value for key, value in yaml_dict.items() if key != "op"}
    extracted_params["working_dir"] = cfg.working_dir
    extracted_params["dataset_path"] = cfg.dataset_path
    extracted_params["export_path"] = cfg.export_path
    return extracted_params


def update_op_process(args):
    op_keys = list(OPERATORS.modules.keys())
    logger.info(f"Loading all operation keys: {op_keys}")

    if args.process is None:
        args.process = []

    with open(args.config_file) as file:
        process_cfg = yaml.safe_load(file)
    for i, process_op in enumerate(process_cfg["process"]):
        if process_op["op"] in op_keys:
            args.process.append(process_op["op"])
            args.process[i] = {process_op["op"]: extract_parameters(process_op, args)}

    return args


def process_parser(args):
    parser_dict = {}
    parser_dict["dataset_path"] = args.dataset_path
    parser_dict["export_path"] = args.export_path
    parser_dict["working_dir"] = args.working_dir
    parser_dict["cpu_required"] = args.cpu_required
    parser_dict["mem_required"] = args.mem_required
    parser_dict["accelerator"] = args.accelerator
    parser_dict["enable_mandatory_ocr"] = args.enable_mandatory_ocr
    parser_dict["concat_csv_rows"] = args.concat_csv_rows
    parser_dict["enable_table_summary"] = args.enable_table_summary
    parser_dict["format_sheet_data_to_json"] = args.format_sheet_data_to_json
    parser_dict["sheet_column_filters"] = args.sheet_column_filters
    args.process.append("rag_parser")
    args.process[0] = {"rag_parser": parser_dict}
    return args


def process_splitter(args):
    splitter_dict = {}
    splitter_dict["dataset_path"] = args.dataset_path
    splitter_dict["export_path"] = args.export_path
    splitter_dict["working_dir"] = args.working_dir
    splitter_dict["cpu_required"] = args.cpu_required
    splitter_dict["mem_required"] = args.mem_required
    splitter_dict["type"] = args.type
    splitter_dict["chunk_size"] = args.chunk_size
    splitter_dict["chunk_overlap"] = args.chunk_overlap
    splitter_dict["enable_multimodal"] = args.enable_multimodal
    args.process.append("rag_splitter")
    args.process[0] = {"rag_splitter": splitter_dict}
    return args


def process_embedder(args):
    embedder_dict = {}
    embedder_dict["dataset_path"] = args.dataset_path
    embedder_dict["export_path"] = args.export_path
    embedder_dict["working_dir"] = args.working_dir
    embedder_dict["cpu_required"] = args.cpu_required
    embedder_dict["mem_required"] = args.mem_required
    embedder_dict["accelerator"] = args.accelerator
    embedder_dict["source"] = args.source
    embedder_dict["model"] = args.model
    embedder_dict["enable_sparse"] = args.enable_sparse
    embedder_dict["multimodal_source"] = args.multimodal_source
    embedder_dict["enable_multimodal"] = args.enable_multimodal
    args.process.append("rag_embedder")
    args.process[0] = {"rag_embedder": embedder_dict}
    return args


def init_configs():
    """
    initialize the argparse parser and parse configs from one of:
        1. POSIX-style commands line args;
        2. environment variables
        3. hard-coded defaults

    :return: a global cfg object used by the Executor or Analyzer
    """
    parser = argparse.ArgumentParser()
    # shared parameters
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Path to datasets with optional weights(0.0-1.0), 1.0 as "
        "default. Accepted format:<w1> dataset1-path <w2> dataset2-path "
        "<w3> dataset3-path ...",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="./outputs/hello_world.jsonl",
        help="Path to export and save the output processed dataset. The "
        "directory to store the processed dataset will be the work "
        "directory of this process.",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        default="/PAI-RAG",
        help="Path to working dir for ray cluster.",
    )
    parser.add_argument(
        "--cpu_required",
        type=int,
        default=1,
        help="Cpu required for each rag operator.",
    )
    parser.add_argument(
        "--mem_required",
        type=str,
        default="1GB",
        help="Memory required for each rag operator.",
    )
    # Only used for multi-operators mode
    parser.add_argument(
        "--config_file",
        help="Path to a dj basic configuration file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--process", default=[], help="list of operator processes to run"
    )

    # Add subparsers
    subparsers = parser.add_subparsers(
        dest="operator", help="Choose a rag operator to run"
    )

    # SubParser for rag_parser
    parser_parser = subparsers.add_parser("rag_parser", help="Run rag_parser")
    parser_parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Accelerator type for pai_rag_parser and pai_rag_embedder operator.",
    )
    parser_parser.add_argument(
        "--enable_mandatory_ocr",
        type=bool,
        default=False,
        help="Whether to enable mandatory OCR for pai_rag_parser operator.",
    )
    parser_parser.add_argument(
        "--concat_csv_rows",
        type=bool,
        default=False,
        help="Whether to concat csv rows for pai_rag_parser operator.",
    )
    parser_parser.add_argument(
        "--enable_table_summary",
        type=bool,
        default=False,
        help="Whether to enable table summary for pai_rag_parser operator.",
    )
    parser_parser.add_argument(
        "--format_sheet_data_to_json",
        type=bool,
        default=False,
        help="Whether to format sheet data to json for pai_rag_parser operator.",
    )
    parser_parser.add_argument(
        "--sheet_column_filters",
        type=List[str],
        default=[],
        help="Column filters for pai_rag_parser operator.",
    )

    # SubParser for rag_splitter
    parser_splitter = subparsers.add_parser("rag_splitter", help="Run rag_splitter")
    parser_splitter.add_argument(
        "--type",
        type=str,
        default="Token",
        help="Split type for pai_rag_splitter operator.",
    )
    parser_splitter.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Chunk size for pai_rag_splitter operator.",
    )
    parser_splitter.add_argument(
        "--chunk_overlap",
        type=int,
        default=20,
        help="Chunk overlap for pai_rag_splitter operator.",
    )
    parser_splitter.add_argument(
        "--enable_multimodal",
        type=bool,
        default=False,
        help="Whether to enable multimodal for pai_rag_splitter and pai_rag_embedder operator.",
    )

    # SubParser for rag_embedder
    parser_embedder = subparsers.add_parser("rag_embedder", help="Run rag_embedder")
    parser_embedder.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Accelerator type for pai_rag_parser and pai_rag_embedder operator.",
    )
    parser_embedder.add_argument(
        "--source",
        type=str,
        default="huggingface",
        help="Embedding model source for pai_rag_embedder operator.",
    )
    parser_embedder.add_argument(
        "--model",
        type=str,
        default="bge-large-zh-v1.5",
        help="Embedding model name for pai_rag_embedder operator.",
    )
    parser_embedder.add_argument(
        "--enable_sparse",
        type=bool,
        default=False,
        help="Whether to enable sparse for pai_rag_embedder operator.",
    )
    parser_embedder.add_argument(
        "--multimodal_source",
        type=str,
        default="cnclip",
        help="Multi-modal embedding model source for pai_rag_embedder operator.",
    )
    parser_embedder.add_argument(
        "--enable_multimodal",
        type=bool,
        default=False,
        help="Whether to enable multimodal for pai_rag_splitter and pai_rag_embedder operator.",
    )

    args = parser.parse_args()

    # Determine which process to run with shared and unique parameters
    if args.config_file is not None:
        args = update_op_process(args)
    elif args.operator == "rag_parser":
        args = process_parser(args)
    elif args.operator == "rag_splitter":
        args = process_splitter(args)
    elif args.operator == "rag_embedder":
        args = process_embedder(args)
    else:
        parser.print_help()

    return args


@logger.catch(reraise=True)
def main():
    cfg = init_configs()
    executor = RayExecutor(cfg)
    executor.run()


if __name__ == "__main__":
    main()
