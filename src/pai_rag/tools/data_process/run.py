import argparse
from loguru import logger
from typing import List
from pai_rag.tools.data_process.ray_executor import RayExecutor


def process_parser(
    dataset_path,
    export_path,
    working_dir,
    cpu_required,
    mem_required,
    accelerator,
    enable_mandatory_ocr,
    concat_csv_rows,
    enable_table_summary,
    format_sheet_data_to_json,
    sheet_column_filters,
):
    parser_dict = {}
    parser_dict["dataset_path"] = dataset_path
    parser_dict["export_path"] = export_path
    parser_dict["working_dir"] = working_dir
    parser_dict["cpu_required"] = cpu_required
    parser_dict["mem_required"] = mem_required
    parser_dict["accelerator"] = accelerator
    parser_dict["enable_mandatory_ocr"] = enable_mandatory_ocr
    parser_dict["concat_csv_rows"] = concat_csv_rows
    parser_dict["enable_table_summary"] = enable_table_summary
    parser_dict["format_sheet_data_to_json"] = format_sheet_data_to_json
    parser_dict["sheet_column_filters"] = sheet_column_filters
    return parser_dict


def process_splitter(
    dataset_path,
    export_path,
    working_dir,
    cpu_required,
    mem_required,
    type,
    chunk_size,
    chunk_overlap,
    enable_multimodal,
):
    splitter_dict = {}
    splitter_dict["dataset_path"] = dataset_path
    splitter_dict["export_path"] = export_path
    splitter_dict["working_dir"] = working_dir
    splitter_dict["cpu_required"] = cpu_required
    splitter_dict["mem_required"] = mem_required
    splitter_dict["type"] = type
    splitter_dict["chunk_size"] = chunk_size
    splitter_dict["chunk_overlap"] = chunk_overlap
    splitter_dict["enable_multimodal"] = enable_multimodal
    return splitter_dict


def process_embedder(
    dataset_path,
    export_path,
    working_dir,
    cpu_required,
    mem_required,
    accelerator,
    source,
    model,
    enable_sparse,
    multimodal_source,
    enable_multimodal,
):
    embedder_dict = {}
    embedder_dict["dataset_path"] = dataset_path
    embedder_dict["export_path"] = export_path
    embedder_dict["working_dir"] = working_dir
    embedder_dict["cpu_required"] = cpu_required
    embedder_dict["mem_required"] = mem_required
    embedder_dict["accelerator"] = accelerator
    embedder_dict["source"] = source
    embedder_dict["model"] = model
    embedder_dict["enable_sparse"] = enable_sparse
    embedder_dict["multimodal_source"] = multimodal_source
    embedder_dict["enable_multimodal"] = enable_multimodal
    return embedder_dict


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
    if args.operator == "rag_parser":
        op_params = process_parser(
            args.dataset_path,
            args.export_path,
            args.working_dir,
            args.cpu_required,
            args.mem_required,
            args.accelerator,
            args.enable_mandatory_ocr,
            args.concat_csv_rows,
            args.enable_table_summary,
            args.format_sheet_data_to_json,
            args.sheet_column_filters,
        )
    elif args.operator == "rag_splitter":
        op_params = process_splitter(
            args.dataset_path,
            args.export_path,
            args.working_dir,
            args.cpu_required,
            args.mem_required,
            args.type,
            args.chunk_size,
            args.chunk_overlap,
            args.enable_multimodal,
        )
    elif args.operator == "rag_embedder":
        op_params = process_embedder(
            args.dataset_path,
            args.export_path,
            args.working_dir,
            args.cpu_required,
            args.mem_required,
            args.accelerator,
            args.source,
            args.model,
            args.enable_sparse,
            args.multimodal_source,
            args.enable_multimodal,
        )
    else:
        parser.print_help()

    return args, op_params


@logger.catch(reraise=True)
def main():
    cfg, op_params = init_configs()
    executor = RayExecutor(cfg, op_params)
    executor.run()


if __name__ == "__main__":
    main()
