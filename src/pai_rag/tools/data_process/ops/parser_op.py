import ray
import threading
from loguru import logger
from pai_rag.core.rag_module import resolve
from pai_rag.utils.oss_client import OssClient
from pai_rag.tools.data_process.ops.base_op import BaseOP, OPERATORS
from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
from pai_rag.integrations.readers.pai.pai_data_reader import BaseDataReaderConfig
from pai_rag.tools.data_process.utils.formatters import convert_document_to_dict
from pai_rag.tools.data_process.utils.download_utils import download_models_via_lock

OP_NAME = "rag_parser"


@OPERATORS.register_module(OP_NAME)
@ray.remote
class Parser(BaseOP):
    """Mapper to generate samples whose captions are generated based on
    another model and the figure."""

    _accelerator = "cpu"
    _batched_op = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        download_models_via_lock(self.model_dir, "PDF-Extract-Kit", self.accelerator)
        logger.info("ParseActor init finished.")

    def process(self, input_file):
        current_thread = threading.current_thread()
        import os

        logger.info(f"当前线程的 ID: {current_thread.ident} 进程ID: {os.getpid()}")

        self.data_reader_config = BaseDataReaderConfig(
            concat_csv_rows=self.kwargs.get("concat_csv_rows", False),
            enable_mandatory_ocr=self.kwargs.get("enable_mandatory_ocr", False),
            enable_table_summary=self.kwargs.get("enable_table_summary", False),
            format_sheet_data_to_json=self.kwargs.get(
                "format_sheet_data_to_json", False
            ),
            sheet_column_filters=self.kwargs.get("sheet_column_filters", None),
        )
        if self.kwargs.get("oss_bucket", None) and self.kwargs.get(
            "oss_endpoint", None
        ):
            self.oss_store = resolve(
                cls=OssClient,
                bucket_name=self.kwargs["oss_bucket"],
                endpoint=self.kwargs["oss_endpoint"],
            )
        else:
            self.oss_store = None

        data_reader = resolve(
            cls=PaiDataReader,
            reader_config=self.data_reader_config,
            oss_store=self.oss_store,
        )

        documents = data_reader.load_data(file_path_or_directory=input_file)
        if len(documents) == 0:
            logger.info(f"No data found in the input file: {input_file}")
            return None
        return convert_document_to_dict(documents[0])
