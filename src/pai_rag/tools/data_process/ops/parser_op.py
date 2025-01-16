import os
import ray
import threading
from typing import List
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

    def __init__(
        self,
        concat_csv_rows: bool = False,
        enable_mandatory_ocr: bool = False,
        format_sheet_data_to_json: bool = False,
        sheet_column_filters: List[str] = None,
        oss_bucket: str = None,
        oss_endpoint: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        download_models_via_lock(self.model_dir, "PDF-Extract-Kit", self.accelerator)
        self.data_reader_config = BaseDataReaderConfig(
            concat_csv_rows=concat_csv_rows,
            enable_mandatory_ocr=enable_mandatory_ocr,
            format_sheet_data_to_json=format_sheet_data_to_json,
            sheet_column_filters=sheet_column_filters,
        )
        if oss_bucket is not None and oss_endpoint is not None:
            self.oss_store = resolve(
                cls=OssClient,
                bucket_name=oss_bucket,
                endpoint=oss_endpoint,
            )
        else:
            self.oss_store = None
        self.data_reader = resolve(
            cls=PaiDataReader,
            reader_config=self.data_reader_config,
            oss_store=self.oss_store,
        )
        logger.info(
            f"""ParserActor [PaiDataReader] init finished with following parameters:
                        concat_csv_rows: {concat_csv_rows}
                        enable_mandatory_ocr: {enable_mandatory_ocr}
                        format_sheet_data_to_json: {format_sheet_data_to_json}
                        sheet_column_filters: {sheet_column_filters}
                        oss_bucket: {oss_bucket}
                        oss_endpoint: {oss_endpoint}
            """
        )

    def process(self, input_file):
        current_thread = threading.current_thread()
        logger.info(f"当前线程的 ID: {current_thread.ident} 进程ID: {os.getpid()}")
        documents = self.data_reader.load_data(file_path_or_directory=input_file)
        if len(documents) == 0:
            logger.info(f"No data found in the input file: {input_file}")
            return None
        return convert_document_to_dict(documents[0])
