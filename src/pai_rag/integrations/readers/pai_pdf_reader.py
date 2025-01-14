"""Read PDF files."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from pai_rag.utils.markdown_utils import (
    transform_local_to_oss,
)
from operator import itemgetter
import tempfile
from PIL import Image
import os
from loguru import logger
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.dict2md.ocr_mkcontent import merge_para_with_text
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.libs.commons import join_path
from urllib.parse import urlparse
from collections import defaultdict


DEFAULT_HEADING_DIFF_THRESHOLD = 2


class PaiPDFReader(BaseReader):
    """Read PDF files including texts, tables, images.

    Args:
        enable_table_summary (bool):  whether to use table_summary to process tables
    """

    def __init__(
        self,
        enable_mandatory_ocr: bool = False,
        enable_table_summary: bool = False,
        oss_cache: Any = None,
    ) -> None:
        self.enable_table_summary = enable_table_summary
        self.enable_mandatory_ocr = enable_mandatory_ocr
        self._oss_cache = oss_cache
        logger.info(
            f"PaiPdfReader created with enable_table_summary : {self.enable_table_summary}"
        )
        logger.info(
            f"PaiPdfReader created with enable_mandatory_ocr : {self.enable_mandatory_ocr}"
        )

    def _transform_local_to_oss(self, pdf_name: str, local_url: str):
        image = Image.open(local_url)
        return transform_local_to_oss(self._oss_cache, image, pdf_name)

    def is_url(self, url: str) -> bool:
        """判断是否为 URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def create_markdown(
        self,
        pdf_name: str,
        pdf_info_dict: list,
        img_buket_path: str = "",
    ):
        output_content = []
        text_height_min = float("inf")
        text_height_max = 0
        title_list = []
        # 存储每个title及其index
        title_dict = defaultdict(list)
        # 记录index
        index_count = 0
        for page_info in pdf_info_dict:
            paras_of_layout = page_info.get("para_blocks")
            if not paras_of_layout:
                continue
            (
                page_markdown,
                text_height_min,
                text_height_max,
                index_count,
            ) = self.create_page_markdown(
                pdf_name,
                paras_of_layout,
                img_buket_path,
                title_list,
                title_dict,
                text_height_min,
                text_height_max,
                index_count,
            )
            output_content.extend(page_markdown)
        self.post_process_multi_level_headings(
            title_list, output_content, title_dict, text_height_min, text_height_max
        )
        markdown_result = "\n\n".join(output_content)
        return markdown_result

    def create_page_markdown(
        self,
        pdf_name,
        paras_of_layout,
        img_buket_path,
        title_list,
        title_dict,
        text_height_min,
        text_height_max,
        index_count,
    ):
        page_markdown = []
        for para_block in paras_of_layout:
            text_height_min, text_height_max = self.collect_title_info(
                para_block, title_list, text_height_min, text_height_max
            )
            para_text = ""
            para_type = para_block["type"]
            if para_type in [BlockType.Text, BlockType.List, BlockType.Index]:
                para_text = merge_para_with_text(para_block)
            elif para_type == BlockType.Title:
                para_text = f"# {merge_para_with_text(para_block)}"
                title_dict[para_text].append(index_count)
            elif para_type == BlockType.InterlineEquation:
                para_text = merge_para_with_text(para_block)
            elif para_type == BlockType.Image:
                for block in para_block["blocks"]:  # 1st.拼image_body
                    if block["type"] == BlockType.ImageBody:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["type"] == ContentType.Image:
                                    if span.get("image_path", "") and not self.is_url(
                                        span.get("image_path", "")
                                    ):
                                        image_path = join_path(
                                            img_buket_path, span["image_path"]
                                        )
                                        oss_url = self._transform_local_to_oss(
                                            pdf_name, image_path
                                        )
                                        para_text += f"\n![]({oss_url})  \n"
                for block in para_block["blocks"]:  # 2nd.拼image_caption
                    if block["type"] == BlockType.ImageCaption:
                        para_text += merge_para_with_text(block) + "  \n"
                for block in para_block["blocks"]:  # 3rd.拼image_footnote
                    if block["type"] == BlockType.ImageFootnote:
                        para_text += merge_para_with_text(block) + "  \n"
            elif para_type == BlockType.Table:
                for block in para_block["blocks"]:  # 1st.拼table_caption
                    if block["type"] == BlockType.TableCaption:
                        para_text += merge_para_with_text(block) + "  \n"
                for block in para_block["blocks"]:  # 2nd.拼table_body
                    if block["type"] == BlockType.TableBody:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["type"] == ContentType.Table:
                                    # if processed by table model
                                    if span.get("latex", ""):
                                        para_text += f"\n\n$\n {span['latex']}\n$\n\n"
                                    elif span.get("html", ""):
                                        para_text += f"\n\n{span['html']}\n\n"
                                    if span.get("image_path", "") and not self.is_url(
                                        span.get("image_path", "")
                                    ):
                                        image_path = join_path(
                                            img_buket_path, span["image_path"]
                                        )
                                        oss_url = self._transform_local_to_oss(
                                            pdf_name, image_path
                                        )
                                        para_text += f"\n![]({oss_url})  \n"
                for block in para_block["blocks"]:  # 3rd.拼table_footnote
                    if block["type"] == BlockType.TableFootnote:
                        para_text += merge_para_with_text(block) + "  \n"

            if para_text.strip() == "":
                continue
            else:
                page_markdown.append(para_text.strip() + "  ")
            index_count += 1

        return page_markdown, text_height_min, text_height_max, index_count

    def collect_title_info(
        self, para_block, title_list, text_height_min, text_height_max
    ):
        if not para_block.get("lines", None) or len(para_block["lines"]) <= 0:
            return text_height_min, text_height_max
        x0, y0, x1, y1 = para_block["lines"][0]["bbox"]
        content_height = y1 - y0
        if para_block["type"] == BlockType.Title:
            title_height = int(content_height)
            title_text = merge_para_with_text(para_block)
            title_list.append((title_text, title_height))
        elif para_block["type"] == BlockType.Text:
            if content_height < text_height_min:
                text_height_min = content_height
            if content_height > text_height_max:
                text_height_max = content_height
        return text_height_min, text_height_max

    def post_process_multi_level_headings(
        self, title_list, output_content, title_dict, text_height_min, text_height_max
    ):
        logger.info(
            "*****************************start process headings*****************************"
        )
        sorted_list = sorted(title_list, key=itemgetter(1), reverse=True)
        diff_list = [
            (sorted_list[i][1] - sorted_list[i + 1][1], i)
            for i in range(len(sorted_list) - 1)
        ]
        sorted_diff = sorted(diff_list, key=itemgetter(0), reverse=True)
        slice_index = []
        for diff, index in sorted_diff:
            # 标题差的绝对值超过2，则认为是下一级标题
            # markdown 中，# 表示一级标题，## 表示二级标题，以此类推，最多有6级标题，最多能有5次切分
            if diff >= DEFAULT_HEADING_DIFF_THRESHOLD and len(slice_index) <= 5:
                slice_index.append(index)
        slice_index.sort(reverse=True)
        rank = 1
        cur_index = 0
        if len(slice_index) > 0:
            cur_index = slice_index.pop()
        for index, (title_text, title_height) in enumerate(sorted_list):
            if index > cur_index:
                rank += 1
                if len(slice_index) > 0:
                    cur_index = slice_index.pop()
                else:
                    cur_index = len(sorted_list) - 1
            title_level = "#" * rank + " "
            if text_height_min <= text_height_max and int(
                text_height_min
            ) <= title_height <= int(text_height_max):
                title_level = ""
            old_title = "# " + title_text
            new_title = title_level + title_text
            if len(title_dict.get(old_title)) > 0:
                md_index = title_dict.get(old_title).pop()
                output_content[md_index] = new_title

        logger.info(
            "*****************************process headings ended*****************************"
        )

    def parse_pdf(
        self,
        pdf_path: str,
    ):
        """
        执行从 pdf 转换到 json、md 的过程，输出 md 和 json 文件到 pdf 文件所在的目录

        :param pdf_path: .pdf 文件的路径，可以是相对路径，也可以是绝对路径
        :param parse_method: 解析方法， 共 auto、ocr两种，默认 auto。auto会根据文件类型选择TXT模式或者OCR模式解析。ocr会直接使用OCR模式。
        :param model_json_path: 已经存在的模型数据文件，如果为空则使用内置模型，pdf 和 model_json 务必对应
        """
        try:
            pdf_name = os.path.basename(pdf_path).split(".")[0]
            pdf_name = pdf_name.replace(" ", "_")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, pdf_name)

                image_writer = FileBasedDataWriter(temp_file_path)
                # parent_dir is "", if the pdf_path is relative path, it will be joined with parent_dir.
                file_reader = FileBasedDataReader("")
                pdf_bytes = file_reader.read(pdf_path)
                ds = PymuDocDataset(pdf_bytes)

                # 选择解析方式
                if (
                    self.enable_mandatory_ocr
                    or ds.classify() == SupportedPdfParseMethod.OCR
                ):
                    infer_result = ds.apply(doc_analyze, ocr=True)
                    pipe_result = infer_result.pipe_ocr_mode(image_writer)
                else:
                    infer_result = ds.apply(doc_analyze, ocr=False)
                    pipe_result = infer_result.pipe_txt_mode(image_writer)

                content_list = pipe_result._pipe_res["pdf_info"]

                md_content = self.create_markdown(
                    pdf_name, content_list, temp_file_path
                )

            return md_content

        except Exception as e:
            logger.error(e)
            return None

    def load_data(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from PDF file and also accepts extra information in dict format."""
        return self.load(file_path, metadata=metadata, extra_info=extra_info)

    def load(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from PDF file and also accepts extra information in dict format.

        Args:
            file_path (Union[Path, str]): file path of PDF file (accepts string or Path).
            metadata (bool, optional): if metadata to be included or not. Defaults to True.
            extra_info (Optional[Dict], optional): extra information related to each document in dict format. Defaults to None.

        Raises:
            TypeError: if extra_info is not a dictionary.
            TypeError: if file_path is not a string or Path.

        Returns:
            List[Document]: list of documents.
        """
        md_content = self.parse_pdf(file_path)
        logger.info(f"[PaiPDFReader] successfully processed pdf file {file_path}.")
        docs = []
        if metadata:
            if not extra_info:
                extra_info = {}
            doc = Document(text=md_content, extra_info=extra_info)

            docs.append(doc)
        else:
            doc = Document(
                text=md_content,
                extra_info=dict(),
            )
            docs.append(doc)
            logger.info(f"processed pdf file {file_path} without metadata")
        logger.info(f"[PaiPDFReader] successfully loaded {len(docs)} nodes.")
        return docs
