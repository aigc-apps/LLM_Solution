import json
from typing import Any
import requests
import httpx
import os
import re
import markdown
import html
import mimetypes
from http import HTTPStatus
from loguru import logger
from pai_rag.app.web.view_model import ViewModel
from pai_rag.app.web.ui_constants import EMPTY_KNOWLEDGEBASE_MESSAGE
from pai_rag.core.rag_config import RagConfig
from pai_rag.core.rag_index_manager import RagIndexEntry, RagIndexMap
from urllib.parse import urljoin

DEFAULT_CLIENT_TIME_OUT = 120
DEFAULT_LOCAL_URL = "http://127.0.0.1:8001/"


class RagApiError(Exception):
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class RagWebClient:
    def __init__(self):
        self.endpoint = DEFAULT_LOCAL_URL  # default link
        self.session_id = None

    def set_endpoint(self, endpoint: str):
        self.endpoint = endpoint if endpoint.endswith("/") else f"{endpoint}/"

    def clear_history(self):
        self.session_id = None

    @property
    def query_url(self):
        return urljoin(self.endpoint, "api/v1/query")

    @property
    def search_url(self):
        return urljoin(self.endpoint, "api/v1/query/search")

    @property
    def data_analysis_url(self):
        return urljoin(self.endpoint, "api/v1/query/data_analysis")

    @property
    def db_loder_url(self):
        return urljoin(self.endpoint, "api/v1/query/load_db_info")

    @property
    def llm_url(self):
        return urljoin(self.endpoint, "api/v1/query/llm")

    @property
    def retrieval_url(self):
        return urljoin(self.endpoint, "api/v1/query/retrieval")

    @property
    def config_url(self):
        return urljoin(self.endpoint, "api/v1/config")

    @property
    def load_data_url(self):
        return urljoin(self.endpoint, "api/v1/upload_data")

    @property
    def load_datasheet_url(self):
        return urljoin(self.endpoint, "api/v1/upload_datasheet")

    @property
    def load_db_history_url(self):
        return urljoin(self.endpoint, "api/v1/upload_db_history")

    @property
    def load_agent_cfg_url(self):
        # return f"{self.endpoint}v1/config/agent"
        return urljoin(self.endpoint, "api/v1/config/agent")

    @property
    def get_load_state_url(self):
        return urljoin(self.endpoint, "api/v1/get_upload_state")

    @property
    def get_evaluate_generate_url(self):
        return urljoin(self.endpoint, "api/v1/evaluate/generate")

    @property
    def get_evaluate_retrieval_url(self):
        return urljoin(self.endpoint, "api/v1/evaluate/retrieval")

    @property
    def get_evaluate_response_url(self):
        return urljoin(self.endpoint, "api/v1/evaluate/response")

    @property
    def index_url(self):
        return urljoin(self.endpoint, "api/v1/indexes/")

    @property
    def list_index_url(self):
        return urljoin(self.endpoint, "api/v1/indexes")

    @property
    def health_check_url(self):
        return urljoin(self.endpoint, "api/v1/health")

    def _format_rag_response(
        self, response, with_history: bool = False, stream: bool = False
    ):
        if stream:
            text = response["delta"]
        else:
            text = response["answer"]
        docs = response.get("docs", []) or []
        session_id = response.get("session_id", None)
        is_finished = response.get("is_finished", True)

        referenced_docs = ""
        if is_finished:
            content_list = []
            self.session_id = session_id
            for i, doc in enumerate(docs):
                filename = doc["metadata"].get("file_name", None)
                sheet_name = doc["metadata"].get("sheet_name", None)
                ref_table = doc["metadata"].get("query_tables", None)
                invalid_flag = doc["metadata"].get("invalid_flag", 0)
                file_url = doc["metadata"].get("file_url", None)
                if doc.get("image_url", None):
                    media_url = doc.get("image_url")
                else:
                    media_url = doc.get("metadata", {}).get("image_info_list", None)
                if media_url and doc["text"] == "":
                    formatted_image_name = re.sub(
                        "^[0-9a-z]{32}_", "", "/".join(media_url.split("/")[-2:])
                    )
                    content = f"""
<span>
    <a href="{media_url}"> [{i+1}]: {formatted_image_name} </a> Score:{doc["score"]}
</span>
<br>
"""
                elif filename:
                    formatted_file_name = re.sub("^[0-9a-z]{32}_", "", filename)
                    if sheet_name:
                        formatted_file_name += f">>{sheet_name}"
                    html_content = html.escape(
                        re.sub(r"<.*?>", "", doc["text"])
                    ).replace("\n", " ")
                    if file_url:
                        formatted_file_name = (
                            f'<a href="{file_url}"> {formatted_file_name} </a>'
                        )
                    content = f"""
<span class="text">
    [{i+1}]: {formatted_file_name} Score:{doc["score"]}
    <span style='color: gray; font-size: 12px;'> ( {html_content} ) </span>
</span>
<br>
"""
                elif ref_table:
                    ref_table_format = ", ".join([i for i in ref_table])
                    formatted_table_name = f"查询数据库中相关表名包括： <b>{ref_table_format}</b>"

                    if invalid_flag == 0:
                        run_flag = " ✓ "
                        ref_sql = doc["metadata"].get("query_code_instruction", None)
                        formatted_sql_query = f"<b>{ref_sql}</b>"
                        content = (
                            f"""<span style="color:grey; font-size: 14px;">{formatted_table_name}</span> \n"""
                            f"""<span style="color:grey; font-size: 14px;">生成的sql语句为：</span> <pre style="color:grey; font-size: 12px;">{formatted_sql_query}</pre> """
                            f"""<span style="color:grey; font-size: 14px;">sql查询是否有效：</span> <span style="color:green; font-size: 14px;">{run_flag}</span>"""
                        )
                    else:
                        run_flag = " ✗ "
                        ref_sql = doc["metadata"].get(
                            "generated_query_code_instruction", None
                        )
                        formatted_sql_query = f"<b>{ref_sql}</b>"
                        content = (
                            f"""<span style="color:grey; font-size: 14px;">{formatted_table_name}</span> \n"""
                            f"""<span style="color:grey; font-size: 14px;">生成的sql语句为：</span> <pre style="color:grey; font-size: 12px;">{formatted_sql_query}</pre> """
                            f"""<span style="color:grey; font-size: 14px;">sql查询是否有效：</span> <span style="color:red; font-size: 14px;">{run_flag}</span>"""
                        )
                else:
                    content = ""
                content_list.append(content)
            referenced_docs = "".join(content_list)

        formatted_answer = text
        if referenced_docs:
            formatted_answer += f"\n\n**Reference**:\n {referenced_docs}"

        response["delta"] = formatted_answer

        return response

    def check_health(self):
        try:
            r = requests.get(self.health_check_url)
            return r.status_code == HTTPStatus.OK
        except Exception as ex:
            logger.error(f"Check health failed: {ex}")
            return False

    def query(
        self,
        text: str,
        with_history: bool = False,
        stream: bool = False,
        citation: bool = False,
        with_intent: bool = False,
        index_name: str = None,
    ):
        session_id = self.session_id if with_history else None
        q = dict(
            question=text,
            session_id=session_id,
            stream=stream,
            citation=citation,
            with_intent=with_intent,
            index_name=index_name,
        )
        r = requests.post(self.query_url, json=q, stream=True)
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=r.text)
        if not stream:
            response = dotdict(json.loads(r.text))
            yield self._format_rag_response(
                response, with_history=with_history, stream=stream
            )
        else:
            for chunk in r.iter_lines(chunk_size=8192, decode_unicode=True):
                if not chunk.startswith("data:"):
                    continue
                chunk_response = dotdict(json.loads(chunk[5:]))
                yield self._format_rag_response(
                    chunk_response, with_history=with_history, stream=stream
                )

    def query_search(
        self,
        text: str,
        with_history: bool = False,
        citation: bool = False,
        stream: bool = False,
    ):
        session_id = self.session_id if with_history else None
        q = dict(
            question=text,
            session_id=session_id,
            stream=stream,
            with_intent=False,
            citation=citation,
        )
        r = requests.post(self.search_url, json=q, stream=True)
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=r.text)
        if not stream:
            response = dotdict(json.loads(r.text))
            yield self._format_rag_response(response, stream=stream)
        else:
            for chunk in r.iter_lines(chunk_size=8192, decode_unicode=True):
                if not chunk.startswith("data:"):
                    continue
                chunk_response = dotdict(json.loads(chunk[5:]))
                yield self._format_rag_response(chunk_response, stream=stream)

    def query_data_analysis(
        self,
        text: str,
        with_history: bool = False,
        stream: bool = False,
    ):
        session_id = self.session_id if with_history else None
        q = dict(
            question=text,
            session_id=session_id,
            stream=stream,
        )
        r = requests.post(self.data_analysis_url, json=q, stream=True)
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=r.text)
        if not stream:
            response = dotdict(json.loads(r.text))
            yield self._format_rag_response(response, stream=stream)
        else:
            for chunk in r.iter_lines(chunk_size=8192, decode_unicode=True):
                if not chunk.startswith("data:"):
                    continue
                chunk_response = dotdict(json.loads(chunk[5:]))
                yield self._format_rag_response(chunk_response, stream=stream)

    def query_llm(
        self,
        text: str,
        with_history: bool = False,
        temperature: float = 0.1,
        stream: bool = False,
    ):
        session_id = self.session_id if with_history else None
        q = dict(
            question=text,
            temperature=temperature,
            session_id=session_id,
            stream=stream,
        )

        r = requests.post(
            self.llm_url, json=q, stream=True, timeout=DEFAULT_CLIENT_TIME_OUT
        )
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=r.text)
        if not stream:
            response = dotdict(json.loads(r.text))
            yield self._format_rag_response(
                response, with_history=with_history, stream=stream
            )
        else:
            for chunk in r.iter_lines(chunk_size=8192, decode_unicode=True):
                if not chunk.startswith("data:"):
                    continue
                chunk_response = dotdict(json.loads(chunk[5:]))
                yield self._format_rag_response(
                    chunk_response, with_history=with_history, stream=stream
                )

    def query_vector(self, text: str, index_name: str = None):
        q = dict(question=text, index_name=index_name)
        r = requests.post(self.retrieval_url, json=q, timeout=DEFAULT_CLIENT_TIME_OUT)
        response = dotdict(json.loads(r.text))
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=response.message)

        formatted_text = (
            "<tr><th>Document</th><th>Score</th><th>Text</th><th>Media</tr>\n"
        )
        if len(response["docs"]) == 0:
            response["delta"] = EMPTY_KNOWLEDGEBASE_MESSAGE.format(query_str=text)
        else:
            for i, doc in enumerate(response["docs"]):
                html_content = markdown.markdown(doc["text"])
                file_url = doc.get("metadata", {}).get("file_url", None)
                if doc.get("image_url", None):
                    media_url = doc.get("image_url")
                else:
                    media_url = doc.get("metadata", {}).get("image_info_list", None)
                if media_url and isinstance(media_url, list):
                    media_url = "<br>".join(
                        [
                            f'<img src="{url.get("image_url", None)}" alt="Image {j + 1}"/>'
                            for j, url in enumerate(media_url)
                        ]
                    )
                elif media_url:
                    media_url = f"""<img src="{media_url}"/>"""
                safe_html_content = html.escape(html_content).replace("\n", "<br>")
                if file_url:
                    safe_html_content = (
                        f"""<a href="{file_url}">{safe_html_content}</a>"""
                    )
                formatted_text += '<tr style="font-size: 13px;"><td>Doc {}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n'.format(
                    i + 1, doc["score"], safe_html_content, media_url
                )
            formatted_text = (
                "<table>\n<tbody>\n" + formatted_text + "</tbody>\n</table>"
            )
            response["delta"] = formatted_text
        yield response

    def add_knowledge(
        self,
        oss_path: str = None,
        input_files: str = None,
        enable_raptor: bool = False,
        enable_multimodal: bool = False,
        index_name: str = None,
    ):
        files = []
        file_obj_list = []
        if input_files:
            for file_name in input_files:
                file_obj = open(file_name, "rb")
                mimetype = mimetypes.guess_type(file_name)[0]
                files.append(
                    ("files", (os.path.basename(file_name), file_obj, mimetype))
                )
                file_obj_list.append(file_obj)

        para = {
            "enable_raptor": enable_raptor,
            "oss_path": oss_path,
            "index_name": index_name,
            "enable_multimodal": enable_multimodal,
        }
        try:
            r = requests.post(
                self.load_data_url,
                files=files,
                data=para,
                timeout=DEFAULT_CLIENT_TIME_OUT,
            )
            response = dotdict(json.loads(r.text))
            if r.status_code != HTTPStatus.OK:
                raise RagApiError(code=r.status_code, msg=response.message)
        finally:
            for file_obj in file_obj_list:
                file_obj.close()

        response = dotdict(json.loads(r.text))
        return response

    def add_datasheet(
        self,
        input_file: str,
    ):
        file_obj = open(input_file, "rb")
        mimetype = mimetypes.guess_type(input_file)[0]
        files = {"file": (input_file, file_obj, mimetype)}
        try:
            r = requests.post(
                self.load_datasheet_url,
                files=files,
                timeout=DEFAULT_CLIENT_TIME_OUT,
            )
            response = dotdict(json.loads(r.text))
            if r.status_code != HTTPStatus.OK:
                raise RagApiError(code=r.status_code, msg=response.message)
        except Exception as e:
            logger.exception(f"add_datasheet failed: {e}")
        finally:
            file_obj.close()

        response = dotdict(json.loads(r.text))
        return response

    def add_db_history(
        self,
        input_file: str,
    ):
        file_obj = open(input_file, "rb")
        mimetype = mimetypes.guess_type(input_file)[0]
        files = {"file": (input_file, file_obj, mimetype)}
        try:
            r = requests.post(
                self.load_db_history_url,
                files=files,
                timeout=DEFAULT_CLIENT_TIME_OUT,
            )
            response = dotdict(json.loads(r.text))
            if r.status_code != HTTPStatus.OK:
                raise RagApiError(code=r.status_code, msg=response.message)
        except Exception as e:
            logger.exception(f"add_db_history failed: {e}")
        finally:
            file_obj.close()

        response = dotdict(json.loads(r.text))
        return response

    def load_db_info(
        self,
    ):
        try:
            r = requests.post(
                self.db_loder_url,
                timeout=DEFAULT_CLIENT_TIME_OUT,
            )
            # response = dotdict(json.loads(r.text))
            # print("response:", r, r.text)
            if r.status_code != HTTPStatus.OK:
                raise RagApiError(code=r.status_code, msg=r.text)
        except Exception as e:
            logger.exception(f"load db info failed: {e}")
            raise  # 重新抛出异常

        return r.text

    async def get_knowledge_state(self, task_id: str):
        async with httpx.AsyncClient(timeout=DEFAULT_CLIENT_TIME_OUT) as client:
            r = await client.get(self.get_load_state_url, params={"task_id": task_id})
            response = dotdict(json.loads(r.text))
            if r.status_code != HTTPStatus.OK:
                raise RagApiError(code=r.status_code, msg=response.message)
            return response

    def patch_config(self, update_dict: Any):
        config = self.get_config()
        # print("config:", config)
        view_model: ViewModel = ViewModel.from_app_config(config)
        view_model.update(update_dict)
        new_config = view_model.to_app_config()
        r = requests.patch(
            self.config_url, json=new_config, timeout=DEFAULT_CLIENT_TIME_OUT
        )
        response = dotdict(json.loads(r.text))
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=response.message)

    def get_config(self):
        r = requests.get(self.config_url, timeout=DEFAULT_CLIENT_TIME_OUT)
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=r.text)
        config = RagConfig.model_validate_json(json_data=r.text)
        return config

    def evaluate_for_generate_qa(self, overwrite):
        r = requests.post(
            self.get_evaluate_generate_url,
            params={"overwrite": overwrite},
            timeout=DEFAULT_CLIENT_TIME_OUT,
        )
        response = dotdict(json.loads(r.text))
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=response.message)
        return response

    def evaluate_for_retrieval_stage(self):
        r = requests.post(
            self.get_evaluate_retrieval_url, timeout=DEFAULT_CLIENT_TIME_OUT
        )
        response = dotdict(json.loads(r.text))

        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=response.message)

        return response

    def evaluate_for_response_stage(self):
        r = requests.post(
            self.get_evaluate_response_url, timeout=DEFAULT_CLIENT_TIME_OUT
        )
        response = dotdict(json.loads(r.text))
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=response.message)
        return response

    def list_indexes(self) -> RagIndexMap:
        r = requests.get(self.list_index_url, timeout=DEFAULT_CLIENT_TIME_OUT)
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=r.text)

        return RagIndexMap.model_validate_json(r.text)

    def add_index(self, index_entry: RagIndexEntry):
        r = requests.post(
            urljoin(self.index_url, index_entry.index_name),
            json=index_entry.model_dump(),
            timeout=DEFAULT_CLIENT_TIME_OUT,
        )
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(
                code=r.status_code,
                msg=f"update index {index_entry.index_name} failed. {r.text}",
            )

    def update_index(self, index_entry: RagIndexEntry):
        r = requests.patch(
            urljoin(self.index_url, index_entry.index_name),
            json=index_entry.model_dump(),
            timeout=DEFAULT_CLIENT_TIME_OUT,
        )
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(
                code=r.status_code,
                msg=f"update index {index_entry.index_name} failed. {r.text}",
            )

    def delete_index(self, index_name: str):
        r = requests.delete(
            urljoin(self.index_url, index_name),
            timeout=DEFAULT_CLIENT_TIME_OUT,
        )
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(
                code=r.status_code, msg=f"delete index {index_name} failed. {r.text}"
            )


rag_client = RagWebClient()
