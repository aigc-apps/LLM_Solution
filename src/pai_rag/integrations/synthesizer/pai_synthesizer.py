from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Sequence, AsyncGenerator, cast

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms import LLM
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.multi_modal_llms import MultiModalLLM
import llama_index.core.instrumentation as instrument
from llama_index.core.schema import (
    ImageNode,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    QueryType,
)
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    Response,
    StreamingResponse,
    AsyncStreamingResponse,
)
from llama_index.core.instrumentation.events.synthesis import (
    SynthesizeStartEvent,
    SynthesizeEndEvent,
)
from llama_index.core.llms.llm import (
    stream_completion_response_to_tokens,
    astream_completion_response_to_tokens,
)
from llama_index.core.prompts import PromptTemplate
from loguru import logger

dispatcher = instrument.get_dispatcher(__name__)

DEFAULT_TEXT_QA_TMPL = (
    "参考内容信息如下"
    "-------\n"
    "{context_str}\n"
    "-------\n"
    "根据提供内容而非其他知识回答问题. "
    "问题: {query_str}\n"
    "请仔细思考，必须使用和提问相同的语言，给出你的答案: \n"
)

DEFAULT_TEXT_QA_TMPL_EN = (
    "Reference content information is as follows"
    "-------\n"
    "{context_str}\n"
    "-------\n"
    "Answer the question based on the provided content rather than other knowledge. "
    "Question: {query_str}\n"
    "Please think carefully and use the same language as the question to give your answer: \n"
)

CITATION_TEXT_QA_TMPL = (
    "请完全根据提供的参考内容回答问题。\n"
    "参考内容由几段文本内容组成,"
    "当你生成的内容引用到了某段文本来源，请在内容中引用对应文本的数字序号来显示相关的信息源，"
    "比如[1]，这样可以让你的回复看起来更加可靠。"
    "你的答案需要包含至少一个相关的引用标记。"
    "只有在你真正引用了文本的时候才会插入引用标记，当你没找到任何值得引用的内容时，请直接回复你不知道。\n"
    "注意仅在引用标记中插入数字。你必须使用和提问相同的语言进行回答。\n\n"
    "例如:\n"
    "参考材料\n"
    "-------\n"
    "Source 1:\n"
    "Model Y 是特斯拉推出的一款电动SUV，具有珍珠白（多涂层）车漆、19英寸双子星轮毂和纯黑色高级内饰（黑色座椅）。此外，它还配备了全景玻璃车顶和双电机全轮驱动系统，提供更好的性能和操控。\n\n"
    "Source 2:\n"
    "Model 3 拥有星空灰车漆，19英寸新星轮毂，深色高级内饰（后轮驱动版），基础版辅助驾驶功能。Model 3 还提供多个选配包，例如全自动驾驶能力包和性能提升包，用户可根据需求进行配置。此外，Model 3 具有高效的空气动力学设计和长续航电池选项，适合长途驾驶。 \n\n"
    "Source 3:\n"
    "除了基本配置，特斯拉所有车型还提供许多个性化选项，例如不同颜色的车漆（包括红色、蓝色、黑色等），多种不同设计的轮毂和车顶设计（全景玻璃车顶或金属车顶），以及多种内饰颜色选择。 \n\n"
    "------\n"
    "问题：model3的轮毂和内饰是什么配置？\n"
    "答案：Model 3 配置了 19 英寸新星轮毂和深色高级内饰。它还提供多个选配包，例如全自动驾驶能力包和性能提升包，用户可根据需求进行配置。此外，Model 3 具有高效的空气动力学设计和长续航电池选项，适合长途驾驶 [2].\n\n"
    "现在轮到你了：\n\n"
    "参考材料\n"
    "-------\n"
    "{context_str}\n"
    "-------\n"
    "问题: {query_str}\n"
    "请仔细思考，必须使用和提问相同的语言，给出你的答案："
)

CITATION_TEXT_QA_TMPL_EN = (
    "Please answer the question based entirely on the references provided.\n"
    "The references consist of several paragraphs of text,"
    "When you generate content that references a text source, please quote the corresponding text number in the content to indicate the relevant information source,"
    "For example, [1], this will make your answer look more reliable."
    "Your answer must contain at least one relevant reference mark."
    "Only insert reference marks when you actually quote the text. If you don't find anything worth citing, just reply that you don't know."
    "Note that only numbers are inserted in reference marks. You must answer in the same language as the question.\n\n"
    "For example:\n"
    "References\n"
    "-------\n"
    "Source 1:\n"
    "Model Y is an electric SUV launched by Tesla, featuring Pearl White (multi-coat) paint, 19-inch Gemini wheels, and a pure black premium interior (black seats). It also comes equipped with a panoramic glass roof and dual motor all-wheel drive system, offering better performance and handling. \n\n"
    "Source 2:\n"
    "Model 3 has starry grey paint, 19-inch nova wheels, and a dark premium interior (rear-wheel drive version) with basic assisted driving features. Model 3 also offers several optional packages, such as the full self-driving capability package and performance upgrade package, allowing users to configure according to their needs. Additionally, Model 3 features an efficient aerodynamic design and long-range battery options suitable for long-distance driving. \n\n"
    "Source 3:\n"
    "In addition to the basic configurations, all Tesla models offer a number of personalization options, including different paint colors (such as red, blue, black, etc.), various wheel designs, and roof designs (panoramic glass roof or metal roof), as well as multiple interior color choices. \n\n"
    "------\n"
    "Question: What are the wheels and interior of model3? \n"
    "Answer: Model 3 is equipped with 19-inch nova wheels and a dark premium interior. It also offers several optional packages, such as the full self-driving capability package and performance upgrade package, allowing users to configure according to their needs. Additionally, Model 3 features an efficient aerodynamic design and long-range battery options suitable for long-distance driving [2]. \n\n"
    "Now it's your turn: \n\n"
    "Reference materials\n"
    "-------\n"
    "{context_str}\n"
    "-------\n"
    "Question: {query_str}\n"
    "Please MUST use the same language as the question to answer. Please think carefully and give your answer:"
)


DEFAULT_LLM_CHAT_TMPL = (
    "You are a helpful assistant."
    "Please answer the following question. \n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL = (
    "根据上面给出的图片和下面给出的参考材料来回答用户的问题。\n"
    "参考材料中包含一组文字描述和一组图片链接，图片链接分别对应到前面给出的图片的地址。\n"
    "请根据给定的材料回答给出的问题，回答中需要有文字描述和图片链接。如果材料中没有答案相关的信息，就回复你不知道。\n"
    "如果上面有图片对你生成答案有帮助，请找到图片链接并用markdown格式给出，如![](image_url)。\n\n"
    "你必须使用和提问相同的语言进行回答。"
    "例如：\n"
    "参考材料\n"
    "------\n"
    "Source 1:\n"
    "Model Y 是特斯拉推出的一款电动SUV，具有珍珠白（多涂层）车漆、19英寸双子星轮毂和纯黑色高级内饰（黑色座椅）。此外，它还配备了全景玻璃车顶和双电机全轮驱动系统，提供更好的性能和操控。\n\n"
    "Source 2:\n"
    "Model 3 拥有星空灰车漆，19英寸新星轮毂，深色高级内饰（后轮驱动版），基础版辅助驾驶功能。Model 3 还提供多个选配包，例如全自动驾驶能力包和性能提升包，用户可根据需求进行配置。此外，Model 3 具有高效的空气动力学设计和长续航电池选项，适合长途驾驶。 \n\n"
    "Image 1:\n"
    "http://www.tesla.cn/model3.jpg\n\n"
    "------\n"
    "问题：model3的轮毂和内饰是什么配置?\n"
    "答案：Model 3 配置了 19 英寸新星轮毂和深色高级内饰。它还提供多个选配包，例如全自动驾驶能力包和性能提升包，用户可根据需求进行配置。此外，Model 3 具有高效的空气动力学设计和长续航电池选项，适合长途驾驶。下图是 Model 3 的图片:"
    "![](http://www.tesla.cn/model3.jpg)\n\n"
    "现在轮到你了：\n\n"
    "参考材料\n"
    "------\n"
    "{context_str}\n"
    "------\n"
    "问题: {query_str}\n"
    "请必须使用和提问相同的语言，仔细思考，给出你的答案："
)


DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL_EN = (
    "Answer the user's question based on the pictures given above and the reference materials given below.\n"
    "The reference materials contain a set of text descriptions and a set of image links, which correspond to the addresses of the pictures given above.\n"
    "Please answer the given questions based on the given materials. The answers need to have text descriptions and image links. If there is no information related to the answer in the materials, reply that you don't know.\n"
    "If there are pictures above that help you generate answers, please find the image link and give it in markdown format, such as ![](image_url).\n\n"
    "You must answer in the same language as the question."
    "For example:\n"
    "Reference materials\n"
    "------\n"
    "Source 1:\n"
    "Model Y is an electric SUV launched by Tesla, featuring Pearl White (multi-coat) paint, 19-inch Gemini wheels, and a pure black premium interior (black seats). It also comes equipped with a panoramic glass roof and dual motor all-wheel drive system, offering better performance and handling. \n\n"
    "Source 2:\n"
    "Model 3 has starry grey paint, 19-inch nova wheels, and a dark premium interior (rear-wheel drive version) with basic assisted driving features. Model 3 also offers several optional packages, such as the full self-driving capability package and performance upgrade package, allowing users to configure according to their needs. Additionally, Model 3 features an efficient aerodynamic design and long-range battery options suitable for long-distance driving. \n\n"
    "Image 1:\n"
    "http://www.tesla.cn/model3.jpg\n\n"
    "------\n"
    "Question: What are the wheels and interior of model3?\n"
    "Answer: Model 3 is equipped with 19-inch nova wheels and a dark premium interior. It also offers several optional packages, such as the full self-driving capability package and performance upgrade package, allowing users to configure according to their needs. Additionally, Model 3 features an efficient aerodynamic design and long-range battery options suitable for long-distance driving. Below is an image of Model 3: "
    "![](http://www.tesla.cn/model3.jpg)\n\n"
    "Now it's your turn:\n\n"
    "Reference materials\n"
    "------\n"
    "{context_str}\n"
    "------\n"
    "Question: {query_str}\n"
    "Must use the same language as the question. Please think carefully and give your answer:"
)


CITATION_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL = (
    "根据上面给出的图片和下面给出的参考材料来回答用户的问题。\n"
    "参考材料中包含一组文字描述和一组图片链接，图片链接分别对应到前面给出的图片的地址。\n"
    "请根据给定的材料回答给出的问题，如果你当前生成的内容引用到了某一段文字描述，请直接在内容里引用他的数字序号，如[1]。\n"
    "如果上面有图片对你生成答案有帮助，请找到图片链接并用markdown格式给出，如![](image_url)。"
    "请至少列出一个文本和图片引用。如果材料中没有答案相关的信息，就回复你不知道。\n"
    "你必须使用和提问相同的语言进行回答。"
    "例如：\n"
    "参考材料\n"
    "------\n"
    "Source 1:\n"
    "Model Y 是特斯拉推出的一款电动SUV，具有珍珠白（多涂层）车漆、19英寸双子星轮毂和纯黑色高级内饰（黑色座椅）。此外，它还配备了全景玻璃车顶和双电机全轮驱动系统，提供更好的性能和操控。\n\n"
    "Source 2:\n"
    "Model 3 拥有星空灰车漆，19英寸新星轮毂，深色高级内饰（后轮驱动版），基础版辅助驾驶功能。Model 3 还提供多个选配包，例如全自动驾驶能力包和性能提升包，用户可根据需求进行配置。此外，Model 3 具有高效的空气动力学设计和长续航电池选项，适合长途驾驶。 \n\n"
    "Image 1:\n"
    "http://www.tesla.cn/model3.jpg\n\n"
    "------\n"
    "问题：model3的轮毂和内饰是什么配置?\n"
    "答案：Model 3 配置了 19 英寸新星轮毂和深色高级内饰。它还提供多个选配包，例如全自动驾驶能力包和性能提升包，用户可根据需求进行配置。此外，Model 3 具有高效的空气动力学设计和长续航电池选项，适合长途驾驶 [2]. 下图是 Model 3 的图片:"
    "![](http://www.tesla.cn/model3.jpg)\n\n"
    "现在轮到你了：\n\n"
    "参考材料\n"
    "------\n"
    "{context_str}\n"
    "------\n"
    "问题: {query_str}\n"
    "请必须使用和提问相同的语言，仔细思考，给出你的答案："
)

CITATION_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL_EN = (
    "Answer the user's question based on the pictures given above and the reference materials given below.\n"
    "The reference materials contain a set of text descriptions and a set of image links. The image links correspond to the addresses of the pictures given above.\n"
    "Please answer the given questions based on the given materials. If the content you are currently generating refers to a certain text description, please directly quote its numerical serial number in the content, such as [1].\n"
    "If there are pictures above that help you generate the answer, please find the image link and give it in markdown format, such as ![](image_url)."
    "Please list at least one text and image reference. If there is no information related to the answer in the material, reply that you don't know.\n"
    "You must answer in the same language as the question."
    "For example:\n"
    "Reference materials\n"
    "------\n"
    "Source 1:\n"
    "Model Y is an electric SUV launched by Tesla, featuring Pearl White (multi-coat) paint, 19-inch Gemini wheels, and a pure black premium interior (black seats). It also comes equipped with a panoramic glass roof and dual motor all-wheel drive system, offering better performance and handling. \n\n"
    "Source 2:\n"
    "Model 3 has starry grey paint, 19-inch nova wheels, and a dark premium interior (rear-wheel drive version) with basic assisted driving features. Model 3 also offers several optional packages, such as the full self-driving capability package and performance upgrade package, allowing users to configure according to their needs. Additionally, Model 3 features an efficient aerodynamic design and long-range battery options suitable for long-distance driving. \n\n"
    "Image 1:\n"
    "http://www.tesla.cn/model3.jpg\n\n"
    "------\n"
    "Question: What are the wheels and interior of model3?\n"
    "Answer: Model 3 is equipped with 19-inch nova wheels and a dark premium interior. It also offers several optional packages, such as the full self-driving capability package and performance upgrade package, allowing users to configure according to their needs. Additionally, Model 3 features an efficient aerodynamic design and long-range battery options suitable for long-distance driving [2]. Below is an image of Model 3: "
    "![](http://www.tesla.cn/model3.jpg)\n\n"
    "Now it's your turn:\n\n"
    "Reference materials\n"
    "------\n"
    "{context_str}\n"
    "------\n"
    "Question: {query_str}\n"
    "Please MUST use the same language as the question, think carefully, and give your answer:"
)
QueryTextType = QueryType


def empty_response_generator() -> Generator[str, None, None]:
    yield "Empty Response"


async def empty_response_agenerator() -> AsyncGenerator[str, None]:
    yield "Empty Response"


@dataclass
class PaiQueryBundle(QueryBundle):
    stream: bool = False
    no_retrieval: bool = False
    citation: bool = False


"""
PaiSynthesizer:
Supports multi-modal inputs synthesizer.
Will use Multi-modal LLM for inputs with images and LLM for pure text inputs.
"""


class PaiSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        multimodal_llm: Optional[MultiModalLLM] = None,
        multimodal_qa_template: Optional[BasePromptTemplate] = None,
        citation_text_qa_template: Optional[BasePromptTemplate] = None,
        citation_multimodal_qa_template: Optional[BasePromptTemplate] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            streaming=streaming,
        )
        self._text_qa_template = text_qa_template or PromptTemplate(
            template=DEFAULT_TEXT_QA_TMPL
        )
        self._multimodal_qa_template = multimodal_qa_template or PromptTemplate(
            template=DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL
        )
        self._citation_text_qa_template = citation_text_qa_template or PromptTemplate(
            template=CITATION_TEXT_QA_TMPL
        )
        self._citation_multimodal_qa_template = (
            citation_multimodal_qa_template
            or PromptTemplate(template=CITATION_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL)
        )
        self._llm_only_template = PromptTemplate(template=DEFAULT_LLM_CHAT_TMPL)
        self._multimodal_llm = multimodal_llm

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"text_qa_template": self._text_qa_template}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_qa_template" in prompts:
            self._text_qa_template = prompts["text_qa_template"]

    @dispatcher.span
    def synthesize(
        self,
        query: PaiQueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )

        if not query.no_retrieval and len(nodes) == 0:
            if query.stream:
                empty_response = StreamingResponse(
                    response_gen=empty_response_generator()
                )
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response
            else:
                empty_response = Response("Empty Response")
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        text_nodes, image_nodes = [], []
        for node in nodes:
            if isinstance(node.node, ImageNode):
                image_nodes.append(node)
            else:
                text_nodes.append(node)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query.query_str},
        ) as event:
            if query.no_retrieval:
                response_str = self.get_llm_only_response(
                    query_str=query.query_str,
                    streaming=query.stream,
                    **response_kwargs,
                )
            else:
                response_str = self.get_response(
                    query_str=query.query_str,
                    text_chunks=[
                        n.node.get_content(metadata_mode=MetadataMode.LLM)
                        for n in text_nodes
                    ],
                    image_url_list=[n.node.image_url for n in image_nodes],
                    streaming=query.stream,
                    citation=query.citation,
                    **response_kwargs,
                )

            additional_source_nodes = additional_source_nodes or []
            source_nodes = list(nodes) + list(additional_source_nodes)

            response = self._prepare_response_output(response_str, source_nodes)

            event.on_end(payload={EventPayload.RESPONSE: response})

        dispatcher.event(
            SynthesizeEndEvent(
                query=query,
                response=response,
            )
        )
        return response

    @dispatcher.span
    async def asynthesize(
        self,
        query: PaiQueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )
        if not query.no_retrieval and len(nodes) == 0:
            if query.stream:
                empty_response = AsyncStreamingResponse(
                    response_gen=empty_response_agenerator()
                )
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response
            else:
                empty_response = Response("Empty Response")
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        text_nodes, image_nodes = [], []
        for node in nodes:
            if isinstance(node.node, ImageNode):
                image_nodes.append(node)
            else:
                text_nodes.append(node)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query.query_str},
        ) as event:
            if query.no_retrieval:
                response_str = await self.aget_llm_only_response(
                    query_str=query.query_str,
                    streaming=query.stream,
                    **response_kwargs,
                )
            else:
                response_str = await self.aget_response(
                    query_str=query.query_str,
                    text_chunks=[
                        n.node.get_content(metadata_mode=MetadataMode.LLM)
                        for n in text_nodes
                    ],
                    image_url_list=[n.node.image_url for n in image_nodes],
                    streaming=query.stream,
                    citation=query.citation,
                    **response_kwargs,
                )

            additional_source_nodes = additional_source_nodes or []
            source_nodes = list(nodes) + list(additional_source_nodes)

            response = self._prepare_response_output(response_str, source_nodes)

            event.on_end(payload={EventPayload.RESPONSE: response})

        dispatcher.event(
            SynthesizeEndEvent(
                query=query,
                response=response,
            )
        )
        return response

    def _get_multi_modal_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        image_url_list: Sequence[str] = None,
        streaming: bool = False,
        citation: bool = False,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        image_documents = load_image_urls(image_url_list)

        context_str = (
            "\n".join(
                [f"Source {i+1}:\n{text}\n" for i, text in enumerate(text_chunks)]
            )
            + "\n"
        )
        context_str += "\n".join(
            [f"Image {i+1}:\n{url}\n" for i, url in enumerate(image_url_list)]
        )

        if not citation:
            fmt_prompt = self._multimodal_qa_template.format(
                context_str=context_str, query_str=query_str
            )
        else:
            fmt_prompt = self._citation_multimodal_qa_template.format(
                context_str=context_str, query_str=query_str
            )

        if streaming:
            completion_response_gen = self._multimodal_llm.stream_complete(
                prompt=fmt_prompt,
                image_documents=image_documents,
                **response_kwargs,
            )
            stream_tokens = stream_completion_response_to_tokens(
                completion_response_gen
            )
            return cast(Generator, stream_tokens)
        else:
            llm_response = self._multimodal_llm.complete(
                prompt=fmt_prompt,
                image_documents=image_documents,
                **response_kwargs,
            )
            response = llm_response.text or "Empty Response"
            return response

    async def _aget_multi_modal_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        image_url_list: Sequence[str] = None,
        streaming: bool = False,
        citation: bool = False,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        image_documents = load_image_urls(image_url_list)

        context_str = (
            "\n".join(
                [f"Source {i+1}:\n{text}\n" for i, text in enumerate(text_chunks)]
            )
            + "\n"
        )
        context_str += "\n".join(
            [f"Image {i+1}:\n{url}\n" for i, url in enumerate(image_url_list)]
        )

        if not citation:
            fmt_prompt = self._multimodal_qa_template.format(
                context_str=context_str, query_str=query_str
            )
        else:
            fmt_prompt = self._citation_multimodal_qa_template.format(
                context_str=context_str, query_str=query_str
            )

        if streaming:
            completion_response_gen = await self._multimodal_llm.astream_complete(
                prompt=fmt_prompt,
                image_documents=image_documents,
                **response_kwargs,
            )
            stream_tokens = await astream_completion_response_to_tokens(
                completion_response_gen
            )
            return cast(Generator, stream_tokens)
        else:
            llm_response = await self._multimodal_llm.acomplete(
                prompt=fmt_prompt,
                image_documents=image_documents,
                **response_kwargs,
            )
            response = llm_response.text or "Empty Response"
            return response

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        image_url_list: Sequence[str] = None,
        streaming: bool = False,
        citation: bool = False,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        if image_url_list and len(image_url_list) > 0:
            assert (
                self._multimodal_llm is not None
            ), "Multi-modal LLM must be provided to understand image documents."

            logger.info(
                f"Synthsize using Multi-modal LLM with images {image_url_list}. citation: {citation}"
            )
            return await self._aget_multi_modal_response(
                query_str=query_str,
                text_chunks=text_chunks,
                image_url_list=image_url_list,
                streaming=streaming,
                citation=citation,
                **response_kwargs,
            )

        logger.info(f"Synthsize using LLM with no image inputs. citation: {citation}")
        if not citation:
            text_qa_template = self._text_qa_template.partial_format(
                query_str=query_str
            )
        else:
            text_qa_template = self._citation_text_qa_template.partial_format(
                query_str=query_str
            )

        context_str = "\n".join(
            [f"Source {i+1}:\n{text}\n" for i, text in enumerate(text_chunks)]
        )

        response: RESPONSE_TEXT_TYPE
        if not streaming:
            response = await self._llm.apredict(
                text_qa_template,
                context_str=context_str,
                **response_kwargs,
            )
        else:
            # customized modify [will be removed]
            response = await self._llm.astream(
                text_qa_template,
                context_str=context_str,
                **response_kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        image_url_list: Sequence[str] = None,
        streaming: bool = False,
        citation: bool = False,
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        if image_url_list and len(image_url_list) > 0:
            assert (
                self._multimodal_llm is not None
            ), "Multi-modal LLM must be provided to understand image documents."
            return self._get_multi_modal_response(
                query_str=query_str,
                text_chunks=text_chunks,
                image_url_list=image_url_list,
                streaming=streaming,
                **kwargs,
            )

        if not citation:
            text_qa_template = self._text_qa_template.partial_format(
                query_str=query_str
            )
        else:
            text_qa_template = self._citation_text_qa_template.partial_format(
                query_str=query_str
            )

        context_str = "\n".join(
            [f"Source {i+1}:\n{text}\n" for i, text in enumerate(text_chunks)]
        )
        response: RESPONSE_TEXT_TYPE
        if not streaming:
            response = self._llm.predict(
                text_qa_template,
                context_str=context_str,
                **kwargs,
            )
        else:
            response = self._llm.stream(
                text_qa_template,
                context_str=context_str,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    async def aget_llm_only_response(
        self,
        query_str: str,
        streaming: bool = False,
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        response: RESPONSE_TEXT_TYPE
        if not streaming:
            response = await self._llm.apredict(
                self._llm_only_template,
                query_str=query_str,
                **kwargs,
            )
        else:
            response = await self._llm.astream(
                self._llm_only_template,
                query_str=query_str,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    def get_llm_only_response(
        self,
        query_str: str,
        streaming: bool = False,
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        response: RESPONSE_TEXT_TYPE
        if not streaming:
            response = self._llm.predict(
                self._llm_only_template,
                query_str=query_str,
                **kwargs,
            )
        else:
            response = self._llm.stream(
                self._llm_only_template,
                query_str=query_str,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response
