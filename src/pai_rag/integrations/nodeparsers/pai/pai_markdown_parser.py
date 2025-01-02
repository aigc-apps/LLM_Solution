"""Markdown node parser."""
from llama_index.core.bridge.pydantic import Field, BaseModel
from typing import Any, Iterator, List, Optional, Sequence
import json

from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.schema import (
    BaseNode,
    ImageNode,
    TextNode,
    NodeRelationship,
    MetadataMode,
)
from pai_rag.integrations.nodeparsers.utils.pai_markdown_tree import (
    build_markdown_tree,
    TreeNode,
)


class ImageInfo(BaseModel):
    image_url: str = Field(description="Image url.")
    image_text: Optional[str] = Field(description="Image text.", default=None)
    image_url_start_pos: Optional[int] = Field(
        description="Image start position.", default=None
    )
    image_url_end_pos: Optional[int] = Field(
        description="Image end position.", default=None
    )


class StructuredNodeParser(BaseModel):
    """Strcutured node parser.

    Will try to detect document struct according to Title information.

    Splits a document into Nodes using custom splitting logic.

    Args:
        chunk_size (int): chunk size
        chunk_overlap_size (int): chunk overlap size
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    chunk_size: int = Field(default=500, description="chunk size.")
    chunk_overlap_size: int = Field(default=10, description="Chunk overlap size.")
    enable_multimodal: bool = Field(
        default=False, description="whether use multimodal."
    )
    base_parser: NodeParser = Field(
        default=SentenceSplitter(chunk_size=500, chunk_overlap=10),
        description="base parser",
    )

    title_stack: List = Field(default_factory=list)
    nodes_list: List[TextNode] = Field(default_factory=list)
    chunk_images_list: List[str] = Field(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "StructuredNodeParser"

    def _cut(self, raw_section: str) -> Iterator[str]:
        # 可能存在单个node 字符数大于chunk_size，此时需要将node进行拆分。拆分元素里不会含有image。
        return [
            f"{self._format_section_header(self.title_stack)}:{chunk}"
            for chunk in self.base_parser.split_text(raw_section)
        ]

    def _format_section_header(self, section_headers):
        return " >> ".join([h.content for h in section_headers])

    def _format_tree_nodes(self, node, doc_node, ref_doc):
        relationships = {NodeRelationship.SOURCE: ref_doc.as_related_node_info()}
        if node.category == "image" and self.enable_multimodal:
            image_node = ImageNode(
                embedding=doc_node.embedding,
                image_url=node.content,
                excluded_embed_metadata_keys=doc_node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=doc_node.excluded_llm_metadata_keys,
                metadata_seperator=doc_node.metadata_seperator,
                metadata_template=doc_node.metadata_template,
                text_template=doc_node.text_template,
                metadata={
                    "image_url": node.content,
                    **doc_node.extra_info,
                },
                relationships=relationships,
            )
            self.nodes_list.append(image_node)
            image_info = ImageInfo(image_url=node.content)
            self.chunk_images_list.append(json.dumps(image_info.__dict__))
            return ""
        if not node.children:
            return node.content
        return node.content + "\n".join(
            [
                self._format_tree_nodes(child, doc_node, ref_doc)
                for child in node.children
            ]
        )

    def _add_text_node(self, chunk_content, doc_node, ref_doc):
        relationships = {NodeRelationship.SOURCE: ref_doc.as_related_node_info()}
        if len(self.chunk_images_list) > 0 and self.enable_multimodal:
            text_node = TextNode(
                text=chunk_content,
                embedding=doc_node.embedding,
                excluded_embed_metadata_keys=doc_node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=doc_node.excluded_llm_metadata_keys,
                metadata_seperator=doc_node.metadata_seperator,
                metadata_template=doc_node.metadata_template,
                text_template=doc_node.text_template,
                metadata={
                    "image_info_list": self.chunk_images_list.copy(),
                    **doc_node.extra_info,
                },
                relationships=relationships,
            )
        else:
            text_node = TextNode(
                text=chunk_content,
                embedding=doc_node.embedding,
                excluded_embed_metadata_keys=doc_node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=doc_node.excluded_llm_metadata_keys,
                metadata_seperator=doc_node.metadata_seperator,
                metadata_template=doc_node.metadata_template,
                text_template=doc_node.text_template,
                meta_data=doc_node.extra_info,
                relationships=relationships,
            )

        self.nodes_list.append(text_node)

    def get_nodes_form_tree(
        self, root: TreeNode, doc_node: BaseNode, ref_doc: Optional[BaseNode] = None
    ):
        ref_doc = ref_doc or doc_node
        # 判断是否可以将整个树节点作为一个chunk
        if root.content_token_count <= self.chunk_size:
            new_chunk_text = self._format_tree_nodes(root, doc_node, ref_doc)
            # 避免插入内容为空的节点
            if len(new_chunk_text) > 0:
                self._add_text_node(new_chunk_text, doc_node, ref_doc)
                self.chunk_images_list.clear()
        else:
            self.traverse_tree(root, doc_node, ref_doc)

        return self.nodes_list.copy()

    def traverse_tree(self, tree_node, doc_node, ref_doc):
        relationships = {NodeRelationship.SOURCE: ref_doc.as_related_node_info()}

        if tree_node.category == "title":
            while self.title_stack and self.title_stack[-1].level >= tree_node.level:
                self.title_stack.pop()
            self.title_stack.append(tree_node)
        elif tree_node.category == "image":
            image_node = ImageNode(
                embedding=doc_node.embedding,
                image_url=tree_node.content,
                excluded_embed_metadata_keys=doc_node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=doc_node.excluded_llm_metadata_keys,
                metadata_seperator=doc_node.metadata_seperator,
                metadata_template=doc_node.metadata_template,
                text_template=doc_node.text_template,
                metadata={
                    "image_url": tree_node.content,
                    **doc_node.extra_info,
                },
                relationships=relationships,
            )
            self.nodes_list.append(image_node)
            image_info = ImageInfo(image_url=tree_node.content)
            self.chunk_images_list.append(json.dumps(image_info.__dict__))

        # 单个节点token数大于chunk_size，则需要将节点进行拆分。拆分元素里不会含有image。
        if not tree_node.children:
            for new_chunk_chunk in self._cut(tree_node.content):
                self._add_text_node(new_chunk_chunk, doc_node, ref_doc)
                self.chunk_images_list.clear()

        tree_tokens = tree_node.content_token_count
        chunk_text = tree_node.content
        for child in tree_node.children:
            if child.category == "image" and self.enable_multimodal:
                image_node = ImageNode(
                    embedding=doc_node.embedding,
                    image_url=child.content,
                    excluded_embed_metadata_keys=doc_node.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=doc_node.excluded_llm_metadata_keys,
                    metadata_seperator=doc_node.metadata_seperator,
                    metadata_template=doc_node.metadata_template,
                    text_template=doc_node.text_template,
                    metadata={
                        "image_url": child.content,
                        **doc_node.extra_info,
                    },
                    relationships=relationships,
                )
                self.nodes_list.append(image_node)
                image_info = ImageInfo(image_url=child.content)
                self.chunk_images_list.append(json.dumps(image_info.__dict__))
                continue
            elif child.category == "title":
                while self.title_stack and self.title_stack[-1].level >= child.level:
                    self.title_stack.pop()
                self.title_stack.append(child)

            if child.content_token_count > self.chunk_size:
                # 避免插入内容为空的节点
                if len(chunk_text) > 0:
                    new_chunk_text = (
                        f"{self._format_section_header(self.title_stack)}:{chunk_text}"
                    )
                    self._add_text_node(new_chunk_text, doc_node, ref_doc)
                    self.chunk_images_list.clear()
                chunk_text = ""
                tree_tokens = 0
                self.traverse_tree(child, doc_node, ref_doc)
            elif tree_tokens + child.content_token_count <= self.chunk_size:
                chunk_text += self._format_tree_nodes(child, doc_node, ref_doc)
                tree_tokens += child.content_token_count
            else:
                if len(chunk_text) > 0:
                    new_chunk_text = (
                        f"{self._format_section_header(self.title_stack)}:{chunk_text}"
                    )
                    self._add_text_node(new_chunk_text, doc_node, ref_doc)
                    self.chunk_images_list.clear()
                chunk_text = self._format_tree_nodes(child, doc_node, ref_doc)
                tree_tokens = child.content_token_count

        # 处理最后的段落
        if len(chunk_text) > 0:
            new_chunk_text = (
                f"{self._format_section_header(self.title_stack)}:{chunk_text}"
            )
            self._add_text_node(new_chunk_text, doc_node, ref_doc)
            self.chunk_images_list.clear()


class MarkdownNodeParser(NodeParser):
    chunk_size: int = Field(default=500, description="chunk size.")
    chunk_overlap_size: int = Field(default=10, description="Chunk overlap size.")
    enable_multimodal: bool = Field(
        default=False, description="whether use multimodal."
    )
    base_parser: NodeParser = Field(
        default=SentenceSplitter(chunk_size=500, chunk_overlap=10),
        description="base parser",
    )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            ast_root = build_markdown_tree(text)
            parser = StructuredNodeParser(
                chunk_size=self.chunk_size,
                chunk_overlap_size=self.chunk_overlap_size,
                enable_multimodal=self.enable_multimodal,
                base_parser=self.base_parser,
            )
            nodes = parser.get_nodes_form_tree(ast_root, node)
            all_nodes.extend(nodes)
        return all_nodes
