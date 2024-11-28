<p align="center">
    <h1>PAI-RAG: 一个易于使用的模块化RAG框架 </h1>
</p>

[![PAI-RAG CI](https://github.com/aigc-apps/PAI-RAG/actions/workflows/main.yml/badge.svg)](https://github.com/aigc-apps/PAI-RAG/actions/workflows/main.yml)

<details open>
<summary></b>📕 目录</b></summary>

- 💡 [什么是PAI-RAG?](#什么是pai-rag)
- 🌟 [主要模块和功能](#主要模块和功能)
- 🔎 [快速开始](#快速开始)
  - [本地环境](#方式一本地环境)
  - [Docker镜像](#方式二docker镜像)
- 🔧 [文档](#文档)

</details>

# 💡 什么是PAI-RAG?

PAI-RAG 是一个易于使用的模块化 RAG（检索增强生成）开源框架，结合 LLM（大型语言模型）提供真实问答能力，支持 RAG 系统各模块灵活配置和定制开发，为基于阿里云人工智能平台（PAI）的任何规模的企业提供生产级的 RAG 系统。

# 🌟 主要模块和功能

![framework](docs/figures/framework.jpg)

- 模块化设计，灵活可配置
- 功能丰富，包括Agentic RAG, 多模态问答和nl2sql等
- 基于社区开源组件构建，定制化门槛低
- 多维度自动评估体系，轻松掌握各模块性能质量
- 集成全链路可观测和评估可视化工具
- 交互式UI/API调用，便捷的迭代调优体验
- 阿里云快速场景化部署/镜像自定义部署/开源私有化部署

# 🔎 快速开始

## 方式一：本地环境

1. 克隆仓库

   ```bash
   git clone git@github.com:aigc-apps/PAI-RAG.git
   ```

2. 配置开发环境

   本项目使用poetry进行管理，若在本地环境下使用，建议在安装环境之前先创建一个空环境。为了确保环境一致性并避免因Python版本差异造成的问题，我们指定Python版本为3.11。

   ```bash
   conda create -n rag_env python==3.11
   conda activate rag_env
   ```

   如果使用macOS且需要处理PPTX文件，需要下载依赖库处理PPTX文件

   ```bash
   brew install mono-libgdiplus
   ```

   直接使用poetry安装项目依赖包：

   ```bash
    pip install poetry
    poetry install
    poetry run aliyun-bootstrap -a install
   ```

- 常见网络超时问题

  注：在安装过程中，若遇到网络连接超时的情况，可以添加阿里云或清华的镜像源，在 pyproject.toml 文件末尾追加以下几行：

  ```bash
  [[tool.poetry.source]]
  name = "mirrors"
  url = "http://mirrors.aliyun.com/pypi/simple/" # 阿里云
  # url = "https://pypi.tuna.tsinghua.edu.cn/simple/" # 清华
  priority = "default"
  ```

  之后，再依次执行以下命令：

  ```bash
  poetry lock
  poetry install
  ```

3. 下载其他模型到本地

   ```bash
   # 支持 model name (默认 ""), 没有参数时, 默认下载上述所有模型。
   load_model [--model-name MODEL_NAME]
   ```

4. 启动RAG服务

   使用DashScope API，需要在命令行引入环境变量

   ```bash
   export DASHSCOPE_API_KEY=""
   ```

   启动:

   ```bash
   # 启动，支持自定义host(默认0.0.0.0), port(默认8001), config(默认src/pai_rag/config/settings.yaml), skip-download-models(不加为False)
   # 默认启动时下载模型 [bge-large-zh-v1.5, easyocr] , 可设置 skip-download-models 避免启动时下载模型.
   # 可使用命令行 "load_model" 下载模型 including [bge-large-zh-v1.5, easyocr, SGPT-125M-weightedmean-nli-bitfit, bge-large-zh-v1.5, bge-m3, bge-reranker-base, bge-reranker-large, paraphrase-multilingual-MiniLM-L12-v2, qwen_1.8b, text2vec-large-chinese]
   pai_rag serve [--host HOST] [--port PORT] [--config CONFIG_FILE] [--skip-download-models]
   ```

   ```bash
   pai_rag serve
   ```

5. 启动RAG WebUI

   ```bash
   # 启动，支持自定义host(默认0.0.0.0), port(默认8002), config(默认localhost:8001)
   pai_rag ui [--host HOST] [--port PORT] [rag-url RAG_URL]
   ```

   你也可以打开http://127.0.0.1:8002/ 来配置RAG服务以及上传本地数据。

6. 【可选】本地工具-上传数据

   向当前索引存储中插入data_path路径下的新文件

   ```bash
   load_data -c src/pai_rag/config/settings.yaml -d data_path -p pattern
   ```

   path examples:

   ```
   a. load_data -d test/example
   b. load_data -d test/example_data/pai_document.pdf
   c. load_data -d test/example_data -p *.pdf

   ```

## 方式二：Docker镜像

为了更方便使用，节省较长时间的环境安装问题，我们也提供了直接基于镜像启动的方式。

1. 配置环境变量

   ```bash
   cd docker
   cp .env.example .env
   ```

   如果你需要使用dashscope api或者OSS存储，可以根据需要修改.env中的环境变量。

2. 启动

```bash
docker-compose up -d
```

3. 打开浏览器中的http://localhost:8000 访问web ui.

# 🔧 文档

## API服务

可以直接通过API服务调用RAG能力（上传数据，RAG查询，检索，NL2SQL, Function call等等）。更多细节可以查看[API文档](./docs/api_zh.md)

## Agentic RAG

您也可以在PAI-RAG中使用支持API function calling功能的Agent，请参考文档：
[Agentic RAG](./docs/agentic_rag.md)

## Data Analysis

您可以在PAI-RAG中使用支持数据库和表格文件的数据分析功能，请参考文档：[Data Analysis](./docs/data_analysis_doc.md)

## 参数配置

如需实现更多个性化配置，请参考文档：

[参数配置说明](./docs/config_guide_cn.md)

## 支持文件类型

| 文件类型 | 文件格式                               |
| -------- | -------------------------------------- |
| 非结构化 | .txt, .docx， .pdf， .html，.pptx，.md |
| 图片     | .gif， .jpg，.png，.jpeg， .webp       |
| 结构化   | .csv，.xls， .xlsx，.jsonl             |
| 其他     | .epub，.mbox，.ipynb                   |

1. .doc格式文档需转化为.docx格式
2. .ppt和.pptm格式需转化为.pptx格式
