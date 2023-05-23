# 基于本地知识库的 ChatGLM 等大语言模型应用实现(只接收文本资料和文本提问)

## Fork说明

本项目fork自 [imClumsyPanda](https://github.com/imClumsyPanda) 主导的开源项目[langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM), 更详细的说明见[上游项目README](https://github.com/imClumsyPanda/langchain-ChatGLM/blob/master/README.md)

## 基于上游项目修改情况

- 除readme文件外的修改在 **分支 `text` 下完成**
- 删除原项目中的 paddlepaddle及paddleocr库依赖，并删除 `local_doc_qa.py` 中加载 `pdf/image` 部分代码, 因此创建资料库时pdf,image文件会读取不到
- `local_doc_qa.py` 中添加 读取 `html` 功能
- `local_doc_qa.py` 中 `configs.model_config` 读取模块被修改
- 删除 REANME_en.md