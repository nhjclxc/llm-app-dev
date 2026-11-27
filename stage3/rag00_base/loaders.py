

# 文档读取和元数据提取



from dataclasses import dataclass, field
from pathlib import Path


### 4.1 先定义统一文档结构
# 索引前把不同来源统一为一个对象，后续切块和入库只处理这个对象：

@dataclass
class SourceDocument:
    source: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

def read_text_document(path: Path) -> SourceDocument:
    """ 只读取 .md 和 .txt 文档"""
    if path.suffix.lower() not in [".md", ".txt"]:
        raise ValueError(f"{path} is not a .md or .txt file")

    text = path.read_text(encoding="utf-8")
    title = path.stem
    return SourceDocument(
        source=str(path.absolute()),
        title=title,
        text=text,
        metadata={"extention": path.suffix},
    )


