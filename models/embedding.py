from pathlib import Path
from faiss import IndexFlatIP
from .load_model import load_model

class Embedding():
    def __init__(self):
        self.model = load_model("embed")
    
    def embed(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def save_index(self, index: IndexFlatIP, path: Path) -> None:
        index.save(str(path))

    def load_index(self, path: Path) -> IndexFlatIP:
        return IndexFlatIP.load(str(path))
