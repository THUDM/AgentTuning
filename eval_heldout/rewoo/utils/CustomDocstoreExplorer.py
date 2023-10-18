from typing import List
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore.base import Docstore


# Rewrite Docstore lookup to match original implementation from author.
class CustomDocstoreExplorer(DocstoreExplorer):
    def __init__(self, docstore: Docstore):
        super().__init__(docstore)
        self.lookup_str = ""
        self.lookup_index = 0

    def lookup(self, term: str) -> str:
        """Lookup a term in document (if saved)."""
        if self.document is None:
            raise ValueError("Cannot lookup without a successful search first")
        if term.lower() != self.lookup_str:
            self.lookup_str = term.lower()
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self._sentence if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"

    @property
    def _sentence(self) -> List[str]:
        if self.document is None:
            raise ValueError("Cannot get paragraphs without a document")
        return self.document.page_content.split(".")
