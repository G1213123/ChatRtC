"""Ask a question to the notion database."""
import re
from typing import Any, Dict, List

from langchain.chains import RetrievalQAWithSourcesChain


class RetrievalQAWithClausesSourcesChain (RetrievalQAWithSourcesChain):
    clauses_key: str = "clauses"  #: :meta private:

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        _output_keys = [self.answer_key, self.clauses_key, self.sources_answer_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"]
        return _output_keys

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        docs = self._get_docs( inputs )
        answer, _ = self.combine_documents_chain.combine_docs( docs, **inputs )
        if len(re.findall( r"SOURCES:\s", answer ))==1:
            answer, sources = re.split( r"SOURCES:\s", answer )
        else:
            sources = ""
        if len(re.findall( r"CLAUSES:\s", answer ))==1:
            answer, clauses = re.split( r"CLAUSES:\s", answer )
        else:
            clauses = ""
        result: Dict[str, Any] = {
            self.answer_key: answer,
            self.sources_answer_key: sources,
            self.clauses_key:clauses
        }
        if self.return_source_documents:
            result["source_documents"] = docs
        return result