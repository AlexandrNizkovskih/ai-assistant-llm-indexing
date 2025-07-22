"""Document preprocessing utilities."""

import re
import pandas as pd
from llama_index.core import Document


def extract_structured_data(text: str):
    """Extract simple table-like structures from text."""
    tables = []
    table_pattern = re.compile(r"(\d+(?:\.\d+)?(?:\s+|,|\t)\d+(?:\.\d+)?(?:\s+|,|\t)\d+(?:\.\d+)?(?:\s+|,|\t)\d+)")
    for match in table_pattern.finditer(text):
        table_data = match.group(0)
        columns = re.split(r"\s+|,|\t", table_data)
        tables.append(columns)

    if tables:
        return pd.DataFrame(tables)
    return None


def preprocess_document(doc: Document):
    """Filter and enrich documents for indexing."""
    if len(doc.text) < 100:
        return None

    structured_data = extract_structured_data(doc.text)
    if isinstance(structured_data, pd.DataFrame):
        structured_data = structured_data.to_dict(orient="list")

    return Document(text=doc.text, metadata={"structured_data": structured_data})
