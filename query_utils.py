"""Query handling helpers."""


def classify_query(query: str, min_length: int = 10, max_length: int = 100) -> str:
    """Return a message describing whether the query length is acceptable."""
    query = query.strip()
    if not query:
        return "Запрос пуст."
    if len(query) < min_length:
        return f"Запрос слишком короткий. Длина запроса: {len(query)} символов."
    if len(query) > max_length:
        return f"Запрос слишком длинный. Длина запроса: {len(query)} символов."
    return "Запрос корректный."
