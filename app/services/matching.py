from typing import Dict, Any, List


CRITICAL_PARAMETERS: Dict[str, float] = {
    # Example weights; in real use tune per domain
    "напряжение": 3.0,
    "ток": 3.0,
    "частота": 1.0,
    "класс_защиты": 2.0,
}


def calculate_match_score(target_char: Dict[str, Any], candidate_char: Dict[str, Any]) -> float:
    score = 0.0
    total = 0.0
    for param, weight in CRITICAL_PARAMETERS.items():
        total += weight
        if param in target_char and param in candidate_char:
            if str(target_char[param]).strip().lower() == str(candidate_char[param]).strip().lower():
                score += weight
    return (score / total) if total > 0 else 0.0


def find_analogs(target_char: Dict[str, Any], catalog: List[Dict[str, Any]], threshold: float = 0.5, limit: int = 6) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for product in catalog:
        candidate_char = product.get("characteristics", {})
        score = calculate_match_score(target_char, candidate_char)
        if score >= threshold:
            product_copy = dict(product)
            product_copy["score"] = score
            scored.append(product_copy)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


