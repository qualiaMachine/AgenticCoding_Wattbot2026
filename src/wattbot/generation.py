from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import Settings, require_model_settings
from .extraction import Chunk


ANSWER_FIELDS = ("answer", "answer_value", "answer_unit", "ref_id", "ref_url", "supporting_materials", "explanation")


def generate_answer(
    question_row: dict[str, str],
    contexts: list[tuple[Chunk, float]],
    metadata_by_id: dict[str, dict[str, str]],
    settings: Settings,
) -> dict[str, str]:
    require_model_settings(settings)
    evidence = "\n\n".join(
        f"[source={chunk.document_id}; page={chunk.page}]\n{chunk.text}" for chunk, _ in contexts
    )
    prompt = f"""Answer the question using only the supplied source excerpts.
Return a JSON object with exactly these strings: {", ".join(ANSWER_FIELDS)}.
answer_value must be a normalized number, a parenthesized stated range, an exact categorical/boolean value,
or is_blank. Cite only source IDs supplied in excerpts. supporting_materials must quote the excerpt verbatim
and identify page(s). If evidence is insufficient, use answer_value, answer_unit, ref_id, ref_url, and
supporting_materials as is_blank; answer must refuse and explanation must explain the abstention.

Question: {question_row["question"]}

Source excerpts:
{evidence}"""
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": "You are a careful evidence extraction system. Never invent facts or citations."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Answer model returned an empty response")
    return normalize_prediction(json.loads(content), question_row, metadata_by_id)


def normalize_prediction(
    generated: dict[str, Any], question_row: dict[str, str], metadata_by_id: dict[str, dict[str, str]]
) -> dict[str, str]:
    output = {"id": question_row["id"], "question": question_row["question"]}
    for field in ANSWER_FIELDS:
        output[field] = str(generated.get(field, "")).strip()
    if not output["explanation"]:
        output["explanation"] = "The retrieved evidence was evaluated against the question."
    if output["answer_value"].lower() == "is_blank":
        output.update(
            answer=output["answer"] or "Unable to answer with confidence based on the provided documents.",
            answer_unit="is_blank",
            ref_id="is_blank",
            ref_url="is_blank",
            supporting_materials="is_blank",
        )
    else:
        cited_ids = [item.strip() for item in output["ref_id"].replace(";", ",").split(",") if item.strip()]
        invalid_ids = sorted(set(cited_ids) - set(metadata_by_id))
        if invalid_ids:
            raise ValueError(f"Model cited source IDs outside metadata: {', '.join(invalid_ids)}")
        output["ref_url"] = ", ".join(_source_url(metadata_by_id[item]) for item in cited_ids)
    return output


def _source_url(row: dict[str, str]) -> str:
    return next((row[column] for column in ("url", "ref_url", "pinned_url", "source_url") if row.get(column)), "")

