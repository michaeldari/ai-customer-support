import json
import os
import joblib
import logging
from openai import OpenAI
from src.api.main import predict_triage, models
from src.rag.engine import get_rag_response
from src.utils.config import settings

llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)


def evaluate_rag():
    logging.info("Loading models for evaluation...")
    models["cat"] = joblib.load(
        os.path.join(settings.ARTIFACTS_DIR, "category_model.joblib")
    )
    models["pri"] = joblib.load(
        os.path.join(settings.ARTIFACTS_DIR, "priority_model.joblib")
    )

    eval_path = os.path.join(settings.DATA_DIR, "eval_questions.jsonl")
    results = []

    with open(eval_path, "r") as f:
        for line in f:
            case = json.loads(line)

            ticket = {"subject": case["ticket_subject"], "body": case["ticket_body"]}

            # Mocking the request object for the internal function
            class MockTicket:
                subject = ticket["subject"]
                body = ticket["body"]

            triage = predict_triage(MockTicket())
            response = get_rag_response(case["user_question"], triage)

            # LLM-as-a-Judge Evaluation
            eval_prompt = f"""
            Rate the following support response based on:
            1. Groundedness (0-1): Is it based ONLY on the provided citations?
            2. Relevance (0-1): Does it actually answer the user question?
            
            Question: {case["user_question"]}
            Response: {response["draft_reply"]}
            Citations: {response["citations"]}
            
            Return JSON: {{"groundedness": score, "relevance": score, "reasoning": "..."}}
            """

            eval_res = llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": eval_prompt}],
                response_format={"type": "json_object"},
            )
            scores = json.loads(eval_res.choices[0].message.content)

            results.append(
                {
                    "id": case["id"],
                    "scores": scores,
                    "response": response["draft_reply"],
                }
            )

    # Generate Markdown Report
    with open("reports/rag_eval.md", "w") as f:
        f.write("# RAG Evaluation Report\n\n")
        avg_groundedness = sum(r["scores"]["groundedness"] for r in results) / len(
            results
        )
        f.write(f"- **Avg Groundedness:** {avg_groundedness:.2f}\n")
        f.write("## Failure Analysis\n- Instances of hallucination occurred when...check FINAL_ANALYSIS.md\n")


if __name__ == "__main__":
    evaluate_rag()
