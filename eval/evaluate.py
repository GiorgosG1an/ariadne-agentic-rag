import json
import asyncio
from llama_index.core.evaluation import CorrectnessEvaluator, AnswerRelevancyEvaluator, ContextRelevancyEvaluator, FaithfulnessEvaluator
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.rate_limiter import TokenBucketRateLimiter
from llama_index.core.workflow import Context  # Προσθήκη για διαχείριση state

from google.genai import types

# --- 1. Import το δικό σου RAGWorkflow αντί για το chat_engine ---
from rag_workflow import RAGWorkflow
from prompts import CORRECTNESS_EVAL_TEMPLATE, FAITHFULNESS_EVAL_TEMPLATE, CONTEXT_EVAL_TEMPLATE, ANSWER_RELEVANCY_EVAL_TEMPLATE
from config import settings

rate_limiter = TokenBucketRateLimiter(
    requests_per_minute=100,
    tokens_per_minute=250000
)

judge_llm = GoogleGenAI(
    model='gemini-3-flash-preview',
    api_key=settings.google_api_key,
    temperature=0.0,
    generation_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0
        )
    ),
    rate_limiter=rate_limiter
)

correctness_evaluator = CorrectnessEvaluator(llm=judge_llm, eval_template=CORRECTNESS_EVAL_TEMPLATE)
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm) # eval_template=FAITHFULNESS_EVAL_TEMPLATE
context_relevancy_evaluator = ContextRelevancyEvaluator(llm=judge_llm, eval_template=CONTEXT_EVAL_TEMPLATE)
answer_relevancy_evaluator = AnswerRelevancyEvaluator(llm=judge_llm, eval_template=ANSWER_RELEVANCY_EVAL_TEMPLATE)

async def run_evaluation():
    dataset = []
    with open('eval_datasets/evaluation_dataset_courses.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line:
                continue 
            
            try:
                dataset.append(json.loads(clean_line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {e}")

    all_results = []
    total_correctness_score = 0
    total_faithfulness_score = 0
    total_context_relevancy_score = 0
    total_answer_relevancy = 0

    for i, item in enumerate(dataset):
        query = item["question"]
        topic = item["topic"]
        complexity = item["complexity"]
        expected_answer = item["expected_answer"]
        
        print(f"\nΑξιολόγηση Ερωτήματος [{i+1}/{len(dataset)}]: {query}")
        max_retries = 4
        success = False
        for attempt in range(max_retries):
            try:
                # --- 2. Δημιουργία φρέσκου Workflow & Context για κάθε ερώτηση ---
                # Χρησιμοποιούμε μοναδικό session_id για να έχουμε καθαρή μνήμη ανά evaluation
                workflow = RAGWorkflow(session_id=f"eval_session_{i}", timeout=120)
                ctx = Context(workflow)

                # Εκτέλεση του workflow (περνώντας το context για να κρατήσουμε το state)
                response_generator = await workflow.run(ctx=ctx, user_msg=query)

                # --- 3. Κατανάλωση του Streaming Generator ---
                generated_answer = ""
                async for chunk in response_generator:
                    generated_answer += chunk.delta

                # --- 4. Εξαγωγή των Retrieved Contexts μέσα από το Workflow State ---
                contexts = await ctx.store.get("retrieved_texts", default=[])

                # --- 5. Ασύγχρονη Αξιολόγηση (χρησιμοποιούμε .aevaluate αντί για .evaluate) ---
                correctness_result = await correctness_evaluator.aevaluate(
                    query=query,
                    response=generated_answer,
                    reference=expected_answer,
                )

                if correctness_result.score is not None:
                    total_correctness_score += correctness_result.score

                faithfulness_result = await faithfulness_evaluator.aevaluate(
                    query=query,
                    response=generated_answer,
                    contexts=contexts,
                )

                if faithfulness_result.score is not None:
                    total_faithfulness_score += faithfulness_result.score

                context_rel_res = await context_relevancy_evaluator.aevaluate(
                    query=query, 
                    contexts=contexts,
                    sleep_time_in_seconds=1,
                )
                if context_rel_res.score is not None:
                    total_context_relevancy_score += context_rel_res.score

                answer_rel_res = await answer_relevancy_evaluator.aevaluate(
                    query=query, 
                    response=generated_answer, 
                    contexts=contexts,
                )

                if answer_rel_res.score is not None:
                    total_answer_relevancy += answer_rel_res.score

                # Καταγραφή αποτελεσμάτων
                record = {
                    "topic": topic,
                    "complexity": complexity,
                    "query": query,
                    "expected_answer": expected_answer,
                    "generated_response": generated_answer,
                    "retrieved_contexts": contexts,
                    "scores": {
                        "correctness": correctness_result.score,
                        "faithfulness": faithfulness_result.score,
                        "context_relevancy": context_rel_res.score,
                        "answer_relevancy": answer_rel_res.score
                    },
                    "passing": {
                        "correctness": correctness_result.passing,
                        "faithfulness": faithfulness_result.passing
                    },
                    "feedback": {
                        "correctness": correctness_result.feedback,
                        "faithfulness": faithfulness_result.feedback,
                        "context_relevancy": context_rel_res.feedback,
                        "answer_relevancy": answer_rel_res.feedback
                    }
                }

                all_results.append(record)
                success = True

                break

            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg or "429" in error_msg or "Unavailable" in error_msg:
                    wait_time = 30 * (attempt + 1) # Περιμένει 30s, μετά 60s, μετά 90s...
                    print(f" Error (503/429). Attempt: {attempt+1}/{max_retries}. Waiting for {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Error at query {i+1}: {e}")
                    break 
        if not success:
            print(f"❌ [ΑΠΟΤΥΧΙΑ] Το ερώτημα {i+1} εγκαταλείφθηκε μετά από {max_retries} προσπάθειες λόγω Google API.")
        # --- Εκτύπωση Στοιχείων ---
        print(f"\n{'='*70}")
        print(f"Correctness: {correctness_result.score}/5.0 | Passing: {correctness_result.passing}")
        print(f"Σχόλιο: {correctness_result.feedback.strip() if correctness_result.feedback else '-'}")
        
        print(f"\nFaithfulness: {faithfulness_result.score} | Passing: {faithfulness_result.passing}")
        print(f"Σχόλιο: {faithfulness_result.feedback.strip() if faithfulness_result.feedback else '-'}")
        
        print(f"\nContext Relevancy: {context_rel_res.score}")
        print(f"Σχόλιο: {context_rel_res.feedback.strip() if context_rel_res.feedback else '-'}")
        
        print(f"\nAnswer Relevancy: {answer_rel_res.score}")
        print(f"Σχόλιο: {answer_rel_res.feedback.strip() if answer_rel_res.feedback else '-'}")
        print(f"{'-'*70}\n")
        
        if (i + 1) % 10 == 0:
            print(f"\n⏳ Ολοκληρώθηκαν {i+1} ερωτήματα. Προληπτικό διάλειμμα 2 λεπτών (Cool-down API)...\n")
            await asyncio.sleep(120)
        else:
            await asyncio.sleep(2)

    output_filename = "eval_results/rag_evaluation_courses.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\n--- Συνολικός Μέσος Όρος Correctness: {total_correctness_score / len(dataset):.2f} / 5.0 ---")
    print(f"\n--- Συνολικός Μέσος Όρος Faithfulness: {total_faithfulness_score / len(dataset):.2f} / 1.0 ---")
    print(f"\n--- Συνολικός Μέσος Όρος Context Relevancy: {total_context_relevancy_score / len(dataset):.2f} / 1.0 ---")
    print(f"\n--- Συνολικός Μέσος Όρος Answer Relevancy: {total_answer_relevancy / len(dataset):.2f} / 1.0 ---")

    print(f"--- Τα αναλυτικά αποτελέσματα αποθηκεύτηκαν στο: {output_filename} ---")

# --- 6. Εκκίνηση του Ασύγχρονου Event Loop ---
if __name__ == "__main__":
    asyncio.run(run_evaluation())