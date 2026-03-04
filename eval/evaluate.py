"""
RAG Evaluation Pipeline — Offline Quality Assessment

Usage:
  python eval/evaluate.py                    # Full evaluation (50 questions)
  python eval/evaluate.py --quick            # Quick run (10 questions)
  python eval/evaluate.py --category factual # Single category
  python eval/evaluate.py --threshold 0.80   # Custom pass threshold
"""

import json, os, sys, time, argparse, re
from typing import Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import anthropic
from app.ingestion import ingest_text, get_doc_count, clear_documents


@dataclass
class EvalResult:
    question_id: str; category: str; question: str; expected_answer: str
    actual_answer: str; required_facts: list; facts_found: list; facts_missing: list
    fact_recall: float; has_citation: bool; is_refusal: bool; expected_refusal: bool
    refusal_correct: bool; faithfulness_score: float; relevancy_score: float
    latency_seconds: float; retrieval_attempts: int; grounding_confidence: float
    pipeline_trace: list = field(default_factory=list)

@dataclass
class EvalSummary:
    total_questions: int; avg_faithfulness: float; avg_relevancy: float
    avg_fact_recall: float; citation_coverage: float; refusal_accuracy: float
    avg_latency: float; pass_fail: str; category_scores: dict; failed_questions: list


FAITHFULNESS_PROMPT = """You are an evaluation judge. Assess whether the generated answer is faithful to the provided context — every claim must be supported.

Context: {context}
Question: {question}
Generated Answer: {answer}

Score 0.0-1.0 (1.0=every claim supported, 0.0=fabricated).
Respond ONLY with JSON: {{"score": 0.85, "reasoning": "one sentence"}}"""

RELEVANCY_PROMPT = """Assess whether the answer addresses the question.

Question: {question}
Answer: {answer}
Expected: {expected}

Score 0.0-1.0 (1.0=completely answers, 0.0=irrelevant).
Respond ONLY with JSON: {{"score": 0.85, "reasoning": "one sentence"}}"""


class RAGEvaluator:
    def __init__(self, dataset_path=None):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.judge_model = "claude-sonnet-4-20250514"
        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
        with open(dataset_path) as f:
            self.dataset = json.load(f)
        self.thresholds = self.dataset["metadata"]["quality_thresholds"]
        self.results = []

    def _call_judge(self, prompt):
        try:
            resp = self.client.messages.create(model=self.judge_model, max_tokens=200,
                messages=[{"role": "user", "content": prompt}])
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1].replace("json", "").strip()
            return json.loads(text)
        except Exception:
            return {"score": 0.5, "reasoning": "Judge evaluation inconclusive"}

    def ingest_test_documents(self):
        clear_documents()
        for doc in self.dataset["test_documents"]:
            ingest_text(doc["content"], source=doc["source"])
        print(f"Ingested {len(self.dataset['test_documents'])} docs ({get_doc_count()} chunks)")

    def check_facts(self, answer, required_facts):
        answer_lower = answer.lower()
        found, missing = [], []
        for fact in required_facts:
            if fact == "REFUSE":
                refusal = ["don't have", "do not have", "not in", "no information",
                    "cannot find", "not mentioned", "not available", "doesn't contain",
                    "does not contain", "unable to find", "no data", "not provided"]
                (found if any(p in answer_lower for p in refusal) else missing).append(fact)
            else:
                f = fact.lower().replace(",", "").replace("$", "")
                a = answer_lower.replace(",", "").replace("$", "")
                (found if f in a else missing).append(fact)
        return found, missing

    def check_citation(self, answer):
        patterns = [r"\[source:", r"\[Source:", r"according to", r"from the",
                    r"based on the", r"the document", r"the report", r"the handbook"]
        return any(re.search(p, answer) for p in patterns)

    def check_refusal(self, answer):
        phrases = ["don't have information", "do not have information", "not in the documents",
            "no information about that", "cannot find", "not mentioned", "I don't have",
            "unable to find", "not provided in", "no relevant documents"]
        return any(p in answer.lower() for p in phrases)

    async def evaluate_single(self, qa, agent):
        start = time.time()
        result = await agent.query(qa["question"])
        latency = time.time() - start
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        metrics = result.get("metrics", {})
        hallucination = result.get("hallucination_check", {})

        found, missing = self.check_facts(answer, qa["required_facts"])
        fact_recall = len(found) / len(qa["required_facts"]) if qa["required_facts"] else 1.0
        has_citation = self.check_citation(answer)
        is_refusal = self.check_refusal(answer)
        expected_refusal = qa["category"] == "unanswerable"
        refusal_correct = (is_refusal == expected_refusal)

        if expected_refusal:
            faithfulness = 1.0 if is_refusal else 0.0
        else:
            ctx = "\n".join([s.get("source", "") for s in sources]) if sources else "No context"
            faithfulness = self._call_judge(FAITHFULNESS_PROMPT.format(
                context=ctx, question=qa["question"], answer=answer)).get("score", 0.5)

        if expected_refusal and is_refusal:
            relevancy = 1.0
        else:
            relevancy = self._call_judge(RELEVANCY_PROMPT.format(
                question=qa["question"], answer=answer, expected=qa["expected_answer"])).get("score", 0.5)

        return EvalResult(question_id=qa["id"], category=qa["category"], question=qa["question"],
            expected_answer=qa["expected_answer"], actual_answer=answer,
            required_facts=qa["required_facts"], facts_found=found, facts_missing=missing,
            fact_recall=fact_recall, has_citation=has_citation, is_refusal=is_refusal,
            expected_refusal=expected_refusal, refusal_correct=refusal_correct,
            faithfulness_score=faithfulness, relevancy_score=relevancy,
            latency_seconds=round(latency, 2), retrieval_attempts=metrics.get("retrieval_attempts", 0),
            grounding_confidence=hallucination.get("confidence", 0),
            pipeline_trace=result.get("pipeline_trace", []))

    async def run_evaluation(self, category=None, max_questions=None):
        from app.agent import AgenticRAG
        agent = AgenticRAG()
        qa_pairs = self.dataset["qa_pairs"]
        if category: qa_pairs = [q for q in qa_pairs if q["category"] == category]
        if max_questions: qa_pairs = qa_pairs[:max_questions]

        print(f"\nRunning evaluation: {len(qa_pairs)} questions")
        print("=" * 60)
        self.results = []
        for i, qa in enumerate(qa_pairs):
            print(f"  [{i+1}/{len(qa_pairs)}] {qa['category']:15s} | {qa['question'][:50]}...", end="")
            r = await self.evaluate_single(qa, agent)
            self.results.append(r)
            s = "PASS" if r.fact_recall >= 0.5 and r.refusal_correct else "FAIL"
            print(f" | facts={r.fact_recall:.0%} faith={r.faithfulness_score:.2f} {s}")
        return self.generate_summary()

    def generate_summary(self):
        if not self.results: return None
        answerable = [r for r in self.results if not r.expected_refusal]
        unanswerable = [r for r in self.results if r.expected_refusal]

        avg_faith = sum(r.faithfulness_score for r in answerable) / len(answerable) if answerable else 0
        avg_relev = sum(r.relevancy_score for r in answerable) / len(answerable) if answerable else 0
        avg_recall = sum(r.fact_recall for r in answerable) / len(answerable) if answerable else 0
        cit_cov = sum(1 for r in answerable if r.has_citation) / len(answerable) if answerable else 0
        ref_acc = sum(1 for r in unanswerable if r.refusal_correct) / len(unanswerable) if unanswerable else 0
        avg_lat = sum(r.latency_seconds for r in self.results) / len(self.results)

        categories = {}
        for cat in ["factual", "reasoning", "unanswerable", "multi_source", "ambiguous"]:
            cr = [r for r in self.results if r.category == cat]
            if not cr: continue
            if cat == "unanswerable":
                categories[cat] = {"count": len(cr), "refusal_accuracy": sum(1 for r in cr if r.refusal_correct) / len(cr)}
            else:
                categories[cat] = {"count": len(cr),
                    "avg_faithfulness": round(sum(r.faithfulness_score for r in cr) / len(cr), 3),
                    "avg_fact_recall": round(sum(r.fact_recall for r in cr) / len(cr), 3),
                    "avg_relevancy": round(sum(r.relevancy_score for r in cr) / len(cr), 3)}

        failed = [{"id": r.question_id, "category": r.category, "question": r.question[:60],
            "fact_recall": r.fact_recall, "faithfulness": r.faithfulness_score,
            "refusal_correct": r.refusal_correct, "missing_facts": r.facts_missing}
            for r in self.results if r.fact_recall < 0.5 or not r.refusal_correct or r.faithfulness_score < 0.5]

        passes = (avg_faith >= self.thresholds["faithfulness_min"]
            and avg_relev >= self.thresholds["answer_relevancy_min"]
            and cit_cov >= self.thresholds["citation_coverage_min"]
            and (ref_acc >= self.thresholds["refusal_accuracy_min"] if unanswerable else True))

        return EvalSummary(total_questions=len(self.results), avg_faithfulness=round(avg_faith, 3),
            avg_relevancy=round(avg_relev, 3), avg_fact_recall=round(avg_recall, 3),
            citation_coverage=round(cit_cov, 3), refusal_accuracy=round(ref_acc, 3),
            avg_latency=round(avg_lat, 2), pass_fail="PASS" if passes else "FAIL",
            category_scores=categories, failed_questions=failed)

    def print_report(self, summary):
        print("\n" + "=" * 60)
        print("  RAG EVALUATION REPORT")
        print("=" * 60)
        print(f"\n  VERDICT: {summary.pass_fail}")
        print(f"  Questions: {summary.total_questions} | Avg latency: {summary.avg_latency}s")
        print(f"\n  {'Metric':<25s} {'Score':>8s} {'Threshold':>10s} {'Status':>8s}")
        print(f"  {'-'*55}")
        for name, score, thr in [
            ("Faithfulness", summary.avg_faithfulness, self.thresholds["faithfulness_min"]),
            ("Answer Relevancy", summary.avg_relevancy, self.thresholds["answer_relevancy_min"]),
            ("Fact Recall", summary.avg_fact_recall, 0.70),
            ("Citation Coverage", summary.citation_coverage, self.thresholds["citation_coverage_min"]),
            ("Refusal Accuracy", summary.refusal_accuracy, self.thresholds["refusal_accuracy_min"])]:
            print(f"  {name:<25s} {score:>7.1%} {thr:>9.0%} {'PASS' if score >= thr else 'FAIL':>8s}")
        print(f"\n  CATEGORY BREAKDOWN:")
        for cat, scores in summary.category_scores.items():
            print(f"    {cat}: {scores}")
        if summary.failed_questions:
            print(f"\n  FAILED ({len(summary.failed_questions)}):")
            for fq in summary.failed_questions[:10]:
                print(f"    {fq['id']}: {fq['question']} | missing={fq['missing_facts']}")
        print("=" * 60)

    def save_results(self, summary, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        sp = os.path.join(output_dir, f"eval_summary_{ts}.json")
        with open(sp, "w") as f:
            json.dump({"timestamp": ts, "verdict": summary.pass_fail,
                "total_questions": summary.total_questions,
                "metrics": {"faithfulness": summary.avg_faithfulness, "relevancy": summary.avg_relevancy,
                    "fact_recall": summary.avg_fact_recall, "citation_coverage": summary.citation_coverage,
                    "refusal_accuracy": summary.refusal_accuracy, "avg_latency": summary.avg_latency},
                "thresholds": self.thresholds, "category_scores": summary.category_scores,
                "failed_questions": summary.failed_questions}, f, indent=2)
        dp = os.path.join(output_dir, f"eval_details_{ts}.json")
        with open(dp, "w") as f:
            json.dump([{"id": r.question_id, "category": r.category, "question": r.question,
                "expected": r.expected_answer, "actual": r.actual_answer,
                "fact_recall": r.fact_recall, "facts_found": r.facts_found,
                "facts_missing": r.facts_missing, "faithfulness": r.faithfulness_score,
                "relevancy": r.relevancy_score, "has_citation": r.has_citation,
                "refusal_correct": r.refusal_correct, "latency": r.latency_seconds,
                "grounding_confidence": r.grounding_confidence} for r in self.results], f, indent=2)
        print(f"\n  Results saved: {sp}")
        return sp


async def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick run (10 questions)")
    parser.add_argument("--category", type=str, help="Evaluate single category")
    parser.add_argument("--threshold", type=float, help="Override faithfulness threshold")
    parser.add_argument("--save", action="store_true", default=True, help="Save results")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY required"); sys.exit(1)

    evaluator = RAGEvaluator()
    if args.threshold:
        evaluator.thresholds["faithfulness_min"] = args.threshold

    print("Ingesting test documents...")
    evaluator.ingest_test_documents()

    summary = await evaluator.run_evaluation(
        category=args.category, max_questions=10 if args.quick else None)
    evaluator.print_report(summary)
    if args.save: evaluator.save_results(summary)

    if summary.pass_fail == "FAIL":
        print("\nEVALUATION FAILED"); sys.exit(1)
    else:
        print("\nEVALUATION PASSED"); sys.exit(0)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
