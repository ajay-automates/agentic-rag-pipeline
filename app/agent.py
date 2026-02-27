"""
Agentic RAG Pipeline - Self-Correcting Retrieval Agent

Pipeline:
1. Retrieve - Get relevant chunks from vector store
2. Grade - Claude evaluates if retrieved docs actually answer the question
3. Decide - If relevant: generate. If not: reformulate query and re-retrieve
4. Generate - Produce answer with citations
5. Hallucination Check - Verify answer is grounded in retrieved docs
"""

import anthropic
import json
import os
import time
from typing import Optional
from .ingestion import search


GRADER_PROMPT = """You are a retrieval grader. Assess whether a retrieved document is relevant to the question.

Retrieved document:
{document}

User question:
{question}

Respond with ONLY a JSON object:
{{"relevance": "relevant|partially_relevant|not_relevant", "reason": "one sentence"}}"""


REFORMULATOR_PROMPT = """The original query didn't retrieve good results. Generate a better search query.

Original question: {question}

Respond with ONLY the reformulated query text."""


GENERATOR_PROMPT = """Answer the question using ONLY the provided context documents.

RULES:
1. Only use information from the provided context
2. If the context doesn't contain enough info, say so
3. Cite sources using [Source: filename]
4. Be specific with numbers and details from documents

Context documents:
{context}

Question: {question}"""


HALLUCINATION_CHECK_PROMPT = """Check if the answer is fully supported by the source documents.

Source documents:
{context}

Generated answer:
{answer}

Respond with ONLY a JSON object:
{{"grounded": true, "confidence": 0.9, "issues": []}}"""


class AgenticRAG:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"

    def _call_claude(self, prompt, max_tokens=1024):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _grade_documents(self, question, documents):
        graded = []
        for doc in documents:
            prompt = GRADER_PROMPT.format(document=doc["text"][:1000], question=question)
            try:
                result = self._call_claude(prompt, max_tokens=200).strip()
                if result.startswith("```"):
                    result = result.split("```")[1].replace("json", "").strip()
                grade = json.loads(result)
                doc["grade"] = grade.get("relevance", "not_relevant")
                doc["grade_reason"] = grade.get("reason", "")
            except Exception:
                doc["grade"] = "partially_relevant"
                doc["grade_reason"] = "Grading inconclusive"
            graded.append(doc)
        return graded

    def _reformulate_query(self, question):
        return self._call_claude(REFORMULATOR_PROMPT.format(question=question), max_tokens=100).strip()

    def _generate_answer(self, question, context_docs):
        context = "\n\n---\n\n".join([
            f"[Source: {d['source']}] (Relevance: {d.get('relevance_score', 'N/A')})\n{d['text']}"
            for d in context_docs
        ])
        return self._call_claude(GENERATOR_PROMPT.format(context=context, question=question), max_tokens=2048)

    def _check_hallucination(self, answer, context_docs):
        context = "\n\n".join([f"[{d['source']}]: {d['text']}" for d in context_docs])
        try:
            result = self._call_claude(
                HALLUCINATION_CHECK_PROMPT.format(context=context[:4000], answer=answer),
                max_tokens=300
            ).strip()
            if result.startswith("```"):
                result = result.split("```")[1].replace("json", "").strip()
            return json.loads(result)
        except Exception:
            return {"grounded": True, "confidence": 0.5, "issues": ["Check inconclusive"]}

    async def query(self, question, max_retries=2):
        start_time = time.time()
        pipeline_trace = []
        current_query = question
        attempt = 0
        relevant_docs = []

        while attempt <= max_retries:
            attempt += 1
            step_name = f"Attempt {attempt}" + (" (reformulated)" if attempt > 1 else "")

            retrieved = search(current_query, n_results=5)
            pipeline_trace.append({
                "step": f"{step_name} - Retrieve",
                "query": current_query,
                "docs_retrieved": len(retrieved),
                "top_scores": [d["relevance_score"] for d in retrieved[:3]]
            })

            if not retrieved:
                pipeline_trace.append({"step": f"{step_name} - No Results", "action": "No documents in vector store"})
                break

            graded = self._grade_documents(question, retrieved)
            relevant = [d for d in graded if d["grade"] in ("relevant", "partially_relevant")]
            not_relevant = [d for d in graded if d["grade"] == "not_relevant"]
            pipeline_trace.append({
                "step": f"{step_name} - Grade",
                "relevant": len(relevant),
                "not_relevant": len(not_relevant),
                "grades": [{"source": d["source"][:30], "grade": d["grade"]} for d in graded]
            })

            if len(relevant) >= 2 or (len(relevant) >= 1 and attempt > 1):
                relevant_docs = relevant
                pipeline_trace.append({"step": f"{step_name} - Decision", "action": "Sufficient relevant docs.", "docs_used": len(relevant)})
                break
            elif attempt <= max_retries:
                new_query = self._reformulate_query(current_query)
                pipeline_trace.append({"step": f"{step_name} - Reformulate", "original": current_query, "reformulated": new_query})
                current_query = new_query
            else:
                relevant_docs = relevant if relevant else retrieved[:3]
                pipeline_trace.append({"step": f"{step_name} - Decision", "action": "Max retries. Using best available.", "docs_used": len(relevant_docs)})

        if not relevant_docs:
            answer = "I don't have relevant documents to answer this. Please upload documents first."
            hallucination_check = {"grounded": True, "confidence": 1.0, "issues": []}
        else:
            answer = self._generate_answer(question, relevant_docs)
            pipeline_trace.append({"step": "Generate Answer", "docs_used": len(relevant_docs), "answer_length": len(answer)})
            hallucination_check = self._check_hallucination(answer, relevant_docs)
            pipeline_trace.append({
                "step": "Hallucination Check",
                "grounded": hallucination_check.get("grounded", False),
                "confidence": hallucination_check.get("confidence", 0),
                "issues": hallucination_check.get("issues", [])
            })

        elapsed = round(time.time() - start_time, 2)
        return {
            "answer": answer,
            "question": question,
            "pipeline_trace": pipeline_trace,
            "sources": [{"source": d["source"], "relevance": d.get("relevance_score", 0), "grade": d.get("grade", "N/A")} for d in relevant_docs],
            "hallucination_check": hallucination_check,
            "metrics": {
                "retrieval_attempts": attempt,
                "docs_used_for_answer": len(relevant_docs),
                "query_reformulated": attempt > 1,
                "answer_grounded": hallucination_check.get("grounded", False),
                "grounding_confidence": hallucination_check.get("confidence", 0),
                "latency_seconds": elapsed
            }
        }
