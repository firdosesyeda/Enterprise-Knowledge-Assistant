"""
Evaluation Script for the Enterprise Knowledge Assistant.
Run this to measure system performance against known Q&A pairs.

Usage:
    python evaluate.py --api http://localhost:8000
"""
import argparse
import json
import time
from typing import List, Dict

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    exit(1)


# ── Test cases ───────────────────────────────────────────────────────────────
TEST_CASES: List[Dict] = [
    # HR Policy tests
    {
        "question": "How many paid leaves do employees get annually?",
        "expected_keywords": ["24", "leaves", "annual"],
        "category": "HR Policy",
        "difficulty": "easy",
    },
    {
        "question": "What is the remote work policy?",
        "expected_keywords": ["2 days", "remote", "manager"],
        "category": "HR Policy",
        "difficulty": "easy",
    },
    {
        "question": "Can unused sick leaves be carried forward?",
        "expected_keywords": ["cannot", "lapse", "sick"],
        "category": "HR Policy",
        "difficulty": "medium",
    },
    {
        "question": "What is the notice period for an employee with 3 years of tenure?",
        "expected_keywords": ["2 months", "notice"],
        "category": "HR Policy",
        "difficulty": "medium",
    },
    {
        "question": "How many weeks of maternity leave are employees entitled to?",
        "expected_keywords": ["26 weeks", "maternity"],
        "category": "HR Policy",
        "difficulty": "easy",
    },
    # Customer FAQ tests
    {
        "question": "What is the refund policy?",
        "expected_keywords": ["30 days", "refund"],
        "category": "Customer FAQ",
        "difficulty": "easy",
    },
    {
        "question": "What is the file upload size limit?",
        "expected_keywords": ["5GB", "file", "upload"],
        "category": "Technical",
        "difficulty": "easy",
    },
    {
        "question": "What certifications does TechCorp have for security?",
        "expected_keywords": ["SOC 2", "ISO 27001"],
        "category": "Customer FAQ",
        "difficulty": "medium",
    },
    {
        "question": "How many API requests can Business plan users make per hour?",
        "expected_keywords": ["10,000", "business", "hour"],
        "category": "Technical",
        "difficulty": "medium",
    },
    # Hallucination tests (answer should NOT be found)
    {
        "question": "What is the company's stock price?",
        "expected_keywords": [],
        "category": "Out-of-scope",
        "difficulty": "hallucination_check",
        "should_say_not_found": True,
    },
    {
        "question": "What is the lunch allowance amount?",
        "expected_keywords": [],
        "category": "Out-of-scope",
        "difficulty": "hallucination_check",
        "should_say_not_found": True,
    },
    # Multi-hop / ambiguous questions
    {
        "question": "If I joined the company 3 months ago, can I work from home?",
        "expected_keywords": ["probation", "not", "90 days"],
        "category": "HR Policy",
        "difficulty": "hard",
    },
    {
        "question": "How do I get my money back if I'm not happy with the product?",
        "expected_keywords": ["30 days", "refund", "billing@"],
        "category": "Customer FAQ",
        "difficulty": "hard",
    },
]


def run_evaluation(api_base: str, verbose: bool = False) -> Dict:
    """Run all test cases against the API and return a report."""
    results = []
    passed = 0
    hallucination_passed = 0
    hallucination_total = 0

    not_found_phrases = [
        "couldn't find", "not find", "no information", "not available",
        "not mentioned", "not specified", "not in the documents",
        "i don't have", "unable to find",
    ]

    print(f"\n{'─'*60}")
    print(f"  ENTERPRISE KNOWLEDGE ASSISTANT - EVALUATION REPORT")
    print(f"{'─'*60}")
    print(f"  API: {api_base}")
    print(f"  Test cases: {len(TEST_CASES)}")
    print(f"{'─'*60}\n")

    for i, tc in enumerate(TEST_CASES, 1):
        question = tc["question"]
        expected_kws = tc.get("expected_keywords", [])
        should_not_find = tc.get("should_say_not_found", False)
        category = tc["category"]
        difficulty = tc["difficulty"]

        start = time.time()
        try:
            resp = requests.post(
                f"{api_base}/api/v1/ask",
                json={"question": question},
                timeout=30,
            )
            elapsed = time.time() - start

            if resp.status_code != 200:
                result = {
                    "question": question, "category": category,
                    "difficulty": difficulty, "status": "API_ERROR",
                    "error": resp.text, "passed": False,
                    "latency_s": elapsed,
                }
                results.append(result)
                print(f"  [{i:02d}] ❌ API ERROR ({resp.status_code}): {question[:60]}")
                continue

            data = resp.json()
            answer = data.get("answer", "")
            confidence = data.get("confidence", 0)
            sources = data.get("sources", [])
            answer_lower = answer.lower()

            if should_not_find:
                hallucination_total += 1
                # Pass if the model says it doesn't know
                test_passed = any(p in answer_lower for p in not_found_phrases)
                if test_passed:
                    hallucination_passed += 1
            elif expected_kws:
                matched = sum(1 for kw in expected_kws if kw.lower() in answer_lower)
                test_passed = matched >= len(expected_kws) * 0.5
            else:
                test_passed = confidence > 0.2

            if test_passed:
                passed += 1

            result = {
                "question": question,
                "category": category,
                "difficulty": difficulty,
                "answer_preview": answer[:150],
                "confidence": confidence,
                "sources_count": len(sources),
                "sources": [s["document"] for s in sources],
                "keywords_matched": sum(1 for kw in expected_kws if kw.lower() in answer_lower) if expected_kws else "N/A",
                "keywords_total": len(expected_kws),
                "passed": test_passed,
                "latency_s": round(elapsed, 2),
            }
            results.append(result)

            status = "✅" if test_passed else "❌"
            diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴", "hallucination_check": "🔵"}
            print(f"  [{i:02d}] {status} {diff_icon.get(difficulty, '')} [{category}] {question[:55]}...")
            print(f"        Confidence: {confidence:.0%} | Sources: {len(sources)} | Latency: {elapsed:.1f}s")
            if verbose:
                print(f"        Answer: {answer[:120]}...")
            print()

        except requests.Timeout:
            results.append({"question": question, "passed": False, "error": "Timeout"})
            print(f"  [{i:02d}] ⏱ TIMEOUT: {question[:60]}")
        except Exception as e:
            results.append({"question": question, "passed": False, "error": str(e)})
            print(f"  [{i:02d}] 💥 ERROR: {str(e)}")

    # Summary
    total = len(TEST_CASES)
    avg_conf = sum(r.get("confidence", 0) for r in results) / max(len(results), 1)
    avg_lat = sum(r.get("latency_s", 0) for r in results) / max(len(results), 1)

    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total tests:       {total}")
    print(f"  Passed:            {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"  Failed:            {total - passed}/{total}")
    print(f"  Avg confidence:    {avg_conf:.0%}")
    print(f"  Avg latency:       {avg_lat:.1f}s")
    if hallucination_total:
        print(f"  Hallucination prevention: {hallucination_passed}/{hallucination_total} ({hallucination_passed/hallucination_total*100:.0f}%)")
    print(f"{'═'*60}\n")

    report = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total * 100, 1),
        "avg_confidence": round(avg_conf, 3),
        "avg_latency_s": round(avg_lat, 2),
        "hallucination_prevention_rate": (
            round(hallucination_passed / hallucination_total * 100, 1)
            if hallucination_total else None
        ),
        "results": results,
    }

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the knowledge assistant")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--verbose", action="store_true", help="Show full answers")
    parser.add_argument("--output", default="evaluation_report.json", help="Output file")
    args = parser.parse_args()

    report = run_evaluation(args.api, verbose=args.verbose)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  📊 Full report saved to: {args.output}\n")
