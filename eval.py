
import json
import requests
import re

API_URL = "http://127.0.0.1:8000/ask"

TESTS = [
    {
        "name": "Invoice amount",
        "question": "What is the outstanding invoice amount?",
        "expected": ["14000"],
        "source": "01_invoice_payment.pdf",
    },
    {
        "name": "Atlas migration",
        "question": "What is the current status of the Atlas migration?",
        "expected": ["amber"],
        "source": "04_project_status_report.pdf",
    },
    {
        "name": "Warehouse inventory",
        "question": "Which warehouse items need reordering?",
        "expected": ["wireless keyboard", "65w adapter"],
        "source": "07_warehouse_inventory.pdf",
    },
    {
        "name": "Scanned PDF - thermal rolls",
        "question": "How many boxes of thermal label rolls are available, and what is the reorder threshold?",
        "expected": ["48", "75"],
        "source": "scanned_ocr_test_document.pdf",
    },
    {
        "name": "Scanned PDF - barcode scanners",
        "question": "How many USB barcode scanners are available, and what is their reorder threshold?",
        "expected": ["32", "20"],
        "source": "scanned_ocr_test_document.pdf",
    },
    {
        "name": "Unsupported question",
        "question": "What is the CEO's home address?",
        "expected": ["do not contain the answer"],
        "source": None,
    },
]


def run_test(test):
    response = requests.post(
        API_URL,
        json={"question": test["question"], "top_k": 3},
        timeout=120,
    )

    if response.status_code != 200:
        return False, f"HTTP {response.status_code}: {response.text[:200]}"

    data = response.json()
    print("      RAW RESPONSE:", json.dumps(data, indent=2))

    answer = str(data.get("answer", "")).lower()

    print(f"      Actual answer: {answer}")
    print(f"      Actual citations: {data.get('citations', [])}")

    citations = json.dumps(data.get("citations", [])).lower()

    # Check expected answer facts.
    def normalize_text(text):
        text = str(text).lower()
        text = text.replace("₹", "")
        text = text.replace(",", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()


    facts_ok = all(
        normalize_text(fact) in normalize_text(answer)
        for fact in test["expected"]
    )

    # Check citation for supported questions.
    source = test["source"]
    citation_ok = (
        source is None or source.lower() in citations
    )

    passed = facts_ok and citation_ok

    details = (
        f"Facts: {'PASS' if facts_ok else 'FAIL'}, "
        f"Citation: {'PASS' if citation_ok else 'FAIL'}"
    )

    return passed, details


def main():
    passed_count = 0

    print("\\nAWS DOCUMENT RAG EVALUATION")
    print("-" * 40)

    for test in TESTS:
        try:
            passed, details = run_test(test)
            if passed:
                passed_count += 1

            print(
                f"[{'PASS' if passed else 'FAIL'}] "
                f"{test['name']}: {details}"
            )

        except requests.RequestException as exc:
            print(f"[FAIL] {test['name']}: Request error: {exc}")
        except (ValueError, TypeError, KeyError) as exc:
            print(f"[FAIL] {test['name']}: Response error: {exc}")

    print("-" * 40)
    print(f"Result: {passed_count}/{len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
