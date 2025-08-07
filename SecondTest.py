import json
import logging
import time
from openai import OpenAI

# === API Settings ===
API_KEY = ""
BASE_URL = ""
MODEL_NAME = "gpt-4o-2024-11-20"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

logging.basicConfig(level=logging.INFO)


def build_prompt(question, ref_solution, ref_answer, student_solution, student_answer, is_proof):
    if is_proof:
        prompt = f"""You are a mathematics instructor. Determine whether the following student's proof is logically valid and correctly proves the statement. Ignore formatting, focus only on mathematical logic.

Question:
{question}

Reference Proof:
{ref_solution}

Student's Proof:
{student_solution}

Return only "Yes" if the proof is correct or "No" otherwise."""
    else:
        prompt = f"""You are a mathematics evaluator. Check whether the student's final answer is mathematically equivalent to the reference answer. Ignore formatting differences.

Question:
{question}

Reference Final Answer:
{ref_answer}

Student's Final Answer:
{student_answer}

Return only "Yes" if the answer is correct or "No" otherwise."""
    return prompt


def get_chatgpt_judgment(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0,
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            logging.warning(f"API error (attempt {attempt + 1}): {e}")
            time.sleep(2)
    return "no"


def evaluate_all(standard_file, student_file, output_file):
    with open(standard_file, "r", encoding="utf-8") as f1, open(student_file, "r", encoding="utf-8") as f2:
        standard_data = [json.loads(line) for line in f1]
        student_data = [json.loads(line) for line in f2]

    assert len(standard_data) == len(student_data), "Mismatch in number of questions!"

    correct_count = 0
    total_count = len(standard_data)
    evaluated = []

    for i, (ref, stu) in enumerate(zip(standard_data, student_data)):
        question = ref["question"]
        ref_sol = ref.get("answer/solution", ref.get("solution", ""))
        stu_sol = stu.get("answer/solution", stu.get("solution", ""))
        ref_gt = ref.get("ground_truth", "")
        stu_gt = stu.get("ground_truth", "")

        #  Determine whether it is a proof-based problem
        proof_flag = isinstance(ref_gt, str) and ref_gt.strip() == ""
        prompt = build_prompt(question, ref_sol, ref_gt, stu_sol, stu_gt, proof_flag)
        judgment = get_chatgpt_judgment(prompt)

        correct = (judgment == "yes")
        if correct:
            correct_count += 1

        print(f"[{i + 1:>3}] {'(Proof)' if proof_flag else '(Answer)'} → {'✔️ Correct' if correct else '❌ Wrong'}")

        evaluated.append({
            "question": question,
            "reference_solution": ref_sol,
            "student_solution": stu_sol,
            "is_proof": proof_flag,
            "judgment": judgment,
            "correct": correct
        })

    acc = correct_count / total_count if total_count else 0
    print("\n====== Summary ======")
    print(f"Total: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {acc:.2%}")

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in evaluated:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Detailed results saved to: {output_file}")


if __name__ == "__main__":
    STANDARD_FILE = r"Textual_200.jsonl"
    STUDENT_FILE = r"answer-textual-gpt-4o-mini.jsonl"
    OUTPUT_FILE = r"judgement-textual.jsonl"

    evaluate_all(STANDARD_FILE, STUDENT_FILE, OUTPUT_FILE)
