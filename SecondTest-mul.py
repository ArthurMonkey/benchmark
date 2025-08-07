import os
import json
import base64
import time
from pathlib import Path
from openai import OpenAI

# ========== GPT-4o API Settings==========
client = OpenAI(
    api_key="",
    base_url=""
)

# ========== Basic Settings ==========
standard_file = r"vision_100.jsonl"
student_file = r"answer-vision.jsonl"
image_base_path = r"image_file"
output_file = r"judgement-vision.jsonl"

# ========== image change to base64 ==========
def encode_image(image_path: str) -> str:
    mime_type = "image/jpeg"
    if image_path.lower().endswith(".png"):
        mime_type = "image/png"
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_data}"

# ========== Determine whether it is a proof-based problem ==========
def is_proof_problem(question_text):
    return "prove" in question_text.lower() or "证明" in question_text

# ==========  prompt ==========
def build_judge_prompt(question, ref_solution, ref_answer, stu_solution, stu_answer):
    if is_proof_problem(question):
        return f"""You are a university-level mathematics instructor. Determine whether the following student's proof is logically valid and correctly proves the statement. Focus strictly on the correctness of reasoning and mathematical logic, and ignore differences in wording or format.

Question:
{question}

Reference Proof:
{ref_solution}

Student's Proof:
{stu_solution}

Return only "Yes" if the proof is complete and logically correct, or "No" if it is missing steps, incorrect, or incomplete."""
    else:
        return f"""You are a mathematics evaluator. Compare the student's final answer to the reference final answer. Evaluate whether they are mathematically equivalent.

Question:
{question}

Reference Final Answer:
{ref_answer}

Student's Final Answer:
{stu_answer}

Rules:
- If either answer is missing or empty, return "No".
- Ignore differences in formatting, variable names, or simplification, as long as the mathematical meaning is identical.
- Only return "Yes" if you are certain they are mathematically equivalent. Otherwise, return "No".

Answer only with "Yes" or "No"."""

# ========== request body ==========
def get_model_judgment(messages, retries=5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=2048
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"⚠️ API error: {e}")
            time.sleep(min(2 ** attempt, 30))
    return "no"

# ========== Main ==========
with open(standard_file, "r", encoding="utf-8") as f1, open(student_file, "r", encoding="utf-8") as f2:
    ref_data = [json.loads(line) for line in f1]
    stu_data = [json.loads(line) for line in f2]

assert len(ref_data) == len(stu_data)

results = []
correct_count = 0

for i, (ref, stu) in enumerate(zip(ref_data, stu_data)):
    question = ref["question"]
    ref_solution = ref.get("answer/solution", "")
    stu_solution = stu.get("answer", "") or stu.get("answer/solution", "")
    ref_gt = ref.get("ground_truth", "").strip()
    stu_gt = stu.get("ground_truth", "").strip()
    proof_flag = is_proof_problem(question)

    image_path = Path(image_base_path).joinpath(ref["image_path"]).as_posix()
    if not os.path.exists(image_path):
        print(f"[{i+1:02}] ❌ lose image, skip this question: {image_path}")
        continue

    if not proof_flag and (not ref_gt or not stu_gt):
        judgment = "no"
    else:
        prompt = build_judge_prompt(question, ref_solution, ref_gt, stu_solution, stu_gt)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
                {"type": "text", "text": prompt}
            ]
        }]
        judgment = get_model_judgment(messages)

    correct = judgment == "yes"
    if correct:
        correct_count += 1

    print(f"[{i+1:02}] {'✔️' if correct else '❌'} → {'Proof' if proof_flag else 'Answer'} | Judgment: {judgment}")

    results.append({
        "question": question,
        "reference_solution": ref_solution,
        "student_solution": stu_solution,
        "ground_truth_ref": ref_gt,
        "ground_truth_stu": stu_gt,
        "judgment": judgment,
        "correct": correct,
        "is_proof": proof_flag
    })

# ========== Output result ==========
with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

acc = correct_count / len(results) if results else 0
print(f"\n✅ Accuracy: {acc:.2%} ({correct_count}/{len(results)})")
print(f"✅ Results saved to: {output_file}")

