import os
import json
import base64
import time
from pathlib import Path
from openai import OpenAI

# ========== API Settings ==========
client = OpenAI(
    api_key="",
    base_url=""
)

# ========== Basic Settings ==========
input_path = r"vision_100.jsonl"
output_path = r"answer-vision-100"
image_base_path = r"image_file"


# ========== image change to base64 ==========
def encode_image(image_path: str) -> str:
    mime_type = "image/jpeg"
    ext = Path(image_path).suffix.lower()
    if ext == ".png":
        mime_type = "image/png"
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

# ========== Prompt  ==========
PROMPT_TEMPLATE = """
You are a university-level mathematics instructor. Given the following question, provide a complete and rigorous solution. The final answer must be clearly marked as `groundtruth`.

If the question is a proof problem, provide the full proof in the solution section, but leave the groundtruth field empty.

Please strictly follow this output format (use English field names exactly as shown):

question:
<Original question text>
answer/solution:
<Your detailed solution>
groundtruth:
<Final answer or leave empty if proof>

Please read the reference image at image_path.
""".strip()

# ========== Exract==========
with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for idx, line in enumerate(infile, start=1):
        data = json.loads(line.strip())
        raw_image_path = data.get("image_path", "").strip()
        image_filename = os.path.basename(raw_image_path)
        image_path = Path(image_base_path).joinpath(image_filename).as_posix()
        question = data.get("question", "").strip()

        print(f"\n📌 Processing problem {idx} ")
        print(f"🖼️ image_path: {image_path}")

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image(image_path),
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": PROMPT_TEMPLATE + "\n\nQuestion:\n" + question
                }
            ]
        }]

        answer = None
        for attempt in range(1, 11):
            try:
                print(f"⏳  Attempting request:{attempt} ", end=" ")
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=2048
                )
                answer = response.choices[0].message.content.strip()
                print("✅ success")
                break
            except Exception as e:
                print(f"❌ Request failed – Encountered an error: {e}")
                if attempt < 10:
                    sleep_time = min(2 ** attempt, 30)
                    print(f"🔁 wait for {sleep_time} seconds and try again...")
                    time.sleep(sleep_time)
                else:
                    print("🚫 Maximum number of retries reached. Skipping this problem.")
                    answer = "Error: API call failed."

        data["answer"] = answer
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"\n✅ All problems have been processed, and the output has been written to: {output_path}")

