import requests
import re
from datasets import load_dataset

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e2b"

def call_ollama(prompt, system_prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "system": system_prompt, "stream": False, "keep_alive": 0}
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR]: {e}"

def run_baseline():
    print("="*60)
    print("🔬 MEMULAI BASELINE EVALUATOR - FISIKA (SINGLE MODEL)")
    print("="*60)

    dataset = load_dataset("cais/mmlu", "high_school_physics", split="test[:20]")
    correct_count = 0
    total_questions = len(dataset)
    
    for index, item in enumerate(dataset):
        question = item['question']
        choices = item['choices']
        correct_index = item['answer']
        correct_letter = chr(65 + correct_index) # Convert 0,1,2,3 to A,B,C,D
        
        choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        
        print(f"\n{'='*60}\n--- Soal {index + 1}/{total_questions} ---")
        print(f"Q: {question}\n{choices_text}")
        print(f"Kunci Jawaban Asli: {correct_letter}")

        sys_prompt = "You are a physics expert. Solve the problem step-by-step. You MUST end your response with the final answer letter inside brackets, like [A], [B], [C], or [D]."
        user_prompt = f"Question: {question}\nChoices:\n{choices_text}\nProvide your answer."
        
        output = call_ollama(user_prompt, sys_prompt)
        print(f"\n>>> JAWABAN: {output}\n")

        # --- EVALUASI PILIHAN GANDA ---
        extracted_letters = re.findall(r'\[([A-D])\]', output.upper())
        
        is_correct = False
        if extracted_letters:
            if extracted_letters[-1] == correct_letter:
                is_correct = True
                print("[ ✅ ] STATUS: BENAR (Pilihan Ganda Tepat)")
        elif f"[{correct_letter}]" in output.upper():
            is_correct = True
            print("[ ✅ ] STATUS: BENAR (Fallback Match)")
            
        if is_correct:
            correct_count += 1
        else:
            print("[ ❌ ] STATUS: SALAH")

    accuracy = (correct_count / total_questions) * 100
    print(f"\n[ 📊 ] AKURASI AKHIR: {accuracy:.2f}%")

if __name__ == "__main__":
    run_baseline()