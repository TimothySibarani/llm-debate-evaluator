import os
import requests
import re
from datasets import load_dataset
from datetime import datetime

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e2b"
LOG_FILE = "baseline_results.txt" # File TXT untuk menyimpan hasil pikiran dan jawaban

def call_ollama(prompt, system_prompt):
    payload = {"model": MODEL_NAME, 
               "prompt": prompt, 
               "system": system_prompt, 
               "stream": False, 
               "keep_alive": 0,
                "options": {
                "seed": 42,          # Mengunci seed ke angka 42
                "temperature": 0.0   # Membuat jawaban LLM menjadi deterministik (paling logis, tidak acak)
                    }
               }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR]: {e}"

def write_to_log(text):
    """Fungsi pembantu untuk menulis/append ke file log txt."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def run_baseline():
    # Inisialisasi file log (Timpa file lama setiap run baru)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"LLM BASELINE EVALUATOR LOG - GEOGRAPHY\n")
        f.write(f"Started at: {timestamp}\n")
        f.write(f"Dataset: MMLU High School Geography | Model: {MODEL_NAME}\n")
        f.write("="*60 + "\n")

    print("="*60)
    print("🔬 MEMULAI BASELINE EVALUATOR - GEOGRAFI (SINGLE MODEL)")
    print("="*60)

    dataset = load_dataset("cais/mmlu", "high_school_geography", split="test[:20]")
    correct_count = 0
    total_questions = len(dataset)
    
    for index, item in enumerate(dataset):
        question = item['question']
        choices = item['choices']
        correct_index = item['answer']
        correct_letter = chr(65 + correct_index) # Convert 0,1,2,3 to A,B,C,D
        
        choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        
        # Tampilkan di Terminal
        print(f"\n{'='*60}\n--- Soal {index + 1}/{total_questions} ---")
        print(f"Q: {question}\n{choices_text}")
        print(f"Kunci Jawaban Asli: {correct_letter}")

        # Simpan ke TXT
        write_to_log(f"\n{'='*60}\n--- Soal {index + 1}/{total_questions} ---")
        write_to_log(f"Q: {question}\n{choices_text}")
        write_to_log(f"Kunci Jawaban Asli: {correct_letter}")

        sys_prompt = "You are a geography expert. Solve the problem step-by-step. You MUST end your response with the final answer letter inside brackets, like [A], [B], [C], or [D]."
        user_prompt = f"Question: {question}\nChoices:\n{choices_text}\nProvide your answer."
        
        output = call_ollama(user_prompt, sys_prompt)
        
        # Tampilkan di Terminal dan Simpan ke TXT
        print(f"\n>>> HASIL PIKIRAN & JAWABAN LLM:\n{output}\n")
        write_to_log(f"\n>>> HASIL PIKIRAN & JAWABAN LLM:\n{output}\n")

        # --- BLOK EVALUASI CERDAS (Multiple Choice) ---
        extracted = re.findall(r'\[([A-D])\]', output.upper())
        is_correct = False
        match_type = ""
        
        # Cek 1: Format Tepat dalam Kurung Siku
        if extracted:
            if extracted[-1] == correct_letter:
                is_correct = True
                match_type = "Format Tepat [...]"
        
        # Cek 2: Fallback Evaluation (Jika LLM lupa kurung siku)
        if not is_correct:
            last_part = output.upper()[-50:]
            fallback_match = re.search(r'(?:ANSWER|OPTION|CHOICE|IS)[\s:]*([A-D])\b', last_part)
            if fallback_match and fallback_match.group(1) == correct_letter:
                is_correct = True
                match_type = "Fallback Deteksi Teks Akhir"
            elif f"[{correct_letter}]" in output.upper():
                is_correct = True
                match_type = "Fallback Pencocokan String"

        if is_correct:
            correct_count += 1
            status_msg = f"[ ✅ ] STATUS: BENAR ({match_type})"
            print(status_msg)
            write_to_log(status_msg)
        else:
            status_msg = "[ ❌ ] STATUS: SALAH"
            print(status_msg)
            write_to_log(status_msg)
        # ------------------------------------------------------------

    # --- REKAPITULASI HASIL AKHIR ---
    accuracy = (correct_count / total_questions) * 100
    recap_text = (
        "\n" + "="*60 + "\n"
        "[ 📊 ] REKAPITULASI HASIL BASELINE (SINGLE MODEL)\n"
        f"Total Soal Diuji  : {total_questions}\n"
        f"Total Jawaban Benar: {correct_count}\n"
        f"Akurasi Akhir     : {accuracy:.2f}%\n"
        + "="*60 + "\n"
    )
    print(recap_text)
    write_to_log(recap_text)

if __name__ == "__main__":
    run_baseline()