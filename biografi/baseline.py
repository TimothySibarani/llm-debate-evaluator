import os
import requests
import re
from datasets import load_dataset
from datetime import datetime

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e2b"
LOG_FILE = "baseline_results.txt" # File TXT untuk menyimpan hasil pikiran dan jawaban

def call_ollama(prompt, system_prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "system": system_prompt, "stream": False, "keep_alive": 0}
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
        f.write(f"LLM BASELINE EVALUATOR LOG - BIOGRAPHY\n")
        f.write(f"Started at: {timestamp}\n")
        f.write(f"Dataset: TriviaQA | Model: {MODEL_NAME}\n")
        f.write("="*60 + "\n")

    print("="*60)
    print("🔬 MEMULAI BASELINE EVALUATOR - BIOGRAFI/TRIVIAQA (SINGLE MODEL)")
    print("="*60)

    # Menggunakan mode streaming untuk mencegah terminal stuck/download bergiga-giga
    dataset_stream = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation", streaming=True)
    dataset = dataset_stream.take(20) # Ambil 20 soal pertama
    
    correct_count = 0
    total_questions = 20
    
    for index, item in enumerate(dataset):
        question = item['question']
        exact_answer = item['answer']['value']
        
        # Kumpulkan semua alias dan ubah ke huruf kecil untuk menghindari masalah case-sensitivity
        aliases = [alias.lower() for alias in item['answer']['aliases']]
        if exact_answer.lower() not in aliases:
            aliases.append(exact_answer.lower())

        # Tampilkan di Terminal
        print(f"\n{'='*60}\n--- Soal {index + 1}/{total_questions} ---")
        print(f"Q: {question}\nKunci Jawaban Asli: {exact_answer}")

        # Simpan ke TXT
        write_to_log(f"\n{'='*60}\n--- Soal {index + 1}/{total_questions} ---")
        write_to_log(f"Q: {question}\nKunci Jawaban Asli: {exact_answer}")

        sys_prompt = "You are a history and biography expert. Think step-by-step. Provide your reasoning, and you MUST enclose your final answer entity or date inside brackets, like [George Washington]."
        user_prompt = f"Question: {question}\nProvide your answer."
        
        output = call_ollama(user_prompt, sys_prompt)
        
        # Tampilkan di Terminal dan Simpan ke TXT
        print(f"\n>>> HASIL PIKIRAN & JAWABAN LLM:\n{output}\n")
        write_to_log(f"\n>>> HASIL PIKIRAN & JAWABAN LLM:\n{output}\n")

        # --- BLOK EVALUASI CERDAS TINGKAT LANJUT ---
        output_lower = output.lower() # Ubah seluruh output ke huruf kecil
        extracted_texts = re.findall(r'\[(.*?)\]', output_lower)
        
        is_correct = False
        match_type = ""

        # Cek 1: Validasi dari dalam kurung siku
        if extracted_texts:
            final_extracted = extracted_texts[-1].strip()
            for alias in aliases:
                if alias == final_extracted or alias in final_extracted:
                    is_correct = True
                    match_type = "Format Tepat [...]"
                    break
        
        # Cek 2: Fallback Evaluation (Jika LLM lupa kurung siku, scan seluruh teks)
        if not is_correct:
            for alias in aliases:
                # Hanya cocokkan jika alias ditemukan dalam teks jawaban
                if alias in output_lower:
                    is_correct = True
                    match_type = "Fallback Pencarian Teks"
                    break
                
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