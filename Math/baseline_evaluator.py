import os
import requests
import re
from datasets import load_dataset

# --- Konfigurasi Baseline ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e2b" # Pastikan nama model sesuai yang ada di Ollama Anda

def call_ollama(prompt, system_prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "system": system_prompt, "stream": False, "keep_alive": 0}
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR]: {e}"

def run_baseline():
    print("="*60)
    print("🔬 MEMULAI BASELINE EVALUATOR (SINGLE MODEL)")
    print(f"Dataset: GSM8K | Model: {MODEL_NAME}")
    print("="*60)

    # Mengambil 20 soal pertama
    dataset = load_dataset("openai/gsm8k", "main", split="test[:20]") 
    correct_count = 0
    total_questions = len(dataset)
    
    for index, item in enumerate(dataset):
        question = item['question']
        exact_answer = item['answer'].split("#### ")[-1].strip()

        print(f"\n{'='*60}\n--- Soal {index + 1}/{total_questions} ---")
        print(f"Q: {question[:150]}...") 
        print(f"Kunci Jawaban Asli: {exact_answer}")

        # Prompt Bahasa Inggris (Agar adil dengan pengujian Multi-Agent)
        sys_prompt = "You are a math expert. Solve the problem step-by-step using Chain of Thought. You MUST end your response with the final numerical answer inside brackets, like [45]."
        user_prompt = f"Question: {question}\nProvide your answer."
        
        print(f"\n[ 🤖 ] {MODEL_NAME} sedang menjawab...")
        output = call_ollama(user_prompt, sys_prompt)
        
        # Tampilkan jawaban lengkap
        print(f"\n>>> JAWABAN LENGKAP SOAL {index + 1}:")
        print(output)
        print("\n" + "-"*30)

# --- BLOK EVALUASI CERDAS (Ganti bagian lama dengan ini) ---
        
        # Bersihkan koma
        output_clean = output.replace(",", "") # Jika di benchmark_evaluator, ganti 'output' jadi 'final_output'
        
        # Ekstrak semua angka (termasuk desimal) dari jawaban AI menggunakan Regex
        found_numbers = re.findall(r'-?\d+\.?\d*', output_clean)
        
        is_correct = False
        try:
            # Ubah kunci jawaban asli menjadi angka matematika (float)
            exact_float = float(exact_answer)
            
            # Cek 1: Apakah ada di dalam kurung siku secara literal?
            if f"[{exact_answer}]" in output_clean or f"[{exact_float}]" in output_clean:
                is_correct = True
                print("[ ✅ ] STATUS: BENAR (Format Tepat)")
                
            # Cek 2: Evaluasi Nilai Matematika di akhir teks
            elif found_numbers:
                # Ambil 3 angka terakhir yang diketik AI (menghindari angka di kalimat basa-basi akhir)
                last_few_numbers = found_numbers[-3:] 
                for num_str in last_few_numbers:
                    if float(num_str) == exact_float:
                        is_correct = True
                        print("[ ✅ ] STATUS: BENAR (Deteksi Nilai Matematika 7.0 == 7)")
                        break
        except ValueError:
            # Jaga-jaga jika kunci jawaban bukan angka murni
            pass 
            
        if is_correct:
            correct_count += 1
        else:
            print("[ ❌ ] STATUS: SALAH")
        # ------------------------------------------------------------

    # Kalkulasi Akurasi
    print("\n" + "="*60)
    print("[ 📊 ] REKAPITULASI HASIL BASELINE (SINGLE MODEL)")
    accuracy = (correct_count / total_questions) * 100
    print(f"Total Soal Diuji  : {total_questions}")
    print(f"Total Jawaban Benar: {correct_count}")
    print(f"Akurasi Akhir     : {accuracy:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_baseline()