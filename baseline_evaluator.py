import os
import requests
from datasets import load_dataset

# --- Konfigurasi Baseline ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b" # Pastikan nama model sesuai yang ada di Ollama Anda

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

        # Evaluasi Ketat (Mencegah Positif Palsu)
        if f"[{exact_answer}]" in output:
            print("[ ✅ ] STATUS: BENAR (Format Tepat)")
            correct_count += 1
        elif output.strip().endswith(str(exact_answer)) or output.strip().endswith(f" {exact_answer}."):
            print("[ ✅ ] STATUS: BENAR (Deteksi Akhir Kalimat)")
            correct_count += 1
        else:
            print("[ ❌ ] STATUS: SALAH")

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