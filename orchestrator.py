import os
import requests
from datasets import load_dataset

# --- Konfigurasi Evaluasi ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
GENERATOR_MODEL = "gemma:2b"   
VERIFIER_MODEL = "gemma:2b"     
MAX_TURNS = 5 # Batas putaran debat per soal

def call_ollama(model_name, prompt, system_prompt):
    payload = {"model": model_name, "prompt": prompt, "system": system_prompt, "stream": False, "keep_alive": 0}
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR]: {e}"

def run_debate_on_question(question, ground_truth):
    current_answer = ""
    feedback = ""
    is_verified = False
    
    for turn in range(1, MAX_TURNS + 1):
        print(f"\n  ➤ Putaran Debat {turn} ".center(50, "."))
        
        # --- GENERATOR ---
        gen_sys = "Anda ahli matematika. Jawab dengan langkah-langkah (Chain of Thought). Di akhir jawaban, WAJIB tuliskan angka akhirnya saja di dalam kurung siku, contoh: [45]."
        if turn == 1:
            gen_prompt = f"Soal: {question}\nBerikan jawaban Anda."
        else:
            gen_prompt = (f"Soal Awal: {question}\n"
                          f"Jawaban Anda Sebelumnya: {current_answer}\n"
                          f"Koreksi Auditor: {feedback}\n\n"
                          f"FOKUS PADA SOAL MATEMATIKA AWAL. Perbaiki perhitungan Anda berdasarkan koreksi.")
        
        current_answer = call_ollama(GENERATOR_MODEL, gen_prompt, gen_sys)
        print(f"  [ 🤖 ] GENERATOR ({GENERATOR_MODEL}):\n  {current_answer[:200]}... [dipotong]")

        # --- VERIFIER ---
        ver_sys = (
            "Anda mesin pemeriksa kunci jawaban. Periksa perhitungan asisten. "
            "Jika hitungannya BENAR dan MASUK AKAL, jawab HANYA dengan satu kata: 'TERVERIFIKASI'. "
            "Jika SALAH, tunjukkan letak kesalahan angka atau logikanya."
        )
        ver_prompt = f"Soal Asli: {question}\nJawaban Asisten yang diperiksa: {current_answer}\nApakah angka akhirnya benar?"
        
        feedback = call_ollama(VERIFIER_MODEL, ver_prompt, ver_sys)
        print(f"  [ 🕵️ ] VERIFIER ({VERIFIER_MODEL}):\n  {feedback[:200]}... [dipotong]")

        if "TERVERIFIKASI" in feedback.upper():
            is_verified = True
            print("  [ ✔️ ] Konsensus Tercapai!")
            break 
            
    return current_answer, is_verified

def run_benchmark():
    print("="*60)
    print("🚀 MEMULAI BENCHMARK EVALUATOR (MULTI-AGENT)")
    print(f"Dataset: GSM8K | Gen: {GENERATOR_MODEL} | Ver: {VERIFIER_MODEL}")
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

        # Jalankan debat dan tampilkan prosesnya
        final_output, consensus = run_debate_on_question(question, exact_answer)
        
        print(f"\n>>> HASIL AKHIR SOAL {index + 1}:")
        print(f"Jawaban Lengkap:\n{final_output}\n")
        if f"[{exact_answer}]" in final_output:
            print("[ ✅ ] STATUS: BENAR (Format Tepat)")
            correct_count += 1
        elif final_output.strip().endswith(str(exact_answer)) or final_output.strip().endswith(f" {exact_answer}."):
            print("[ ✅ ] STATUS: BENAR (Deteksi Akhir Kalimat)")
            correct_count += 1
        else:
            print("[ ❌ ] STATUS: SALAH")

    # Kalkulasi Akurasi
    print("\n" + "="*60)
    print("[ 📊 ] REKAPITULASI HASIL BENCHMARK (MULTI-AGENT)")
    accuracy = (correct_count / total_questions) * 100
    print(f"Total Soal Diuji  : {total_questions}")
    print(f"Total Jawaban Benar: {correct_count}")
    print(f"Akurasi Akhir     : {accuracy:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()