import os
import requests
import re
from datasets import load_dataset
from datetime import datetime

# --- Konfigurasi Evaluasi ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
GENERATOR_MODEL = "gemma4:e2b"   
VERIFIER_MODEL = "gemma4:e2b"     
MAX_TURNS = 5 # Batas putaran debat per soal
LOG_FILE = "debate_results.txt"

def call_ollama(model_name, prompt, system_prompt):
    payload = {"model": model_name, "prompt": prompt, "system": system_prompt, "stream": False, "keep_alive": 0}
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR]: {e}"

def write_to_log(text):
    """Fungsi pembantu untuk menulis/append ke file log txt."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def run_debate_on_question(question, ground_truth, question_number):
    current_answer = ""
    feedback = ""
    is_verified = False
    
    # Tulis header soal ke file log
    write_to_log(f"\n{'='*60}\n--- SOAL {question_number} ---")
    write_to_log(f"Q: {question}")
    write_to_log(f"Kunci Jawaban Asli: {ground_truth}\n")
    
    for turn in range(1, MAX_TURNS + 1):
        print(f"\n  ➤ Putaran Debat {turn} ".center(50, "."))
        write_to_log(f"  ➤ Putaran Debat {turn} ".center(50, "."))
        
        # --- GENERATOR ---
        gen_sys = "You are a math expert. Solve the problem step-by-step using Chain of Thought. You MUST end your response with the final numerical answer inside brackets, like [45]."
        if turn == 1:
            gen_prompt = f"Question: {question}\nProvide your step-by-step answer."
        else:
            gen_prompt = (f"Original Question: {question}\n"
                          f"Your Previous Attempt: {current_answer}\n"
                          f"Auditor Feedback: {feedback}\n\n"
                          f"Please reconsider the problem and correct your calculation.")
        
        current_answer = call_ollama(GENERATOR_MODEL, gen_prompt, gen_sys)
        print(f"  [ 🤖 ] GENERATOR ({GENERATOR_MODEL}):\n  {current_answer[:200]}... [dipotong]")
        # Catat full jawaban Generator ke log
        write_to_log(f"\n[ 🤖 ] GENERATOR ({GENERATOR_MODEL}):\n{current_answer}\n")

        # --- VERIFIER ---
        # PERBAIKAN: Instruksi Verifier diperketat agar tidak sembarangan bilang VERIFIED
        ver_sys = (
            "You are a strict math auditor. Check the assistant's work. "
            "If the answer and logic are 100% correct, reply EXACTLY and ONLY with the word 'VERIFIED'. "
            "If it is wrong, DO NOT use the word 'VERIFIED'. Instead, explain the specific error."
        )
        ver_prompt = f"Soal Asli: {question}\nJawaban Asisten yang diperiksa: {current_answer}\nApakah angka akhirnya benar?"
        
        feedback = call_ollama(VERIFIER_MODEL, ver_prompt, ver_sys)
        print(f"  [ 🕵️ ] VERIFIER ({VERIFIER_MODEL}):\n  {feedback[:200]}... [dipotong]")
        # Catat full jawaban Verifier ke log
        write_to_log(f"\n[ 🕵️ ] VERIFIER ({VERIFIER_MODEL}):\n{feedback}\n")

        if "TERVERIFIKASI" in feedback.upper() or "VERIFIED" in feedback.upper():
            is_verified = True
            print("  [ ✔️ ] Konsensus Tercapai!")
            write_to_log("  [ ✔️ ] Konsensus Tercapai!\n")
            break 
            
    return current_answer, is_verified

def run_benchmark():
    # Inisialisasi file log (Timpa file lama setiap run baru)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"LLM DEBATE EVALUATOR LOG\n")
        f.write(f"Started at: {timestamp}\n")
        f.write(f"Dataset: GSM8K | Gen: {GENERATOR_MODEL} | Ver: {VERIFIER_MODEL}\n")
        f.write("="*60 + "\n")

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

        # Jalankan debat dan tampilkan prosesnya (kirim nomor soal untuk dilog)
        final_output, consensus = run_debate_on_question(question, exact_answer, index + 1)
        
        print(f"\n>>> HASIL AKHIR SOAL {index + 1}:")
        print(f"Jawaban Lengkap:\n{final_output}\n")

        # --- BLOK EVALUASI CERDAS ---
        output_clean = final_output.replace(",", "") 
        
        # PERBAIKAN: Hanya ekstrak teks yang ada DI DALAM kurung siku [...]
        bracket_contents = re.findall(r'\[(.*?)\]', output_clean)
        
        is_correct = False
        try:
            exact_float = float(exact_answer)
            
            # Cek 1: Pencocokan string langsung untuk format yang benar-benar pas
            if f"[{exact_answer}]" in output_clean or f"[{exact_float}]" in output_clean:
                is_correct = True
                status_msg = "[ ✅ ] STATUS: BENAR (Format Tepat)"
                print(status_msg)
                write_to_log(status_msg)
                
            # Cek 2: Jika format tidak persis sama, cari nilai matematika di dalam kurung siku terakhir
            elif bracket_contents:
                # Ambil isi dari kurung siku yang paling terakhir diketik AI
                last_bracket_text = bracket_contents[-1]
                
                # Ekstrak angka dari dalam kurung siku tersebut
                nums_in_bracket = re.findall(r'-?\d+\.?\d*', last_bracket_text)
                
                if nums_in_bracket:
                    # Ambil angka terakhir di dalam kurung siku
                    final_num = float(nums_in_bracket[-1])
                    if final_num == exact_float:
                        is_correct = True
                        status_msg = "[ ✅ ] STATUS: BENAR (Deteksi Nilai Matematika dalam [...])"
                        print(status_msg)
                        write_to_log(status_msg)
                        
        except ValueError:
            pass 
            
        if is_correct:
            correct_count += 1
        else:
            status_msg = "[ ❌ ] STATUS: SALAH"
            print(status_msg)
            write_to_log(status_msg)
        # ------------------------------------------------------------

    # Kalkulasi Akurasi
    accuracy = (correct_count / total_questions) * 100
    
    recap_text = (
        "\n" + "="*60 + "\n"
        "[ 📊 ] REKAPITULASI HASIL BENCHMARK (MULTI-AGENT)\n"
        f"Total Soal Diuji  : {total_questions}\n"
        f"Total Jawaban Benar: {correct_count}\n"
        f"Akurasi Akhir     : {accuracy:.2f}%\n"
        + "="*60 + "\n"
    )
    
    print(recap_text)
    write_to_log(recap_text)

if __name__ == "__main__":
    run_benchmark()