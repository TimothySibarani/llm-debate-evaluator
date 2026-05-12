import os
import requests
import re
from datasets import load_dataset
from datetime import datetime

# --- Konfigurasi Evaluasi ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
GENERATOR_MODEL = "gemma4:e2b"   
VERIFIER_MODEL = "gemma4:e2b"     
MAX_TURNS = 5 
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

def run_debate_on_question(question, exact_answer, index):
    current_answer = ""
    feedback = ""
    is_verified = False
    
    # Tampilkan di Terminal
    print(f"\n{'='*60}\n--- Soal {index} ---")
    print(f"Q: {question}")
    print(f"Kunci Jawaban Asli: {exact_answer}\n")
    
    # Tulis header soal ke file log
    write_to_log(f"\n{'='*60}\n--- SOAL {index} ---")
    write_to_log(f"Q: {question}")
    write_to_log(f"Kunci Jawaban Asli: {exact_answer}\n")
    
    for turn in range(1, MAX_TURNS + 1):
        print(f"  ➤ Putaran Debat {turn} ".center(50, "."))
        write_to_log(f"  ➤ Putaran Debat {turn} ".center(50, "."))
        
        # --- GENERATOR ---
        gen_sys = "You are a history and biography expert. You MUST end your response with the final target entity or date inside brackets, like [Entity Name]."
        if turn == 1:
            gen_prompt = f"Question: {question}\nAnswer:"
        else:
            gen_prompt = (f"Original Question: {question}\n"
                          f"Your Previous Attempt: {current_answer}\n"
                          f"Auditor Feedback: {feedback}\n\n"
                          f"Please reconsider and correct your factual data.")
        
        current_answer = call_ollama(GENERATOR_MODEL, gen_prompt, gen_sys)
        
        # Tampilkan sebagian di terminal (mirip script Math)
        print(f"  [ 🤖 ] GENERATOR ({GENERATOR_MODEL}):\n  {current_answer[:200]}... [dipotong]")
        # Catat full jawaban Generator ke log
        write_to_log(f"\n[ 🤖 ] GENERATOR ({GENERATOR_MODEL}):\n{current_answer}\n")

        # --- VERIFIER ---
        ver_sys = (
            "You are a strict historical auditor. Check the facts. "
            "If the fact and final bracketed entity are 100% correct, reply EXACTLY and ONLY with the word 'VERIFIED'. "
            "If it is wrong, DO NOT use the word 'VERIFIED'. Instead, specify the historical inaccuracy."
        )
        ver_prompt = f"Question: {question}\nAssistant Answer: {current_answer}\nAre the facts and the final bracketed entity accurate?"
        
        feedback = call_ollama(VERIFIER_MODEL, ver_prompt, ver_sys)
        
        # Tampilkan sebagian di terminal
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
        f.write(f"LLM DEBATE EVALUATOR LOG - BIOGRAPHY\n")
        f.write(f"Started at: {timestamp}\n")
        f.write(f"Dataset: TriviaQA | Gen: {GENERATOR_MODEL} | Ver: {VERIFIER_MODEL}\n")
        f.write("="*60 + "\n")

    print("="*60)
    print("🚀 MEMULAI BENCHMARK EVALUATOR - BIOGRAFI (MULTI-AGENT)")
    print(f"Dataset: TriviaQA | Gen: {GENERATOR_MODEL} | Ver: {VERIFIER_MODEL}")
    print("="*60)

    # Menggunakan mode streaming untuk efisiensi RAM/Storage
    dataset_stream = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation", streaming=True)
    dataset = dataset_stream.take(20)
    
    correct_count = 0
    total_questions = 20
    
    for index, item in enumerate(dataset):
        question = item['question']
        exact_answer = item['answer']['value']
        
        # Ekstraksi alias untuk pengecekan nanti
        aliases = [alias.lower() for alias in item['answer']['aliases']]
        if exact_answer.lower() not in aliases:
            aliases.append(exact_answer.lower())
            
        # Jalankan debat dan log prosesnya
        final_output, consensus = run_debate_on_question(question, exact_answer, index + 1)
        
        print(f"\n>>> HASIL AKHIR SOAL {index + 1}:")
        print(f"Jawaban Lengkap:\n{final_output}\n")
        
        # --- BLOK EVALUASI CERDAS ---
        output_lower = final_output.lower()
        extracted_texts = re.findall(r'\[(.*?)\]', output_lower)
        
        is_correct = False
        match_type = ""

        # Cek 1: Validasi format dari dalam kurung siku
        if extracted_texts:
            final_extracted = extracted_texts[-1].strip()
            for alias in aliases:
                if alias == final_extracted or alias in final_extracted:
                    is_correct = True
                    match_type = "Format Tepat [...]"
                    break
        
        # Cek 2: Fallback Evaluation (Mencegah false negative jika AI lupa kurung)
        if not is_correct:
            for alias in aliases:
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