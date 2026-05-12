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

def run_debate_on_question(item, index):
    question = item['question']
    choices = item['choices']
    correct_letter = chr(65 + item['answer'])
    choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
    
    current_answer = ""
    feedback = ""
    is_verified = False
    
    # Tampilkan di Terminal
    print(f"\n{'='*60}\n--- Soal {index} ---")
    print(f"Q: {question}\n{choices_text}")
    print(f"Kunci Jawaban Asli: {correct_letter}\n")
    
    # Tulis header soal ke file log
    write_to_log(f"\n{'='*60}\n--- SOAL {index} ---")
    write_to_log(f"Q: {question}\n{choices_text}")
    write_to_log(f"Kunci Jawaban Asli: {correct_letter}\n")
    
    for turn in range(1, MAX_TURNS + 1):
        print(f"  ➤ Putaran Debat {turn} ".center(50, "."))
        write_to_log(f"  ➤ Putaran Debat {turn} ".center(50, "."))
        
        # --- GENERATOR ---
        gen_sys = (
            "You are a strict geography expert taking a multiple-choice test. "
            "Solve the problem step-by-step. Focus strictly on geographical facts. "
            "You MUST end your response with the final multiple choice letter inside brackets, like [A], [B], [C], or [D]."
        )
        if turn == 1:
            gen_prompt = f"Question: {question}\nChoices:\n{choices_text}\nAnswer:"
        else:
            gen_prompt = (f"Original Question: {question}\nChoices:\n{choices_text}\n"
                          f"Your Previous Attempt: {current_answer}\n"
                          f"Auditor Feedback: {feedback}\n\n"
                          f"The auditor has flagged an error in your factual knowledge. Re-evaluate the facts and provide a corrected answer.")
        
        current_answer = call_ollama(GENERATOR_MODEL, gen_prompt, gen_sys)
        
        # Tampilkan sebagian di terminal
        print(f"  [ 🤖 ] GENERATOR ({GENERATOR_MODEL}):\n  {current_answer[:200]}... [dipotong]")
        # Catat full jawaban Generator ke log
        write_to_log(f"\n[ 🤖 ] GENERATOR ({GENERATOR_MODEL}):\n{current_answer}\n")

        # --- VERIFIER ---
        ver_sys = (
            "You are a strict and highly skeptical geographical fact-checker. Your job is to verify the assistant's answer. "
            "CRITICAL RULES: "
            "1) Assume the premise of the multiple-choice question is 100% valid. DO NOT debate the validity of the question itself. "
            "2) Focus ONLY on whether the final chosen letter and the core fact are correct. Do not be swayed by convincing but factually wrong explanations. "
            "3) If the final letter and fact are absolutely correct, reply EXACTLY and ONLY with 'VERIFIED'. "
            "4) If the answer is wrong, DO NOT use the word 'VERIFIED'. State the correct geographical fact and the correct letter choice."
        )
        ver_prompt = f"Question: {question}\nChoices:\n{choices_text}\nAssistant Answer: {current_answer}\nIs the selected letter factually correct?"
        
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
    # Inisialisasi file log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"LLM DEBATE EVALUATOR LOG - GEOGRAPHY\n")
        f.write(f"Started at: {timestamp}\n")
        f.write(f"Dataset: MMLU High School Geography | Gen: {GENERATOR_MODEL} | Ver: {VERIFIER_MODEL}\n")
        f.write("="*60 + "\n")

    print("="*60)
    print("🚀 MEMULAI BENCHMARK EVALUATOR - GEOGRAFI (MULTI-AGENT)")
    print(f"Dataset: MMLU High School Geography | Gen: {GENERATOR_MODEL} | Ver: {VERIFIER_MODEL}")
    print("="*60)

    dataset = load_dataset("cais/mmlu", "high_school_geography", split="test[:20]")
    correct_count = 0
    total_questions = len(dataset)
    
    for index, item in enumerate(dataset):
        correct_letter = chr(65 + item['answer'])
        
        # Jalankan debat
        final_output, consensus = run_debate_on_question(item, index + 1)
        
        print(f"\n>>> HASIL AKHIR SOAL {index + 1}:")
        print(f"Jawaban Lengkap:\n{final_output}\n")
        
        # --- BLOK EVALUASI CERDAS (Multiple Choice) ---
        extracted = re.findall(r'\[([A-D])\]', final_output.upper())
        is_correct = False
        match_type = ""
        
        # Cek 1: Format Tepat dalam Kurung Siku
        if extracted:
            if extracted[-1] == correct_letter:
                is_correct = True
                match_type = "Format Tepat [...]"
        
        # Cek 2: Fallback Evaluation (Jika LLM lupa kurung siku, cari format "Answer: X" atau kemunculan huruf tunggal yang menonjol di akhir)
        if not is_correct:
            # Mencari pola seperti "answer is A", "Option B", dll di 50 karakter terakhir
            last_part = final_output.upper()[-50:]
            fallback_match = re.search(r'(?:ANSWER|OPTION|CHOICE|IS)[\s:]*([A-D])\b', last_part)
            if fallback_match and fallback_match.group(1) == correct_letter:
                is_correct = True
                match_type = "Fallback Deteksi Teks Akhir"
            elif f"[{correct_letter}]" in final_output.upper():
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

    # Kalkulasi Akurasi & Cetak Rekapitulasi
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