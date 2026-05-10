import os
import time
import requests
import json

# --- Konfigurasi File State ---
GENERATOR_FILE = "generator_output.txt"
VERIFIER_FILE = "verifier_feedback.txt"
PROMPT_FILE = "user_prompt.txt"

# --- Konfigurasi Ollama ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Anda bisa mengganti model di bawah ini sesuai yang Anda unduh di Ollama
# Contoh untuk heterogeneous (model berbeda): GENERATOR = "llama3", VERIFIER = "qwen"
GENERATOR_MODEL = "gemma:2b" 
VERIFIER_MODEL = "gemma:2b"

def setup_workspace():
    """Membersihkan workspace sebelum eksperimen dimulai."""
    for file in [GENERATOR_FILE, VERIFIER_FILE, PROMPT_FILE]:
        if os.path.exists(file):
            os.remove(file)
    print("[SYSTEM] Workspace dibersihkan. Memulai sekuens...")

def read_file(filepath):
    """Membaca isi file teks."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def write_file(filepath, content):
    """Menulis konten ke file teks."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def call_ollama(model_name, prompt, system_prompt):
    """Fungsi pembantu untuk memanggil API lokal Ollama."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False, # Kita set False agar jawaban diterima
        "keep_alive": 0 #utuh sekaligus (bukan per kata)
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status() # Akan memicu error jika server Ollama mati
        result = response.json()
        return result.get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Gagal menghubungi Ollama. Pastikan Ollama menyala! Detail: {e}")
        return "[ERROR SERVER LLM]"

# --- Agen Berbasis LLM ---
def generator_agent(turn, user_prompt):
    """Agen 1: Bertugas menjawab pertanyaan dan memperbaiki jawaban berdasarkan kritik."""
    print(f"\n[Agent 1 - Generator ({GENERATOR_MODEL})] Sedang memikirkan jawaban (Turn {turn})...")
    
    system_prompt = "Anda adalah asisten ahli yang logis dan detail. Jawab pertanyaan pengguna dengan seakurat mungkin. Jika diberikan kritik, perbaiki jawaban Anda sebelumnya."
    
    if turn == 1:
        # Putaran pertama: Langsung jawab pertanyaan user
        prompt = f"Pertanyaan Pengguna: {user_prompt}\nBerikan jawaban terbaik Anda."
    else:
        # Putaran selanjutnya: Jawab berdasarkan kritik dari Verifier
        feedback = read_file(VERIFIER_FILE)
        previous_answer = read_file(GENERATOR_FILE)
        prompt = (f"Pertanyaan Awal: {user_prompt}\n"
                  f"Jawaban Anda Sebelumnya: {previous_answer}\n\n"
                  f"Kritik dari Auditor: {feedback}\n\n"
                  f"Berdasarkan kritik di atas, tolong perbaiki dan berikan jawaban akhir Anda yang lebih akurat dan logis.")
    
    output = call_ollama(GENERATOR_MODEL, prompt, system_prompt)
    write_file(GENERATOR_FILE, output)
    print(f"[Agent 1] Selesai! Output disimpan ke {GENERATOR_FILE}")

def verifier_agent(turn):
    """Agen 2: Bertugas mencari celah, cacat logika, atau halusinasi dari Generator."""
    print(f"\n[Agent 2 - Verifier ({VERIFIER_MODEL})] Menganalisis output Generator (Turn {turn})...")
    
    system_prompt = "Anda adalah auditor logika yang skeptis dan kritis. Tugas Anda HANYA MENGKRITIK jawaban, mencari cacat logika, ketidakakuratan fakta, atau halusinasi. Jangan memberikan jawaban langsung kepada pengguna. Berikan kritik maksimal dalam 3 kalimat yang padat."
    
    generator_output = read_file(GENERATOR_FILE)
    user_prompt = read_file(PROMPT_FILE)
    
    prompt = (f"Pertanyaan Awal Pengguna: {user_prompt}\n"
              f"Jawaban Asisten yang harus dievaluasi: {generator_output}\n\n"
              f"Berikan kritik tajam Anda terhadap jawaban asisten tersebut.")
    
    feedback = call_ollama(VERIFIER_MODEL, prompt, system_prompt)
    write_file(VERIFIER_FILE, feedback)
    print(f"[Agent 2] Selesai! Kritik disimpan ke {VERIFIER_FILE}")

# --- Logika Looping Orchestrator ---
def run_debate(initial_prompt, max_turns=2):
    write_file(PROMPT_FILE, initial_prompt)
    
    for turn in range(1, max_turns + 1):
        print(f"\n{'='*15} TURN {turn} {'='*15}")
        
        # 1. Generator
        generator_agent(turn, initial_prompt)
        
        # Berhenti jika ini turn terakhir
        if turn == max_turns:
            print("\n[SYSTEM] Batas maksimal turn tercapai. Evaluasi selesai.")
            break
            
        # 2. Verifier
        verifier_agent(turn)

if __name__ == "__main__":
    setup_workspace()
    
    # Prompt pengujian (Sengaja dibuat agak menjebak untuk menguji Verifier)
    test_prompt = "Jika saya butuh waktu 10 menit untuk merebus 1 telur, berapa lama waktu yang dibutuhkan untuk merebus 5 telur secara bersamaan?"
    
    print("=== MEMULAI SISTEM MULTI-AGENT DEBATE (OLLAMA) ===")
    # Kita set 2 putaran saja untuk tes awal (1x Jawab awal, 1x Kritik, 1x Jawab Revisi)
    run_debate(initial_prompt=test_prompt, max_turns=2)
    
    print("\n=== HASIL AKHIR ===")
    print("Final Output dari Generator:")
    print(read_file(GENERATOR_FILE))