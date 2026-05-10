import os
import time

# --- Konfigurasi File State ---
GENERATOR_FILE = "generator_output.txt"
VERIFIER_FILE = "verifier_feedback.txt"
PROMPT_FILE = "user_prompt.txt"

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

# --- Mock-up Agen (Pengganti LLM Sementara) ---
def mock_generator_agent(turn, prompt):
    """
    Simulasi SLM Generator. 
    Nantinya fungsi ini diganti dengan API Call ke Ollama (Model 1).
    """
    print(f"\n[Agent 1 - Generator] Sedang memproses (Turn {turn})...")
    time.sleep(1) # Simulasi waktu komputasi
    
    if turn == 1:
        output = f"JAWABAN AWAL untuk prompt: '{prompt}'. Ini adalah draf pertama saya."
    else:
        feedback = read_file(VERIFIER_FILE)
        output = f"JAWABAN REVISI (Berdasarkan feedback: '{feedback}'). Logika telah diperbaiki."
    
    write_file(GENERATOR_FILE, output)
    print("[Agent 1 - Generator] Output disimpan ke generator_output.txt")

def mock_verifier_agent(turn):
    """
    Simulasi SLM Verifier. 
    Nantinya fungsi ini diganti dengan API Call ke Ollama (Model 2).
    """
    print(f"\n[Agent 2 - Verifier] Menganalisis output Generator (Turn {turn})...")
    time.sleep(1) # Simulasi waktu komputasi
    
    generator_output = read_file(GENERATOR_FILE)
    
    # Simulasi logika pengecekan (misal: mencari dimensi fisika atau strict matching matematika)
    feedback = f"KRITIK: Evaluasi logika pada kalimat '{generator_output}'. Pastikan perhitungan/fakta konsisten."
    
    write_file(VERIFIER_FILE, feedback)
    print("[Agent 2 - Verifier] Feedback disimpan ke verifier_feedback.txt")

# --- Logika Looping Orchestrator ---
def run_debate(initial_prompt, max_turns=2):
    """
    Fungsi utama yang menjalankan siklus evaluasi.
    Sequential Loading bisa diterapkan di sini pada tahap produksi.
    """
    write_file(PROMPT_FILE, initial_prompt)
    
    for turn in range(1, max_turns + 1):
        print(f"\n{'='*15} TURN {turn} {'='*15}")
        
        # 1. Generator menghasilkan atau merevisi jawaban
        mock_generator_agent(turn, initial_prompt)
        
        # Jika ini adalah turn terakhir, Verifier tidak perlu mengkritik lagi
        if turn == max_turns:
            print("\n[SYSTEM] Batas maksimal turn tercapai. Evaluasi selesai.")
            break
            
        # 2. Verifier mengevaluasi jawaban
        mock_verifier_agent(turn)

if __name__ == "__main__":
    setup_workspace()
    
    # Contoh kasus uji awal
    test_prompt = "Berapa hasil dari integrasi f(x) dan validasi hukum termodinamika?"
    
    print("=== MEMULAI SISTEM MULTI-AGENT DEBATE ===")
    run_debate(initial_prompt=test_prompt, max_turns=3)
    
    print("\n=== HASIL AKHIR ===")
    print("Final Output dari Generator:")
    print(read_file(GENERATOR_FILE))