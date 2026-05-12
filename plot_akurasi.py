import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mengumpulkan data akurasi berdasarkan hasil log evaluasi
data = {
    # NAMA KOLOM DIUBAH KE BAHASA INGGRIS AGAR COCOK DENGAN BARPLOT
    'Category': [
        'Math\n(GSM8K)', 'Math\n(GSM8K)',
        'Physics\n(MMLU)', 'Physics\n(MMLU)',
        'Geography\n(MMLU)', 'Geography\n(MMLU)',
        'Biography\n(TriviaQA)', 'Biography\n(TriviaQA)'
    ],
    'Method': [
        'Single Agent (Baseline)', 'Multi-Agent (Debate)',
        'Single Agent (Baseline)', 'Multi-Agent (Debate)',
        'Single Agent (Baseline)', 'Multi-Agent (Debate)',
        'Single Agent (Baseline)', 'Multi-Agent (Debate)'
    ],
    'Accuracy (%)': [
        70.0, 80.0,  # Matematika
        80.0, 85.0,  # Fisika
        85.0, 90.0,  # Geografi
        25.0, 30.0   # Biografi
    ]
}

df = pd.DataFrame(data)

# Mengatur tema visual Seaborn
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(11, 7))

# Membuat diagram batang dengan palet warna yang profesional
ax = sns.barplot(
    x='Category', 
    y='Accuracy (%)', 
    hue='Method', 
    data=df, 
    palette=['#4C72B0', '#DD8452']
)

# Menambahkan anotasi label persentase tepat di atas setiap batang
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontweight='bold',
                    fontsize=10)

# Kustomisasi judul dan label sumbu
plt.title('Accuracy Evaluation: Single Agent vs Multi-Agent', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Accuracy (%)', fontweight='bold', labelpad=10)
plt.xlabel('Domain & Dataset', fontweight='bold', labelpad=10)

# Menyesuaikan batas sumbu Y agar anotasi tidak terpotong
plt.ylim(0, 105)

# Konfigurasi legend
plt.legend(title='Method of approach', loc='upper right', framealpha=1)

# Merapikan tata letak sebelum disimpan
plt.tight_layout()

# Menyimpan hasil plot sebagai file gambar
plt.savefig('Perbandingan_Akurasi_Memora.png', dpi=300)
plt.show()