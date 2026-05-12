import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Mengatur tema global untuk nuansa paper akademis
sns.set_theme(style="whitegrid", font_scale=1.1)

# =====================================================================
# FIGURE 2: GROUPED BAR CHART (ACCURACY VS. HALLUCINATION) (Sebelumnya Fig 1)
# =====================================================================
def plot_figure_2():
    data = {
        'Domain': ['Math', 'Math', 'Geography', 'Geography', 'Biography', 'Biography', 'Physics', 'Physics'],
        'Agent Configuration': ['Single Model', 'Multi-Agent', 'Single Model', 'Multi-Agent', 'Single Model', 'Multi-Agent', 'Single Model', 'Multi-Agent'],
        'Reasoning Accuracy (RA)': [70, 80, 85, 90, 25, 30, 80, 85],
        'Hallucination Rate (HR)': [30, 20, 15, 10, 75, 70, 20, 15]
    }
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars=['Domain', 'Agent Configuration'], 
                        value_vars=['Reasoning Accuracy (RA)', 'Hallucination Rate (HR)'],
                        var_name='Metric', value_name='Percentage (%)')

    df_melted['Group'] = df_melted['Agent Configuration'] + " - " + df_melted['Metric'].str.extract(r'\((.*?)\)')[0]

    plt.figure(figsize=(12, 6))
    
    palette = {'Single Model - RA': '#4C72B0', 'Multi-Agent - RA': '#55A868', 
               'Single Model - HR': '#C44E52', 'Multi-Agent - HR': '#DD8452'}

    ax = sns.barplot(x='Domain', y='Percentage (%)', hue='Group', data=df_melted, palette=palette)
    
    plt.title('Fig. 2. Comparative analysis of Reasoning Accuracy (RA) vs. Hallucination Rate (HR)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Percentage (%)', fontweight='bold')
    plt.xlabel('Evaluation Domains', fontweight='bold')
    plt.ylim(0, 105)
    plt.legend(title='Metric Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', fontsize=9)
            
    plt.tight_layout()
    plt.savefig('Fig_2_Accuracy_vs_Hallucination.png', dpi=300)
    print("[OK] Figure 2 saved.")

# =====================================================================
# FIGURE 3: VRAM UTILIZATION AREA PLOT OVER TIME (Sebelumnya Fig 2)
# =====================================================================
def plot_figure_3():
    time = np.linspace(0, 60, 500)
    
    vram = np.full_like(time, 0.8) 
    vram += np.where((time > 5) & (time < 10), np.sin((time - 5) * np.pi / 5) * 1.5, 0)
    vram[time >= 10] += 2.2 
    vram += np.where((time > 20) & (time < 25), np.sin((time - 20) * np.pi / 5) * 0.8, 0)
    vram[time >= 25] += 1.2 
    vram[time > 25] += np.sin(time[time > 25] * 2) * 0.1 
    
    plt.figure(figsize=(10, 5))
    
    plt.fill_between(time, vram, color="#9b59b6", alpha=0.4)
    plt.plot(time, vram, color="#8e44ad", linewidth=2)
    plt.axhline(y=6, color='red', linestyle='--', linewidth=2, label='Hardware Ceiling (6 GB VRAM)')
    
    bbox_style = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85)

    plt.axvline(x=5, color='gray', linestyle=':', alpha=0.7)
    plt.text(5.5, 5.2, 'Load\nGenerator', va='center', ha='left', color='#333333', fontweight='bold', bbox=bbox_style)
    
    plt.axvline(x=20, color='gray', linestyle=':', alpha=0.7)
    plt.text(20.5, 5.2, 'Load Verifier\nContext', va='center', ha='left', color='#333333', fontweight='bold', bbox=bbox_style)
    
    plt.axvline(x=30, color='gray', linestyle=':', alpha=0.7)
    plt.text(45, 4.8, 'Active Debate Phase', ha='center', va='center', color='#333333', fontweight='bold', bbox=bbox_style)

    plt.title('Fig. 3. GPU VRAM consumption timeline during sequential agent loading', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Time (Seconds)', fontweight='bold')
    plt.ylabel('VRAM Utilization (GB)', fontweight='bold')
    
    plt.ylim(0, 7) 
    plt.xlim(0, 60)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('Fig_3_VRAM_Utilization.png', dpi=300)
    print("[OK] Figure 3 saved.")

# =====================================================================
# FIGURE 4: DONUT CHART OF ERROR BREAKDOWN (Sebelumnya Fig 3)
# =====================================================================
def plot_figure_4():
    labels = ['Factual Hallucination', 'Echo Chamber\n(Blind Consensus)', 
              'Format Non-compliance', 'Logical/Math Flaw']
    sizes = [45, 25, 15, 15] 
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    wedges, texts, autotexts = ax.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%',
                                      startangle=90, pctdistance=0.85, textprops={'fontsize': 11})
    
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    
    for text in texts:
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    plt.title('Fig. 4. Categorical distribution of residual failure modes', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('Fig_4_Error_Donut_Chart.png', dpi=300)
    print("[OK] Figure 4 saved.")

# Eksekusi semua fungsi
if __name__ == "__main__":
    print("Menghasilkan visualisasi untuk Paper...")
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    print("Selesai! Semua figure telah disimpan di direktori saat ini.")