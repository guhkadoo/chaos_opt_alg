import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregamento dos dados a partir do arquivo CSV local
df = pd.read_csv('minimos_locais_coa.csv')

# 2. Configurações visuais do Matplotlib para estilo acadêmico/pôster
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True, dpi=300)

# Identifica a transição entre a Fase 1 (Exploração) e Fase 2 (Refinamento)
idx_segunda_onda = df[df['fase'].str.contains('segunda_onda')].index[0]
eval_divisao = df.loc[idx_segunda_onda, 'avaliacao']

# Sombreamento visual para indicar as fases do algoritmo
for ax in (ax1, ax2, ax3):
    ax.axvspan(df['avaliacao'].min(), eval_divisao, color='#e6f2ff', alpha=0.5, 
               label='Fase 1: Exploração Global' if ax == ax1 else "")
    ax.axvspan(eval_divisao, df['avaliacao'].max(), color='#eafaf1', alpha=0.5, 
               label='Fase 2: Refinamento Local Adaptativo' if ax == ax1 else "")
    ax.axvline(x=eval_divisao, color='#7f8c8d', linestyle='--', linewidth=1.2)
    ax.grid(True, linestyle='--', alpha=0.5, color='#b2bec3')

# --- PAINEL 1: Trajetória de Perdas de Potência Ativa (MW) ---
ax1.plot(df['avaliacao'], df['perdas_mw'], color='#0984e3', linewidth=2, marker='o', markersize=3, label='Perdas de Potência Ativa (MW)')
ax1.scatter(df.loc[0, 'avaliacao'], df.loc[0, 'custo'], color='#d63031', s=60, zorder=5, label='Caso Base (com violação)')
ax1.scatter(df.iloc[-1]['avaliacao'], df.iloc[-1]['perdas_mw'], color='#27ae60', s=70, zorder=5, edgecolor='black', 
            label=f"Ótimo Final: {df.iloc[-1]['perdas_mw']:.4f} MW")

ax1.set_ylabel('Perdas de Potência (MW)', fontweight='bold')
ax1.set_title('Trajetória Real de Convergência COA - Sistema IEEE 14 Barras', fontsize=12, fontweight='bold', pad=12)
ax1.legend(loc='upper right', framealpha=0.9)

# --- PAINEL 2: Envelope do Perfil de Tensão (pu) ---
ax2.plot(df['avaliacao'], df['vm_min'], color='#e67e22', linestyle='-', linewidth=1.5, label='V_min (Rede)')
ax2.plot(df['avaliacao'], df['vm_max'], color='#c0392b', linestyle='-', linewidth=1.5, label='V_max (Rede)')
ax2.plot(df['avaliacao'], df['vm_slack'], color='#2980b9', linestyle=':', label='V_slack')
ax2.plot(df['avaliacao'], df['vm_gen_0'], color='#8e44ad', linestyle='--', alpha=0.7, label='V_gen_0')
ax2.plot(df['avaliacao'], df['vm_gen_1'], color='#16a085', linestyle='--', alpha=0.7, label='V_gen_1')

# Limites operacionais (0,95 a 1,10 pu)
ax2.axhline(y=0.95, color='#d63031', linestyle='-', linewidth=1, alpha=0.7)
ax2.axhline(y=1.10, color='#d63031', linestyle='-', linewidth=1, alpha=0.7)

ax2.set_ylabel('Tensão no Barramento (p.u.)', fontweight='bold')
ax2.set_ylim(0.94, 1.11)
ax2.legend(loc='lower right', ncol=3, framealpha=0.9, fontsize=8)

# --- PAINEL 3: Evolução das Variáveis Discretas (Taps e Shunt) ---
ax3.step(df['avaliacao'], df['tap_0'], where='post', color='#2c3e50', linewidth=1.5, label='Tap 0')
ax3.step(df['avaliacao'], df['tap_1'], where='post', color='#d35400', linewidth=1.5, label='Tap 1')
ax3.step(df['avaliacao'], df['tap_2'], where='post', color='#27ae60', linewidth=1.5, label='Tap 2')
ax3.step(df['avaliacao'], df['shunt_step'], where='post', color='#8e44ad', linestyle='--', linewidth=1.5, label='Shunt Step')

ax3.set_xlabel('Número de Avaliações / Trials', fontweight='bold')
ax3.set_ylabel('Posição dos Controles', fontweight='bold')
ax3.legend(loc='center right', ncol=4, framealpha=0.9, fontsize=8)

plt.tight_layout()
plt.savefig('ieee14_convergencia_real_coa.png', bbox_inches='tight')
plt.show()