"""
Análisis de Tanques - Caso TR25 a TR5
Visualización de costos, volúmenes y eficiencia
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuración de estilo
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

def main():
    # Leer datos
    df = pd.read_csv('optimization_results_t5/sequence_tracking.csv')
    
    # Filtrar solo los pasos donde se agregó un tanque (step > 0)
    df_tanques = df[df['step'] > 0].copy()
    
    # Crear etiquetas para cada tanque
    df_tanques['label'] = df_tanques.apply(
        lambda x: f"{x['added_node']}\n(Predio {int(x['added_predio'])})", axis=1
    )
    
    # Calcular métricas adicionales
    df_tanques['costo_por_m3'] = df_tanques['current_tank_cost'] / df_tanques['current_tank_volume']
    df_tanques['costo_total_tanque'] = df_tanques['current_tank_cost'] + df_tanques['current_tank_land']
    df_tanques['costo_links_por_m'] = df_tanques['_marginal_cost'] / df_tanques['derivation_links_length']
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Gráfico de barras - Costo por tanque
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(range(len(df_tanques)), df_tanques['current_tank_cost']/1e6, 
                    color='steelblue', edgecolor='navy', alpha=0.8)
    ax1.set_xticks(range(len(df_tanques)))
    ax1.set_xticklabels([f"T{i+1}" for i in range(len(df_tanques))], rotation=45)
    ax1.set_ylabel('Costo (Millones USD)')
    ax1.set_title('Costo de Construcción por Tanque', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Agregar valores encima de barras
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}M',
                ha='center', va='bottom', fontsize=8)
    
    # 2. Gráfico de barras - Volumen por tanque
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(range(len(df_tanques)), df_tanques['current_tank_volume'], 
                    color='forestgreen', edgecolor='darkgreen', alpha=0.8)
    ax2.set_xticks(range(len(df_tanques)))
    ax2.set_xticklabels([f"T{i+1}" for i in range(len(df_tanques))], rotation=45)
    ax2.set_ylabel('Volumen (m³)')
    ax2.set_title('Volumen de Almacenamiento por Tanque', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Costo por m³
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(range(len(df_tanques)), df_tanques['costo_por_m3'], 
                    color='coral', edgecolor='darkred', alpha=0.8)
    ax3.set_xticks(range(len(df_tanques)))
    ax3.set_xticklabels([f"T{i+1}" for i in range(len(df_tanques))], rotation=45)
    ax3.set_ylabel('USD/m³')
    ax3.set_title('Costo Unitario por m³ Almacenado', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=df_tanques['costo_por_m3'].mean(), color='red', linestyle='--', 
                label=f'Media: ${df_tanques["costo_por_m3"].mean():.0f}/m³')
    ax3.legend()
    
    # 4. Longitud de tuberías de derivación
    ax4 = plt.subplot(2, 3, 4)
    bars4 = ax4.bar(range(len(df_tanques)), df_tanques['derivation_links_length'], 
                    color='mediumpurple', edgecolor='indigo', alpha=0.8)
    ax4.set_xticks(range(len(df_tanques)))
    ax4.set_xticklabels([f"T{i+1}" for i in range(len(df_tanques))], rotation=45)
    ax4.set_ylabel('Longitud (m)')
    ax4.set_title('Longitud de Tuberías de Derivación', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Acumulado - Inversión vs Reducción de Flooding
    ax5 = plt.subplot(2, 3, 5)
    ax5_twin = ax5.twinx()
    
    # Inversión acumulada
    inv_acum = df_tanques['cost_investment_total'].cumsum() / 1e6
    line1 = ax5.plot(range(len(df_tanques)), inv_acum, 'b-o', linewidth=2, 
                     markersize=6, label='Inversión Acumulada')
    ax5.set_ylabel('Inversión Total (Millones USD)', color='b')
    ax5.tick_params(axis='y', labelcolor='b')
    ax5.grid(alpha=0.3)
    
    # Reducción de flooding acumulada
    red_acum = df_tanques['flooding_reduction'].cumsum()
    line2 = ax5_twin.plot(range(len(df_tanques)), red_acum, 'r-s', linewidth=2, 
                          markersize=6, label='Reducción Flooding')
    ax5_twin.set_ylabel('Reducción Acumulada Flooding (m³)', color='r')
    ax5_twin.tick_params(axis='y', labelcolor='r')
    
    ax5.set_xticks(range(len(df_tanques)))
    ax5.set_xticklabels([f"T{i+1}" for i in range(len(df_tanques))], rotation=45)
    ax5.set_title('Inversión vs Reducción de Flooding', fontweight='bold')
    
    # Combinar leyendas
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='center right')
    
    # 6. Tabla resumen
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Crear tabla resumen
    tabla_data = []
    for idx, row in df_tanques.iterrows():
        tabla_data.append([
            f"T{int(row['n_tanks'])}",
            row['added_node'][:8],
            f"${row['current_tank_cost']/1e6:.2f}M",
            f"{row['current_tank_volume']:.0f}m³",
            f"${row['costo_por_m3']:.0f}/m³",
            row['target_model'][:10]
        ])
    
    tabla = ax6.table(
        cellText=tabla_data,
        colLabels=['Tanque', 'Nodo', 'Costo', 'Volumen', 'Costo/m³', 'Objetivo'],
        loc='center',
        cellLoc='center',
        colColours=['#4472C4']*6,
        colWidths=[0.1, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1.2, 1.5)
    
    # Colorear header
    for i in range(6):
        tabla[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax6.set_title('Resumen de Tanques Instalados', fontweight='bold', pad=20)
    
    # Título general
    fig.suptitle('Análisis de Tanques de Tormenta - Reducción TR25 a TR5\n' + 
                 f'Total: {len(df_tanques)} tanques | Inversión: ${df_tanques["cost_investment_total"].iloc[-1]/1e6:.1f}M | ' +
                 f'Reducción Flooding: {df_tanques["flooding_reduction"].iloc[-1]/1e3:.0f}k m³',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Guardar figura
    output_path = 'optimization_results_t5/analisis_tanques_completo.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Gráfico guardado en: {output_path}")
    
    # Mostrar estadísticas clave
    print("\n" + "="*60)
    print("RESUMEN EJECUTIVO - ANÁLISIS DE TANQUES")
    print("="*60)
    print(f"Total de tanques instalados: {len(df_tanques)}")
    print(f"Inversión total: ${df_tanques['cost_investment_total'].iloc[-1]/1e6:.2f} millones USD")
    print(f"Volumen total almacenado: {df_tanques['total_tank_volume'].iloc[-1]/1e3:.1f} mil m³")
    print(f"Reducción total de flooding: {df_tanques['flooding_reduction'].iloc[-1]/1e3:.1f} mil m³")
    print(f"Costo promedio por m³: ${df_tanques['costo_por_m3'].mean():.0f}/m³")
    print(f"Longitud total tuberías derivación: {df_tanques['derivation_links_length'].sum()/1e3:.1f} km")
    print("\nDesglose por objetivo:")
    for obj in df_tanques['target_model'].unique():
        count = df_tanques[df_tanques['target_model'] == obj].shape[0]
        print(f"  - {obj}: {count} tanques")
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    main()
