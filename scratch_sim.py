import pandas as pd

df = pd.read_csv('output/reports/reporte_20260603_195229.csv')
anom = df[df['estado']=='ANOMALIA'].copy()

def lines_prob_from_top3(top3_str):
    if not isinstance(top3_str, str): return 0.0
    for part in top3_str.split('|'):
        p = part.strip()
        if p.startswith('lines:'):
            try: return float(p.split(':')[1].rstrip('%')) / 100
            except: return 0.0
    return 0.0

def top2_labels(top3_str):
    if not isinstance(top3_str, str): return set()
    parts = [p.strip() for p in top3_str.split('|')][:2]
    return {p.split(':')[0] for p in parts}

DIGITS = {'0','1','2','3','4','5','6','7','8','9'}

anom['lines_prob'] = anom['top3'].apply(lines_prob_from_top3)
anom['top2'] = anom['top3'].apply(top2_labels)

anom['digit_on_lines'] = anom['top2'].apply(lambda t: bool(t & DIGITS) and 'lines' in t)
anom['dot_on_lines']   = anom.apply(lambda r: 'dot' in r['top2'] and 'lines' in r['top2'] and r['lines_prob'] >= 0.10, axis=1)
anom['multi_lines']    = anom.apply(lambda r: 'multiples_componentes' in str(r['motivo']) and r['lines_prob'] >= 0.15, axis=1)
anom['seria_eliminado'] = anom['digit_on_lines'] | anom['dot_on_lines'] | anom['multi_lines']

print('=== SIMULACION DE NUEVOS FILTROS ===')
print(f'Anomalias originales: {len(anom)}')
print(f'  Eliminadas por digit_on_lines: {anom["digit_on_lines"].sum()}')
print(f'  Eliminadas por dot_on_lines:   {anom["dot_on_lines"].sum()}')
print(f'  Eliminadas por multi_lines:    {anom["multi_lines"].sum()}')
print(f'  Total a eliminar (union):      {anom["seria_eliminado"].sum()}')
print(f'  Anomalias reales restantes:    {(~anom["seria_eliminado"]).sum()}')
print()
restantes = anom[~anom['seria_eliminado']]
print('=== ANOMALIAS QUE QUEDARIAN ===')
print('Motivos:')
print(restantes['motivo'].value_counts())
print()
print('Prediccion:')
print(restantes['prediccion'].value_counts())
print()
print('Muestra detalle:')
print(restantes[['prediccion','confianza','motivo','top3']].head(15).to_string())
