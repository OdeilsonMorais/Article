# 16_figuras_artigo

Figuras do artigo geradas só a partir dos CSVs das outras pastas (não roda FEM).

    python3 16_figuras_artigo/figuras_artigo.py        # todas
    python3 16_figuras_artigo/figuras_artigo.py 7 12   # só as figuras 7 e 12

Saída em `figuras/` (PDF vetorial e PNG 600 dpi). Figuras: 1 esquema (célula, arranjos, arquiteturas), 2 convergência, 3 α crítico,
4 R_K AMM/DMM, 5 par, 6 aglomerados, 7 dispersões aleatórias, 8 ocas (mesma massa),
9 nanocristalinas, 10 mapa de projeto, 11 sensibilidade a k_p, 12 espessura crítica, 13 campo em torno da oca (lê os .npz de
09/estudos/ocas/dados, gerados por cortes_tc.py --calcular).
`figuras/_conferencia/` são folhas de conferência e pode ser apagada.
