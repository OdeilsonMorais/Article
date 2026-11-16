#!/usr/bin/env bash
# Monta o repositório público do artigo a partir da pasta kapitza. Rodar no Git Bash:
#   bash montar_repositorio.sh /e/Area/.../Simulações/kapitza /c/caminho/Article
# O segundo argumento é o clone do github.com/OdeilsonMorais/Article.
set -euo pipefail
ORIGEM="$1"; DESTINO="$2"
PASTAS="04_duas_particulas 05_N_particulas 06_dispersao_aleatoria 07_resistencia_interface \
09_particulas_ocas 11_muitas_particulas 12_nanocristalino 14_reforcos_teoria 15_bidispersa \
16_figuras_artigo 17_tc_alfa 18_vacancias 19_espumas_sinteticas"
AQUI="$(cd "$(dirname "$0")" && pwd)"
cp "$AQUI/LICENSE" "$AQUI/README.md" "$AQUI/CITATION.cff" "$AQUI/.gitignore" "$DESTINO/"
for p in $PASTAS; do
  echo "copiando $p"
  cp -r "$ORIGEM/$p" "$DESTINO/"
done
# remove o que o .gitignore já ignoraria, para o tamanho ficar visível
find "$DESTINO" \( -name "__pycache__" -o -name "*.msh" -o -name "*.npz" -o -name "*.npy" \
  -o -name "*.h5" -o -name "*.xdmf" -o -name "*.log" \) -prune -exec rm -rf {} +
du -sh "$DESTINO"/* | sort -h | tail -15
echo "Confira o conteúdo e depois: cd \"$DESTINO\" && git add . && git status"
