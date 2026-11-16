# 05_N_particulas

Mesma formulação do 04 (RVE periódica, dois campos P1 com salto de Kapitza,
três níveis de h_bulk e Aitken), generalizada para N esferas iguais.

## O que mudou em relação ao 04

- `geometria.py`: `geometria_de_centros(lado, raio, centros)` aceita qualquer
  lista de centros e calcula a menor lacuna interna e até as imagens
  periódicas. `preparar_geometria` (o par) continua igual.
- `malha.py`: a partícula j recebe o volume 2 + j e a interface 11 + j (no par,
  2, 3, 11 e 12, como antes); até 9 partículas. Cada par vizinho com
  s < 4 h_int recebe as esferas concêntricas de refino da lacuna. A opção
  `engrossar_longe` faz o tamanho crescer até 4 h_bulk longe das esferas,
  como em `04/estudos/uma_esfera`.
- `periodicidade.py`, `solver.py`, `pos_processamento.py`: as etiquetas das
  partículas e interfaces são lidas da malha, então servem para qualquer N.
- `solver.resolver_kapitza_gradientes`: resolve vários gradientes com uma só
  fatoração LU (o tensor sai pelo custo de pouco mais de um caso).

Para s/R <= 0,4 e sem `engrossar_longe`, a malha do par é idêntica à do 04
(conferido no Gmsh, nó por nó).

## Estudo de aglomerados

`estudos/aglomerados/aglomerados.py`: isolada, par, cadeia3, cadeia4,
triangulo, quadrado, tetraedro, octaedro, cubo, todos com a mesma phi total
(padrão 2,44 %, a do par com L = 7R). Cada aglomerado é comparado com
Hasselman–Johnson no mesmo phi. Detalhes e exemplos no cabeçalho do script.
