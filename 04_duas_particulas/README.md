# Duas nanopartículas periódicas com resistência de Kapitza

RVE cúbica 3D periódica com duas esferas idênticas em uma matriz. Resolve a
condução estacionária com contato perfeito e com resistência de Kapitza nas
interfaces e calcula `k_eff,xx`.

Dependências: FEniCSx 0.11 (dolfinx, basix, ufl), Gmsh, mpi4py, petsc4py,
NumPy e SciPy. O `dolfinx_mpc` **não** é mais necessário (ver abaixo).

## Organização

```text
duas_particulas_periodico/
  main.py                 # entrada: prepara e mostra um caso
  parametros.py           # propriedades, dimensões e validação
  geometria.py            # tamanho da célula e centros das esferas
  malha.py                # malha Gmsh/FEniCSx e marcação de regiões
  periodicidade.py        # identificação periódica da flutuação térmica
  kapitza.py              # condutância térmica de Kapitza
  solver.py               # contato perfeito e Kapitza (dois campos P1)
  pos_processamento.py    # fluxo médio, k_eff,xx e salto RMS na interface
  varreduras.py           # combinações de distância e orientação
  resultados/             # reservado para saídas geradas
```

## Executar a preparação e a malha

Na pasta deste projeto, execute `python main.py`.
Para listar a varredura preliminar, execute `python main.py --varredura`.
No Linux com FEniCSx 0.11, Gmsh, mpi4py e petsc4py, execute
`python3 main.py --malha` para gerar a malha inicial em
`resultados/malha_inicial.msh`.
Essa primeira malha usa `h = R/2` e o algoritmo paralelo HXT. Para limitar o
uso de CPU, por exemplo a oito núcleos, execute `python3 main.py --malha --threads 8`.
Sem `--threads`, o Gmsh recebe o número de CPUs lógicas visível no sistema.
Para refinar a malha, defina `h/R` com `--h-relativo`; por exemplo,
`python3 main.py --malha --h-relativo 0.25 --threads 8` gera elementos com
comprimento característico de `R/4`.
Internamente, o Gmsh constrói a geometria em unidades de raio para evitar
problemas numéricos com dimensões nanométricas; a malha importada no FEniCSx é
convertida de volta para metros.

### Malha adaptativa

Use a malha uniforme apenas como referência. Para os cálculos físicos, ative
o refinamento local com:

```bash
python3 main.py --malha --malha-adaptativa --h-relativo 0.5 \
  --h-interface-relativo 0.1 --threads 8
```

Nesse comando, `h/R = 0,5` é o tamanho máximo no volume distante das esferas;
as interfaces recebem `h_int/R = 0,1`. O Gmsh combina um campo `Distance` das
superfícies esféricas com um `Threshold` e uma caixa local na lacuna interna
entre as partículas. Por padrão, o tamanho na lacuna é
`min(h_int, s/4)`: há ao menos quatro elementos através da menor lacuna.

Os controles são `--h-interface-relativo`, `--h-lacuna-relativo` e
`--alcance-refino-relativo`. Para a separação atual `s/R = 0,5`, o comando
acima usa `h_lacuna/R = 0,1` e produz cinco elementos através da lacuna. Para
uma separação futura `s/R = 0,05`, o valor automático passa a `0,0125`.

Para resolver diretamente com essa malha, execute:

```bash
python3 main.py --solver-kapitza --rk 1e-8 --malha-adaptativa \
  --h-relativo 0.5 --h-interface-relativo 0.1 --threads 8
```

Faça a convergência reduzindo `--h-interface-relativo` (por exemplo `0.125`,
`0.10`, `0.075`) e mantenha o caso somente quando a variação de `k_eff,xx`
entre os dois últimos níveis ficar abaixo de 0,05%. Verifique também se a área
das duas interfaces se aproxima de `8πR²`.

Os dois primeiros comandos apenas imprimem parâmetros e geometria. O comando
`--malha` cria a malha tetraédrica 3D com as etiquetas físicas. O comando
`python3 main.py --periodicidade` gera a malha em memória e verifica os pares
periódicos da flutuação térmica.

O comando `python3 main.py --solver-perfeito --threads 8` resolve o caso de
referência de contato perfeito (temperatura contínua nas interfaces).

O comando `python3 main.py --solver-kapitza --rk 1e-8 --h-relativo 0.25 --threads 8`
resolve a mesma RVE com a lei `q_n = [T]/R_K` nas duas interfaces e informa
também a área das interfaces e o salto RMS de temperatura.

### Tensor de condutividade efetiva

Acrescente `--tensor` a um dos solvers para impor, sucessivamente,
`G = (1,0,0)`, `(0,1,0)` e `(0,0,1)` K/m. As colunas de `K_eff` são os
fluxos médios resultantes:

```bash
python3 main.py --solver-kapitza --tensor --rk 1e-8 --malha-adaptativa \
  --h-relativo 0.5 --h-interface-relativo 0.075 --threads 8
```

O resultado é a matriz completa `K_eff` em W/(m K). Para a orientação
longitudinal, espera-se `K_xx` diferente de `K_yy ≈ K_zz`; para a transversal,
espera-se `K_yy` diferente de `K_xx ≈ K_zz`. Os termos fora da diagonal devem
ser pequenos para a geometria simétrica.

Os dois solvers funcionam com `mpirun -np N python3 main.py ...`. A fatoração
do sistema reduzido é feita no processo 0 (LU, MUMPS se disponível).

## Formulação numérica

- `T(x) = -G·x + theta(x)`, com `theta` periódica; `<q> = K_eff G`.
- **Kapitza**: dois campos contínuos P1, `theta_m` (matriz) e `theta_p`
  (partículas), que coexistem nos nós da interface. O termo
  `∫_Γ (1/R_K) [theta][v] dS`, com `[theta] = theta_m - theta_p`, é a forma
  fraca da lei de Kapitza. DOFs fora da região de cada campo são fixados em
  zero. É a formulação de `03_rve_periodica/mapa_R_RK_phi`.
- **Malha periódica**: `gmsh.model.mesh.setPeriodic` faz as faces opostas
  terem nós coincidentes. A periodicidade vira a identificação exata
  `theta(escravo) = theta(mestre)` (coeficiente 1), aplicada pela redução
  `K^T A K u_r = K^T b`.
- **Âncora** `theta(0,0,0) = 0`: imposta no sistema já reduzido.
- **Ghost mode** `shared_facet`: necessário para integrar `dS` em paralelo.

### Por que não `dolfinx_mpc`

1. Na versão 0.11, `dolfinx_mpc.assemble_matrix` não aceita integrais em
   facetas internas (`dS`): "Not implemented yet".
2. `create_periodic_constraint_geometrical` interpola o ponto-imagem e cria
   mestres espúrios (coeficientes ~1e-15), às vezes também escravos. Isso
   deixava o sistema reduzido singular.
3. Com malha não periódica, a periodicidade por interpolação dava
   `k_eff = 0,19965` num meio homogêneo (`k_p = k_m = 0,2`), em vez de 0,2.

### Verificações (h = R/2, φ = 0,05, s/R = 0,5, longitudinal)

| Teste | Resultado |
|---|---|
| Meio homogêneo, contato perfeito | k = 0,2000000 (exato) |
| Meio homogêneo, Kapitza R_K = 1e-16 | k = 0,2000000 |
| Kapitza R_K → 0 × contato perfeito | diferença ~1e-9 |
| Kapitza R_K → ∞ | tende ao limite de esferas isolantes (Maxwell: 0,1854) |
| Componentes q_y/q_x, q_z/q_x | ~1e-4 (simetria) |
| Serial × `mpirun -np 2` | mesmos valores (a menos da malha) |

### Convergência de malha (malha uniforme)

| h/R | k perfeito | k Kapitza (R_K = 1e-8) | tempo |
|---|---|---|---|
| 0,50 | 0,23923 | 0,23270 | 1 s |
| 0,35 | 0,23680 | 0,23116 | 2 s |
| 0,25 | 0,23566 | 0,23047 | 2 s |
| 0,18 | 0,23491 | 0,23002 | 5 s |

A convergência uniforme é monotônica, mas lenta (~h^1,3): a esfera facetada
perde até 4 % de área com h = R/2 e a lacuna entre as esferas não é resolvida.
O refinamento adaptativo acima substitui essa malha para a próxima série de
convergência.

## Convenções

- Unidades SI: comprimento em m, condutividade em W/(m K), resistência
  interfacial por área em m² K/W e gradiente em K/m.
- Raio inicial: 50 nm; matriz: 0,2 W/(m K); partícula: 5 W/(m K).
- Fração volumétrica inicial: 0,05, escolhida apenas para preparar os casos.
- O valor de R_K permanece indefinido até escolhermos um caso físico.
- `s/R` é a distância entre superfícies dividida pelo raio. A orientação
  longitudinal alinha os centros em x; a transversal, em y; fluxo imposto em x.
- L = (8 pi R³ / (3 phi))^(1/3); mover as esferas mantém a fração volumétrica.
- A varredura mantém `φ` fixa, portanto mantém `L` fixo. Para cada caso,
  `Geometria` registra a lacuna interna, a lacuna até a imagem periódica mais
  próxima e a menor delas.
- Para `φ = 0,05`, `L/R = 5,51293`. A lista de separações é
  `s/R = 1,0; 0,5; 0,2; 0,1; 0,05` em cada orientação. O antigo caso `1,5`
  foi removido: deixa somente `0,01293R = 0,647 nm` entre as superfícies de
  uma partícula e a imagem periódica da outra (ou `0,323 nm` até a face).
- Contato entre esferas (s = 0) fica fora desta etapa: exige modelo próprio.

## Próximas etapas

1. Repetir a convergência com a malha adaptativa e fixar o nível que dá
   variação inferior a 0,05% em `k_eff,xx`.
2. Executar as varreduras de distância, orientação e R_K com o tensor.
