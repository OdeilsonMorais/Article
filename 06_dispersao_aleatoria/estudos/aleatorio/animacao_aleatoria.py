"""Animação de um sorteio: o plano de corte varre a célula e mostra T, q no plano e q normal.

G é sempre ao longo de x (face "quente" em x = 0). O plano é xy (varrendo z,
padrão) ou xz (varrendo y), da face 0 até a face L. Cada quadro tem quatro
painéis:
  * a célula em 3D com o plano na posição atual;
  * a temperatura T/(G R), com isotermas;
  * |q|/(k_m G) em escala log, com as linhas de fluxo da parte de q no plano;
  * a componente de q normal ao plano (q_z ou q_y), em escala divergente:
    vermelho sai pela face de cima do plano, azul pela de baixo.
As escalas de cor são as mesmas em todos os quadros, para dar para comparar.

Duas etapas, como em cortes_aleatorios.py (rodar de 06_dispersao_aleatoria):

1. Calcular (FEniCSx, em série; resolve uma vez e amostra todos os planos):
       python3 estudos/aleatorio/animacao_aleatoria.py --calcular --phi 0.1 --semente 6
2. Desenhar (só Matplotlib; refaz o vídeo sem resolver de novo):
       python3 estudos/aleatorio/animacao_aleatoria.py --phi 0.1 --semente 6

Saída em estudos/aleatorio/figuras/animacao_..._plano_xy.mp4 (ou .gif, se o
ffmpeg não estiver instalado; no WSL: sudo apt install ffmpeg). Com --formato
gif força o GIF.
"""

import argparse
import shutil
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))

import numpy as np  # noqa: E402

from cortes_aleatorios import (EIXOS, TINTA_SECUNDARIA, _circulos, _estilo, escolher_corte,  # noqa: E402
                               ler_sorteios, nome_do_caso, vista_3d)


def calcular(linha, phi, rk, normal, quadros, pontos, h_bulk):
    from cortes_aleatorios import amostrar_plano, resolver

    lado = float(linha["lado_R"])
    ctx = resolver(linha["centros"], lado, phi, rk, h_bulk, eixo=0)
    posicoes = np.linspace(0, lado, quadros)
    pilhas = {}
    for k, plano in enumerate(posicoes):
        d = amostrar_plano(ctx, normal, plano, pontos)
        for chave in ("T", "q_u", "q_v", "q_n", "q_modulo"):
            pilhas.setdefault(chave, []).append(d[chave].astype(np.float32))
        print(f"  plano {EIXOS[normal]} = {plano:.3f} R ({k + 1}/{quadros})", flush=True)
    return dict(U=d["U"], V=d["V"], posicoes=posicoes, normal=normal, eixo=0,
                **{chave: np.stack(valores) for chave, valores in pilhas.items()})


def desenhar(plt, linha, d, destino: Path, formato: str, fps: int) -> Path:
    from matplotlib import animation
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    centros, lado = linha["centros"], float(linha["lado_R"])
    normal = int(d["normal"])
    a, b = [EIXOS[i] for i in range(3) if i != normal]
    U, V = d["U"], d["V"]
    extensao = (U.min(), U.max(), V.min(), V.max())
    par = escolher_corte(centros)[0]
    # Escalas fixas em todos os quadros.
    T_min, T_max = float(d["T"].min()), float(d["T"].max())
    niveis = np.linspace(T_min, T_max, 22)
    norma_q = LogNorm(vmin=0.1, vmax=max(float(np.nanpercentile(d["q_modulo"], 99.5)), 1.5))
    limite_n = max(float(np.nanpercentile(np.abs(d["q_n"]), 99.5)), 0.05)
    norma_n = TwoSlopeNorm(vcenter=0, vmin=-limite_n, vmax=limite_n)
    phi = 100 * float(linha["phi"])
    titulo = (f"Sorteio {linha['semente']}   ·   φ = {phi:.3g} %   ·   G ao longo de x   ·   "
              f"k médio {float(linha['desvio_HJ_percent']):+.2f} % vs HJ").replace(".", ",")

    figura = plt.figure(figsize=(19.2, 5.6), dpi=100)
    grade = figura.add_gridspec(1, 4, width_ratios=(0.85, 1, 1, 1), left=0.01, right=0.97,
                                bottom=0.11, top=0.84, wspace=0.32)
    eixo3d = figura.add_subplot(grade[0, 0], projection="3d")
    eixo_T, eixo_q, eixo_n = (figura.add_subplot(grade[0, k]) for k in (1, 2, 3))
    # Barras de cor desenhadas uma vez só.
    for eixo, mapa, norma, rotulo in ((eixo_T, "inferno", plt.Normalize(T_min, T_max), "T / (G R)"),
                                      (eixo_q, "magma", norma_q, "|q| / (k_m G)"),
                                      (eixo_n, "RdBu_r", norma_n, f"q_{EIXOS[normal]} / (k_m G)")):
        figura.colorbar(plt.cm.ScalarMappable(norm=norma, cmap=mapa), ax=eixo, shrink=0.85, label=rotulo)
    figura.suptitle(titulo, fontsize=15, color=TINTA_SECUNDARIA)

    def quadro(k):
        plano = float(d["posicoes"][k])
        corte = {"normal": normal, "plano": plano}
        for eixo in (eixo3d, eixo_T, eixo_q, eixo_n):
            eixo.cla()
        vista_3d(eixo3d, centros, lado, (par, normal, plano, 0))
        eixo3d.set_title(f"{EIXOS[normal]} = {plano:.2f} R".replace(".", ","), fontsize=14)

        eixo_T.imshow(d["T"][k], origin="lower", extent=extensao, cmap="inferno",
                      vmin=T_min, vmax=T_max, interpolation="bilinear")
        eixo_T.contour(U, V, d["T"][k], levels=niveis, colors="white", linewidths=0.6, alpha=0.7,
                       linestyles="solid")
        eixo_T.set_title("Temperatura")

        eixo_q.imshow(np.clip(d["q_modulo"][k], 0.1, None), origin="lower", extent=extensao,
                      cmap="magma", norm=norma_q, interpolation="bilinear")
        eixo_q.streamplot(U[0, :], V[:, 0], d["q_u"][k], d["q_v"][k], color=(1, 1, 1, 0.55),
                          linewidth=0.7, density=1.0, arrowsize=0.7)
        eixo_q.set_title("|q| e linhas de fluxo no plano")

        eixo_n.imshow(d["q_n"][k], origin="lower", extent=extensao, cmap="RdBu_r", norm=norma_n,
                      interpolation="bilinear")
        eixo_n.set_title(f"Fluxo normal ao plano, q_{EIXOS[normal]}")

        for eixo, cor in ((eixo_T, "white"), (eixo_q, "white"), (eixo_n, "#1f2a44")):
            _circulos(eixo, centros, corte, cor)
            eixo.set_xlim(extensao[:2]), eixo.set_ylim(extensao[2:])
            eixo.set_xlabel(f"{a}/R"), eixo.set_ylabel(f"{b}/R"), eixo.set_aspect("equal")
        return []

    if formato == "mp4" and shutil.which("ffmpeg") is None:
        print("ffmpeg não encontrado: gravando GIF (no WSL: sudo apt install ffmpeg).")
        formato = "gif"
    destino = destino.with_name(f"{destino.name}.{formato}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    escritor = (animation.FFMpegWriter(fps=fps, bitrate=4000) if formato == "mp4"
                else animation.PillowWriter(fps=fps))
    filme = animation.FuncAnimation(figura, quadro, frames=len(d["posicoes"]), blit=False)
    filme.save(destino, writer=escritor,
               progress_callback=lambda i, n: print(f"  quadro {i + 1}/{n}", end="\r", flush=True))
    plt.close(figura)
    print(f"\nAnimação: {destino}")
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calcular", action="store_true", help="Resolver e amostrar os planos (FEniCSx).")
    parser.add_argument("--phi", type=float, default=0.1)
    parser.add_argument("--semente", type=int, default=6)
    parser.add_argument("--rk", type=float, default=1e-10)
    parser.add_argument("--plano", choices=("xy", "xz"), default="xy",
                        help="xy varre z (padrão); xz varre y.")
    parser.add_argument("--quadros", type=int, default=60)
    parser.add_argument("--pontos", type=int, default=260, help="Pontos por lado em cada plano.")
    parser.add_argument("--h-bulk", type=float, default=0.25)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--formato", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--csv", default=str(PASTA / "aleatorio.csv"))
    args = parser.parse_args()

    sorteios = ler_sorteios(Path(args.csv), args.phi, args.rk)
    if args.semente not in sorteios:
        raise SystemExit(f"Semente {args.semente} sem linha no CSV para phi={args.phi:g}, R_K={args.rk:g}.")
    linha = sorteios[args.semente]
    normal = {"xy": 2, "xz": 1}[args.plano]
    nome = f"animacao_{nome_do_caso(args.phi, args.semente, args.rk)[len('aleatorio_'):]}_plano_{args.plano}"
    arquivo = PASTA / "dados" / f"{nome}.npz"
    if args.calcular:
        if arquivo.exists():
            print(f"Já existe {arquivo.name}; desenhando a partir dele.")
        else:
            arquivo.parent.mkdir(exist_ok=True)
            d = calcular(linha, args.phi, args.rk, normal, args.quadros, args.pontos, args.h_bulk)
            np.savez_compressed(arquivo, **d)
            print(f"Dados: {arquivo}")
    if not arquivo.exists():
        raise SystemExit(f"Falta {arquivo.name}: rode com --calcular primeiro.")
    d = dict(np.load(arquivo))
    desenhar(_estilo(), linha, d, PASTA / "figuras" / nome, args.formato, args.fps)


if __name__ == "__main__":
    main()
