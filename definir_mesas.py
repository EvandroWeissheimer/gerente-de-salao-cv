"""Ferramenta para desenhar as mesas sobre um quadro do vídeo/câmera.

Uso:
    python definir_mesas.py --fonte videos/salao.mp4 --saida config/mesas.json

Controles:
    clique esquerdo   adiciona um ponto ao polígono da mesa atual
    clique direito    remove o último ponto
    ENTER ou n        fecha a mesa atual (mínimo 3 pontos)
    u                 desfaz a última mesa fechada
    a / d             volta / avança 1 segundo no vídeo
    s                 salva o JSON e a imagem de referência
    q ou ESC          sai

Dica: desenhe cada área cobrindo a mesa E as cadeiras (onde os clientes
ficam sentados) e salve num quadro em que as mesas estejam LIMPAS e
VAZIAS: esse quadro vira a referência usada para detectar mesa suja.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from salao.config import ConfigMesa, Parametros, carregar_config, salvar_config
from salao.janela import maximizar

JANELA = "Definir mesas"
AJUDA = ["Clique: ponto | Dir.: desfaz ponto | ENTER/n: fecha mesa",
         "u: desfaz mesa | a/d: -/+ 1s no video | s: salvar | q: sair"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Desenha as áreas das mesas")
    ap.add_argument("--fonte", required=True, help="vídeo, índice da webcam (0) ou URL RTSP")
    ap.add_argument("--saida", default="config/mesas.json")
    ap.add_argument("--largura", type=int, default=1280, help="mesma largura usada em monitorar.py")
    args = ap.parse_args()

    captura = cv2.VideoCapture(int(args.fonte) if args.fonte.isdigit() else args.fonte)
    eh_arquivo = Path(args.fonte).exists()
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0

    def ler_quadro():
        ok, q = captura.read()
        if not ok:
            return None
        if args.largura and q.shape[1] != args.largura:
            escala = args.largura / q.shape[1]
            q = cv2.resize(q, (args.largura, round(q.shape[0] * escala)), interpolation=cv2.INTER_AREA)
        return q

    quadro = ler_quadro()
    if quadro is None:
        sys.exit(f"Não foi possível ler a fonte: {args.fonte}")
    altura, largura = quadro.shape[:2]

    # Carrega mesas existentes para permitir editar/adicionar
    mesas: list[list[tuple[int, int]]] = []
    nomes: list[str] = []
    parametros = Parametros()
    saida = Path(args.saida)
    if saida.exists():
        cfg = carregar_config(saida)
        parametros = cfg.parametros
        nomes = [m.nome for m in cfg.mesas]
        mesas = [[(round(x * largura), round(y * altura)) for x, y in m.poligono] for m in cfg.mesas]
        print(f"{len(mesas)} mesa(s) carregada(s) de {saida}")

    atual: list[tuple[int, int]] = []

    def mouse(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN:
            atual.append((x, y))
        elif evento == cv2.EVENT_RBUTTONDOWN and atual:
            atual.pop()

    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(JANELA, mouse)

    maximizada = False
    while True:
        tela = quadro.copy()
        for i, pts in enumerate(mesas):
            arr = np.array(pts, np.int32)
            cv2.polylines(tela, [arr], True, (80, 190, 40), 2)
            cx, cy = arr.mean(axis=0).astype(int)
            cv2.putText(tela, f"Mesa {i + 1}", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if atual:
            cv2.polylines(tela, [np.array(atual, np.int32)], False, (0, 200, 255), 2)
            for pt in atual:
                cv2.circle(tela, pt, 4, (0, 200, 255), -1)
        for i, linha in enumerate(AJUDA + [f"Mesas: {len(mesas)}  |  desenhando a Mesa {len(mesas) + 1}"]):
            cv2.putText(tela, linha, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(tela, linha, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(JANELA, tela)
        if not maximizada:
            cv2.waitKey(1)
            maximizar(JANELA)
            maximizada = True

        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord("q"), 27):
            break
        elif tecla in (13, 10, ord("n")):
            if len(atual) >= 3:
                mesas.append(list(atual))
                atual.clear()
            else:
                print("A mesa precisa de pelo menos 3 pontos.")
        elif tecla == ord("u") and mesas:
            mesas.pop()
        elif tecla in (ord("a"), ord("d")):
            if eh_arquivo:
                atual_idx = captura.get(cv2.CAP_PROP_POS_FRAMES) - 1  # quadro exibido
                alvo = atual_idx + (fps if tecla == ord("d") else -fps)
                captura.set(cv2.CAP_PROP_POS_FRAMES, max(0, alvo))
            novo = ler_quadro()
            if novo is not None:
                quadro = novo
        elif tecla == ord("s"):
            if not mesas:
                print("Nenhuma mesa para salvar.")
                continue
            saida.parent.mkdir(parents=True, exist_ok=True)
            nome_ref = saida.stem + "_referencia.jpg"
            cv2.imwrite(str(saida.parent / nome_ref), quadro)
            config_mesas = [
                ConfigMesa(i + 1, nomes[i] if i < len(nomes) else f"Mesa {i + 1}",
                           [[x / largura, y / altura] for x, y in pts])
                for i, pts in enumerate(mesas)
            ]
            salvar_config(saida, config_mesas, parametros, nome_ref)
            print(f"Salvo: {saida} ({len(mesas)} mesas) e {saida.parent / nome_ref}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
