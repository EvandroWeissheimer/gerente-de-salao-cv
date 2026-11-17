"""Monitoramento das mesas do salão a partir de um vídeo ou câmera.

Exemplos:
    python monitorar.py --fonte videos/salao.mp4 --mesas config/mesas.json
    python monitorar.py --fonte 0 --mesas config/mesas.json            # webcam
    python monitorar.py --fonte rtsp://usuario:senha@ip:554/stream ...  # câmera IP / CFTV
    python monitorar.py --fonte videos/salao.mp4 --mesas config/mesas.json --salvar saida/demo.mp4 --privacidade

Fluxo:  ENTRADA (quadro) -> YOLO detecta pessoas -> associa pessoas às mesas
        -> verifica limpeza (diferença p/ referência) -> atualiza estados
        -> SAÍDA (vídeo anotado + painel + CSV de eventos + resumo)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from salao.config import carregar_config
from salao.detector import DetectorPessoas
from salao.limpeza import VerificadorLimpeza
from salao.mesas import Estado, Mesa, associar_pessoas, mascara_poligono, poligono_para_pixels
from salao.painel import desenhar
from salao.relatorio import Relatorio


def argumentos() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Câmera de segurança como gerente de salão")
    ap.add_argument("--fonte", required=True, help="arquivo de vídeo, índice da webcam (0) ou URL RTSP")
    ap.add_argument("--mesas", default="config/mesas.json", help="arquivo JSON com as mesas")
    ap.add_argument("--modelo", default="yolo11n.pt", help="pesos do YOLO (yolo11n.pt, yolo11s.pt, ...)")
    ap.add_argument("--largura", type=int, default=1280, help="redimensiona o quadro para esta largura")
    ap.add_argument("--tamanho-imagem", type=int, default=640, help="tamanho de entrada do YOLO")
    ap.add_argument("--fator-tempo", type=float, default=1.0,
                    help="para vídeos em timelapse: 1 s de vídeo equivale a N s reais (só para arquivos)")
    ap.add_argument("--pular", type=int, default=1, help="roda o YOLO a cada N quadros (mais rápido)")
    ap.add_argument("--rastrear", action="store_true", help="ativa rastreamento (ByteTrack) com IDs")
    ap.add_argument("--privacidade", action="store_true", help="pixeliza as pessoas no vídeo exibido/salvo")
    ap.add_argument("--processamento", action="store_true", help="começa mostrando a diferença de limpeza")
    ap.add_argument("--salvar", help="salva o vídeo anotado (ex.: saida/demo.mp4)")
    ap.add_argument("--sem-janela", action="store_true", help="não abre janela (servidor/sem monitor)")
    ap.add_argument("--dispositivo", help="cpu, 0 (GPU), mps ...")
    ap.add_argument("--saida", default="saida", help="pasta dos relatórios")
    return ap.parse_args()


def abrir_fonte(fonte: str) -> tuple[cv2.VideoCapture, bool]:
    """Retorna (captura, eh_arquivo)."""
    if fonte.isdigit():
        return cv2.VideoCapture(int(fonte)), False
    eh_arquivo = Path(fonte).exists()
    if not eh_arquivo and "://" not in fonte:
        sys.exit(f"Fonte não encontrada: {fonte}")
    return cv2.VideoCapture(fonte), eh_arquivo


def redimensionar(quadro, largura: int):
    if largura and quadro.shape[1] != largura:
        escala = largura / quadro.shape[1]
        quadro = cv2.resize(quadro, (largura, round(quadro.shape[0] * escala)), interpolation=cv2.INTER_AREA)
    return quadro


def main() -> None:
    args = argumentos()
    cfg = carregar_config(args.mesas)
    p = cfg.parametros

    captura, eh_arquivo = abrir_fonte(args.fonte)
    ok, quadro = captura.read()
    if not ok:
        sys.exit(f"Não foi possível ler a fonte: {args.fonte}")
    quadro = redimensionar(quadro, args.largura)
    altura, largura = quadro.shape[:2]
    fps_video = captura.get(cv2.CAP_PROP_FPS) or 30.0
    if fps_video <= 1 or fps_video > 240:
        fps_video = 30.0

    # Mesas, máscaras e verificadores de limpeza
    mesas, mascaras, verificadores = [], [], []
    for m in cfg.mesas:
        poligono = poligono_para_pixels(m.poligono, largura, altura)
        mesas.append(Mesa(m.id, m.nome, poligono, p))
        mascaras.append(mascara_poligono(poligono, largura, altura))
        verificadores.append(VerificadorLimpeza(
            mascara_poligono(poligono, largura, altura, p.fator_area_limpeza), p))

    if cfg.referencia and cfg.referencia.exists():
        ref = redimensionar(cv2.imread(str(cfg.referencia)), largura)
        if ref.shape[:2] == (altura, largura):
            for v in verificadores:
                v.definir_referencia(ref)
            print(f"Referência das mesas limpas: {cfg.referencia}")
        else:
            print("[aviso] referência com proporção diferente do vídeo; será capturada ao vivo")
    else:
        print("[aviso] sem imagem de referência: a mesa limpa será aprendida quando estiver livre e vazia")

    print("Carregando modelo YOLO...")
    detector = DetectorPessoas(args.modelo, p.confianca, args.tamanho_imagem, args.rastrear, args.dispositivo)
    relatorio = Relatorio(args.saida, ao_vivo=not eh_arquivo)

    gravador = None
    mostrar_proc = args.processamento
    pausado = False
    pessoas, destino = [], []
    indice, inicio, fps_proc = 0, time.monotonic(), 0.0
    t = 0.0
    tempos_proc = []
    ultimo_quadro = None

    if not args.sem_janela:
        cv2.namedWindow("Gerente de Salao", cv2.WINDOW_NORMAL)

    try:
        while True:
            if not pausado:
                if indice > 0:
                    ok, quadro = captura.read()
                    if not ok:
                        break
                    quadro = redimensionar(quadro, args.largura)
                t0 = time.perf_counter()
                # Tempo do vídeo (arquivo) ou tempo real (câmera)
                t = indice / fps_video * args.fator_tempo if eh_arquivo else time.monotonic() - inicio

                # 1) Detecção de pessoas
                if indice % max(1, args.pular) == 0:
                    pessoas = detector.detectar(quadro)

                # 2) Associação pessoa -> mesa
                por_mesa, destino = associar_pessoas(pessoas, mascaras, p.sobreposicao_minima)

                # 3) Limpeza + 4) máquina de estados
                for mesa, verif, grupo in zip(mesas, verificadores, por_mesa):
                    sujeira = None
                    if mesa.estado == Estado.LIMPEZA or mostrar_proc:
                        sujeira = verif.medir(quadro, pessoas)  # mede só a parte visível
                    if mesa.estado == Estado.LIVRE and not grupo:
                        verif.atualizar_referencia(quadro, pessoas)
                    relatorio.registrar(mesa.atualizar(t, len(grupo), sujeira))

                tempos_proc.append(time.perf_counter() - t0)
                indice += 1
                # FPS real do laço (inclui quadros sem detecção e o desenho)
                agora = time.perf_counter()
                if ultimo_quadro is not None:
                    inst = 1 / max(agora - ultimo_quadro, 1e-6)
                    fps_proc = 0.9 * fps_proc + 0.1 * inst if fps_proc else inst
                ultimo_quadro = agora

            # 5) Saída visual
            tela = desenhar(quadro, mesas, pessoas, destino, t, fps_proc, verificadores,
                            mostrar_proc, args.privacidade, pausado)
            if args.salvar and not pausado:
                if gravador is None:
                    Path(args.salvar).parent.mkdir(parents=True, exist_ok=True)
                    gravador = cv2.VideoWriter(args.salvar, cv2.VideoWriter_fourcc(*"mp4v"),
                                               fps_video, (tela.shape[1], tela.shape[0]))
                gravador.write(tela)

            if args.sem_janela:
                if indice % 100 == 0:
                    print(f"  quadro {indice}  t={t:.0f}s  {fps_proc:.1f} FPS", flush=True)
                continue

            cv2.imshow("Gerente de Salao", tela)
            tecla = cv2.waitKey(1 if not pausado else 50) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("p"):
                pausado = not pausado
            elif tecla == ord("v"):
                mostrar_proc = not mostrar_proc
            elif ord("1") <= tecla <= ord("9"):
                n = tecla - ord("0")
                for mesa in mesas:
                    if mesa.id == n:
                        relatorio.registrar(mesa.marcar_limpa(t))
    except KeyboardInterrupt:
        pass
    finally:
        captura.release()
        if gravador is not None:
            gravador.release()
            print(f"Vídeo salvo em {args.salvar}")
        cv2.destroyAllWindows()
        decorrido = time.monotonic() - inicio
        fps_medio = len(tempos_proc) / decorrido if tempos_proc and decorrido > 0 else 0.0
        relatorio.finalizar(mesas, t, fps_medio)


if __name__ == "__main__":
    main()
