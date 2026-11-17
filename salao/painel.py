"""Desenho do resultado: mesas coloridas sobre o vídeo + painel lateral.

Obs.: o cv2.putText não desenha acentos, por isso os textos na tela
estão sem acento.
"""
from __future__ import annotations

import cv2
import numpy as np

from .detector import Pessoa
from .limpeza import VerificadorLimpeza
from .mesas import Estado, Mesa

# Cores em BGR
CORES = {
    Estado.LIVRE: (80, 190, 40),      # verde
    Estado.OCUPADA: (60, 60, 230),    # vermelho
    Estado.LIMPEZA: (0, 200, 255),    # amarelo
}
CINZA = (160, 160, 160)
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
FUNDO_PAINEL = (40, 32, 28)
FONTE = cv2.FONT_HERSHEY_SIMPLEX
LARGURA_PAINEL = 360
ROTULO_CURTO = {Estado.LIVRE: "LIVRE", Estado.OCUPADA: "OCUPADA", Estado.LIMPEZA: "LIMPEZA"}


def formatar_tempo(segundos: float) -> str:
    segundos = int(max(0, segundos))
    h, resto = divmod(segundos, 3600)
    m, s = divmod(resto, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def anonimizar(quadro: np.ndarray, pessoas: list[Pessoa]) -> None:
    """Pixeliza as pessoas no vídeo de saída (privacidade / LGPD)."""
    altura, largura = quadro.shape[:2]
    for p in pessoas:
        x1, y1, x2, y2 = max(0, p.x1), max(0, p.y1), min(largura, p.x2), min(altura, p.y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        regiao = quadro[y1:y2, x1:x2]
        pequena = cv2.resize(regiao, (max(1, (x2 - x1) // 16), max(1, (y2 - y1) // 16)),
                             interpolation=cv2.INTER_LINEAR)
        quadro[y1:y2, x1:x2] = cv2.resize(pequena, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def _texto(img, texto, org, escala=0.5, cor=BRANCO, espessura=1, fundo=None):
    if fundo is not None:
        (w, h), base = cv2.getTextSize(texto, FONTE, escala, espessura)
        x, y = org
        cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 2), fundo, -1)
    cv2.putText(img, texto, org, FONTE, escala, cor, espessura, cv2.LINE_AA)


def desenhar(quadro: np.ndarray, mesas: list[Mesa], pessoas: list[Pessoa],
             destino: list[int | None], t: float, fps: float,
             verificadores: list[VerificadorLimpeza] | None = None,
             mostrar_processamento: bool = False, privacidade: bool = False,
             pausado: bool = False) -> np.ndarray:
    img = quadro.copy()
    piscar = int(t * 2) % 2 == 0  # alterna 2x por segundo

    if privacidade:
        anonimizar(img, pessoas)

    # 1) Áreas das mesas preenchidas com transparência
    camada = img.copy()
    for mesa in mesas:
        cv2.fillPoly(camada, [mesa.poligono], CORES[mesa.estado])
    cv2.addWeighted(camada, 0.28, img, 0.72, 0, img)

    # 2) Visualização do processamento de imagem (diferença para a mesa limpa)
    if mostrar_processamento and verificadores:
        for v in verificadores:
            if v.ultima_diferenca is None:
                continue
            regiao = img[v.y1:v.y2, v.x1:v.x2]
            marcados = v.ultima_diferenca > 0
            regiao[marcados] = (255, 0, 255)  # magenta = diferente da referência
            cv2.rectangle(img, (v.x1, v.y1), (v.x2, v.y2), (255, 0, 255), 1)

    # 3) Caixas das pessoas
    for pessoa, idx in zip(pessoas, destino):
        cor = CORES[mesas[idx].estado] if idx is not None else CINZA
        cv2.rectangle(img, (pessoa.x1, pessoa.y1), (pessoa.x2, pessoa.y2), cor, 2)
        rotulo = f"#{pessoa.id}" if pessoa.id is not None else f"{pessoa.confianca:.2f}"
        _texto(img, rotulo, (pessoa.x1 + 2, pessoa.y1 + 14), 0.45, BRANCO, 1, fundo=cor)

    # 4) Contorno e rótulo de cada mesa
    for i, mesa in enumerate(mesas):
        cor = CORES[mesa.estado]
        alerta = mesa.alerta(t)
        espessura = 5 if (alerta and piscar) else 2
        cv2.polylines(img, [mesa.poligono], True, cor, espessura, cv2.LINE_AA)

        cx, cy = mesa.poligono.mean(axis=0).astype(int)
        linhas = [mesa.nome, f"{ROTULO_CURTO[mesa.estado]} {formatar_tempo(mesa.tempo_no_estado(t))}"]
        if mesa.estado == Estado.OCUPADA:
            linhas.append(f"{mesa.pessoas} pessoa(s)")
        if mostrar_processamento and verificadores and verificadores[i].ultimo_indice is not None:
            linhas.append(f"sujeira {100 * verificadores[i].ultimo_indice:.0f}%")
        if alerta:
            linhas.append(alerta)
        for j, linha in enumerate(linhas):
            cor_fundo = (0, 0, 200) if (alerta and linha == alerta) else (30, 30, 30)
            _texto(img, linha, (cx - 45, cy - 10 + j * 18), 0.45, BRANCO, 1, fundo=cor_fundo)

    if pausado:
        _texto(img, "PAUSADO", (10, 30), 0.8, BRANCO, 2, fundo=(0, 0, 0))

    painel = _painel(img.shape[0], mesas, t, fps, piscar)
    if painel.shape[0] > img.shape[0]:  # muitas mesas: estende o quadro para baixo
        img = cv2.copyMakeBorder(img, 0, painel.shape[0] - img.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=FUNDO_PAINEL)
    return np.hstack([img, painel])


def _painel(altura: int, mesas: list[Mesa], t: float, fps: float, piscar: bool) -> np.ndarray:
    altura = max(altura, 60 + 34 * len(mesas) + 120)
    p = np.full((altura, LARGURA_PAINEL, 3), FUNDO_PAINEL, dtype=np.uint8)
    _texto(p, "GERENTE DE SALAO", (14, 32), 0.75, BRANCO, 2)
    _texto(p, f"Tempo {formatar_tempo(t)}   {fps:4.1f} FPS", (14, 58), 0.5, CINZA)

    contagem = {e: sum(m.estado == e for m in mesas) for e in Estado}
    y = 92
    for estado, rotulo in ((Estado.LIVRE, "Livres"), (Estado.OCUPADA, "Ocupadas"), (Estado.LIMPEZA, "Limpeza")):
        x = 14 + [Estado.LIVRE, Estado.OCUPADA, Estado.LIMPEZA].index(estado) * 115
        cv2.rectangle(p, (x, y - 14), (x + 12, y - 2), CORES[estado], -1)
        _texto(p, f"{rotulo}: {contagem[estado]}", (x + 18, y - 2), 0.48, BRANCO)
    cv2.line(p, (14, y + 12), (LARGURA_PAINEL - 14, y + 12), (90, 90, 90), 1)

    y += 40
    for mesa in mesas:
        alerta = mesa.alerta(t)
        cv2.rectangle(p, (14, y - 14), (28, y), CORES[mesa.estado], -1)
        _texto(p, mesa.nome[:12], (36, y), 0.5, BRANCO)
        _texto(p, ROTULO_CURTO[mesa.estado], (150, y), 0.5, CORES[mesa.estado])
        _texto(p, formatar_tempo(mesa.tempo_no_estado(t)), (260, y), 0.5, BRANCO)
        if alerta:
            _texto(p, alerta, (36, y + 16), 0.42, (80, 80, 255) if piscar else (160, 160, 255))
            y += 16
        y += 30

    rodape = ["q: sair   p: pausar", "v: ver processamento", "1-9: marcar mesa como limpa"]
    for i, linha in enumerate(rodape):
        _texto(p, linha, (14, altura - 60 + i * 18), 0.42, CINZA)
    return p
