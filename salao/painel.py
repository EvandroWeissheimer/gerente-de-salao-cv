"""Desenho do resultado: mesas coloridas sobre o vídeo + painel lateral.

O OpenCV desenha as formas (polígonos, caixas) e o Pillow desenha os
textos e cartões, porque o cv2.putText não tem acentos nem fontes boas.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detector import Pessoa
from .limpeza import VerificadorLimpeza
from .mesas import Estado, Mesa

LARGURA_PAINEL = 380

# Paleta (RGB) — tema escuro
FUNDO = (17, 24, 39)
CARTAO = (31, 41, 55)
BORDA = (55, 65, 81)
TEXTO = (249, 250, 251)
SUAVE = (156, 163, 175)
ALERTA = (244, 63, 94)
CORES = {
    Estado.LIVRE: (34, 197, 94),     # verde
    Estado.OCUPADA: (239, 68, 68),   # vermelho
    Estado.LIMPEZA: (245, 158, 11),  # âmbar
}
ROTULO = {Estado.LIVRE: "Livre", Estado.OCUPADA: "Ocupada", Estado.LIMPEZA: "Aguardando limpeza"}
ROTULO_CURTO = {Estado.LIVRE: "Livre", Estado.OCUPADA: "Ocupada", Estado.LIMPEZA: "Limpeza"}


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


def _misturar(cor, fundo, alfa):
    return tuple(int(c * alfa + f * (1 - alfa)) for c, f in zip(cor, fundo))


# ----------------------------------------------------------------------
# Fontes: usa a do sistema (Segoe UI no Windows, DejaVu no Linux, Arial no Mac)
# ----------------------------------------------------------------------
_FONTES = {
    "regular": ["segoeui.ttf", "C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf", "arial.ttf"],
    "negrito": ["segoeuib.ttf", "C:/Windows/Fonts/segoeuib.ttf", "DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "arialbd.ttf"],
}


@lru_cache(maxsize=None)
def fonte(tamanho: int, negrito: bool = False) -> ImageFont.FreeTypeFont:
    for caminho in _FONTES["negrito" if negrito else "regular"]:
        try:
            return ImageFont.truetype(caminho, tamanho)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=tamanho)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _largura(d: ImageDraw.ImageDraw, texto: str, f) -> int:
    x1, _, x2, _ = d.textbbox((0, 0), texto, font=f)
    return x2 - x1


def _pilula(d, x, y, texto, cor, f, fundo_base=CARTAO, direita=False):
    """Etiqueta arredondada com fundo translúcido da cor e texto na cor."""
    w = _largura(d, texto, f) + 16
    h = f.size + 8
    if direita:
        x -= w
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=_misturar(cor, fundo_base, 0.22))
    d.text((x + 8, y + h / 2), texto, font=f, fill=cor, anchor="lm")
    return w


# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
def desenhar(quadro: np.ndarray, mesas: list[Mesa], pessoas: list[Pessoa],
             destino: list[int | None], t: float, fps: float,
             verificadores: list[VerificadorLimpeza] | None = None,
             mostrar_processamento: bool = False, privacidade: bool = False,
             pausado: bool = False) -> np.ndarray:
    img = quadro.copy()
    piscar = int(t * 2) % 2 == 0

    if privacidade:
        anonimizar(img, pessoas)

    # 1) Áreas das mesas (OpenCV): preenchimento translúcido + contorno
    camada = img.copy()
    for mesa in mesas:
        cv2.fillPoly(camada, [mesa.poligono], _bgr(CORES[mesa.estado]))
    cv2.addWeighted(camada, 0.22, img, 0.78, 0, img)

    # 2) Processamento de imagem: pixels diferentes da referência em magenta
    if mostrar_processamento and verificadores:
        for v in verificadores:
            if v.ultima_diferenca is None:
                continue
            regiao = img[v.y1:v.y2, v.x1:v.x2]
            regiao[v.ultima_diferenca > 0] = (255, 0, 255)

    for mesa in mesas:
        alerta = mesa.alerta(t)
        espessura = 4 if (alerta and piscar) else 2
        cv2.polylines(img, [mesa.poligono], True, _bgr(CORES[mesa.estado]), espessura, cv2.LINE_AA)

    # 3) Caixas das pessoas
    for pessoa, idx in zip(pessoas, destino):
        cor = _bgr(CORES[mesas[idx].estado]) if idx is not None else (170, 170, 170)
        cv2.rectangle(img, (pessoa.x1, pessoa.y1), (pessoa.x2, pessoa.y2), cor, 2, cv2.LINE_AA)

    # 4) Etiquetas sobre o vídeo (Pillow, com transparência)
    base = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    camada_txt = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(camada_txt)
    for i, mesa in enumerate(mesas):
        _etiqueta_mesa(d, mesa, t, piscar,
                       verificadores[i] if (mostrar_processamento and verificadores) else None)
    for pessoa, idx in zip(pessoas, destino):
        if pessoa.id is not None:
            d.text((pessoa.x1 + 4, pessoa.y1 + 2), f"#{pessoa.id}", font=fonte(12, True), fill=TEXTO)
    if pausado:
        f = fonte(16, True)
        w = _largura(d, "PAUSADO", f) + 24
        d.rounded_rectangle((12, 12, 12 + w, 44), radius=16, fill=(0, 0, 0, 190))
        d.text((24, 28), "PAUSADO", font=f, fill=TEXTO, anchor="lm")
    img = cv2.cvtColor(np.array(Image.alpha_composite(base, camada_txt).convert("RGB")), cv2.COLOR_RGB2BGR)

    # 5) Painel lateral
    painel = _painel(img.shape[0], mesas, t, fps, piscar)
    if painel.shape[0] > img.shape[0]:  # muitas mesas: estende o quadro para baixo
        img = cv2.copyMakeBorder(img, 0, painel.shape[0] - img.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=_bgr(FUNDO))
    return np.hstack([img, painel])


def _etiqueta_mesa(d: ImageDraw.ImageDraw, mesa: Mesa, t: float, piscar: bool,
                   verif: VerificadorLimpeza | None) -> None:
    cor = CORES[mesa.estado]
    f_nome, f_info = fonte(14, True), fonte(13)
    linha2 = f"{ROTULO_CURTO[mesa.estado]} · {formatar_tempo(mesa.tempo_no_estado(t))}"
    if mesa.estado == Estado.OCUPADA:
        linha2 += f" · {mesa.pessoas} {'pessoa' if mesa.pessoas == 1 else 'pessoas'}"
    linhas = [(mesa.nome, f_nome, TEXTO), (linha2, f_info, cor)]
    if verif is not None and verif.ultimo_indice is not None:
        linhas.append((f"sujeira {100 * verif.ultimo_indice:.0f}%", f_info, (232, 121, 249)))

    w = max(_largura(d, txt, f) for txt, f, _ in linhas) + 32
    h = 10 + sum(f.size + 5 for _, f, _ in linhas)
    cx, cy = mesa.poligono.mean(axis=0).astype(int)
    x, y = int(cx - w / 2), int(cy - h / 2)
    d.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=(17, 24, 39, 205))
    d.rounded_rectangle((x, y, x + 4, y + h), radius=2, fill=cor + (255,))
    yy = y + 6
    for txt, f, c in linhas:
        d.text((x + 14, yy), txt, font=f, fill=c)
        yy += f.size + 5

    alerta = mesa.alerta(t)
    if alerta:
        f = fonte(12, True)
        wa = _largura(d, alerta, f) + 20
        fundo = ALERTA + (255 if piscar else 170,)
        d.rounded_rectangle((x, y + h + 4, x + wa, y + h + 26), radius=11, fill=fundo)
        d.text((x + 10, y + h + 15), alerta, font=f, fill=TEXTO, anchor="lm")


def _painel(altura: int, mesas: list[Mesa], t: float, fps: float, piscar: bool) -> np.ndarray:
    L, M = LARGURA_PAINEL, 18  # largura e margem

    # Altura dos cartões: modo compacto se não couber
    topo_cartoes, rodape = 250, 52
    alertas = sum(1 for m in mesas if m.alerta(t))
    normal = len(mesas) * 72 + alertas * 26
    compacto = topo_cartoes + normal + rodape > altura
    alt_cartao = 44 if compacto else 62
    necessario = topo_cartoes + len(mesas) * (alt_cartao + 10) + (0 if compacto else alertas * 26) + rodape
    altura = max(altura, necessario)

    im = Image.new("RGB", (L, altura), FUNDO)
    d = ImageDraw.Draw(im)

    # Cabeçalho
    d.text((M, 18), "Gerente de Salão", font=fonte(22, True), fill=TEXTO)
    d.text((M, 48), "Monitoramento das mesas em tempo real", font=fonte(12), fill=SUAVE)
    d.text((L - M, 22), formatar_tempo(t), font=fonte(18, True), fill=TEXTO, anchor="ra")
    d.text((L - M, 48), f"{fps:.1f} FPS", font=fonte(12), fill=SUAVE, anchor="ra")

    # Resumo: três blocos
    contagem = {e: sum(m.estado == e for m in mesas) for e in Estado}
    larg_bloco = (L - 2 * M - 16) // 3
    for i, (estado, rotulo) in enumerate(((Estado.LIVRE, "Livres"), (Estado.OCUPADA, "Ocupadas"),
                                         (Estado.LIMPEZA, "Limpeza"))):
        x = M + i * (larg_bloco + 8)
        d.rounded_rectangle((x, 80, x + larg_bloco, 146), radius=10, fill=CARTAO)
        d.text((x + 12, 86), str(contagem[estado]), font=fonte(28, True), fill=CORES[estado])
        d.text((x + 12, 124), rotulo, font=fonte(12), fill=SUAVE)

    # Barra de ocupação do salão + indicadores do dia
    total = max(1, len(mesas))
    taxa = contagem[Estado.OCUPADA] / total
    d.text((M, 162), "Ocupação do salão", font=fonte(12), fill=SUAVE)
    d.text((L - M, 162), f"{100 * taxa:.0f}%", font=fonte(12, True), fill=TEXTO, anchor="ra")
    d.rounded_rectangle((M, 182, L - M, 190), radius=4, fill=CARTAO)
    if taxa > 0:
        d.rounded_rectangle((M, 182, M + int((L - 2 * M) * taxa), 190), radius=4, fill=CORES[Estado.OCUPADA])

    ocupacoes = [x for m in mesas for x in m.duracoes_ocupacao]
    limpezas = [x for m in mesas for x in m.duracoes_limpeza]
    media_ocup = formatar_tempo(sum(ocupacoes) / len(ocupacoes)) if ocupacoes else "--:--"
    media_limp = formatar_tempo(sum(limpezas) / len(limpezas)) if limpezas else "--:--"
    d.text((M, 202), f"Giros: {len(ocupacoes)}   ·   Permanência média: {media_ocup}",
           font=fonte(12), fill=SUAVE)
    d.text((M, 220), f"Tempo médio até liberar: {media_limp}", font=fonte(12), fill=SUAVE)
    d.line((M, 242, L - M, 242), fill=BORDA, width=1)

    # Cartões das mesas
    y = topo_cartoes + 4
    for mesa in mesas:
        cor = CORES[mesa.estado]
        alerta = mesa.alerta(t)
        h = alt_cartao + (26 if (alerta and not compacto) else 0)
        d.rounded_rectangle((M, y, L - M, y + h), radius=10, fill=CARTAO)
        d.rounded_rectangle((M, y, M + 5, y + h), radius=3, fill=cor)

        tempo = formatar_tempo(mesa.tempo_no_estado(t))
        if compacto:
            meio = y + h // 2
            d.text((M + 16, meio), mesa.nome, font=fonte(14, True), fill=TEXTO, anchor="lm")
            rotulo, cor_rot = (alerta, ALERTA) if alerta else (ROTULO_CURTO[mesa.estado], cor)
            _pilula(d, M + 110, meio - 11, rotulo, cor_rot, fonte(11, True))
            d.text((L - M - 14, meio), tempo, font=fonte(16, True), fill=TEXTO, anchor="rm")
        else:
            d.text((M + 16, y + 10), mesa.nome, font=fonte(15, True), fill=TEXTO)
            largura_nome = _largura(d, mesa.nome, fonte(15, True))
            _pilula(d, M + 26 + largura_nome, y + 10, ROTULO[mesa.estado], cor, fonte(11, True))
            if mesa.estado == Estado.OCUPADA:
                detalhe = f"{mesa.pessoas} {'pessoa' if mesa.pessoas == 1 else 'pessoas'} na mesa"
            elif mesa.estado == Estado.LIMPEZA:
                detalhe = "Clientes saíram"
            else:
                detalhe = "Disponível para novos clientes"
            d.text((M + 16, y + 36), detalhe, font=fonte(12), fill=SUAVE)
            d.text((L - M - 14, y + 12), tempo, font=fonte(20, True), fill=TEXTO, anchor="ra")
            d.text((L - M - 14, y + 38), "no estado", font=fonte(10), fill=SUAVE, anchor="ra")
            if alerta:
                fundo = ALERTA if piscar else _misturar(ALERTA, CARTAO, 0.6)
                d.rounded_rectangle((M + 14, y + 60, L - M - 12, y + 82), radius=11, fill=fundo)
                d.text((M + 26, y + 71), f"!  {alerta}", font=fonte(12, True), fill=TEXTO, anchor="lm")
        y += h + 10

    # Rodapé com atalhos
    d.line((M, altura - rodape + 6, L - M, altura - rodape + 6), fill=BORDA, width=1)
    d.text((M, altura - 30), "Q sair   ·   P pausar   ·   F tela cheia", font=fonte(11), fill=SUAVE, anchor="lm")
    d.text((M, altura - 13), "V processamento   ·   1–9 marcar mesa limpa", font=fonte(11), fill=SUAVE, anchor="lm")

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
