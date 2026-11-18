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


def _pilula(d, x, y, texto, cor, f, fundo_base=CARTAO, direita=False, k=1.0):
    """Etiqueta arredondada com fundo translúcido da cor e texto na cor."""
    w = _largura(d, texto, f) + int(16 * k)
    h = f.size + int(8 * k)
    if direita:
        x -= w
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=_misturar(cor, fundo_base, 0.22))
    d.text((x + w / 2, y + h / 2), texto, font=f, fill=cor, anchor="mm")
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
             pausado: bool = False, tamanho_saida: tuple[int, int] | None = None) -> np.ndarray:
    """Monta a tela: vídeo anotado + painel lateral.

    tamanho_saida=(largura, altura) desenha tudo já no tamanho da janela,
    para o OpenCV não precisar esticar a imagem (o que serrilha os textos).
    Sem ele, usa a resolução do quadro (vídeo salvo).
    """
    piscar = int(t * 2) % 2 == 0
    fh, fw = quadro.shape[:2]

    # Escalas: s para o vídeo, k para textos e painel
    if tamanho_saida:
        larg_saida, alt_saida = tamanho_saida
        k = min(max(alt_saida / 720, 0.8), 1.8)
        larg_painel = int(LARGURA_PAINEL * k)
        s = min((larg_saida - larg_painel) / fw, alt_saida / fh)
        s = max(s, 0.1)
    else:
        k, s = 1.0, 1.0
        larg_painel = LARGURA_PAINEL

    img = quadro.copy()
    if privacidade:
        anonimizar(img, pessoas)
    if abs(s - 1) > 1e-3:
        img = cv2.resize(img, (round(fw * s), round(fh * s)),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)

    def esc(pts):
        return (np.asarray(pts, dtype=np.float32) * s).round().astype(np.int32)

    poligonos = [esc(m.poligono) for m in mesas]
    linha = max(2, round(2 * k))

    # 1) Áreas das mesas: preenchimento translúcido
    camada = img.copy()
    for mesa, pol in zip(mesas, poligonos):
        cv2.fillPoly(camada, [pol], _bgr(CORES[mesa.estado]))
    cv2.addWeighted(camada, 0.22, img, 0.78, 0, img)

    # 2) Processamento de imagem: pixels diferentes da referência em magenta
    if mostrar_processamento and verificadores:
        for v in verificadores:
            if v.ultima_diferenca is None:
                continue
            x1, y1, x2, y2 = (round(c * s) for c in (v.x1, v.y1, v.x2, v.y2))
            x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue
            marc = cv2.resize(v.ultima_diferenca, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) > 0
            img[y1:y2, x1:x2][marc] = (255, 0, 255)

    # 3) Contornos das mesas e caixas das pessoas
    for mesa, pol in zip(mesas, poligonos):
        espessura = linha * 2 if (mesa.alerta(t) and piscar) else linha
        cv2.polylines(img, [pol], True, _bgr(CORES[mesa.estado]), espessura, cv2.LINE_AA)
    for pessoa, idx in zip(pessoas, destino):
        cor = _bgr(CORES[mesas[idx].estado]) if idx is not None else (170, 170, 170)
        p1 = (round(pessoa.x1 * s), round(pessoa.y1 * s))
        p2 = (round(pessoa.x2 * s), round(pessoa.y2 * s))
        cv2.rectangle(img, p1, p2, cor, linha, cv2.LINE_AA)

    # 4) Etiquetas sobre o vídeo (Pillow, com transparência), já no tamanho final
    base = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    camada_txt = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(camada_txt)
    for i, (mesa, pol) in enumerate(zip(mesas, poligonos)):
        _etiqueta_mesa(d, mesa, pol, t, piscar, k,
                       verificadores[i] if (mostrar_processamento and verificadores) else None)
    for pessoa in pessoas:
        if pessoa.id is not None:
            d.text((pessoa.x1 * s + 4 * k, pessoa.y1 * s + 2 * k), f"#{pessoa.id}",
                   font=fonte(round(12 * k), True), fill=TEXTO)
    if pausado:
        f = fonte(round(16 * k), True)
        w = _largura(d, "PAUSADO", f) + round(24 * k)
        d.rounded_rectangle((12 * k, 12 * k, 12 * k + w, 44 * k), radius=16 * k, fill=(0, 0, 0, 190))
        d.text((24 * k, 28 * k), "PAUSADO", font=f, fill=TEXTO, anchor="lm")
    img = cv2.cvtColor(np.array(Image.alpha_composite(base, camada_txt).convert("RGB")), cv2.COLOR_RGB2BGR)

    # 5) Painel lateral na mesma altura (ou na altura da janela)
    alt_total = tamanho_saida[1] if tamanho_saida else img.shape[0]
    painel = _painel(alt_total, mesas, t, fps, piscar, k)
    alt_total = max(alt_total, painel.shape[0], img.shape[0])
    if img.shape[0] < alt_total:   # centraliza o vídeo na vertical
        sobra = alt_total - img.shape[0]
        img = cv2.copyMakeBorder(img, sobra // 2, sobra - sobra // 2, 0, 0, cv2.BORDER_CONSTANT, value=_bgr(FUNDO))
    if painel.shape[0] < alt_total:
        painel = cv2.copyMakeBorder(painel, 0, alt_total - painel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=_bgr(FUNDO))
    tela = np.hstack([img, painel])

    if tamanho_saida:   # ajusta ao pixel exato da janela (sem nenhum redimensionamento depois)
        larg_saida, alt_saida = tamanho_saida
        if tela.shape[1] < larg_saida:
            sobra = larg_saida - tela.shape[1]
            tela = cv2.copyMakeBorder(tela, 0, 0, sobra // 2, sobra - sobra // 2,
                                      cv2.BORDER_CONSTANT, value=_bgr(FUNDO))
        tela = tela[:alt_saida, :larg_saida]
    return tela


def _etiqueta_mesa(d: ImageDraw.ImageDraw, mesa: Mesa, poligono: np.ndarray, t: float,
                   piscar: bool, k: float, verif: VerificadorLimpeza | None) -> None:
    u = lambda v: round(v * k)  # noqa: E731
    cor = CORES[mesa.estado]
    f_nome, f_info = fonte(u(14), True), fonte(u(13))
    linha2 = f"{ROTULO_CURTO[mesa.estado]} · {formatar_tempo(mesa.tempo_no_estado(t))}"
    if mesa.estado == Estado.OCUPADA:
        linha2 += f" · {mesa.pessoas} {'pessoa' if mesa.pessoas == 1 else 'pessoas'}"
    linhas = [(mesa.nome, f_nome, TEXTO), (linha2, f_info, cor)]
    if verif is not None and verif.ultimo_indice is not None:
        linhas.append((f"sujeira {100 * verif.ultimo_indice:.0f}%", f_info, (232, 121, 249)))

    w = max(_largura(d, txt, f) for txt, f, _ in linhas) + u(32)
    h = u(10) + sum(f.size + u(5) for _, f, _ in linhas)
    cx, cy = poligono.mean(axis=0)
    x, y = int(cx - w / 2), int(cy - h / 2)
    d.rounded_rectangle((x, y, x + w, y + h), radius=u(8), fill=(17, 24, 39, 205))
    d.rounded_rectangle((x, y, x + u(4), y + h), radius=u(2), fill=cor + (255,))
    yy = y + u(6)
    for txt, f, c in linhas:
        d.text((x + u(14), yy), txt, font=f, fill=c)
        yy += f.size + u(5)

    alerta = mesa.alerta(t)
    if alerta:
        f = fonte(u(12), True)
        wa = _largura(d, alerta, f) + u(20)
        fundo = ALERTA + (255 if piscar else 170,)
        d.rounded_rectangle((x, y + h + u(4), x + wa, y + h + u(26)), radius=u(11), fill=fundo)
        d.text((x + u(10), y + h + u(15)), alerta, font=f, fill=TEXTO, anchor="lm")


def _painel(altura: int, mesas: list[Mesa], t: float, fps: float, piscar: bool, k: float = 1.0) -> np.ndarray:
    u = lambda v: round(v * k)  # noqa: E731
    L, M = u(LARGURA_PAINEL), u(18)  # largura e margem

    # Altura dos cartões: modo compacto se não couber
    topo_cartoes, rodape = u(250), u(56)
    alertas = sum(1 for m in mesas if m.alerta(t))
    normal = len(mesas) * u(72) + alertas * u(26)
    compacto = topo_cartoes + normal + rodape > altura
    alt_cartao = u(44) if compacto else u(62)
    necessario = (topo_cartoes + len(mesas) * (alt_cartao + u(10))
                  + (0 if compacto else alertas * u(26)) + rodape)
    altura = max(altura, necessario)

    im = Image.new("RGB", (L, altura), FUNDO)
    d = ImageDraw.Draw(im)

    # Cabeçalho
    d.text((M, u(18)), "Gerente de Salão", font=fonte(u(22), True), fill=TEXTO)
    d.text((M, u(48)), "Monitoramento das mesas em tempo real", font=fonte(u(12)), fill=SUAVE)
    d.text((L - M, u(22)), formatar_tempo(t), font=fonte(u(18), True), fill=TEXTO, anchor="ra")
    d.text((L - M, u(48)), f"{fps:.1f} FPS", font=fonte(u(12)), fill=SUAVE, anchor="ra")

    # Resumo: três blocos
    contagem = {e: sum(m.estado == e for m in mesas) for e in Estado}
    larg_bloco = (L - 2 * M - u(16)) // 3
    for i, (estado, rotulo) in enumerate(((Estado.LIVRE, "Livres"), (Estado.OCUPADA, "Ocupadas"),
                                         (Estado.LIMPEZA, "Limpeza"))):
        x = M + i * (larg_bloco + u(8))
        d.rounded_rectangle((x, u(80), x + larg_bloco, u(146)), radius=u(10), fill=CARTAO)
        d.text((x + u(12), u(86)), str(contagem[estado]), font=fonte(u(28), True), fill=CORES[estado])
        d.text((x + u(12), u(124)), rotulo, font=fonte(u(12)), fill=SUAVE)

    # Barra de ocupação do salão + indicadores do dia
    total = max(1, len(mesas))
    taxa = contagem[Estado.OCUPADA] / total
    d.text((M, u(162)), "Ocupação do salão", font=fonte(u(12)), fill=SUAVE)
    d.text((L - M, u(162)), f"{100 * taxa:.0f}%", font=fonte(u(12), True), fill=TEXTO, anchor="ra")
    d.rounded_rectangle((M, u(182), L - M, u(190)), radius=u(4), fill=CARTAO)
    if taxa > 0:
        d.rounded_rectangle((M, u(182), M + int((L - 2 * M) * taxa), u(190)), radius=u(4),
                            fill=CORES[Estado.OCUPADA])

    ocupacoes = [x for m in mesas for x in m.duracoes_ocupacao]
    limpezas = [x for m in mesas for x in m.duracoes_limpeza]
    media_ocup = formatar_tempo(sum(ocupacoes) / len(ocupacoes)) if ocupacoes else "--:--"
    media_limp = formatar_tempo(sum(limpezas) / len(limpezas)) if limpezas else "--:--"
    d.text((M, u(202)), f"Giros: {len(ocupacoes)}   ·   Permanência média: {media_ocup}",
           font=fonte(u(12)), fill=SUAVE)
    d.text((M, u(220)), f"Tempo médio até liberar: {media_limp}", font=fonte(u(12)), fill=SUAVE)
    d.line((M, u(242), L - M, u(242)), fill=BORDA, width=1)

    # Cartões das mesas
    y = topo_cartoes + u(4)
    for mesa in mesas:
        cor = CORES[mesa.estado]
        alerta = mesa.alerta(t)
        h = alt_cartao + (u(26) if (alerta and not compacto) else 0)
        d.rounded_rectangle((M, y, L - M, y + h), radius=u(10), fill=CARTAO)
        d.rounded_rectangle((M, y, M + u(5), y + h), radius=u(3), fill=cor)

        tempo = formatar_tempo(mesa.tempo_no_estado(t))
        if compacto:
            meio = y + h // 2
            d.text((M + u(16), meio), mesa.nome, font=fonte(u(14), True), fill=TEXTO, anchor="lm")
            rotulo, cor_rot = (alerta, ALERTA) if alerta else (ROTULO_CURTO[mesa.estado], cor)
            _pilula(d, M + u(110), meio - u(11), rotulo, cor_rot, fonte(u(11), True), k=k)
            d.text((L - M - u(14), meio), tempo, font=fonte(u(16), True), fill=TEXTO, anchor="rm")
        else:
            d.text((M + u(16), y + u(10)), mesa.nome, font=fonte(u(15), True), fill=TEXTO)
            largura_nome = _largura(d, mesa.nome, fonte(u(15), True))
            _pilula(d, M + u(26) + largura_nome, y + u(10), ROTULO[mesa.estado], cor, fonte(u(11), True), k=k)
            if mesa.estado == Estado.OCUPADA:
                detalhe = f"{mesa.pessoas} {'pessoa' if mesa.pessoas == 1 else 'pessoas'} na mesa"
            elif mesa.estado == Estado.LIMPEZA:
                detalhe = "Clientes saíram"
            else:
                detalhe = "Disponível para novos clientes"
            d.text((M + u(16), y + u(36)), detalhe, font=fonte(u(12)), fill=SUAVE)
            d.text((L - M - u(14), y + u(12)), tempo, font=fonte(u(20), True), fill=TEXTO, anchor="ra")
            d.text((L - M - u(14), y + u(38)), "no estado", font=fonte(u(10)), fill=SUAVE, anchor="ra")
            if alerta:
                fundo = ALERTA if piscar else _misturar(ALERTA, CARTAO, 0.6)
                d.rounded_rectangle((M + u(14), y + u(60), L - M - u(12), y + u(82)), radius=u(11), fill=fundo)
                d.text((M + u(26), y + u(71)), f"!  {alerta}", font=fonte(u(12), True), fill=TEXTO, anchor="lm")
        y += h + u(10)

    # Rodapé com atalhos
    d.line((M, altura - rodape + u(6), L - M, altura - rodape + u(6)), fill=BORDA, width=1)
    d.text((M, altura - u(34)), "Q sair   ·   P pausar   ·   F tela cheia", font=fonte(u(11)), fill=SUAVE, anchor="lm")
    d.text((M, altura - u(16)), "V processamento   ·   1–9 marcar mesa limpa", font=fonte(u(11)), fill=SUAVE, anchor="lm")

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
