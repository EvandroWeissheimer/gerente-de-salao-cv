"""Regras de negócio: estado de cada mesa e associação pessoa -> mesa.

Cada mesa segue uma máquina de estados simples:

    LIVRE --(pessoa presente por X s)--> OCUPADA
    OCUPADA --(ninguém por Y s)--> AGUARDANDO LIMPEZA
    AGUARDANDO LIMPEZA --(imagem igual à mesa limpa por Z s)--> LIVRE
    AGUARDANDO LIMPEZA --(pessoa presente por muito tempo)--> OCUPADA

Os filtros de tempo (X, Y, Z) evitam que uma detecção falha de um único
quadro mude o estado da mesa.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .config import Parametros
from .detector import Pessoa


class Estado(str, Enum):
    LIVRE = "LIVRE"
    OCUPADA = "OCUPADA"
    LIMPEZA = "AGUARDANDO LIMPEZA"


@dataclass
class Evento:
    tempo: float          # segundos desde o início do vídeo/câmera
    mesa: str
    tipo: str             # OCUPOU | DESOCUPOU | LIBEROU
    detalhe: str = ""
    duracao: float | None = None  # duração do estado que terminou


@dataclass
class Mesa:
    id: int
    nome: str
    poligono: np.ndarray                  # pontos em pixels, shape (N, 2), int32
    params: Parametros
    estado: Estado = Estado.LIVRE
    desde: float = 0.0                    # instante em que entrou no estado atual
    pessoas: int = 0
    # Estatísticas
    duracoes_ocupacao: list[float] = field(default_factory=list)
    duracoes_limpeza: list[float] = field(default_factory=list)
    # Controle interno dos filtros de tempo
    _presenca_desde: float | None = None
    _ausencia_desde: float | None = None
    _limpa_desde: float | None = None

    # ------------------------------------------------------------------
    def atualizar(self, t: float, n_pessoas: int, sujeira: float | None) -> list[Evento]:
        """Atualiza o estado da mesa no instante t.

        n_pessoas: pessoas associadas à mesa neste quadro.
        sujeira: fração da mesa diferente da referência limpa
                 (None quando não dá para medir, ex.: alguém na frente).
        """
        p = self.params
        eventos: list[Evento] = []
        self.pessoas = n_pessoas

        # Contadores de presença/ausência contínuas
        if n_pessoas > 0:
            self._ausencia_desde = None
            if self._presenca_desde is None:
                self._presenca_desde = t
        else:
            self._presenca_desde = None
            if self._ausencia_desde is None:
                self._ausencia_desde = t
        presenca = t - self._presenca_desde if self._presenca_desde is not None else 0.0
        ausencia = t - self._ausencia_desde if self._ausencia_desde is not None else 0.0

        if self.estado == Estado.LIVRE:
            if presenca >= p.tempo_confirmar_ocupacao:
                # A ocupação começou quando a pessoa apareceu, não agora.
                self._mudar(Estado.OCUPADA, self._presenca_desde)
                eventos.append(Evento(self.desde, self.nome, "OCUPOU"))

        elif self.estado == Estado.OCUPADA:
            if ausencia >= p.tempo_confirmar_saida:
                inicio_saida = self._ausencia_desde
                duracao = inicio_saida - self.desde
                self.duracoes_ocupacao.append(duracao)
                self._mudar(Estado.LIMPEZA, inicio_saida)
                eventos.append(Evento(inicio_saida, self.nome, "DESOCUPOU", duracao=duracao))

        elif self.estado == Estado.LIMPEZA:
            if presenca >= p.tempo_presenca_reocupar:
                # Ficou gente tempo demais para ser só o garçom limpando:
                # novos clientes sentaram na mesa sem ela ter sido liberada.
                self._mudar(Estado.OCUPADA, self._presenca_desde)
                eventos.append(Evento(self.desde, self.nome, "OCUPOU", "sem limpeza detectada"))
            elif n_pessoas == 0 and sujeira is not None and sujeira <= p.limiar_sujeira:
                if self._limpa_desde is None:
                    self._limpa_desde = t
                if t - self._limpa_desde >= p.tempo_confirmar_limpa:
                    eventos.append(self._liberar(self._limpa_desde, "automatico"))
            else:
                self._limpa_desde = None

        return eventos

    def marcar_limpa(self, t: float) -> list[Evento]:
        """Liberação manual (ex.: garçom confirma pelo teclado)."""
        if self.estado != Estado.LIMPEZA:
            return []
        return [self._liberar(t, "manual")]

    def tempo_no_estado(self, t: float) -> float:
        return max(0.0, t - self.desde)

    def alerta(self, t: float) -> str | None:
        """Texto de alerta para o painel, ou None."""
        p, dt = self.params, self.tempo_no_estado(t)
        if self.estado == Estado.OCUPADA and dt < p.alerta_nova_ocupacao:
            return "NOVA: ATENDER"
        if self.estado == Estado.OCUPADA and dt >= p.alerta_permanencia:
            return "PERMANENCIA LONGA"
        if self.estado == Estado.LIMPEZA and dt >= p.alerta_limpeza:
            return "LIMPAR!"
        return None

    # ------------------------------------------------------------------
    def _liberar(self, t: float, detalhe: str) -> Evento:
        duracao = t - self.desde
        self.duracoes_limpeza.append(duracao)
        self._mudar(Estado.LIVRE, t)
        return Evento(t, self.nome, "LIBEROU", detalhe, duracao)

    def _mudar(self, novo: Estado, t: float) -> None:
        self.estado = novo
        self.desde = t
        self._limpa_desde = None


# ----------------------------------------------------------------------
# Geometria
# ----------------------------------------------------------------------
def poligono_para_pixels(poligono_norm: list[list[float]], largura: int, altura: int) -> np.ndarray:
    pts = np.array(poligono_norm, dtype=np.float32) * [largura, altura]
    return pts.round().astype(np.int32)


def mascara_poligono(poligono_px: np.ndarray, largura: int, altura: int, escala: float = 1.0) -> np.ndarray:
    """Máscara binária (0/1, uint8) do polígono no tamanho do quadro.

    escala < 1 encolhe o polígono em direção ao seu centro.
    """
    pts = poligono_px.astype(np.float32)
    if escala != 1.0:
        centro = pts.mean(axis=0)
        pts = centro + (pts - centro) * escala
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    cv2.fillPoly(mascara, [pts.round().astype(np.int32)], 1)
    return mascara


def fracao_dentro(pessoa: Pessoa, mascara: np.ndarray) -> float:
    """Fração da caixa da pessoa que está dentro da máscara da mesa."""
    altura, largura = mascara.shape
    x1, y1 = max(0, pessoa.x1), max(0, pessoa.y1)
    x2, y2 = min(largura, pessoa.x2), min(altura, pessoa.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mascara[y1:y2, x1:x2].mean())


def associar_pessoas(pessoas: list[Pessoa], mascaras: list[np.ndarray],
                     minimo: float) -> tuple[list[list[Pessoa]], list[int | None]]:
    """Associa cada pessoa à mesa com a qual ela mais se sobrepõe.

    Retorna (pessoas_por_mesa, indice_da_mesa_de_cada_pessoa).
    Uma pessoa conta para no máximo uma mesa.
    """
    por_mesa: list[list[Pessoa]] = [[] for _ in mascaras]
    destino: list[int | None] = []
    for pessoa in pessoas:
        fracoes = [fracao_dentro(pessoa, m) for m in mascaras]
        melhor = int(np.argmax(fracoes)) if fracoes else -1
        if melhor >= 0 and fracoes[melhor] >= minimo:
            por_mesa[melhor].append(pessoa)
            destino.append(melhor)
        else:
            destino.append(None)
    return por_mesa, destino
