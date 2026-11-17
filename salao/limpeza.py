"""Verificação de limpeza da mesa por processamento de imagem clássico.

Ideia: guardamos uma imagem de referência da mesa LIMPA e VAZIA. Depois
que os clientes saem, comparamos a mesa atual com essa referência:
pratos, copos e guardanapos esquecidos aparecem como regiões diferentes.

Etapas (todas do OpenCV):
  1. Recorte da região da mesa
  2. Conversão para tons de cinza
  3. Filtro gaussiano (remove ruído da câmera)
  4. Diferença absoluta entre atual e referência, compensando o brilho médio
  5. Limiarização (pixel "mudou" se a diferença > limiar)
  6. Abertura morfológica (remove pontinhos e bordas finas)
  7. Fração de pixels que mudaram -> "índice de sujeira"
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Parametros
from .detector import Pessoa

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class VerificadorLimpeza:
    def __init__(self, mascara: np.ndarray, params: Parametros):
        """mascara: máscara 0/1 do quadro inteiro com a região a verificar."""
        ys, xs = np.nonzero(mascara)
        if len(xs) == 0:
            raise ValueError("Região de limpeza vazia")
        self.x1, self.y1 = int(xs.min()), int(ys.min())
        self.x2, self.y2 = int(xs.max()) + 1, int(ys.max()) + 1
        self.mascara = mascara[self.y1:self.y2, self.x1:self.x2].astype(bool)
        self.area = int(self.mascara.sum())
        self.params = params
        self.referencia: np.ndarray | None = None
        self.ultima_diferenca: np.ndarray | None = None  # para visualização
        self.ultimo_indice: float | None = None

    # ------------------------------------------------------------------
    def _preparar(self, quadro: np.ndarray) -> np.ndarray:
        recorte = quadro[self.y1:self.y2, self.x1:self.x2]
        cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        suave = cv2.GaussianBlur(cinza, (7, 7), 0)
        return suave.astype(np.float32)

    def definir_referencia(self, quadro: np.ndarray) -> None:
        self.referencia = self._preparar(quadro)

    def _visivel(self, pessoas: list[Pessoa]) -> np.ndarray:
        """Região da mesa que não está tapada por nenhuma pessoa."""
        visivel = self.mascara.copy()
        for p in pessoas:
            x1, y1 = max(p.x1, self.x1) - self.x1, max(p.y1, self.y1) - self.y1
            x2, y2 = min(p.x2, self.x2) - self.x1, min(p.y2, self.y2) - self.y1
            if x2 > x1 and y2 > y1:
                visivel[y1:y2, x1:x2] = False
        return visivel

    def atualizar_referencia(self, quadro: np.ndarray, pessoas: list[Pessoa] = ()) -> None:
        """Média móvel: adapta a referência a mudanças lentas de luz.

        Só atualiza os pixels visíveis; a primeira referência só é criada
        se a mesa estiver quase toda visível.
        """
        visivel = self._visivel(list(pessoas))
        atual = self._preparar(quadro)
        if self.referencia is None:
            if visivel.sum() >= 0.9 * self.area:
                self.referencia = atual
            return
        cv2.accumulateWeighted(atual, self.referencia, self.params.taxa_atualizacao_referencia,
                               mask=visivel.astype(np.uint8))

    def limpar_visualizacao(self) -> None:
        self.ultima_diferenca = None
        self.ultimo_indice = None

    def medir(self, quadro: np.ndarray, pessoas: list[Pessoa] = ()) -> float | None:
        """Índice de sujeira: fração (0 a 1) da parte VISÍVEL da mesa que
        está diferente da referência. None se não há referência ou se
        pessoas tapam mais do que o permitido."""
        if self.referencia is None:
            self.limpar_visualizacao()
            return None
        visivel = self._visivel(list(pessoas))
        n_visivel = int(visivel.sum())
        if n_visivel < self.params.visibilidade_minima * self.area:
            self.limpar_visualizacao()
            return None
        atual = self._preparar(quadro)
        # Subtrair a média de cada imagem compensa mudanças gerais de brilho
        diferenca = np.abs((atual - atual[visivel].mean()) - (self.referencia - self.referencia[visivel].mean()))
        binaria = (diferenca > self.params.limiar_pixel).astype(np.uint8) * 255
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, KERNEL)
        binaria[~visivel] = 0
        self.ultima_diferenca = binaria
        self.ultimo_indice = float(np.count_nonzero(binaria)) / n_visivel
        return self.ultimo_indice
