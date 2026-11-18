"""Funções de vídeo compartilhadas por monitorar.py e definir_mesas.py."""
from __future__ import annotations

import cv2
import numpy as np


def redimensionar(quadro: np.ndarray, lado_maximo: int) -> np.ndarray:
    """Redimensiona o quadro para que o MAIOR lado tenha `lado_maximo` pixels.

    Em vídeo deitado isso fixa a largura; em vídeo em pé (celular na
    vertical), fixa a altura. Os dois scripts precisam usar a mesma regra
    para as mesas desenhadas caírem no mesmo lugar.
    """
    if not lado_maximo:
        return quadro
    altura, largura = quadro.shape[:2]
    escala = lado_maximo / max(altura, largura)
    if abs(escala - 1) < 1e-3:
        return quadro
    novo = (round(largura * escala), round(altura * escala))
    interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    return cv2.resize(quadro, novo, interpolation=interp)
