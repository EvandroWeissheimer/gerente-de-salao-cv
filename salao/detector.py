"""Detecção (e rastreamento opcional) de pessoas com YOLO.

Usamos um modelo YOLO pré-treinado no conjunto COCO, que já reconhece a
classe "person" (índice 0). Não é preciso treinar nada: filtramos só
essa classe. O sistema NÃO identifica quem é a pessoa, apenas onde ela
está, o que é suficiente para saber se a mesa está ocupada (LGPD).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASSE_PESSOA = 0  # índice de "person" no COCO


@dataclass
class Pessoa:
    x1: int
    y1: int
    x2: int
    y2: int
    confianca: float = 1.0
    id: int | None = None  # preenchido quando o rastreamento está ativo


class DetectorPessoas:
    def __init__(self, modelo: str = "yolo11n.pt", confianca: float = 0.35,
                 tamanho_imagem: int = 640, rastrear: bool = False,
                 dispositivo: str | None = None):
        # Import aqui dentro para os testes rodarem sem o ultralytics instalado
        from ultralytics import YOLO

        self.modelo = YOLO(modelo)  # baixa os pesos automaticamente na 1ª vez
        self.confianca = confianca
        self.tamanho_imagem = tamanho_imagem
        self.rastrear = rastrear
        self.dispositivo = dispositivo

    def detectar(self, quadro: np.ndarray) -> list[Pessoa]:
        opcoes = dict(classes=[CLASSE_PESSOA], conf=self.confianca,
                      imgsz=self.tamanho_imagem, verbose=False)
        if self.dispositivo:
            opcoes["device"] = self.dispositivo

        if self.rastrear:
            # ByteTrack mantém o mesmo ID para a mesma pessoa entre quadros
            resultado = self.modelo.track(quadro, persist=True, tracker="bytetrack.yaml", **opcoes)[0]
        else:
            resultado = self.modelo.predict(quadro, **opcoes)[0]

        caixas = resultado.boxes
        if caixas is None or len(caixas) == 0:
            return []
        xyxy = caixas.xyxy.cpu().numpy().round().astype(int)
        confs = caixas.conf.cpu().numpy()
        ids = caixas.id.cpu().numpy().astype(int) if caixas.id is not None else [None] * len(xyxy)
        return [
            Pessoa(int(x1), int(y1), int(x2), int(y2), float(c), None if i is None else int(i))
            for (x1, y1, x2, y2), c, i in zip(xyxy, confs, ids)
        ]
