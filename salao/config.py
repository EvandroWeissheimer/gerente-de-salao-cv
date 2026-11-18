"""Leitura e escrita do arquivo de configuração das mesas (JSON).

As coordenadas dos polígonos são guardadas NORMALIZADAS (0 a 1), ou seja,
como fração da largura e da altura da imagem. Assim a mesma configuração
funciona com o vídeo em qualquer resolução.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Parametros:
    """Parâmetros ajustáveis do sistema (todos os tempos em segundos)."""

    # --- Detecção de pessoas (YOLO) ---
    confianca: float = 0.35
    """Confiança mínima para aceitar uma detecção de pessoa."""

    sobreposicao_minima: float = 0.30
    """Fração da caixa da pessoa que precisa estar dentro da área da mesa
    para a pessoa ser associada àquela mesa."""


    # --- Filtros de tempo (evitam que o estado "pisque") ---
    tolerancia_falha_deteccao: float = 1.5
    """O detector às vezes perde uma pessoa por alguns quadros (oclusão,
    ângulo). Falhas mais curtas que isso não interrompem a presença."""

    tempo_confirmar_ocupacao: float = 3.0
    """Tempo com pessoa presente para considerar a mesa OCUPADA."""

    tempo_confirmar_saida: float = 10.0
    """Tempo sem ninguém para considerar que os clientes foram embora
    (cobre quem levanta para ir ao banheiro ou ao buffet)."""

    tempo_confirmar_limpa: float = 3.0
    """Tempo que a mesa precisa parecer limpa para voltar a LIVRE."""

    tempo_presenca_reocupar: float = 30.0
    """Na mesa aguardando limpeza, presença por mais que este tempo é
    tratada como novos clientes (e não como o garçom limpando)."""

    # --- Alertas ---
    alerta_nova_ocupacao: float = 60.0
    """Durante esse tempo após a ocupação a mesa pisca 'ATENDER'."""

    alerta_permanencia: float = 3600.0
    """Ocupação acima desse tempo gera alerta de permanência longa."""

    alerta_limpeza: float = 120.0
    """Mesa aguardando limpeza há mais que esse tempo gera alerta."""

    # --- Verificação de limpeza (processamento de imagem) ---
    fator_area_limpeza: float = 0.6
    """Escala do polígono usada para verificar limpeza (0.6 = região
    central, onde ficam pratos e copos; evita confundir cadeiras fora
    do lugar com sujeira). Use 1.0 para usar a área inteira."""

    limiar_pixel: int = 35
    """Diferença mínima de intensidade (0-255) para um pixel contar como
    'diferente' da referência da mesa limpa."""

    limiar_sujeira: float = 0.06
    """Fração máxima de pixels diferentes para a mesa ser considerada limpa."""

    visibilidade_minima: float = 0.5
    """Fração mínima da região da mesa que precisa estar visível (sem
    pessoas na frente) para medir a limpeza. Só a parte visível é medida."""

    taxa_atualizacao_referencia: float = 0.01
    """Velocidade com que a imagem de referência se adapta à iluminação
    enquanto a mesa está livre (média móvel)."""


@dataclass
class ConfigMesa:
    id: int
    nome: str
    poligono: list[list[float]]  # pontos normalizados [[x, y], ...]


@dataclass
class Configuracao:
    mesas: list[ConfigMesa]
    parametros: Parametros = field(default_factory=Parametros)
    referencia: Path | None = None  # imagem do salão com as mesas limpas


def carregar_config(caminho: str | Path) -> Configuracao:
    caminho = Path(caminho)
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)

    mesas = [
        ConfigMesa(id=int(m["id"]), nome=m.get("nome", f"Mesa {m['id']}"), poligono=m["poligono"])
        for m in dados.get("mesas", [])
    ]
    if not mesas:
        raise ValueError(f"Nenhuma mesa definida em {caminho}. Use definir_mesas.py.")
    for m in mesas:
        if len(m.poligono) < 3:
            raise ValueError(f"{m.nome}: o polígono precisa de pelo menos 3 pontos.")

    # Só aceita parâmetros conhecidos (avisa sobre erros de digitação)
    validos = {f.name for f in fields(Parametros)}
    brutos = dados.get("parametros", {})
    desconhecidos = set(brutos) - validos
    if desconhecidos:
        print(f"[aviso] parâmetros ignorados (nome desconhecido): {sorted(desconhecidos)}")
    parametros = Parametros(**{k: v for k, v in brutos.items() if k in validos})

    referencia = None
    if dados.get("referencia"):
        referencia = (caminho.parent / dados["referencia"]).resolve()

    return Configuracao(mesas=mesas, parametros=parametros, referencia=referencia)


def salvar_config(caminho: str | Path, mesas: list[ConfigMesa],
                  parametros: Parametros, referencia: str | None) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    dados = {
        "referencia": referencia,
        "mesas": [
            {"id": m.id, "nome": m.nome,
             "poligono": [[round(x, 4), round(y, 4)] for x, y in m.poligono]}
            for m in mesas
        ],
        "parametros": asdict(parametros),
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
