"""Compara os eventos detectados com um gabarito anotado manualmente.

Gera os indicadores quantitativos dos testes: acertos, eventos perdidos,
falsos alarmes, precisão, revocação e erro médio de tempo.

Uso:
    python avaliar.py --eventos saida/eventos_20261120_101500.csv --gabarito testes/gabarito_video1.csv

Formato do gabarito (CSV), anotado assistindo ao vídeo:
    mesa,evento,tempo
    Mesa 1,OCUPOU,00:12
    Mesa 1,DESOCUPOU,02:40
    Mesa 1,LIBEROU,03:05

eventos: OCUPOU, DESOCUPOU, LIBEROU   |   tempo: segundos ou mm:ss
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def para_segundos(valor: str) -> float:
    valor = valor.strip()
    if ":" in valor:
        partes = [float(x) for x in valor.split(":")]
        segundos = 0.0
        for parte in partes:
            segundos = segundos * 60 + parte
        return segundos
    return float(valor)


def ler(caminho: str, coluna_tempo: str) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return [
            {"mesa": l["mesa"].strip(), "evento": l["evento"].strip().upper(),
             "tempo": para_segundos(l[coluna_tempo])}
            for l in csv.DictReader(f)
        ]


def avaliar(detectados: list[dict], gabarito: list[dict], tolerancia: float) -> dict:
    usados = set()
    por_tipo = defaultdict(lambda: {"acertos": 0, "perdidos": 0, "falsos": 0, "erros": []})

    for g in sorted(gabarito, key=lambda e: e["tempo"]):
        candidatos = [
            (abs(d["tempo"] - g["tempo"]), i) for i, d in enumerate(detectados)
            if i not in usados and d["mesa"] == g["mesa"] and d["evento"] == g["evento"]
            and abs(d["tempo"] - g["tempo"]) <= tolerancia
        ]
        if candidatos:
            erro, i = min(candidatos)
            usados.add(i)
            por_tipo[g["evento"]]["acertos"] += 1
            por_tipo[g["evento"]]["erros"].append(erro)
        else:
            por_tipo[g["evento"]]["perdidos"] += 1

    for i, d in enumerate(detectados):
        if i not in usados:
            por_tipo[d["evento"]]["falsos"] += 1
    return por_tipo


def main() -> None:
    ap = argparse.ArgumentParser(description="Avalia os eventos detectados contra um gabarito")
    ap.add_argument("--eventos", required=True, help="CSV gerado pelo monitorar.py")
    ap.add_argument("--gabarito", required=True, help="CSV anotado manualmente")
    ap.add_argument("--tolerancia", type=float, default=15.0, help="diferença máxima de tempo (s) para contar acerto")
    args = ap.parse_args()

    resultado = avaliar(ler(args.eventos, "tempo_s"), ler(args.gabarito, "tempo"), args.tolerancia)

    print(f"\nTolerância de tempo: {args.tolerancia:.0f} s\n")
    print(f"{'Evento':<11}{'Acertos':>8}{'Perdidos':>10}{'Falsos':>8}{'Precisao':>10}{'Revocacao':>11}{'Erro medio':>12}")
    total = {"acertos": 0, "perdidos": 0, "falsos": 0, "erros": []}
    for tipo in ("OCUPOU", "DESOCUPOU", "LIBEROU"):
        r = resultado.get(tipo)
        if not r:
            continue
        for k in total:
            total[k] += r[k]
        _linha(tipo, r)
    print("-" * 70)
    _linha("TOTAL", total)


def _linha(nome: str, r: dict) -> None:
    a, p, f = r["acertos"], r["perdidos"], r["falsos"]
    precisao = f"{100 * a / (a + f):.0f}%" if a + f else "-"
    revocacao = f"{100 * a / (a + p):.0f}%" if a + p else "-"
    erro = f"{sum(r['erros']) / len(r['erros']):.1f} s" if r["erros"] else "-"
    print(f"{nome:<11}{a:>8}{p:>10}{f:>8}{precisao:>10}{revocacao:>11}{erro:>12}")


if __name__ == "__main__":
    main()
