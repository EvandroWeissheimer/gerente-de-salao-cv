"""Registro de eventos (CSV) e resumo final com indicadores."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

from .mesas import Estado, Evento, Mesa
from .painel import formatar_tempo


class Relatorio:
    def __init__(self, pasta: str | Path, ao_vivo: bool):
        self.pasta = Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ao_vivo = ao_vivo
        self.inicio = datetime.now()
        self.caminho_csv = self.pasta / f"eventos_{self.carimbo}.csv"
        self._arquivo = open(self.caminho_csv, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._arquivo)
        self._csv.writerow(["tempo_s", "tempo", "data_hora", "mesa", "evento", "detalhe", "duracao_s", "duracao"])

    def registrar(self, eventos: list[Evento]) -> None:
        for e in eventos:
            data_hora = (self.inicio + timedelta(seconds=e.tempo)).strftime("%Y-%m-%d %H:%M:%S") if self.ao_vivo else ""
            self._csv.writerow([
                f"{e.tempo:.1f}", formatar_tempo(e.tempo), data_hora, e.mesa, e.tipo, e.detalhe,
                "" if e.duracao is None else f"{e.duracao:.1f}",
                "" if e.duracao is None else formatar_tempo(e.duracao),
            ])
            duracao = f" (durou {formatar_tempo(e.duracao)})" if e.duracao is not None else ""
            print(f"[{formatar_tempo(e.tempo)}] {e.mesa}: {e.tipo} {e.detalhe}{duracao}")
        if eventos:
            self._arquivo.flush()

    def finalizar(self, mesas: list[Mesa], t_final: float, fps_medio: float) -> dict:
        self._arquivo.close()
        resumo_mesas = []
        for m in mesas:
            ocupacoes = list(m.duracoes_ocupacao)
            if m.estado == Estado.OCUPADA:  # ocupação em andamento conta no tempo total
                ocupacoes.append(m.tempo_no_estado(t_final))
            total_ocupado = sum(ocupacoes)
            resumo_mesas.append({
                "mesa": m.nome,
                "estado_final": m.estado.value,
                "ocupacoes_concluidas": len(m.duracoes_ocupacao),
                "tempo_medio_ocupacao_s": _media(m.duracoes_ocupacao),
                "tempo_medio_ate_liberar_s": _media(m.duracoes_limpeza),
                "tempo_total_ocupada_s": round(total_ocupado, 1),
                "taxa_ocupacao_pct": round(100 * total_ocupado / t_final, 1) if t_final > 0 else 0.0,
            })

        todas_ocup = [d for m in mesas for d in m.duracoes_ocupacao]
        todas_limp = [d for m in mesas for d in m.duracoes_limpeza]
        resumo = {
            "duracao_analisada_s": round(t_final, 1),
            "fps_medio_processamento": round(fps_medio, 1),
            "total_ocupacoes": len(todas_ocup),
            "tempo_medio_ocupacao_s": _media(todas_ocup),
            "tempo_medio_ate_liberar_s": _media(todas_limp),
            "mesas": resumo_mesas,
            "arquivo_eventos": str(self.caminho_csv),
        }
        caminho_json = self.pasta / f"resumo_{self.carimbo}.json"
        caminho_json.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
        self._imprimir(resumo)
        print(f"\nEventos: {self.caminho_csv}\nResumo:  {caminho_json}")
        return resumo

    @staticmethod
    def _imprimir(r: dict) -> None:
        def fmt(v):
            return "-" if v is None else formatar_tempo(v)
        print("\n" + "=" * 72)
        print(f"RESUMO  |  tempo analisado {formatar_tempo(r['duracao_analisada_s'])}"
              f"  |  {r['fps_medio_processamento']} FPS  |  {r['total_ocupacoes']} ocupacoes")
        print("=" * 72)
        print(f"{'Mesa':<12}{'Ocup.':>6}{'Media ocup.':>13}{'Media ate liberar':>19}{'Taxa ocup.':>12}")
        for m in r["mesas"]:
            print(f"{m['mesa']:<12}{m['ocupacoes_concluidas']:>6}{fmt(m['tempo_medio_ocupacao_s']):>13}"
                  f"{fmt(m['tempo_medio_ate_liberar_s']):>19}{m['taxa_ocupacao_pct']:>11}%")


def _media(valores: list[float]) -> float | None:
    return round(sum(valores) / len(valores), 1) if valores else None
