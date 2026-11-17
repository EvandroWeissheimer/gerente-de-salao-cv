"""Testes da máquina de estados, da associação e da verificação de limpeza.

Rodar com:  python -m pytest -q
"""
import numpy as np
import pytest

from salao.config import Parametros
from salao.detector import Pessoa
from salao.limpeza import VerificadorLimpeza
from salao.mesas import Estado, Mesa, associar_pessoas, mascara_poligono

PARAMS = Parametros(tempo_confirmar_ocupacao=3, tempo_confirmar_saida=10,
                    tempo_confirmar_limpa=3, tempo_presenca_reocupar=30)
QUADRADO = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], np.int32)


def nova_mesa():
    return Mesa(1, "Mesa 1", QUADRADO, PARAMS)


def simular(mesa, inicio, fim, pessoas, sujeira=None, passo=0.5):
    """Roda a mesa de 'inicio' a 'fim' segundos com entrada constante."""
    eventos, t = [], inicio
    while t < fim:
        eventos += mesa.atualizar(t, pessoas, sujeira)
        t += passo
    return eventos


def test_ocupa_somente_apos_tempo_de_confirmacao():
    mesa = nova_mesa()
    simular(mesa, 0, 2.5, pessoas=2)
    assert mesa.estado == Estado.LIVRE
    eventos = simular(mesa, 2.5, 4, pessoas=2)
    assert mesa.estado == Estado.OCUPADA
    assert eventos[0].tipo == "OCUPOU"
    assert eventos[0].tempo == 0  # ocupação conta desde que a pessoa chegou


def test_deteccao_falsa_de_um_quadro_nao_ocupa():
    mesa = nova_mesa()
    simular(mesa, 0, 5, pessoas=0)
    mesa.atualizar(5, 1, None)          # 1 quadro com detecção falsa
    simular(mesa, 5.5, 20, pessoas=0)
    assert mesa.estado == Estado.LIVRE


def test_cliente_que_levanta_rapido_nao_libera_a_mesa():
    mesa = nova_mesa()
    simular(mesa, 0, 60, pessoas=2)
    simular(mesa, 60, 68, pessoas=0)     # 8 s fora (< 10 s)
    simular(mesa, 68, 90, pessoas=1)
    assert mesa.estado == Estado.OCUPADA


def test_saida_dos_clientes_vai_para_limpeza_com_duracao_correta():
    mesa = nova_mesa()
    simular(mesa, 0, 120, pessoas=3)
    eventos = simular(mesa, 120, 135, pessoas=0)
    assert mesa.estado == Estado.LIMPEZA
    desocupou = [e for e in eventos if e.tipo == "DESOCUPOU"][0]
    assert desocupou.tempo == 120
    assert desocupou.duracao == pytest.approx(120)


def test_mesa_suja_continua_aguardando_e_limpa_libera():
    mesa = nova_mesa()
    simular(mesa, 0, 60, pessoas=2)
    simular(mesa, 60, 100, pessoas=0, sujeira=0.30)   # pratos na mesa
    assert mesa.estado == Estado.LIMPEZA
    eventos = simular(mesa, 100, 110, pessoas=0, sujeira=0.01)
    assert mesa.estado == Estado.LIVRE
    assert eventos[-1].tipo == "LIBEROU"
    assert eventos[-1].detalhe == "automatico"


def test_garcom_limpando_nao_conta_como_nova_ocupacao():
    mesa = nova_mesa()
    simular(mesa, 0, 60, pessoas=2)
    simular(mesa, 60, 80, pessoas=0, sujeira=0.3)
    simular(mesa, 80, 100, pessoas=1)                  # 20 s limpando (< 30 s)
    assert mesa.estado == Estado.LIMPEZA
    simular(mesa, 100, 110, pessoas=0, sujeira=0.0)
    assert mesa.estado == Estado.LIVRE


def test_clientes_sentando_em_mesa_nao_liberada():
    mesa = nova_mesa()
    simular(mesa, 0, 60, pessoas=2)
    simular(mesa, 60, 80, pessoas=0, sujeira=0.3)
    eventos = simular(mesa, 80, 115, pessoas=2)
    assert mesa.estado == Estado.OCUPADA
    assert eventos[0].detalhe == "sem limpeza detectada"


def test_sem_medicao_de_limpeza_so_libera_manual():
    mesa = nova_mesa()
    simular(mesa, 0, 30, pessoas=1)
    simular(mesa, 30, 200, pessoas=0, sujeira=None)
    assert mesa.estado == Estado.LIMPEZA
    eventos = mesa.marcar_limpa(200)
    assert mesa.estado == Estado.LIVRE and eventos[0].detalhe == "manual"


def test_alertas():
    p = Parametros(tempo_confirmar_ocupacao=1, alerta_nova_ocupacao=60, alerta_permanencia=600,
                   tempo_confirmar_saida=5, alerta_limpeza=120)
    mesa = Mesa(1, "Mesa 1", QUADRADO, p)
    simular(mesa, 0, 2, pessoas=1)
    assert mesa.alerta(10) == "Cliente novo: atender"
    assert mesa.alerta(100) is None
    assert mesa.alerta(700) == "Permanência longa"


def test_associacao_escolhe_mesa_com_maior_sobreposicao():
    m1 = mascara_poligono(QUADRADO, 800, 600)
    m2 = mascara_poligono(QUADRADO + [300, 0], 800, 600)   # mesa à direita
    dentro_m1 = Pessoa(120, 120, 220, 280)
    entre = Pessoa(280, 120, 460, 280)                        # mais na mesa 2
    longe = Pessoa(650, 400, 750, 590)
    por_mesa, destino = associar_pessoas([dentro_m1, entre, longe], [m1, m2], minimo=0.3)
    assert destino == [0, 1, None]
    assert len(por_mesa[0]) == 1 and len(por_mesa[1]) == 1


def test_verificador_detecta_objeto_e_ignora_mudanca_de_brilho():
    rng = np.random.default_rng(0)
    limpa = np.full((600, 800, 3), 120, np.uint8)
    limpa += rng.integers(0, 6, limpa.shape, dtype=np.uint8)      # ruído da câmera
    verif = VerificadorLimpeza(mascara_poligono(QUADRADO, 800, 600), Parametros())
    verif.definir_referencia(limpa)

    assert verif.medir(limpa) < 0.01

    mais_clara = np.clip(limpa.astype(int) + 25, 0, 255).astype(np.uint8)
    assert verif.medir(mais_clara) < 0.01                        # só mudou a luz

    suja = limpa.copy()
    suja[150:210, 150:230] = 230                                  # "prato" branco
    assert verif.medir(suja) > Parametros().limiar_sujeira


def test_verificador_mede_so_a_parte_visivel():
    limpa = np.full((600, 800, 3), 120, np.uint8)
    verif = VerificadorLimpeza(mascara_poligono(QUADRADO, 800, 600), Parametros())
    verif.definir_referencia(limpa)
    suja = limpa.copy()
    suja[220:290, 220:290] = 230                       # prato no canto inferior direito
    suja[100:300, 100:160] = 20                        # pessoa escura tapando a esquerda
    pessoa = Pessoa(100, 100, 160, 300)
    assert verif.medir(suja, [pessoa]) > Parametros().limiar_sujeira   # vê o prato
    sem_prato = limpa.copy()
    sem_prato[100:300, 100:160] = 20
    assert verif.medir(sem_prato, [pessoa]) < 0.01                     # pessoa não conta como sujeira
    tapando_quase_tudo = Pessoa(90, 90, 280, 310)
    assert verif.medir(suja, [tapando_quase_tudo]) is None             # não dá para medir
