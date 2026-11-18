# Câmera de Segurança como Gerente de Salão

Protótipo de **visão computacional** que usa a câmera de segurança de um restaurante para acompanhar, em tempo real, o estado de cada mesa do salão: **livre**, **ocupada** ou **aguardando limpeza**, com o tempo de cada estado e alertas para a equipe.

Trabalho da disciplina **Robótica e Visão Computacional**, Ciência da Computação, Unochapecó, 2026/2, Prof. Wagner Titon.

> 🎥 **Vídeo de apresentação:** _(colocar o link do YouTube aqui)_

---

## 1. O problema

Em restaurantes com atendimento à mesa, a equipe controla o salão só com os olhos e a memória. Ninguém sabe, de forma objetiva, quais mesas estão livres, há quanto tempo um cliente está sentado ou qual mesa precisa ser limpa. Isso gera três problemas:

- clientes esperando na entrada enquanto há **mesas vazias que ninguém percebeu**;
- clientes sentados **esperando atendimento**, principalmente em mesas fora do campo de visão dos garçons;
- **mesas sujas paradas** no horário de pico porque ninguém notou que os clientes saíram.

Muitos restaurantes já têm câmeras (CFTV), mas as imagens só são usadas para segurança. A investigação completa está no documento da Etapa 1.

## 2. A solução

O sistema lê o vídeo da câmera e, para cada quadro:

1. **detecta as pessoas** com um modelo YOLO pré-treinado;
2. **associa cada pessoa a uma mesa**, com base em áreas (polígonos) desenhadas sobre a imagem;
3. **verifica se a mesa está limpa**, comparando a imagem atual com uma referência da mesa limpa;
4. **atualiza o estado** de cada mesa com filtros de tempo, para evitar alarmes falsos;
5. **mostra o resultado**: mesas coloridas sobre o vídeo, painel lateral, alertas, registro de eventos e indicadores.

| Estado | Cor | Quando |
|---|---|---|
| **LIVRE** | 🟩 verde | mesa vazia e limpa |
| **OCUPADA** | 🟥 vermelho | pessoas na mesa por pelo menos 3 s |
| **AGUARDANDO LIMPEZA** | 🟨 amarelo | clientes saíram (ninguém por 10 s) e a mesa ainda não voltou a parecer limpa |

**Alertas:** `Cliente novo: atender` nos primeiros 60 s de uma ocupação (avisa o garçom que chegou cliente), `Permanência longa` e `Limpeza atrasada` quando a mesa espera limpeza há muito tempo.

**Indicadores gerados:** número de ocupações (giro) por mesa, tempo médio de ocupação, tempo médio até a mesa ser liberada e taxa de ocupação, que hoje o restaurante não mede.

### Fluxo: entrada → processamento → saída

```mermaid
flowchart LR
    A[Câmera / vídeo / RTSP] --> B[Redimensiona quadro]
    B --> C[YOLO: detecta pessoas]
    C --> D[Associa pessoa à mesa<br/>sobreposição com o polígono]
    B --> E[Verifica limpeza<br/>diferença para a referência]
    D --> F[Máquina de estados<br/>com filtros de tempo]
    E --> F
    F --> G[Vídeo anotado + painel]
    F --> H[CSV de eventos]
    F --> I[Resumo com indicadores]
```

### Máquina de estados de cada mesa

```mermaid
stateDiagram-v2
    [*] --> LIVRE
    LIVRE --> OCUPADA: pessoa presente ≥ 3 s
    OCUPADA --> AGUARDANDO_LIMPEZA: ninguém por ≥ 10 s
    AGUARDANDO_LIMPEZA --> LIVRE: parece limpa por ≥ 3 s (ou tecla manual)
    AGUARDANDO_LIMPEZA --> OCUPADA: pessoas por ≥ 30 s (novos clientes)
```

## 3. Técnicas de visão computacional

| Método | O que faz no projeto | Por que foi escolhido | Limitações |
|---|---|---|---|
| **YOLO11n** (detecção de objetos, CNN pré-treinada no COCO) | Encontra as pessoas no quadro (classe `person`) | Rápido em CPU, preciso, não precisa de treino: a classe "pessoa" já existe no COCO | Pessoas muito encobertas (atrás de pilares, de costas, aglomeradas) podem não ser detectadas |
| **Regiões de interesse poligonais + sobreposição** | Decide de qual mesa é cada pessoa: fração da caixa da pessoa dentro do polígono da mesa | Simples, explicável, funciona com qualquer ângulo de câmera fixa | Depende de a câmera não se mover; mesas muito próximas podem confundir |
| **Filtros temporais (máquina de estados)** | Só muda o estado após X segundos consistentes | Elimina detecções falsas de um quadro e cliente que levanta para ir ao banheiro | Introduz atraso proposital de alguns segundos |
| **Processamento clássico de imagem** (tons de cinza, filtro gaussiano, diferença absoluta, limiarização, abertura morfológica) | Mede o "índice de sujeira": quanto da parte **visível** da mesa está diferente da referência limpa (pratos, copos); o que está atrás de pessoas é ignorado | Não precisa de modelo nem de treino; compensa variação de luz | Sombras fortes ou objetos da cor da mesa enganam; cadeiras fora do lugar podem contar como diferença |
| **ByteTrack** (rastreamento, opcional `--rastrear`) | Mantém um ID para cada pessoa entre quadros | Já integrado ao Ultralytics | Troca de IDs em oclusões |
| **Pixelização** (opcional `--privacidade`) | Anonimiza as pessoas no vídeo exibido/salvo | O sistema não precisa saber quem é a pessoa (LGPD) | — |

## 4. Estrutura do repositório

```
├── monitorar.py          # programa principal
├── definir_mesas.py      # ferramenta para desenhar as mesas sobre o vídeo
├── avaliar.py            # compara os eventos detectados com um gabarito manual
├── salao/
│   ├── config.py         # leitura do JSON e parâmetros
│   ├── detector.py       # YOLO + rastreamento
│   ├── mesas.py          # máquina de estados e associação pessoa → mesa
│   ├── limpeza.py        # verificação de limpeza (processamento de imagem)
│   ├── painel.py         # desenho das mesas e do painel
│   ├── relatorio.py      # CSV de eventos e resumo
│   └── janela.py         # janela maximizada / tela cheia
├── config/               # arquivos de mesas (JSON) e imagens de referência
├── testes/               # gabaritos dos vídeos de teste
│   └── mixkit/           # mesas e gabarito do teste com vídeo público
├── tests/                # testes automatizados (pytest)
├── videos/               # vídeos de entrada (não versionados)
└── saida/                # eventos, resumos e vídeos gerados (não versionados)
```

## 5. Instalação

Requisitos: **Python 3.10 ou superior** (testado com 3.13). Funciona em CPU; GPU é opcional.

```bash
git clone https://github.com/EvandroWeissheimer/gerente-de-salao-cv.git
cd gerente-de-salao-cv
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest
```

> Para não precisar ativar o ambiente (o PowerShell costuma bloquear o script `activate`), os comandos chamam o Python do `.venv` direto. Nos exemplos abaixo, troque `python` por `.venv\Scripts\python.exe`.

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

A instalação baixa o PyTorch (alguns minutos). Na primeira execução, o Ultralytics baixa automaticamente os pesos `yolo11n.pt` (~5 MB).

## 6. Como usar

### Passo 1: desenhar as mesas (uma vez por câmera)

```bash
python definir_mesas.py --fonte videos/salao.mp4 --saida config/mesas.json
```

- **clique esquerdo** adiciona pontos; **ENTER** (ou `n`) fecha a mesa; **clique direito** desfaz o ponto; `u` desfaz a mesa
- `a` / `d` voltam ou avançam 1 s no vídeo: escolha um quadro com as **mesas limpas e vazias**
- `s` salva o `config/mesas.json` e a imagem `config/mesas_referencia.jpg` (referência da mesa limpa)

> Desenhe cada área cobrindo **a mesa e as cadeiras**, onde os clientes ficam sentados.

### Passo 2: monitorar

```bash
# vídeo gravado
python monitorar.py --fonte videos/salao.mp4 --mesas config/mesas.json

# webcam
python monitorar.py --fonte 0 --mesas config/mesas.json

# câmera IP / CFTV (não publique a URL com senha no GitHub)
python monitorar.py --fonte "rtsp://usuario:senha@192.168.0.10:554/stream1" --mesas config/mesas.json

# gerar vídeo para a apresentação, com pessoas anonimizadas e visualização do processamento
python monitorar.py --fonte videos/salao.mp4 --mesas config/mesas.json --salvar saida/demo.mp4 --privacidade --processamento
```

| Tecla | Ação |
|---|---|
| `q` / `ESC` | sair (gera o resumo) |
| `p` | pausar |
| `f` | alterna entre tela cheia e janela maximizada |
| `v` | liga/desliga a visualização do processamento de imagem (pixels diferentes da referência em magenta + índice de sujeira) |
| `1`–`9` | marca a mesa N como limpa manualmente |

| Opção | Padrão | Descrição |
|---|---|---|
| `--modelo` | `yolo11n.pt` | `yolo11s.pt` é mais preciso e mais lento |
| `--pular N` | 1 | roda o YOLO a cada N quadros (use 2–3 em notebook fraco) |
| `--largura` | 1280 | largura de processamento (use a mesma no `definir_mesas.py`) |
| `--fator-tempo N` | 1 | para vídeos em timelapse: 1 s de vídeo = N s reais |
| `--rastrear` | desligado | IDs persistentes por pessoa (ByteTrack) |
| `--privacidade` | desligado | pixeliza as pessoas |
| `--salvar` | — | salva o vídeo anotado |
| `--tela-cheia` | desligado | abre em tela cheia (por padrão a janela abre maximizada) |
| `--sem-janela` | desligado | roda sem interface (servidor) |

### Passo 3: avaliar a precisão (testes)

1. Assista ao vídeo de teste e anote os momentos reais em um CSV (veja `testes/gabarito_exemplo.csv`):
   ```csv
   mesa,evento,tempo
   Mesa 1,OCUPOU,00:05
   Mesa 1,DESOCUPOU,00:40
   Mesa 1,LIBEROU,01:08
   ```
2. Rode o monitoramento no vídeo e compare:
   ```bash
   python avaliar.py --eventos saida/eventos_AAAAMMDD_HHMMSS.csv --gabarito testes/gabarito_video1.csv
   ```
   Saída: acertos, eventos perdidos, falsos alarmes, precisão, revocação e erro médio de tempo por tipo de evento.

### Testes automatizados

```bash
python -m pytest
```

Os testes cobrem a máquina de estados (ocupação, saída temporária, limpeza pelo garçom, mesa reocupada sem limpeza), a associação pessoa → mesa e a verificação de limpeza com mudança de iluminação.

## 7. Parâmetros

Ficam no bloco `"parametros"` do `config/mesas.json` (todos os tempos em segundos):

| Parâmetro | Padrão | Significado |
|---|---|---|
| `confianca` | 0.35 | confiança mínima da detecção |
| `sobreposicao_minima` | 0.30 | fração da pessoa que precisa estar dentro da mesa |
| `tempo_confirmar_ocupacao` | 3 | presença necessária para OCUPADA |
| `tempo_confirmar_saida` | 10 | ausência necessária para considerar que saíram |
| `tempo_confirmar_limpa` | 3 | tempo parecendo limpa para voltar a LIVRE |
| `tempo_presenca_reocupar` | 30 | presença em mesa suja tratada como novos clientes |
| `alerta_nova_ocupacao` | 60 | duração do aviso "ATENDER" |
| `alerta_permanencia` | 3600 | alerta de permanência longa |
| `alerta_limpeza` | 120 | alerta de mesa esperando limpeza |
| `fator_area_limpeza` | 0.6 | parte central do polígono usada na verificação de limpeza |
| `visibilidade_minima` | 0.5 | parte da mesa que precisa estar visível para medir a limpeza |
| `limiar_pixel` | 35 | diferença (0–255) para um pixel contar como mudado |
| `limiar_sujeira` | 0.06 | fração máxima de pixels mudados para a mesa estar limpa |

## 8. Saídas

- **Janela/vídeo anotado:** mesas coloridas, caixas das pessoas, cronômetros, alertas e painel com o resumo do salão.
- **`saida/eventos_*.csv`:** cada mudança de estado (`OCUPOU`, `DESOCUPOU`, `LIBEROU`) com horário e duração.
- **`saida/resumo_*.json`** e resumo no terminal: ocupações, tempo médio de ocupação, tempo médio até liberar e taxa de ocupação por mesa.

## 9. Resultados dos testes

### 9.1 Vídeo público (timelapse de restaurante)

Vídeo: [Customers in a minimalist style restaurant (Mixkit)](https://mixkit.co/free-stock-video/customers-in-a-minimalist-style-restaurant-4385/): câmera fixa no alto, 15 s em timelapse. Considerando 1 s de vídeo = 20 s reais, são cerca de 5 minutos de salão. O vídeo não está no repositório (a licença não permite redistribuir): baixe-o pelo link e salve como `videos/mixkit_restaurante.mp4`.

```bash
python monitorar.py --fonte videos/mixkit_restaurante.mp4 --mesas testes/mixkit/mesas.json --fator-tempo 20 --processamento
python avaliar.py --eventos saida/eventos_*.csv --gabarito testes/mixkit/gabarito.csv --tolerancia 20
```

O gabarito foi anotado assistindo ao vídeo quadro a quadro (4 mesas da fileira da direita):

| Evento | Esperado | Detectado | Resultado |
|---|---|---|---|
| Mesa 2 ocupada | 00:00 | 00:00 | ✅ |
| Mesa 3 ocupada | 00:00 | 00:00 | ✅ |
| Mesa 2 desocupada | ~02:00 | 02:02 | ✅ |
| Mesa 4 ocupada | ~03:00 | 02:55 | ✅ |
| Mesa 2 liberada (limpa) | ~02:40 | não detectada* | ❌ |
| Mesa 1 (sempre vazia) | nenhum evento | nenhum evento | ✅ |

**4 de 5 eventos corretos, nenhum alarme falso, erro médio de 1,7 s, 12,5 FPS em CPU.**

\* A Mesa 2 já começa ocupada, então o sistema nunca viu essa mesa limpa e não tem referência para comparar. Fornecendo uma foto de referência, a liberação foi detectada às 03:56: depois de limpa, a mesa foi arrumada diferente da foto (toalha, sem taças) e só ficou igual quando as taças voltaram. Ou seja, a referência precisa mostrar a mesa **arrumada como o restaurante a deixa**.

O que esse teste ensinou:
- Com a câmera alinhada com a fileira de mesas, as pessoas de uma mesa tapam parte da mesa de trás. Por isso a verificação de limpeza passou a medir só a parte visível da mesa.
- Associar a pessoa pela parte de cima da caixa (cabeça e tronco) foi testado e **piorou** o resultado nesse ângulo; a caixa inteira funcionou melhor.
- Aumentar a entrada do YOLO de 640 para 960 não mudou os eventos e deixou o processamento quase 2× mais lento.

### 9.2 Testes do grupo

Os vídeos gravados pelo grupo não ficam no repositório por causa do tamanho. Baixe-os em _(colocar o link do Google Drive aqui)_ e coloque na pasta `videos/`. As configurações das mesas e os gabaritos de cada vídeo ficam em `testes/`.

> _Preencher com os vídeos gravados pelo grupo: situação testada, resultado esperado × obtido e a saída do `avaliar.py`._

| Teste | Vídeo | Situação | Esperado | Obtido |
|---|---|---|---|---|
| 1 | | Clientes sentam e saem | | |
| 2 | | Cliente levanta e volta em < 10 s | Mesa continua ocupada | |
| 3 | | Clientes saem deixando pratos | Aguardando limpeza até retirarem | |
| 4 | | Garçom limpa a mesa | Mesa volta a livre | |
| 5 | | Pouca iluminação / muitas pessoas | | |

## 10. Limitações e melhorias

**Limitações**
- A câmera precisa ser **fixa**; se ela se mover, as áreas das mesas precisam ser redesenhadas.
- Pessoas encobertas (pilares, aglomeração) podem não ser detectadas; uma câmera do alto funciona melhor.
- O sistema não distingue **cliente de garçom**: um garçom parado muito tempo numa mesa livre pode ocupá-la.
- A verificação de limpeza por diferença de imagem é sensível a sombras fortes e a objetos da cor da mesa, e depende de a mesa ser arrumada igual à foto de referência.
- Mesas enfileiradas na direção da câmera se tapam parcialmente; o ideal é a câmera bem alta e de lado para a fileira.
- O sistema mede **há quanto tempo a mesa está ocupada**, não se o cliente já foi atendido.

**Melhorias possíveis**
- Detectar pratos e copos com o próprio YOLO (classes `cup`, `bowl`, `wine glass` do COCO) para confirmar a sujeira.
- Distinguir garçons pelo uniforme (classificador de cor ou modelo treinado).
- Estimativa de pose para saber se a pessoa está sentada ou em pé.
- Painel web e notificação no celular dos garçons; histórico para dimensionar a equipe por horário.
- Várias câmeras para cobrir o salão inteiro.

## 11. Privacidade (LGPD)

O sistema **não identifica pessoas**: usa apenas a posição das caixas detectadas para saber se há alguém na mesa. Nenhum rosto é reconhecido nem armazenado, e os relatórios contêm só eventos das mesas. A opção `--privacidade` anonimiza as pessoas no vídeo exibido e salvo.

## 12. Integrantes

- Bruno Guilherme de Moura
- Daniel Henrique de Moura
- Evandro Weissheimer
- Rafael Belchor Hein
- Rafael Krämer da Silva

## 13. Referências

- JOCHER, G.; QIU, J. **Ultralytics YOLO11**. 2024. Disponível em: https://github.com/ultralytics/ultralytics.
- ZHANG, Y. et al. **ByteTrack: Multi-Object Tracking by Associating Every Detection Box**. ECCV, 2022.
- LIN, T.-Y. et al. **Microsoft COCO: Common Objects in Context**. ECCV, 2014.
- BRADSKI, G. **The OpenCV Library**. Dr. Dobb's Journal of Software Tools, 2000.
- BRASIL. **Lei nº 13.709, de 14 de agosto de 2018**. Lei Geral de Proteção de Dados Pessoais (LGPD).
