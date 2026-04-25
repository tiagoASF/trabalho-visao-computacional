# Trabalho Visão Computacional — Detecção de Armas de Fogo
 
Detecção e classificação de armas de fogo em imagens usando **YOLOv8n** com transfer learning, distinguindo entre **armas curtas** (pistolas, revólveres) e **armas longas** (fuzis, espingardas).
 
Projeto desenvolvido como trabalho final de disciplina do **MBA em IA Generativa**.
 
---

## Objetivo
 
Treinar um detector de objetos capaz de localizar e classificar armas de fogo em imagens, em duas categorias:
 
> **ArmaCurta**
> 
> **ArmaLonga**
  
A classificação adotada é tem viés **operacional** (porte e ocultabilidade), não estritamente morfológico.
 
---
## Arquitetura e Decisões Técnicas
 
| Componente | Escolha | Justificativa |
|------------|---------|---------------|
| Modelo base | YOLOv8n (`yolov8n.pt`) | Versão *nano* da família YOLOv8: balanço entre velocidade e acurácia, viável para treino em hardware Apple Silicon |
| Framework | Ultralytics 8.4.41 | API estável, integração nativa com PyTorch e métricas padrão |
| Hardware | Apple M3 Pro com aceleração MPS | Plataforma de desenvolvimento; AMP desabilitado por instabilidade conhecida em MPS |
| Imagem | 640×640 | Default do YOLOv8, equilíbrio entre detalhe e custo computacional |
| Batch size | 8 | Ajustado à memória unificada do M3 Pro |
| Otimizador | AdamW (auto) | Selecionado automaticamente pelo Ultralytics; lr inicial ≈ 0.00167 |
| Early stopping | `patience=15` | Proteção contra overfitting (com ressalvas) |
 
---

## Estrutura do Repositório
 
```
trabalho_visao_computacional/
├── README.md
├── pyproject.toml          # Dependências e metadados (uv)
├── uv.lock                 # Lock de dependências (versões exatas)
├── .python-version         # Python 3.11.9
├── .gitignore
├── config.yaml             # Configuração do dataset YOLO
├── classes_dataset.txt     # Referência das classes
├── treino.py               # Script principal de treino e avaliação
├── dataset/                # NÃO versionado
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── runs/                   # NÃO versionado
    └── detect/
        └── transfer_ep30/
            └── yolo_transfer/
                ├── weights/
                │   ├── best.pt
                │   └── last.pt
                ├── results.png
                ├── PR_curve.png
                ├── F1_curve.png
                ├── confusion_matrix.png
                └── ...
```
 
---
## Setup do Ambiente
 
O projeto usa [**uv**] para gerenciamento de dependências e ambiente virtual.
 
### Pré-requisitos
 
- macOS, Linux ou Windows
- [uv](https://docs.astral.sh/uv/getting-started/installation/) instalado
- Git
  
### Instalação
 
```bash
# Clonar o repositório do github
git clone <url-do-repo>
cd trabalho_visao_computacional
 
# Criar ambiente e instalar dependências
uv venv
uv sync
```
 
O `uv sync` instala todas as dependências definidas no `pyproject.toml` e congeladas no `uv.lock`, garantindo reprodutibilidade.
 
### Dependências principais
 
- `ultralytics` — framework YOLOv8
- `numpy` — operações numéricas (dependência transitiva já presente, listada explicitamente)
---
 
## Dataset
 
O dataset **não está versionado** neste repositório, mas foi usado o fornecido pelo professor.
 
### Composição
 
| Split | Imagens | ArmaCurta | ArmaLonga | Total instâncias |
|-------|---------|-----------|-----------|------------------|
| train | 101 | 55 | 59 | 114 |
| val   | 15  | 10 | 6  | 16  |
| test  | 15  | 10 | 10 | 20  |
| **Total** | **131** | **75** | **75** | **150** |
 
 
### Preparação local
 
Coloque o dataset em `dataset/` seguindo a estrutura mostrada acima. Edite o campo `path` do `config.yaml` para apontar para o caminho absoluto local:
 
```yaml
path: /caminho/para/dataset
```
 
---
 
## Como Treinar
 
```bash
uv run treino.py
```
 
O script `treino.py`:
 
1. Carrega o modelo pré-treinado `yolov8n.pt`
2. Treina por até 100 épocas com early stopping (`patience=15`)
3. Salva o melhor modelo em `runs/detect/transfer_ep100/yolo_transfer/weights/best.pt`
4. Avalia automaticamente nos splits `val` e `test`, imprimindo as métricas
   
Parâmetros principais estão hardcoded no script:
 
```python
epochs=100, imgsz=640, batch=8, device='mps',
patience=15, plots=True, amp=False
```
 
---
 
## Modelo Recomendado
 
Para uso final, recomenda-se o modelo do **Experimento 1 (30 épocas)** pelos seguintes motivos:
 
1. Melhor desempenho no conjunto de teste (verdadeiro indicador de generalização)
2. Menor disparidade entre val e test (mais robusto)
3. Tempo de treino significativamente menor (~3 min vs ~8 min)
Métricas finais (modelo recomendado, conjunto de teste):
 
| Métrica | Valor |
|---------|-------|
| mAP@0.5 | 0.644 |
| mAP@0.5:0.95 | 0.301 |
| Precisão média | 0.719 |
| Recall médio | 0.550 |
 
---
## Limitações e melhorias
 
### Limitações reconhecidas
 
- **Tamanho do dataset** insuficiente para estimativas estatisticamente robustas
- **Splits aleatórios** sem estratificação por classe ou por fonte (risco de leakage)
- **Avaliação em split único** (15 imagens de teste) com alta variância amostral
- **Classificação operacional** que mistura subtipos visualmente heterogêneos
- **Possível viés de fonte** se imagens vieram de poucos vídeos ou origens

### Melhorias possíveis
 
- **Expansão do dataset** com diversidade de ângulos, iluminação, contextos e modelos de armas
- **Split estratificado** por classe e por origem das imagens
- **Calibração de threshold de confiança** com base em curvas PR/F1 do conjunto de validação
- **Ablation studies**: efeito de freeze de camadas, learning rate scheduling, augmentations específicas
---
 
## Reprodutibilidade
 
Para reproduzir os experimentos:
 
```bash
uv sync                    # ambiente via uv.lock
uv run treino.py           # treina e avalia
```
 
---
 
## Autor
 
Tiago André da Silveira Fialho
Mat.: 192.028-6
