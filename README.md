# Detecção de placas veiculares com YOLOv8n: treino do zero vs. fine-tuning

Trabalho 2 de **Redes Neurais e Aprendizado Profundo (EMA9824)**, semestre 2026/2. Grupo 7.

**Pergunta:** partir de pesos pré-treinados no COCO compensa, em relação a treinar o YOLOv8n do zero, para detectar placas veiculares com o mesmo orçamento de treino?

**Resposta curta:** sim. Na validação, o fine-tuning eleva o mAP@0.5:0.95 de **0,623 ± 0,011** para **0,698 ± 0,005**, com o mesmo tempo de treino. O ganho se concentra nas imagens com várias placas, um cenário que não aparece no treino: nelas, o mAP@0.5:0.95 vai de **0,085 para 0,252**.

---

## Dados

| | |
|---|---|
| Fonte | [License Plates — Roboflow Universe](https://universe.roboflow.com/license-plates-acy8e/license-plates-xfeyr), versão 2, formato YOLOv8 (exportação de 22/05/2025) |
| Licença | CC BY 4.0 |
| Classes | 1 (placa) |
| Treino | 14.984 imagens / 14.984 placas (7.492 fotos de origem) |
| Validação | 4.012 imagens / 4.608 placas (2.006 fotos de origem) |
| Teste | 2.030 imagens / 2.030 placas (1.015 fotos de origem) |

O notebook baixa os dados direto do Roboflow e verifica a composição dos splits. Nenhuma foto de origem aparece em mais de um split. Cada foto aparece duas vezes dentro do seu split (augmentação aplicada pelo Roboflow).

**Atenção à composição dos splits:** o treino e o teste têm **exatamente uma placa por imagem**. As imagens com duas ou mais placas aparecem **só na validação**. Isso explica por que o teste satura e por que a validação é o conjunto mais informativo (ver Resultados).

## Método

| | Baseline | Melhoria |
|---|---|---|
| Modelo | YOLOv8n (`yolov8n.yaml`), pesos aleatórios | YOLOv8n (`yolov8n.pt`), pré-treinado no COCO (Ultralytics) |
| O que é treinado | rede inteira | rede inteira (fine-tuning completo) |

**Hipótese:** o backbone pré-treinado no COCO já extrai bordas, texturas e formas genéricas, e o COCO tem muitas cenas com vários objetos. Por isso esperamos convergência mais rápida, mAP maior e melhor generalização para imagens diferentes das do treino.

**Protocolo (um fator por vez):** a única diferença entre os dois modelos são os pesos iniciais. Os dois usam:
- 15 épocas, `imgsz=416`, `batch=16` e os demais hiperparâmetros padrão do Ultralytics;
- **Ultralytics 8.4.208** (versão fixa);
- **3 sementes** (42, 43, 44), com resultados em média ± desvio padrão;
- a validação para escolher a melhor época (`best.pt`) e o **teste consultado uma única vez**, no final.

Ambiente: Google Colab, GPU NVIDIA Tesla T4.

## Resultados

### Validação (média ± desvio, 3 sementes)

| Modelo | Precisão | Recall | mAP@0.5 | mAP@0.5:0.95 | Treino (min) |
|---|---|---|---|---|---|
| Baseline (do zero) | 0,953 ± 0,009 | 0,708 ± 0,024 | 0,763 ± 0,018 | 0,623 ± 0,011 | 48,3 ± 0,4 |
| Fine-tuning (COCO) | 0,959 ± 0,004 | 0,824 ± 0,013 | 0,862 ± 0,015 | **0,698 ± 0,005** | 49,3 ± 0,9 |

### Validação por número de placas na imagem

| Subconjunto | Modelo | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| 1 placa | Baseline | 0,870 ± 0,010 | 0,915 ± 0,012 | 0,775 ± 0,010 |
| 1 placa | Fine-tuning | 0,943 ± 0,007 | 0,966 ± 0,011 | 0,826 ± 0,001 |
| 2+ placas | Baseline | 0,225 ± 0,049 | 0,204 ± 0,064 | 0,085 ± 0,022 |
| 2+ placas | Fine-tuning | 0,451 ± 0,032 | 0,490 ± 0,038 | **0,252 ± 0,024** |

### Teste (consultado uma vez)

| Modelo | Precisão | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Baseline (do zero) | 0,998 ± 0,002 | 0,995 ± 0,002 | 0,995 ± 0,000 | 0,885 ± 0,007 |
| Fine-tuning (COCO) | 0,999 ± 0,001 | 1,000 ± 0,000 | 0,995 ± 0,000 | 0,899 ± 0,013 |

O teste **satura**: os dois modelos atingem o teto do mAP@0.5 (0,995). Ele tem a mesma distribuição do treino (uma placa por imagem) e não distingue os modelos. No mAP@0.5:0.95, o ganho (+0,013) é da ordem da dispersão entre sementes.

### Conclusão

O pré-treino no COCO melhora todos os cenários avaliados sem custo extra de treino. O efeito é pequeno nas imagens parecidas com as do treino e grande nas imagens com várias placas, onde o mAP@0.5:0.95 triplica.

## Limitações

- O teste só tem imagens com uma placa e satura. A comparação mais informativa é a da validação, que também foi usada para escolher a melhor época de cada treino. Isso deixa os números da validação levemente otimistas, igualmente para os dois modelos.
- Nenhuma imagem de treino tem mais de uma placa, então ambos os modelos ainda erram muito nessas cenas (recall ≤ 0,45).
- Os modelos foram treinados por 15 épocas. Na validação, o fine-tuning atinge o pico por volta da época 5; o baseline ainda melhorava no fim.
- O treino foi feito em mais de uma sessão do Colab. Por isso, os logs das células de treino mostram só as sementes da última sessão. Os pesos e `results.csv` das 3 sementes ficam no Google Drive e são lidos pelas células de avaliação.

**Com mais tempo:** treinar com mosaic durante todo o treino (`close_mosaic=0`), que gera cenas artificiais com várias placas. Com 15 épocas e o padrão `close_mosaic=10`, ele só fica ativo nas 5 primeiras.

## Como reproduzir

1. Abra `T2_Grupo7.ipynb` no Google Colab com GPU (**Ambiente de execução → Alterar tipo → T4 GPU**).
2. Em **Secrets** (ícone de chave no menu lateral), crie `ROBOFLOW_KEY` com a sua chave de API do Roboflow (conta gratuita).
3. Execute **Ambiente de execução → Executar tudo**. O notebook:
   - instala `ultralytics==8.4.208` e `roboflow`;
   - monta o Google Drive, onde os treinos são salvos em `MyDrive/T2_Grupo7_baseline` e `MyDrive/T2_Grupo7_finetuning`;
   - baixa o dataset e verifica os splits;
   - treina o baseline e o fine-tuning, com 3 sementes cada (cerca de 50 min por treino numa T4, ~5 h no total);
   - avalia todos os modelos na validação e no teste, com a estratificação por número de placas;
   - gera as figuras.

A **figura principal** (convergência do mAP na validação, baseline vs. fine-tuning) é gerada pela última célula e salva em `/content/convergencia_baseline_vs_ft.png`.

As dependências estão em `requirements.txt`. No Colab, a célula de instalação do notebook já fixa as versões.

## Autores

| Integrante | Contribuição |
|---|---|
| Guilherme Bisse Alves Rodrigues | Escolha do tema e do dataset, código de treinamento e avaliação (em conjunto) |
| Pietro Ferreira Franca | Escolha do tema e do dataset, código de treinamento e avaliação (em conjunto) |

## Referências

- Jocher, G., Chaurasia, A., Qiu, J. *Ultralytics YOLOv8*, 2023. https://github.com/ultralytics/ultralytics
- License Plates Dataset (v2). Roboflow Universe, 2025. https://universe.roboflow.com/license-plates-acy8e/license-plates-xfeyr — licença CC BY 4.0.
- Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context*. ECCV, 2014.
