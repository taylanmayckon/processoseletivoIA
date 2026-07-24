# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 📝 Relatório do Candidato

👤 **Nome Completo: Taylan Mayckon Oliveira Araujo**

### 1️⃣ Resumo da Abordagem

<!-- Descreva os hiperparâmetros de fine-tuning utilizados (épocas, tamanho de
imagem, batch size) e quaisquer ajustes feitos para lidar com o desbalanceamento
de classes, se houver. -->
O fine-tuning foi realizado a partir do modelo pré-treinado YOLO11n, utilizando o dataset de detecção de máscaras fornecido pelo processo seletivo, já convertido para o formato YOLO. Antes do treinamento, conduziu-se uma análise exploratória para verificar a distribuição das classes e a consistência das anotações.

O conjunto de dados é composto por 683 imagens de treinamento e 170 de validação, distribuídas em três classes:

| Classe | Quantidade de objetos | Distribuição | 
|----------|----------|----------|
| with_mask   | 3232 | 79,37% | 
| without_mask | 717 | 17,61% | 
| mask_weared_incorrect | 123 | 3,02% |

Observa-se um desbalanceamento acentuado: a classe mask_weared_incorrect possui cerca de 26 vezes menos exemplos que with_mask. Uma inspeção visual de amostras das anotações, realizada com um script auxiliar, não indicou problemas de posicionamento das bounding boxes ou de conversão do formato original.

Os hiperparâmetros utilizados no treinamento foram:

| Parâmetro | Valor |
|----------|----------|
| Modelo base | YOLO11n |
| Pesos iniciais | Pré-treinados (pretrained=True) |
| Épocas | 20 |
| Tamanho da imagem (imgsz) | 640x640 |
| Batch size | 8 |
| Dispositivo | CPU |
| Otimizador | Auto (seleção automática da Ultralytics) |
| Seed | 42 |
| Workers | 0 |
| Early stopping (patience) | 10 |

A definição de 20 épocas buscou equilibrar tempo de treinamento e convergência, considerando que o modelo parte de pesos pré-treinados e que o dataset é de tamanho reduzido. A resolução de entrada foi mantida em 640x640 pixels em função de uma constatação da análise exploratória: as bounding boxes ocupam, em média, apenas 1,64% da área das imagens, o que caracteriza um cenário de detecção de objetos pequenos, reduzir a resolução tenderia a comprometer ainda mais a informação disponível sobre os rostos. O batch size 8 foi adotado como valor conservador para viabilizar o treinamento em CPU sem comprometer a estabilidade.

Não foram aplicadas técnicas específicas para mitigar o desbalanceamento entre classes, como oversampling ou ponderação de perda. Optou-se por manter o pipeline padrão da Ultralytics e avaliar posteriormente o impacto do desbalanceamento nas métricas por classe, já que era esperado desempenho inferior para mask_weared_incorrect dada sua baixa representatividade. O parâmetro patience=10 habilitou o early stopping, mecanismo que não chegou a ser acionado, uma vez que o treinamento apresentou melhora contínua ao longo das 20 épocas executadas.


### 2️⃣ Bibliotecas Utilizadas

<!-- Liste as principais bibliotecas utilizadas, preferencialmente com suas versões. -->

| Biblioteca | Versão | Finalidade |
|----------|----------|----------|
| Python | 3.10.3 | Linguagem utilizada no desenvolvimento |
| Ultralytics | 8.4.26 | Treinamento, validação, exportação e inferência do modelo YOLO11n |
| PyTorch | 2.13.0 | Backend utilizado pela Ultralytics durante o treinamento |
| TensorFlow | 2.19.0 | Execução da inferência com o modelo TensorFlow Lite|
| TensorFlow Lite | 2.19.0 | Execução do modelo model.tflite |

As bibliotecas os e shutil, empregadas na manipulação de arquivos e diretórios, integram a biblioteca padrão do Python e não possuem versionamento independente.


### 3️⃣ Técnica de Otimização do Modelo

<!-- Explique o processo de exportação para TFLite realizado em `optimize_model.py`. -->
O modelo treinado (model.pt) foi convertido para o formato TensorFlow Lite pelo script optimize_model.py, por meio da função model.export(format="tflite") da biblioteca Ultralytics.

Foram avaliadas duas estratégias de quantização: Float16, que reduz a precisão dos pesos de 32 para 16 bits mantendo a representação em ponto flutuante, e Int8, que converte pesos e operações para uma representação inteira de 8 bits. A segunda costuma permitir reduções de tamanho maiores e é bastante empregada em aplicações de Edge AI com restrição de memória e armazenamento.

Nas versões recentes da Ultralytics, o processo de exportação passou a gerar uma pasta intermediária (model_saved_model) contendo múltiplos artefatos da conversão, entre eles as versões Float32, Float16 e Int8 do modelo TensorFlow Lite, além de arquivos auxiliares como o modelo ONNX e dados de calibração. O script localiza automaticamente o arquivo model_int8.tflite, copia-o para a raiz do projeto e o renomeia para model.tflite, essa versão foi a escolhida como entrega final, pelas razões apresentadas na seção seguinte.

### 4️⃣ Resultados Obtidos

<!-- Informe o mAP50 (e, se possível, o mAP50-95) obtido na validação, por classe se
possível, e o tamanho dos arquivos `model.pt` e `model.tflite`. -->

As métricas de validação obtidas ao final do treinamento foram:

| Métrica | Valor |
|----------|----------|
| Precision | 0,772 |
| Recall | 0,707 |
| mAP50 | 0,754 |
| mAP50-95 | 0,535 |

O desempenho por classe é apresentado a seguir:

| Classe | Precision | Recall | mAP50 | mAP50-95 |
|----------|----------|----------|----------|----------|
| with_mask | 0,908 | 0,940 | 0,964 | 0,686 |
| without_mask | 0,716 | 0,700 | 0,763 | 0,506 |
| mask_weared_incorrect | 0,692 | 0,473 | 0,534 | 0,412 |

A classe with_mask apresentou o melhor desempenho, seguida por without_mask. O resultado inferior de mask_weared_incorrect era esperado, dado que essa classe representa apenas cerca de 3% dos objetos anotados no conjunto de dados.

Os tamanhos dos modelos gerados foram:

| Arquivo | Tamanho |
|----------|----------|
| model.pt | 5,20 MB |
| model.tflite (Float16) | 5,11 MB |
| model.tflite (Int8) | 2,84 MB |

Além da validação do modelo original, foram realizados testes de inferência utilizando as versões Float16 e Int8 exportadas para TensorFlow Lite. Nas cinco imagens selecionadas para avaliação, ambas produziram as mesmas detecções, incluindo a identificação de uma ocorrência da classe minoritária mask_weared_incorrect.

A validação da versão model.tflite (Int8) resultou em mAP50 de 0,737 e mAP50-95 de 0,528, valores ligeiramente inferiores aos obtidos pelo modelo original. Considerando a redução de aproximadamente 45% no tamanho do arquivo e a pequena perda de desempenho observada, optou-se pela entrega da versão quantizada em Int8, priorizando um modelo mais adequado para execução em dispositivos embarcados.

### 5️⃣ Comentários Adicionais (Opcional)

<!-- Dificuldades encontradas, decisões técnicas importantes, limitações do modelo
(ex: desempenho na classe minoritária), aprendizados durante o desafio. -->

A principal limitação observada ao longo do desenvolvimento decorre do desbalanceamento do conjunto de dados: a classe with_mask responde por cerca de 79% dos objetos anotados, contra apenas 3% de mask_weared_incorrect. Esse desequilíbrio afetou sobretudo o recall da classe minoritária, mas ainda assim o modelo foi capaz de aprender padrões distintos para as três categorias, considerando a quantidade reduzida de exemplos disponíveis para ela.

A exportação para TensorFlow Lite teve uma pequena diferença em relação ao que foi sugerido na documentação do processo seletivo. Versões recentes da Ultralytics passaram a gerar múltiplos artefatos intermediários durante a conversão, em vez de produzir diretamente um único arquivo .tflite, o que tornou necessária uma etapa extra em optimize_model.py para localizar e selecionar automaticamente a versão desejada entre os artefatos gerados. Testar Float16 e Int8 lado a lado acabou sendo útil justamente por isso, foi possível comparar tamanho e comportamento das duas antes de decidir qual deveria ser escolhida.

De maneira geral, o projeto permitiu percorrer as etapas centrais de um pipeline de detecção de objetos voltado a Edge AI, desde a análise do conjunto de dados ao fine-tuning, exportação, otimização e validação do modelo final.



### 6️⃣ Exemplo de Inferência

<!-- Cole a saída do terminal ao rodar `run_inference.py` (número de detecções por
imagem), e comente brevemente sobre o que observou ao abrir as imagens
anotadas em `runs/detect/inferencia_exemplos/predicoes/` — por exemplo, se as
caixas ficaram bem localizadas, se houve confusão entre classes, ou se a
classe minoritária (`mask_weared_incorrect`) teve desempenho visivelmente pior. -->

O script run_inference.py foi executado sobre cinco imagens do conjunto de validação, utilizando exclusivamente o modelo model.tflite. A saída obtida foi:

```console
Results saved to C:\Users\tayla\Documents\PNAAT\Processo Seletivo\processoseletivoIA\projetos\3-deteccao-mascaras\runs\detect\inferencia_exemplos\predicoes
maksssksksss105.jpg                          9  [9x with_mask]
Results saved to C:\Users\tayla\Documents\PNAAT\Processo Seletivo\processoseletivoIA\projetos\3-deteccao-mascaras\runs\detect\inferencia_exemplos\predicoes
maksssksksss107.jpg                          1  [1x with_mask]
Results saved to C:\Users\tayla\Documents\PNAAT\Processo Seletivo\processoseletivoIA\projetos\3-deteccao-mascaras\runs\detect\inferencia_exemplos\predicoes
maksssksksss11.jpg                          24  [1x mask_weared_incorrect, 21x with_mask, 2x without_mask]
Results saved to C:\Users\tayla\Documents\PNAAT\Processo Seletivo\processoseletivoIA\projetos\3-deteccao-mascaras\runs\detect\inferencia_exemplos\predicoes
maksssksksss113.jpg                          4  [3x with_mask, 1x without_mask]
Results saved to C:\Users\tayla\Documents\PNAAT\Processo Seletivo\processoseletivoIA\projetos\3-deteccao-mascaras\runs\detect\inferencia_exemplos\predicoes
maksssksksss12.jpg                          16  [12x with_mask, 4x without_mask]
----------------------------------------------------------------------
TOTAL                                       54

✅ Imagens anotadas salvas em: runs/detect/inferencia_exemplos/predicoes/
```

Em formato tabular, para facilitar a visualização:

| Imagem | Detecções | Detalhes |
|----------|----------|----------|
| maksssksksss105.jpg | 9 | [9x with_mask] |
| maksssksksss107.jpg  | 1 | [1x with_mask] |
| maksssksksss11.jpg | 24 | [1x mask_weared_incorrect, 21x with_mask, 2x without_mask] |
| maksssksksss113.jpg  | 4 | [3x with_mask, 1x without_mask] |
| maksssksksss12.jpg  | 16 | [12x with_mask, 4x without_mask] |
| TOTAL | 54 |  |

As imagens anotadas foram então comparadas visualmente com as imagens originais. As bounding boxes apresentaram boa localização sobre os rostos detectados, inclusive em imagens com grande número de pessoas, nas quais o modelo conseguiu identificar corretamente diversos rostos simultaneamente.

Não foram observadas confusões evidentes entre classes nas amostras analisadas. Registrou-se uma detecção da classe mask_weared_incorrect, com confiança de aproximadamente 0,93 e bounding box corretamente posicionada. Ainda que essa classe apresente o desempenho mais baixo nas métricas agregadas de validação, resultado do forte desbalanceamento do conjunto de dados, essa limitação não se manifestou de forma clara nas imagens utilizadas para inferência.

Em síntese, a inspeção visual corroborou as métricas de validação, indicando que o modelo exportado para TensorFlow Lite manteve comportamento compatível com o modelo original treinado em PyTorch.

---

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).
