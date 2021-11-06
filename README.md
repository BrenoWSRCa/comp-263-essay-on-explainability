# comp-263-essay-on-explainability



## Crie um python environment e instale as dependências

```bash
conda create -f environment.yml
conda activate cmp_263

```

## Ideia para os exps:

1. **Árvores de decisão:**
    1. Usar o conjunto de validação para achar os parâmetros e uma árvore curta que tenha boa acurácia
    2. Mostrar que random forest pode ser usado mas ai fica mais difícil de interpretar
        - Aqui espero conseguirmos resultados similares a métodos black-box, mas pode ser que não consigamos
2. **CNN Vanilla:**
    1. Podemos usar qualquer CNN da literatura que tenha bons resultados e tentar explicar:
        - Podemos usar GradCam: [http://gradcam.cloudcv.org/](http://gradcam.cloudcv.org/)
        - Outros métodos para extrair uma interpretação
3. **Discussão**
    1. O que é uma explicação e o que é uma interpretação
    2. Ganhos ou perdas em performance
    3. Melhor interpretar/explicar algo black-box ou criar algo interpretável/explicável desde o início?