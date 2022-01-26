# Análise de Sentimentos - Base Twitter

<sub>Projeto para a disciplina de **Machine Learning** (Módulo 5) do Data Science Degree (turma de julho de 2020)</sub>

## Case

Vamos desenvolver um modelo para detectar o sentimento de uma publicação do Twitter a classificando em uma das três categorias: **positiva**, **negativa** ou **neutra**. Vamos testar pelo menos 2 técnicas de NLP diferentes e avaliar a performance do modelo segundo algumas métricas pertinentes.  

## Plano de Trabalho

Vamos abordar o problema em fases, a saber:

* [**Análise de consistência dos dados**](notebooks_exploration/1_consistencia.ipynb): analisaremos se os dados estão fazendo sentido, se os campos estão completos e se há dados duplicados ou faltantes;

* [**Análise exploratória**](notebooks_exploration/2_eda.ipynb): analisaremos a base de treino como um todo, verificando o balanceamento entre as classes e focando, principalmente, na coluna *`tweet_text`*;

* [**Pré-processamento e transformações**](notebooks_exploration/3_preproc.ipynb): projetos de NLP exigem um considerável pré-processamento. Por esse motivo, focaremos no tratamento da *string* do texto. Vamos começar com tratamentos simples e adicionaremos complexidade gradualmente. Nessa etapa testaremos diferentes técnicas de transformações, como o ***Bag Of Words*** e o ***TF-IDF***, bem como o ***Word2Vec*** e o ***Doc2Vec***;

* [**Treinamento do modelo**](notebooks_exploration/4_train.ipynb): depois das transformações, treinaremos os modelos classificadores candidatos. Nessa etapa o problema se torna semelhante aos abordados na primeira parte do módulo. Testaremos diversos classificadores como *RandomForest*, *AdaBoost*, entre outros. Otimizaremos também os hiperparâmetros do modelo com técnicas como a *GridSearch* e a *RandomizedSearch*;

* **Conclusões**: por fim, descreveremos as conclusões sobre os estudos. O modelo é capaz de identificar o sentimento das publicações? É possível extrapolar o modelo para outros contextos, como a análise de sentimento de uma frase qualquer? Tentaremos imaginar questões pertinentes e relevantes que você tenha obtido durante o desenvolvimento do projeto!

Ao final, o modelo produzirá uma base de dados classificando cada tuíte, como a seguir:

|id|sentiment_predict
|-|-|
|12123232|0
|323212|1
|342235|2

Salvaremos a tabela acima em um formato `csv` com o nome ```<nome>_<sobrenome>_nlp_degree.csv```.  

## *Dataset*

Há dois *datasets*: um de treino (com a informação de sentimento já codificada) e um para submissão (sem a informação). Os dois *datasets* consistem de 100 mil postagens no *Twitter* entre agosto e outubro de 2018.

Fora a variável *target*, `sentiment`, os seguintes campos estão presentes:

* **`id`**: um número identificador único para cada tuíte;
* **`tweet_text`**: o conteúdo do tuíte;
* **`tweet_date`**: a data em que o tuíte foi publicado;
* **`sentiment`**: variável codificando o sentimento de cada tuíte (0, se negativo; 1, se positivo; 2, se neutro)
* **`query_used`**: o filtro utilizado para buscar a publicação

---

## Conclusões

Vamos agora destacar algumas características relevantes e/ou surpreendentes que notamos no decorrer da execução da análise.

### Linha do tempo

Os *tweets* começam em 9 de agosto, mas nessa data há somente um único *tweet*. Os *tweets* começam de fato em número mais expressivo em 17 de agosto:

![linha do tempo](assets/linha_tempo.png)

Além disso, parece haver uma distribuição peculiar dos *tweets* ao longo do tempo. Os *tweets* de tom neutro parecem se manter constantes (descontando um padrão de sazonalidade) ao longo do tempo. No entanto, *tweets* negativos ou positivos se concentram em um período de duas semanas, entre o fim de setembro e meados de outubro, em grande quantidade.

Mais especificamente, *tweets* de tom negativo só aparecem em 4 dias no período estudado. Nos dias em que eles aparecem, a proporção de *tweets* negativos (em relação ao total) é significativa, variando de 60% até 95%.

Fora desse período de duas semanas, a maioria esmagadora dos *tweets* são de tom neutro. Além disso, esse período de duas semanas é marcado por uma quantidade maior de *tweets* (os de tom neutro mantêm a tendência constante, enquanto que os de tom positivo e negativo aumentam muito).

Um evento de grande relevância que ocorreu no Brasil no período de aparente maior polarização (maior proporção de *tweets* com sentimento negativo ou positivo, descontados os *tweets* de tom neutro) foi o 1° turno da eleição para presidente. É razoável que a polarização no sentimento da base de *tweets* seja em parte reflexo da alta polarização da campanha eleitoral.

Ressaltamos que, **se for esse o caso, o eventual modelo construído será severamente enviesado**. Esperávamos que as palavras associadas a *tweets* negativos ou positivos fossem tomadas por expressões relativas às eleições, como `bolsonaro`, `haddad`, `lula`, `pt` etc. Isso não se materializou (as palavras foram associadas ao tom neutro, como explicaremos mais a frente), mas ainda assim não podemos afirmar que estes são uma amostra representativa do universo de todos os *tweets* na língua portuguesa.

### Frequência de palavras

A palavra `não` foi a mais frequente, e parece ter conteúdo sentimental alto (sentimento positivo ou negativo):

![frequencia de palavras](assets/frequencia_palavras.png)

No entanto, há quantidade grande de *tweets* positivos contendo a palavra `não`.

Palavras relacionadas à eleição (`bolsonaro`, `haddad`, `pt`) são proeminentes na base. No entanto, surpreendentemente, aparentam ter conteúdo neutro. Isso se deve em parte a uma característica da base discutida mais a frente.

Outras palavras frequentes estão nos seguintes *word cluods*:

#### Sentimento negativo

![wordcloud das palavras com sentimento negativo](assets/wordcloud_negativo.png)

#### Sentimento positivo

![wordcloud das palavras com sentimento positivo](assets/wordcloud_positivo.png)

#### Sentimento neutro

![wordcloud das palavras com sentimento neutro](assets/wordcloud_neutro.png)

### Estatísticas do texto

A média de caracteres na frase completa na base completa, excluindo *@mentions* e *links*, é de 64.8 caracteres (o limite de caracteres do *Twitter* é 280 caracteres), com frases curtas dominando. O formato de *microblogging* estimula a postagem de *tweets* curtos

![histograma do numero de caracteres](assets/histograma_caract.png)

A média de palavras em um *tweet*, também excluindo *@mentions* e *links*, é de 15 palavras, com *tweets* com poucas palavras dominando também.

![histograma da contagem de palavras](assets/histograma_palavras.png)

No entanto, há uma diferença entre *tweets* com tom neutro e *tweets* com tom positivo ou negativo: **os *tweets* com sentimento neutro tem maior número de caracteres e maior número de palavras do que *tweets* com sentimento positivo ou negativo**.

Além disso, a distribuição das contagens de caracteres e de palavras são mais enviesadas a esquerda. Em outras palavras, quando o *tweet* tem sentimento negativo ou positivo, as frases curtas dominam mais do que quando o *tweet* é de tom neutro.

Logo, um *feature* que é potencialmente útil para distinguir *tweets* neutros de *tweets* positivos ou negativos é seu número de palavras/caracteres. No entanto, esse *feature* não foi utilizado no modelo final.

### Buscas feitas

Parece que, fora as *queries* `:)` e `:(`, todas as outras resultaram em *tweets* exclusivamente com tom neutro. A *query* `:)` resultou em *tweets* classificados com tom exclusivamente positivo, e a *query* `:(` resultou em *tweets* com tom exclusivamente negativo.

As outras buscas parecem ser buscas por notícias; não surpreende que sejam classificadas como neutras. Como os resultados destas buscas por notícias retornaram todos os *tweets* com conteúdo político, as palavras relacionadas à eleição foram classificadas como sendo de tom neutro.

Generalizando, na base de treino, conseguimos prever o tom do *tweet* exclusivamente pela *query* utilizada para produzí-lo. No entanto, é improvável que isso se reproduza no universo de *tweets* em geral: um *tweet* negativo ou positivo pode ser retornado mesmo com a busca sendo relacionada a notícias, por exemplo. Em outras palavras, corremos o risco de a base de treino não ser representativa do universo de *tweets* em geral. Por esse motivo, **usaremos uma mistura dos radicais com a busca utilizada para treinar o modelo final**.

### Pré-processamento de texto

Modelos de aprendizado de máquina não conseguem trabalhar com texto, somente números. Temos que ter algumas técnicas para transformar o texto em números.

Treinamos quatro modelos que transformam texto puro em vetores de *features* com os quais os modelos de aprendizado de máquina conseguem trabalhar:

* *Bag of Words* puro, ou `CountVectorizer`;
* *Bag of Words* TF-IDF (ou seja, considerando a influência de cada palavra do *tweet* bem como do *tweet* como um todo);
* *Word2vec* (com duas maneiras de combinar os vetores de cada palavra); e
* *Doc2vec*

### Treinamento de modelos

Treinamos 4 modelos de classificação multiclasse:

* Regressão logística (no esquema *one-vs-rest*, *default* para esse classificador como implementado pelo `scikit-learn`);
* *Random Forests*
* *AdaBoost*
* *XGBoost*

Combinamos cada *transformer* (que transformam texto em números) com cada estimador (que transformam números em predições), perfazendo 16 *pipelines*, e os comparamos entre si.

Aprendemos que

* o *transformer* ***Word2Vec*** performa muito bem nessa base: as três melhores combinações usam esse *transformer*;
* especificamente, **o modelo de regressão logística com *transformer* *Word2Vec* performou de forma excelente** no conjunto de teste; e
* o modelo com *Random Forest* e o modelo com *XGBoost* também performam muito bem no conjunto de teste.

Escolhemos essas 3 combinações (regressão logística com *Word2Vec*, *Random Forest* com *Word2Vec* e *XGBoost* com *Word2Vec*) para tentar melhorar através da otimização dos hiperparâmetros de cada um. Surpreendentemente, **o modelo com estimador *XGBoost* e *transformer* *Word2Vec* teve uma melhora significativa na *performance*, ultrapassando o modelo de regressão logística**. Vamos então utilizá-lo, com os parâmetros otimizados, para predizer o sentimento de novos *tweets*.
