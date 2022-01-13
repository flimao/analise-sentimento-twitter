# Análise de Sentimentos - Base Twitter

<sub>Projeto para a disciplina de **Machine Learning** (Módulo 5) do Data Science Degree (turma de julho de 2020)</sub>

## Case

Vamos desenvolver um modelo para detectar o sentimento de uma publicação do Twitter a classificando em uma das três categorias: **positiva**, **negativa** ou **neutra**. Vamos testar pelo menos 2 técnicas de NLP diferentes e avaliar a performance do modelo segundo algumas métricas pertinentes.  

## Plano de Trabalho

Vamos abordar o problema em fases, a saber:

* **Análise de consistência dos dados**: analisaremos se os dados estão fazendo sentido, se os campos estão completos e se há dados duplicados ou faltantes;

* **Análise exploratória**: analisaremos a base de treino como um todo, verificando o balanceamento entre as classes e focando, principalmente, na coluna *`tweet_text`*;

* **Pré-processamento e transformações**: projetos de NLP exigem um considerável pré-processamento. Por esse motivo, focaremos no tratamento da *string* do texto. Vamos começar com tratamentos simples e adicionaremos complexidade gradualmente. Nessa etapa testaremos diferentes técnicas de transformações, como o ***Bag Of Words*** e o ***TF-IDF***, bem como o ***Word2Vec*** e o ***Doc2Vec***;

* **Treinamento do modelo**: depois das transformações, treinaremos os modelos classificadores candidatos. Nessa etapa o problema se torna semelhante aos abordados na primeira parte do módulo. Testaremos diversos classificadores como *RandomForest*, *AdaBoost*, entre outros. Otimizaremos também os hiperparâmetros do modelo com técnicas como a *GridSearch* e a *RandomizedSearch*;

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

Trabalho em andamento
