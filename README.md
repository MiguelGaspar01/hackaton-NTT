Para garantir o correto fluxo de trabalho, os ficheiros devem ser executados pela seguinte ordem: 

--EDA -> analises para perceber formato dos dados
--Preprocessing -> tratar dados para ajudar modelos a treinar melhor
--Feature Selection -> escolher features (como são poucos acabámos por usar todos)
--Modeling -> invocar os modelos e CrossValidation
--Evaluation -> Final predictions


## Próximos passos,
Os modelos estão a ter dificuldade na classificação de um valor do target, presumo que tenha a ver com a class imbalance
Temos de ver possiveis melhorias no pre processamento que possam ajudar o modelo

## Nota:
Adicionei alguns ficheiros ao gitignore para que não sejam adicionados ao repositório (os ficheiros de dados) pq estava a exceder o limite de espaço online
