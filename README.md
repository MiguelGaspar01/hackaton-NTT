## Ordem de execução

Para garantir o correto fluxo de trabalho, os ficheiros devem ser executados pela seguinte ordem:

- **EDA**  
  - Análises para perceber o formato dos dados  
  - Distribuições e possíveis problemas  

- **Preprocessing**  
  - Tratar os dados para ajudar os modelos a treinar melhor  
  - Transformações, missing values e outliers  

- **Feature Selection**  
  - Escolher features relevantes  
  - Como são poucas, acabámos por utilizar todas  

- **Modeling**  
  - Invocar os modelos  
  - Executar Cross-Validation  
  - Ajustar hiperparâmetros  

- **Evaluation**  
  - Avaliação final no test set  
  - Final predictions  
  - Análise de métricas e interpretação  

---

## Próximos passos

- Os modelos estão a ter dificuldade na classificação de um valor do target  
- Este comportamento poderá estar relacionado com **class imbalance**  
- Devemos focar-nos em melhorias no **preprocessing**  
- Testar transformações adicionais das variáveis  
- Avaliar novas abordagens para melhorar a separação entre classes  

---

## Nota

- Foram adicionados alguns ficheiros ao `.gitignore`  
- Os ficheiros de dados deixam de ser adicionados ao repositório  
- Esta decisão foi tomada porque o repositório estava a exceder o limite de espaço online



## Morning Routine Mix

https://youtu.be/8B4soa7z098 
