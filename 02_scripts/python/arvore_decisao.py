# Árvore de Decisão - Desafio Pessoal - Simpsons
# Curso Noções Data Science - Aldeia
  
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
    
treino = pd.read_csv('/home/ds/git/201907/03_dados/desafio pessoal/treino.csv', sep= ';', header = None) 
teste = pd.read_csv('/home/ds/git/201907/03_dados/desafio pessoal/teste.csv', sep= ';', header = None)
    
#Carrega variáveis preditivas e rótulos
X_treino, y_treino = treino.values[:, 0:5], treino.values[:, 6]
X_teste, y_teste = teste.values[:, 0:5], teste.values[:, 6]

#Configura classificador (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
classificador = DecisionTreeClassifier(random_state = 0) 

# Ajusta e gera o modelo treinado com base nas features/características 
classificador.fit(X_treino, y_treino) 

# Realiza a predição com novos valores / teste
y_predito = classificador.predict(X_teste) 

# Calcula acurácia entre valor predito e valor real rotulado no teste
accur = accuracy_score(y_teste,y_predito)*100
print ("Acurácia: %.2f%%" %accur)

# Apresenta a matriz confusão - https://en.wikipedia.org/wiki/Confusion_matrix
#         ROTULO
# PRED    Homer    Bart
# Homer   OK       NOK
# Bart    NOK      OK
#
cm = confusion_matrix(y_teste, y_predito, labels=['homer','bart'])
print ("Matriz Confusão: \n")
print (cm)