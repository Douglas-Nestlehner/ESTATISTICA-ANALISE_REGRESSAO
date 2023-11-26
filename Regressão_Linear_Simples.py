# Bibliotecas utilizadas
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from random import sample
import scipy.stats
from scipy.stats import t
import math
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Dados csv
# df = pd.read_csv("doc.csv")
# df = pd.read_excel("doc.xlsx")

# Dados digitados
Gasto_publicidade = [0, 2, 3, 4, 4.5, 5, 6 ]
Volumes_vendas = [89, 110, 110.6, 111.2, 112.7, 113, 112.2]

df = pd.DataFrame(list(zip(Gasto_publicidade,Volumes_vendas)), columns=["x", "Y"])                  
df

# csv apartir daqui
X = df.iloc[:, 0].values
y = df.iloc[:, 1].values


# Calculo da correlação entre X e y (Matriz var-cov)
correlacao = np.corrcoef(X,y)
correlacao

X = X.reshape(-1,1)
X

# Contar o numero de linhas do data frame
n  = df.shape[0]

# Contar o numero de colunas do data frame
m  = df.shape[1]

# fit para fazer o treinamento (Ajustar o modelo)
regressor = LinearRegression()
regressor.fit(X,y)

# Valor de b1 
regressor.coef_

# Valor de b0
regressor.intercept_

# plotar o grafico de regressão linear simples
plt.scatter(X,y)
plt.title("Regressão Linear Simples")
plt.xlabel("Gasto com Publicidade (X)")
plt.ylabel("Volume de Vendas (y)")


# Adicionar a linha da regressão
plt.plot(X, regressor.predict(X), color = "red")
plt.show()


# Criar uma previsão
# Exemplo queremos prever o volume de venda (y) se o investimento for 4 (X)
previsao1 = regressor.predict(np.array([4]).reshape(1, 1))
previsao1


# Medir o quão bom esta o modelo (quanto mais proximo de 1 melhor)
score = regressor.score(X, y)
score


### INFORMAÇÔES ADICIONAIS SOBRE RESIDUOS
from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor)
visualizador.fit(X, y)
visualizador.poof() # mostra a distancia dos pontos da reta


####################################
#####  Intervalo de confiança  #####
####################################

# Calculo dos valores ajustados de Y_i (chapeu)
yi_chapeu = regressor.intercept_ + regressor.coef_*X[:,0]
print(yi_chapeu)

# Calculo dos residuos
residuos = y - yi_chapeu
print(residuos)

# Calculo do SSE e do MSE 
n = 7 # n = numero de 0bservações
graus_liberdade= n-2
SSE= sum((residuos)**2)
print(SSE)
MSE= SSE/graus_liberdade
print(MSE)

x_media = sum(X)/n
print(x_media)

# Calculo dos estimadores da variancia de b0 e b1
est_varb1 = MSE/sum((X-x_media)**2)
print(est_varb1)
est_varb0 = MSE*(1/n + (x_media**2/sum((X-x_media)**2)))
print(est_varb0)

# confiança
t_student = t.ppf(1-0.05/2, graus_liberdade)

# Intervalo de Confianca para beta1
lim_inf_beta1 = regressor.coef_ - t_student*math.sqrt(est_varb1)
lim_sup_beta1 = regressor.coef_ + t_student*math.sqrt(est_varb1)
print(lim_inf_beta1[0],lim_sup_beta1[0])


# Intervalo de Confianca para beta0
lim_inf_beta0 = regressor.intercept_ - t_student*math.sqrt(est_varb0)
lim_sup_beta0 = regressor.intercept_ + t_student*math.sqrt(est_varb0)
print(lim_inf_beta0,lim_sup_beta0)


################################################################
#### IC para beta0 e beta1 para uma amostra de tamanho 4800 ####
################################################################

# Amostra aleatoria de tamanho 4800 retirada de x3
amostra = df.sample(n=3,replace=False)
print(amostra)

XA = amostra.iloc[:, 0].values
yA = amostra.iloc[:, 1].values

correlacao_amostra = np.corrcoef(XA,yA)
correlacao_amostra

XA = XA.reshape(-1,1)
XA

regressor_amostra = LinearRegression()
regressor.fit(XA,yA)

regressor_amostra.coef_
regressor_amostra.intercept_


# Calculo dos novos valores de Y_i
y_novo_amostra = regressor.intercept_ + regressor.coef_*X[:,0]
print(y_novo_amostra)

# Regressao Linear Simples
modelo2 = LinearRegression()
modelo2.fit(amostra,y_novo_amostra)
print(modelo2.coef_[0]) # valor estimado de beta1
print(modelo2.intercept_) # valor estimado de beta0

# Calculo dos valores ajustados de Y_i
#yi_chapeu2 = modelo2.intercept_ + modelo2.coef_*
#print(yi_chapeu2)

# Calculo dos residuos
residuos2 = y_novo_amostra #- yi_chapeu2
print(residuos2)

# Calculo do SSE e do MSE
graus_liberdade2= 4800-2
SSE_2= sum((residuos2)**2)
print(SSE_2)
MSE_2= SSE_2/graus_liberdade2
print(MSE_2)

x_media2 = sum(amostra)/4800
print(x_media2)

# Calculo dos estimadores da variancia de b0 e b1
est_varb1_2 = MSE_2/sum((amostra-x_media2)**2)
print(est_varb1_2)
est_varb0_2 = MSE_2*(1/4800 + (x_media2**2/sum((amostra-x_media2)**2)))
print(est_varb0_2)

t_student2 = t.ppf(1-0.05/2, 4800)

# Intervalo de Confianca para beta1
lim_inf_beta1_2 = modelo2.coef_ - t_student2*math.sqrt(est_varb1_2)
lim_sup_beta1_2 = modelo2.coef_ + t_student2*math.sqrt(est_varb1_2)
print(lim_inf_beta1_2[0],lim_sup_beta1_2[0])

# Intervalo de Confianca para beta0
lim_inf_beta0_2 = modelo2.intercept_ - t_student2*math.sqrt(est_varb0_2)
lim_sup_beta0_2 = modelo2.intercept_ + t_student2*math.sqrt(est_varb0_2)
print(lim_inf_beta0_2,lim_sup_beta0_2)



#################################################################                      
########    Tabela ANOVA e teste de hipotese    #################
#################################################################
base_amostra = np.hstack((np.array(amostra),np.array(y_novo_amostra).reshape(-1,1)))
print(base_amostra)
df2 = pd.DataFrame(base_amostra,columns=['x', 'y'])
print(df2)

mod = ols('y ~ x', data=df2).fit()
print(mod.summary())
aov_table = sm.stats.anova_lm(mod, typ=1)
print(aov_table)


##################################
######   IC para E[Y_0]   ########
##################################
# Calculo do valor da media de x
print(x_media2[0])

# Escolha de um x proximo da média e um longe
x0_proximo_media = -0.0149
x0_longe_media = 1.5555

# Calculo dos valores ajustados de Y_i proximo e longe
y0_proximo_media = modelo2.intercept_ + modelo2.coef_*x0_proximo_media
print(y0_proximo_media[0])
y0_longe_media = modelo2.intercept_ + modelo2.coef_*x0_longe_media
print(y0_longe_media[0])

# Calculo dos estimadores da variancia de Y_0chapeu proximo e longe
est_var_y0_proximo = MSE_2*(1/4800 + (x0_proximo_media - x_media2)**2/sum((amostra-x_media2)**2))
print(est_var_y0_proximo[0])
est_var_y0_longe = MSE_2*(1/4800 + (x0_longe_media - x_media2)**2/sum((amostra-x_media2)**2))
print(est_var_y0_longe[0])

# IC para E[Y_0 proximo] 
lim_inf_y0_proximo = y0_proximo_media - t_student2*math.sqrt(est_var_y0_proximo)
lim_sup_y0_proximo = y0_proximo_media + t_student2*math.sqrt(est_var_y0_proximo)
print(lim_inf_y0_proximo[0],lim_sup_y0_proximo[0])

# IC para E[Y_0 longe] 
lim_inf_y0_longe = y0_longe_media - t_student2*math.sqrt(est_var_y0_longe)
lim_sup_y0_longe = y0_longe_media + t_student2*math.sqrt(est_var_y0_longe)
print(lim_inf_y0_longe[0],lim_sup_y0_longe[0])

# Calculo de E[Y_0 proximo]
#esperanca_y0_proximo = 10 + betas[1]*x0_proximo_media
#print(esperanca_y0_proximo)

# Calculo de E[Y_0 longe]
#esperanca_y0_longe = 10 + betas[1]*x0_longe_media
#print(esperanca_y0_longe)



###############################################################
######   Intervalo de Predição para uma nova resposta   #######
###############################################################

# Gerando uma nova observacao da covariavel x3
xnovogerado = np.random.normal(0,1)
print(xnovogerado)

# Calculo de Ynovo
ynovo = 10 # + betas[1]*xnovogerado + np.random.normal(0,1)
print(ynovo)

# Calculo de Ynovo_chapeu
#ychapeu_novo_gerado = model.intercept_+ model.coef_*xnovogerado
#print(ychapeu_novo_gerado[0])

# IP para uma nova resposta
#lim_inf_ychapeu_novo_g = ychapeu_novo_gerado - t_student*math.sqrt(MSE*(1+1/n + (xnovogerado-x_media)**2/sum((X3-x_media)**2)))
#lim_sup_ychapeu_novo_g = ychapeu_novo_gerado + t_student*math.sqrt(MSE*(1+1/n + (xnovogerado-x_media)**2/sum((X3-x_media)**2)))
#print(lim_inf_ychapeu_novo_g[0],lim_sup_ychapeu_novo_g[0])

            
##########################################################################            
######   Intervalo de Predicao para a media de m novas respostas   #######
##########################################################################

# Escolha de um m
m = 150

#y_m = []
#for w in range(0,m):
#    y_m.append(10 + betas[1]*xnovogerado + np.random.normal(0,1))

# Calculo da media das m respostas
#media_y_m = sum(y_m)/m
#print(media_y_m)

# IP para a media de m novas respostas
#lim_inf_ychapeu_novo_m = ychapeu_novo_gerado  - t_student*math.sqrt(MSE*(1/m+1/n + (xnovogerado-x_media)**2/sum((X3-x_media)**2)))
#lim_sup_ychapeu_novo_m = ychapeu_novo_gerado  + t_student*math.sqrt(MSE*(1/m+1/n + (xnovogerado-x_media)**2/sum((X3-x_media)**2)))
#print(lim_inf_ychapeu_novo_m[0],lim_sup_ychapeu_novo_m[0])