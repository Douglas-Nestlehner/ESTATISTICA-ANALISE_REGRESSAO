import random
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from random import sample
from scipy.stats import t
from statsmodels.formula.api import ols

np.random.seed(44)
random.seed(44)

# Numero de observacoes
n = 1000000

# Gerando 199 covariaveis oriundas de distribuicoes Normal, Uniforme, Exponencial, Beta e Gamma.
dset = {}
for i in range(1, 40):  # Distribuicao Normal
    chave = "x" + str(i)
    valor = np.random.normal(0, random.random(), n)
    dset[chave] = valor

for i in range(40, 80):  # Distribuicao Uniforme
    chave = "x" + str(i)
    valor = np.random.rand(n)
    dset[chave] = valor

for i in range(80, 120):  # Distribuicao Exponencial
    chave = "x" + str(i)
    valor = np.random.exponential(random.randrange(1, 11), n)
    dset[chave] = valor

for i in range(120, 160):  # Distribuicao Beta
    chave = "x" + str(i)
    valor = np.random.beta(random.randrange(1, 5), random.randrange(1, 5), n)
    dset[chave] = valor

for i in range(160, 200):  # Distribuicao Gamma
    chave = "x" + str(i)
    valor = np.random.gamma(random.randrange(1, 5), random.randrange(1, 5), n)
    dset[chave] = valor

# Base de dados das covariaveis
df = pd.DataFrame(dset)
print(df)

X = df.iloc[:, list(range(199))].values
type(X)

# Gerando betas por meio de uma Normal(0,2)
betas = []
for j in range(0, 199):
    b = np.random.normal(0, 2)
    betas.append(b)
print(betas)

soma = 0
for k in range(0, 199):
    soma = soma + betas[k] * X[:, k]

# Gerando valores para os erros E_i
E_i = np.random.normal(0, 1, n)

# Calculando os valores de Y_i
y = 10 + soma + E_i
print(y)

# Regressao Linear Multipla
reg = LinearRegression()
reg.fit(X, y)

# Coeficientes betachapeu_1,...,betachapeu_199
print(reg.coef_)

# Intercepto betachapeu_0
print(reg.intercept_)

# Calculo dos valores ajustados
yi_ajustado = 0
for g in range(0, 199):
    yi_ajustado = yi_ajustado + reg.coef_[g] * X[:, g]

# Calculo dos residuos
residuos = y - reg.intercept_ - yi_ajustado
print(residuos)

#### ATIVIDADE 2 ####

## IC para beta0 e beta1 considerando toda a amostra ##

# Escolha da covariavel x3
X3 = np.array(X[:, 2]).reshape((-1, 1))
print(X3)

# Calculo dos valores de Y_i
y_novo = 10 + betas[1] * X[:, 2] + np.random.normal(0, 1, n)
y_novo = np.array(y_novo)
print(y_novo)

# Regressao Linear Simples
model = LinearRegression()
model.fit(X3, y_novo)
print(model.coef_[0])  # valor estimado de beta1
print(model.intercept_)  # valor estimado de beta0

# Calculo dos valores ajustados de Y_i
yi_chapeu = model.intercept_ + model.coef_ * X3[:, 0]
print(yi_chapeu)

# Calculo dos residuos
residuos = y_novo - yi_chapeu
print(residuos)

# Calculo do SSE e do MSE
graus_liberdade = n - 2
SSE = sum((residuos) ** 2)
print(SSE)
MSE = SSE / graus_liberdade
print(MSE)

x_media = sum(X3) / n
print(x_media)

# Calculo dos estimadores da variancia de b0 e b1
est_varb1 = MSE / sum((X3 - x_media) ** 2)
print(est_varb1)
est_varb0 = MSE * (1 / n + (x_media**2 / sum((X3 - x_media) ** 2)))
print(est_varb0)

t_student = t.ppf(1 - 0.05 / 2, graus_liberdade)

# Intervalo de Confianca para beta1
lim_inf_beta1 = model.coef_ - t_student * math.sqrt(est_varb1)
lim_sup_beta1 = model.coef_ + t_student * math.sqrt(est_varb1)
print(lim_inf_beta1[0], lim_sup_beta1[0])


# Intervalo de Confianca para beta0
lim_inf_beta0 = model.intercept_ - t_student * math.sqrt(est_varb0)
lim_sup_beta0 = model.intercept_ + t_student * math.sqrt(est_varb0)
print(lim_inf_beta0, lim_sup_beta0)

## IC para beta0 e beta1 para uma amostra de tamanho 4800 ##

# Amostra aleatoria de tamanho 4800 retirada de x3
amostra = df["x3"].sample(n=4800, replace=False)
amostra = np.array(amostra).reshape(-1, 1)
print(amostra)

# Calculo dos novos valores de Y_i
y_novo_amostra = 10 + betas[1] * amostra[:, 0] + np.random.normal(0, 1, 4800)
print(y_novo_amostra)

# Regressao Linear Simples
modelo2 = LinearRegression()
modelo2.fit(amostra, y_novo_amostra)
print(modelo2.coef_[0])  # valor estimado de beta1
print(modelo2.intercept_)  # valor estimado de beta0

# Calculo dos valores ajustados de Y_i
yi_chapeu2 = modelo2.intercept_ + modelo2.coef_ * amostra[:, 0]
print(yi_chapeu2)

# Calculo dos residuos
residuos2 = y_novo_amostra - yi_chapeu2
print(residuos2)

# Calculo do SSE e do MSE
graus_liberdade2 = 4800 - 2
SSE_2 = sum((residuos2) ** 2)
print(SSE_2)
MSE_2 = SSE_2 / graus_liberdade2
print(MSE_2)

x_media2 = sum(amostra) / 4800
print(x_media2)

# Calculo dos estimadores da variancia de b0 e b1
est_varb1_2 = MSE_2 / sum((amostra - x_media2) ** 2)
print(est_varb1_2)
est_varb0_2 = MSE_2 * (1 / 4800 + (x_media2**2 / sum((amostra - x_media2) ** 2)))
print(est_varb0_2)

t_student2 = t.ppf(1 - 0.05 / 2, 4800)

# Intervalo de Confianca para beta1
lim_inf_beta1_2 = modelo2.coef_ - t_student2 * math.sqrt(est_varb1_2)
lim_sup_beta1_2 = modelo2.coef_ + t_student2 * math.sqrt(est_varb1_2)
print(lim_inf_beta1_2[0], lim_sup_beta1_2[0])

# Intervalo de Confianca para beta0
lim_inf_beta0_2 = modelo2.intercept_ - t_student2 * math.sqrt(est_varb0_2)
lim_sup_beta0_2 = modelo2.intercept_ + t_student2 * math.sqrt(est_varb0_2)
print(lim_inf_beta0_2, lim_sup_beta0_2)

## Tabela ANOVA ##

base_amostra = np.hstack((np.array(amostra), np.array(y_novo_amostra).reshape(-1, 1)))
print(base_amostra)
df2 = pd.DataFrame(base_amostra, columns=["x", "y"])
print(df2)

mod = ols("y ~ x", data=df2).fit()
print(mod.summary())
aov_table = sm.stats.anova_lm(mod, typ=1)
print(aov_table)

## IC para E[Y_0] ##

# Calculo do valor da media de x
print(x_media2[0])

# Escolha de um x proximo da media e um longe
x0_proximo_media = -0.0149
x0_longe_media = 1.5555

# Calculo dos valores ajustados de Y_i proximo e longe
y0_proximo_media = modelo2.intercept_ + modelo2.coef_ * x0_proximo_media
print(y0_proximo_media[0])
y0_longe_media = modelo2.intercept_ + modelo2.coef_ * x0_longe_media
print(y0_longe_media[0])

# Calculo dos estimadores da variancia de Y_0chapeu proximo e longe
est_var_y0_proximo = MSE_2 * (
    1 / 4800 + (x0_proximo_media - x_media2) ** 2 / sum((amostra - x_media2) ** 2)
)
print(est_var_y0_proximo[0])
est_var_y0_longe = MSE_2 * (
    1 / 4800 + (x0_longe_media - x_media2) ** 2 / sum((amostra - x_media2) ** 2)
)
print(est_var_y0_longe[0])

# IC para E[Y_0 proximo]
lim_inf_y0_proximo = y0_proximo_media - t_student2 * math.sqrt(est_var_y0_proximo)
lim_sup_y0_proximo = y0_proximo_media + t_student2 * math.sqrt(est_var_y0_proximo)
print(lim_inf_y0_proximo[0], lim_sup_y0_proximo[0])

# IC para E[Y_0 longe]
lim_inf_y0_longe = y0_longe_media - t_student2 * math.sqrt(est_var_y0_longe)
lim_sup_y0_longe = y0_longe_media + t_student2 * math.sqrt(est_var_y0_longe)
print(lim_inf_y0_longe[0], lim_sup_y0_longe[0])

# Calculo de E[Y_0 proximo]
esperanca_y0_proximo = 10 + betas[1] * x0_proximo_media
print(esperanca_y0_proximo)

# Calculo de E[Y_0 longe]
esperanca_y0_longe = 10 + betas[1] * x0_longe_media
print(esperanca_y0_longe)


## IP para uma nova resposta ##

# Gerando uma nova observacao da covariavel x3
xnovogerado = np.random.normal(0, 1)
print(xnovogerado)

# Calculo de Ynovo
ynovo = 10 + betas[1] * xnovogerado + np.random.normal(0, 1)
print(ynovo)

# Calculo de Ynovo_chapeu
ychapeu_novo_gerado = model.intercept_ + model.coef_ * xnovogerado
print(ychapeu_novo_gerado[0])

# IP para uma nova resposta
lim_inf_ychapeu_novo_g = ychapeu_novo_gerado - t_student * math.sqrt(
    MSE * (1 + 1 / n + (xnovogerado - x_media) ** 2 / sum((X3 - x_media) ** 2))
)
lim_sup_ychapeu_novo_g = ychapeu_novo_gerado + t_student * math.sqrt(
    MSE * (1 + 1 / n + (xnovogerado - x_media) ** 2 / sum((X3 - x_media) ** 2))
)
print(lim_inf_ychapeu_novo_g[0], lim_sup_ychapeu_novo_g[0])

## IP para a media de m novas respostas ##

# Escolha de um m
m = 150

y_m = []
for w in range(0, m):
    y_m.append(10 + betas[1] * xnovogerado + np.random.normal(0, 1))

# Calculo da media das m respostas
media_y_m = sum(y_m) / m
print(media_y_m)

# IP para a media de m novas respostas
lim_inf_ychapeu_novo_m = ychapeu_novo_gerado - t_student * math.sqrt(
    MSE * (1 / m + 1 / n + (xnovogerado - x_media) ** 2 / sum((X3 - x_media) ** 2))
)
lim_sup_ychapeu_novo_m = ychapeu_novo_gerado + t_student * math.sqrt(
    MSE * (1 / m + 1 / n + (xnovogerado - x_media) ** 2 / sum((X3 - x_media) ** 2))
)
print(lim_inf_ychapeu_novo_m[0], lim_sup_ychapeu_novo_m[0])

## Teste Linear Geral ##

np.random.seed(91)
random.seed(91)

# Escolhendo 3 betas proximos de 0
b = sorted(betas)
ordenado = [i for i in b if i > 0]
beta1 = ordenado[0]
beta2 = ordenado[1]
beta3 = ordenado[2]

# Pegando uma amostra aleatoria de 800 observacoes de 8 covariaveis escolhidas, sendo x79, x127 e x194 as que acompanham os betas proximos de 0
x79 = np.array(df["x79"].sample(n=800, replace=False)).reshape(-1, 1)
x127 = np.array(df["x127"].sample(n=800, replace=False)).reshape(-1, 1)
x194 = np.array(df["x194"].sample(n=800, replace=False)).reshape(-1, 1)
x60 = np.array(df["x60"].sample(n=800, replace=False)).reshape(-1, 1)
x120 = np.array(df["x120"].sample(n=800, replace=False)).reshape(-1, 1)
x145 = np.array(df["x145"].sample(n=800, replace=False)).reshape(-1, 1)
x160 = np.array(df["x160"].sample(n=800, replace=False)).reshape(-1, 1)
x175 = np.array(df["x175"].sample(n=800, replace=False)).reshape(-1, 1)

### MODELO COMPLETO

# Juntando os dados dessas covariaveis
amostra_comp = np.hstack((x79, x127, x194, x60, x120, x145, x160, x175))
amostra_comp = pd.DataFrame(amostra_comp)
amostra_comp = amostra_comp.iloc[:, list(range(8))].values
print(amostra_comp)

# Calculo dos Y_i
soma2 = 0
betas_comp = [
    beta1,
    beta2,
    beta3,
    betas[60],
    betas[120],
    betas[145],
    betas[160],
    betas[175],
]
for q in range(0, 8):
    soma2 = soma2 + betas_comp[q] * amostra_comp[:, q]

Y_F = 10 + soma2 + np.random.normal(0, 1, 800)
print(Y_F)

# Modelo de Regressao Linear Multipla
modelo_completo = LinearRegression()
modelo_completo.fit(amostra_comp, Y_F)
print(modelo_completo.coef_)  # valor estimado de beta1,...,beta8
print(modelo_completo.intercept_)  # valor estimado de beta0

# Calculo dos valores ajustados
yi_estimado_comp = 0
for gi in range(0, 8):
    yi_estimado_comp = (
        yi_estimado_comp + modelo_completo.coef_[gi] * amostra_comp[:, gi]
    )
print(yi_estimado_comp)

# Calculo dos residuos
residuos_comp = Y_F - modelo_completo.intercept_ - yi_estimado_comp
print(residuos_comp)

# Calculo do SSE_F
SSE_F = sum((residuos_comp) ** 2)
print(SSE_F)

### MODELO REDUZIDO

# Juntando os dados das 5 covariaveis que nao acompanham os 3 betas proximos de 0
amostra_red = np.hstack((x60, x120, x145, x160, x175))
amostra_red = pd.DataFrame(amostra_red)
amostra_red = amostra_red.iloc[:, list(range(5))].values
print(amostra_red)

# Calculo dos Y_i
soma3 = 0
betas_red = [betas[60], betas[120], betas[145], betas[160], betas[175]]
for qi in range(0, 5):
    soma3 = soma3 + betas_red[qi] * amostra_red[:, qi]

Y_R = 10 + soma3 + np.random.normal(0, 1, 800)
print(Y_R)

# Modelo de Regressao Linear Multipla
modelo_reduzido = LinearRegression()
modelo_reduzido.fit(amostra_red, Y_R)
print(modelo_reduzido.coef_)  # valor estimado de beta4,...,beta8
print(modelo_reduzido.intercept_)  # valor estimado de beta0

# Calculo dos valores ajustados
yi_estimado_red = 0
for gj in range(0, 5):
    yi_estimado_red = yi_estimado_red + modelo_reduzido.coef_[gj] * amostra_red[:, gj]
print(yi_estimado_red)

# Calculo dos residuos
residuos_red = Y_R - modelo_reduzido.intercept_ - yi_estimado_red
print(residuos_red)

# Calculo do SSE_R
SSE_R = sum((residuos_red) ** 2)
print(SSE_R)

# Graus de liberdade do modelo completo e do reduzido, respectivamente
glF = 800 - 9
glR = 800 - 6

# Calculo de SSE_R - SSE_F
diferenca = SSE_R - SSE_F
print(diferenca)

# Calculo do valor observado da estatistica teste (F_0)
estat_teste = ((diferenca) / (glR - glF)) / (SSE_F / glF)
print(estat_teste)

# Calculo da F
F = scipy.stats.f.ppf(0.95, glR - glF, glF, scale=1)
print(F)
