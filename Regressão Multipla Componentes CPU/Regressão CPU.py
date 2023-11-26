import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from yellowbrick.regressor import ResidualsPlot
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from numpy.random import seed
from numpy.random import randn
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import math
import scipy.stats as stats

dados = pd.read_excel("")
dados

dados.describe()


Y = dados["Y"]
Y = Y.values

covariaveis_df = dados.drop(columns=["Y"])
covariaveis = covariaveis_df.iloc[:, list(range(6))].values
# print(Y)
# print(covariaveis)

# Ajustar o modelo
# modelo = LinearRegression()
# modelo.fit(covariaveis,Y)
# print(modelo.coef_)
# print(modelo.intercept_)

# ou
modelo = ols(formula="Y~x1+x2+x3+x4+x5+x6", data=dados).fit()
modelo_intercept = modelo.params["Intercept"]
modelo_coefs = [
    modelo.params["x1"],
    modelo.params["x2"],
    modelo.params["x3"],
    modelo.params["x4"],
    modelo.params["x5"],
    modelo.params["x6"],
]
print(modelo_intercept)
print(modelo_coefs)
# print(modelo.summary())

# Analise de residuos
Y_ajustado = modelo.predict()
residuos = Y - Y_ajustado
print(residuos)
print(Y_ajustado)


# Gráfico Resíduos vs Y_i
plt.plot(Y_ajustado, residuos, "bo")
plt.xlabel("Y ajustado")
plt.ylabel("Resíduos")
plt.show()
# Observa-se uma curva

# Verificar normalidade
sns.displot(residuos)
plt.show()
qqplot(residuos, line="s")
pyplot.show()

# Tabela anova
aov = sm.stats.anova_lm(modelo, typ=1)
print(aov)

# Teste homocedasticidade
res1 = np.sort(residuos)[0:104]
res2 = np.sort(residuos)[104:209]
stats.bartlett(res1, res2)


# Como as suposiço~es não foram satisfeitas, fazemos transformação de Y ou X ou YeX
Y_transf = []
for j in range(0, 209):
    Y_transf.append(math.log(Y[j] + 0.001))

Y_transf

modelo2 = ols(formula="Y_transf~x1+x2+x3+x4+x5+x6", data=dados).fit()
modelo2_intercept = modelo2.params["Intercept"]
modelo2_coefs = [
    modelo2.params["x1"],
    modelo2.params["x2"],
    modelo2.params["x3"],
    modelo2.params["x4"],
    modelo2.params["x5"],
    modelo2.params["x6"],
]
print(modelo2_intercept)
print(modelo2_coefs)
# print(modelo2.summary())

Y_ajustado2 = modelo2.predict()
residuos2 = Y_transf - Y_ajustado2

# Gráfico Resíduos vs Y_i
plt.plot(Y_ajustado2, residuos2, "bo")
plt.plot([min(Y_ajustado2), max(Y_ajustado2)], [0, 0], color="red")
plt.xlabel("Y' ajustado")
plt.ylabel("Resíduos")
plt.show()

qqplot(residuos2, line="s")
pyplot.show()

# TEstes de normalidade e homocedasticidade
# Bartlet (Se p-valor maior que 0.05 deu bom)
import scipy.stats as stats

res1_ = np.sort(residuos2)[0:104]
res2_ = np.sort(residuos2)[104:209]
stats.bartlett(res1_, res2_)
# Stats.leven (homocedasticidade)
stats.levene(res1_, res2_)

# Normalidade Shapiro (Se p-valor > 0.05 deu bom)
from scipy.stats import shapiro

shapiro(residuos2)[1]
from scipy.stats import anderson

anderson(residuos2)

# Identificar o que devemos tranformar
"""
plt.scatter(covariaveis_df["x1"],Y_transf)
plt.xlabel("x1")
plt.ylabel("Y'")
plt.show()

plt.scatter(covariaveis_df["x2"],Y_transf)
plt.xlabel("x2")
plt.ylabel("Y'")
plt.show()

plt.scatter(covariaveis_df["x3"],Y_transf)
plt.xlabel("x3")
plt.ylabel("Y'")
plt.show()

plt.scatter(covariaveis_df["x4"],Y_transf)
plt.xlabel("x4")
plt.ylabel("Y'")
plt.show()

plt.scatter(covariaveis_df["x5"],Y_transf)
plt.xlabel("x5")
plt.ylabel("Y'")
plt.show()

plt.scatter(covariaveis_df["x6"],Y_transf)
plt.xlabel("x6")
plt.ylabel("Y'")
plt.show()
"""

# Transformando as covariaveis
x1 = list(covariaveis_df["x1"])
x1_transf = []
for k in range(0, 209):
    x1_transf.append(1 / x1[k])

x2 = list(covariaveis_df["x2"])
x2_transf = []
for k in range(0, 209):
    x2_transf.append(math.sqrt(x2[k]))

x3 = list(covariaveis_df["x3"])
x3_transf = []
for k in range(0, 209):
    x3_transf.append(math.sqrt(x3[k]))

x4 = list(covariaveis_df["x4"])
x4_transf = []
for k in range(0, 209):
    x4_transf.append(math.sqrt(x4[k]))

x5 = list(covariaveis_df["x5"])
x5_transf = []
for k in range(0, 209):
    x5_transf.append(math.sqrt(x5[k]))

x6 = list(covariaveis_df["x6"])
x6_transf = []
for k in range(0, 209):
    x6_transf.append(math.sqrt(x6[k]))

x1_transf = np.array(x1_transf).reshape(-1, 1)
x2_transf = np.array(x2_transf).reshape(-1, 1)
x3_transf = np.array(x3_transf).reshape(-1, 1)
x4_transf = np.array(x4_transf).reshape(-1, 1)
x5_transf = np.array(x5_transf).reshape(-1, 1)
x6_transf = np.array(x6_transf).reshape(-1, 1)

covariaveis_transf = np.hstack(
    (x1_transf, x2_transf, x3_transf, x4_transf, x5_transf, x6_transf)
)
covariaveis_transf_df = pd.DataFrame(
    covariaveis_transf,
    columns=[
        "x1_transf",
        "x2_transf",
        "x3_transf",
        "x4_transf",
        "x5_transf",
        "x6_transf",
    ],
)
covariaveis_transf = covariaveis_transf_df.iloc[:, list(range(6))].values

covariaveis_transf_df

# Ajusta o modelo e faz a analise de resiudos dnv
modelo3 = ols(
    formula="Y_transf~x1_transf+x2_transf+x3_transf+x4_transf+x5_transf+x6_transf",
    data=covariaveis_transf_df,
).fit()
modelo3_intercept = modelo3.params["Intercept"]
modelo3_coefs = [
    modelo3.params["x1_transf"],
    modelo3.params["x2_transf"],
    modelo3.params["x3_transf"],
    modelo3.params["x4_transf"],
    modelo3.params["x5_transf"],
    modelo3.params["x6_transf"],
]
print(modelo3_intercept)
print(modelo3_coefs)
# print(modelo3.summary())

Y_ajustado3 = modelo3.predict()
residuos3 = Y_transf - Y_ajustado3

# Gráfico Resíduos vs Y_i
plt.plot(Y_ajustado3, residuos3, "bo")
plt.plot([min(Y_ajustado3), max(Y_ajustado3)], [0, 0], color="red")
plt.xlabel("Y' ajustado")
plt.ylabel("Resíduos")

qqplot(residuos3, line="s")
pyplot.show()

# Faz todos os testes denovo
import scipy.stats as stats

res1 = np.sort(residuos3)[0:104]
res2 = np.sort(residuos3)[104:209]
stats.bartlett(res1, res2)

stats.levene(res1, res2)

from scipy.stats import shapiro

shapiro(residuos3)[1]

from scipy.stats import anderson

anderson(residuos3)

# Depois de verificar todas as suposiçõs, devemos observar se todas as variaveis são signifacativas para o modelo
print(modelo3.summary())
# Se o p-valor (P>|t|) for grande conclui-se que a covariavel não é significativa. e retira ela do modelo

# Então ajusta-se o modelo novamente sem a covarariavel que não é siginificativa
modelo_final = ols(
    formula="Y_transf~x1_transf+x2_transf+x3_transf+x4_transf+x6_transf",
    data=covariaveis_transf_df,
).fit()
modelo_final_intercept = modelo_final.params["Intercept"]
modelo_final_coefs = [
    modelo_final.params["x1_transf"],
    modelo_final.params["x2_transf"],
    modelo_final.params["x3_transf"],
    modelo_final.params["x4_transf"],
    modelo_final.params["x6_transf"],
]
print(modelo_final_intercept)
print(modelo_final_coefs)
# print(modelo_final.summary())

# Pode fazer a analise de resiudos novamente, mais vai ta tudo certo.
