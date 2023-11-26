library(GLMsData)
library(dplyr)
library(corrplot)
library(faraway)
library(ggplot2)
library(lmtest)
library(glmtoolbox)
library(gridExtra)
library(hnp)

### Importando a base de dados
data(cyclones)
base_dados <- cyclones
head(base_dados)

### Análise descritiva dos dados
summary(base_dados)

box_resp <- ggplot(data=base_dados) + geom_boxplot(aes(y=Severe), fill = 'turquoise') + 
  labs(y = 'Número de Furacões')

bar_resp <- ggplot(data=base_dados) + geom_bar(aes(x=Severe), fill = 'turquoise') + 
  labs(y = 'Frequência Absoluta', x = 'Número de Furacões')

grid.arrange(box_resp, bar_resp, ncol = 2)

### Modelo Linear Generalizado Poisson

attach(base_dados)

## Função de ligação

# Modelo 1 com função de ligação logarítmica
modelo1 <- glm(Severe~JFM+AMJ+JAS+OND, family = poisson(link = 'log'))
summary(modelo1)

anova(modelo1, test = 'Chisq')

# Modelo 2 com função de ligação logarítmica
modelo2 <- glm(Severe~JFM+AMJ+JAS+OND+(JFM*AMJ)+(JFM*JAS)+(JFM*OND)+(AMJ*JAS)+
                 (AMJ*OND)+(JAS*OND), family = poisson(link = 'log'))
summary(modelo2)

# Modelo 2 com função de ligação logarítmica
modelo3 <- glm(Severe~JFM+AMJ+JAS+OND+(JFM*AMJ)+(JFM*JAS)+(JFM*OND)+(AMJ*JAS)+
                 (AMJ*OND)+(JAS*OND)+(JFM*AMJ*JAS), family = poisson(link = 'log'))
summary(modelo3)


# Modelo 3 com função de ligação logarítmica
modelo3 <- glm(Severe~JFM+AMJ+JAS+OND+(JFM*AMJ*JAS*OND), family = poisson(link = 'log'))
summary(modelo3)

### Retirando a covariável menos significativa
## Retirando AMJ
mod <- glm(Severe~JFM+JAS+OND, family = poisson(link = 'log'))
summary(mod)
anova(mod, test = 'Chisq')

## Retirando JFM
mod <- glm(Severe~JAS+OND, family = poisson(link = 'log'))
summary(mod)

anova(mod, test = 'Chisq')

## Retirando JFM
mod <- glm(Severe~OND, family = poisson(link = 'log'))
summary(mod)

anova(mod, test = 'Chisq')

## Retirando OND
mod <- glm(Severe~1, family = poisson(link = 'log'))
summary(mod)

anova(mod, test = 'Chisq')

## Análise de Diagnóstico
# Envelope 
hnp(modelo1$residuals, sim = 99,resid.type ='deviance',how.many.out=T ,
    conf = 0.95,scale = T, ylab = 'Deviance Residuals', xlab = 'Half-normal Scores')


# Valores preditos, componente de desvio e resíduos de pearson
desvio  <-   rstandard(modelo1,type="deviance")  
pred  <-  modelo1$fitted.values 
pearson  <-  rstandard(modelo1,type="pearson")

# Valores preditos x Resíduos de Pearson
ggplot(mapping = aes(pred, pearson)) +
  geom_point() +
  geom_hline(linetype= "dashed") + 
  labs(x = "Valores ajustados", y = "Resíduos de Pearson")

# Valores preditos x Componentes do Desvio
ggplot(mapping = aes(pred, desvio)) +
  geom_point() +
  geom_hline(linetype= "dashed") + 
  labs(x = "Valores ajustados", y = "Componente do desvio")

# Independência
ggplot() + geom_point(aes(x = Year, y = pearson)) + 
  labs(y = "Resíduos de Person", x = "Ordem de Coleta") + theme_get()

# Pontos de alavanca
X = model.matrix(~JFM+AMJ+JAS+OND)
H = X%*%solve(t(X)%*%X)%*%t(X)
h <- diag(H)
h = lm.influence(mod)$hat
h = hat(X,T)
plot(h, xlab='Índice', ylab= 'Alavanca')
abline(h = 0.3, col = "red")
which(h > 0.3)