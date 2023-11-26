agents = c(rep(1,9), rep(0,9))
concentracao = c(rep(c(5,10,15,20,30,40,60,80,100),2))
resposta = c(118, 58, 42, 35, 27, 25, 21, 19, 18, 69, 35, 26, 21, 18, 16, 13, 12, 12)

dados = data.frame(concentracao, agents, resposta)
X = data.frame(c(rep(1,18)), agents, log(concentracao))

# Ajuste

m1 = glm(resposta ~X$agents + X$log.concentracao.,family = Gamma(link="inverse"))
m1
summary(m1)

m2 = glm(resposta ~X$agents + X$log.concentracao.,family = Gamma(link="identity"))
m2
summary(m2)


m3 = glm(resposta ~X$agents + X$log.concentracao.,family = Gamma(link="log"))
m3
summary(m3)


# Comparacao 

modelo = c("Reciproco", "Identidade", "Log")
AIC = c(m1$aic, m2$aic, m3$aic)
AIC
deviance = c(m1$deviance, m2$deviance, m3$deviance)
deviance

data.frame(modelo, AIC, deviance)