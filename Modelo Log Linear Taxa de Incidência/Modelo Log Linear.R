age = c("5-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "+85")
idad = rep(age, 2)

idades = factor(idad,levels=unique(age),ordered=TRUE)
idades

casos_minneapolis = c(1, 16, 30, 71, 102, 130, 133, 40)
pop_minneapolis = c(172675, 123065, 96216, 92051, 72159, 54722, 32185, 8328)

casos_dallas = c(4, 38, 119, 221, 259, 310, 226, 65)
pop_dallas = c(181343, 146207, 121374, 111353, 83004, 55932, 29007, 7538)

casos = c(casos_minneapolis, casos_dallas)
pop = c(pop_minneapolis, pop_dallas)

cidade = c(rep("Minneapolis",8), rep("Dallas",8))

dados = data.frame(idades, casos, pop, cidade)
dados


# Plotar o grafico 
logg = log(dados$casos/dados$pop)
logg

library(tidyverse)

ggplot(dados, aes(x = dados$idades, y =logg, color = as.factor(dados$cidade))) +
  geom_point() +
  labs(x="Idade", y="Log(contagem/população)", title="Log(contagem/população) X Idade")+
  scale_colour_brewer(palette="Dark2", name="Cidade")+
  theme(plot.title = element_text(hjust = 0.5))


# Ajuste do modelo (Idade Qualitativa)
dados2 = dados
dados2$cidade = as.factor(dados2$cidade)
dados2$idades = as.factor(age)
str(dados2)
dados2
attach(dados2)

modelo1 = glm(casos ~ idades + cidade + offset(log(pop)), data = dados2, family = poisson("log"))
summary(modelo1)


# envelope
library(hnp)
hnp(modelo1)



###### Ajuste do modelo Qauntitativo
dados3 = dados2
dados3$idades = c(15,30,40,50,60,70,80,85)

str(dados3)
attach(dados3)

modelo2 = glm(casos ~ idades + cidade + offset(log(pop)), data = dados3, family = poisson("log"))
summary(modelo2)

hnp(modelo2)

# OR
exp(coef(modelo1))