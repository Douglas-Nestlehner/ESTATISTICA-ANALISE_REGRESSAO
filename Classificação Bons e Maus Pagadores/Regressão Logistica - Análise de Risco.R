library(tidyverse)
library(gridExtra)
library(glmnet)
library(pROC)

setwd("")

# Dados
df = read.table("Dados_trabalho.txt", sep = " ")
df$Y = ifelse(df$V21 == 1, "Bom pagador", "Mau pagador")
df$V21 = NULL
df$V1 = as.factor(df$V1)
df$V2 = as.numeric(df$V2)
df$V3 = as.factor(df$V3)
df$V4 = as.factor(df$V4)
df$V5 = as.numeric(df$V5)
df$V6 = as.factor(df$V6)
df$V7 = as.factor(df$V7)
df$V8 = as.numeric(df$V8)
df$V9 = as.factor(df$V9)
df$V10 = as.factor(df$V10)
df$V11 = as.numeric(df$V11)
df$V12 = as.factor(df$V12)
df$V13 = as.numeric(df$V13)
df$V14 = as.factor(df$V14)
df$V15 = as.factor(df$V15)
df$V16 = as.numeric(df$V16)
df$V17 = as.factor(df$V17)
df$V18 = as.numeric(df$V18)
df$V19 = as.factor(df$V19)
df$V20 = as.factor(df$V20)
df$Y = as.factor(df$Y)

str(df)
summary(df)

# Descritiva dos dados

# prorpoção de classificação
str(df$Y)
summary(df$Y)
ggplot(data = df, aes(x=Y,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  geom_text(aes(label=..count..),
            stat="count", vjust = 1.6, color = "black", size = 5)+
  ggtitle("Ocorrência de Inadimplência") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 


# Influencia das varaiveis
library(tidyverse)

# V1
V1 = ggplot(data = df, aes(x=V1,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Conta corrente") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V2
V2 = ggplot(df, aes(x=V2, color=Y, fill=Y)) +
  geom_histogram(aes(y=..density..),binwidth = 1, boundary = 1, position="identity", alpha=0.5)+
  geom_density(alpha=0.6)+
  scale_color_manual(values=c("darkturquoise", "salmon"))+
  scale_fill_manual(values=c("darkturquoise", "salmon"))+
  labs(title="Duração (Meses)")+
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V3
V3 = ggplot(data = df, aes(x=V3,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Categoria Histórico") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V4 
V4 = ggplot(data = df, aes(x=V4,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Propósito") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

V5 = # V5
  ggplot(df, aes(x=V5, color=Y, fill=Y)) +
  geom_histogram(aes(y=..density..),binwidth = 1000, boundary = 1, position="identity", alpha=0.5)+
  geom_density(alpha=0.6)+
  scale_color_manual(values=c("darkturquoise", "salmon"))+
  scale_fill_manual(values=c("darkturquoise", "salmon"))+
  labs(title="Quantidade Crédito")+
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V6
V6 = ggplot(data = df, aes(x=V6,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Tipo Conta popupança") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V7
V7 = ggplot(data = df, aes(x=V7,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Emprego Atual") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V8
V8 = ggplot(df, aes(x=V8, color=Y, fill=Y)) +
  geom_histogram(aes(y=..density..), binwidth = 1, boundary = 1,position="identity", alpha=0.5)+
  geom_density(alpha=0.6)+
  scale_color_manual(values=c("darkturquoise", "salmon"))+
  scale_fill_manual(values=c("darkturquoise", "salmon"))+
  labs(title="Taxa")+
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V9
V9 = ggplot(data = df, aes(x=V9,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Status pessoal") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 


# V10
V10 = ggplot(data = df, aes(x=V10,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Tipo cliente") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V11
V11 = ggplot(df, aes(x=V11, color=Y, fill=Y)) +
  geom_histogram(aes(y=..density..), binwidth = 1, boundary = 1,position="identity", alpha=0.5)+
  geom_density(alpha=0.6)+
  scale_color_manual(values=c("darkturquoise", "salmon"))+
  scale_fill_manual(values=c("darkturquoise", "salmon"))+
  labs(title="Residencia atual")+
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 


# V12
V12 = ggplot(data = df, aes(x=V12,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Propriedade") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V13
V13 = ggplot(df, aes(x=V13, color=Y, fill=Y)) +
  geom_histogram(aes(y=..density..), binwidth = 1, boundary = 1,position="identity", alpha=0.5)+
  geom_density(alpha=0.6)+
  scale_color_manual(values=c("darkturquoise", "salmon"))+
  scale_fill_manual(values=c("darkturquoise", "salmon"))+
  labs(title="Residencia atual")+
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V14
V14 = ggplot(data = df, aes(x=V14,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Outros parcelamentos") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V15
V15 = ggplot(data = df, aes(x=V15,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Residencia") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V16
V16 = ggplot(df, aes(x=V16, color=Y, fill=Y)) +
  geom_histogram(aes(y=..density..), binwidth = 0.5, boundary = 1,position="identity", alpha=0.5)+
  geom_density(alpha=0.6)+
  scale_color_manual(values=c("darkturquoise", "salmon"))+
  scale_fill_manual(values=c("darkturquoise", "salmon"))+
  labs(title="Número de créditos")+
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V17
V17 = ggplot(data = df, aes(x=V17,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Tipo de Trabalho") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V18
V18 = ggplot(df, aes(x=V18, color=Y, fill=Y)) +
  geom_histogram(aes(y=..density..), binwidth = 0.5, boundary = 1,position="identity", alpha=0.5)+
  geom_density(alpha=0.6)+
  scale_color_manual(values=c("darkturquoise", "salmon"))+
  scale_fill_manual(values=c("darkturquoise", "salmon"))+
  labs(title="Responsaveis manutenção")+
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V19
V19 = ggplot(data = df, aes(x=V19,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Status Telefone") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# V20
V20 = ggplot(data = df, aes(x=V17,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Status trabalho estrangeiro") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

# Y
ggplot(data = df, aes(x=df$Y,fill=Y)) + 
  geom_bar()+
  scale_fill_brewer(palette="Set1")+
  #geom_text(aes(label=..count..))+
  ggtitle("Y") + 
  cowplot::theme_cowplot() +
  theme(plot.title = element_text(hjust = 0.5)) 

grid.arrange(V1,V2,V3,V4)
grid.arrange(V5,V6, V7, V8)
grid.arrange(V9,V10, V11, V12)
grid.arrange(V13,V14, V15, V16)
grid.arrange(V17,V18, V19, V20)

###############################################################################

# Data-spliting
split = sample(c("Treinamento",
                 "Teste"),
               prob = c(0.7,0.3),
               size = nrow(df),
               replace = T)

df_treino = df[split == "Treinamento",]
df_treino

df_teste = df[split == "Teste",]
df_teste



## LOGISTICA COM PENALIZAÇÃO
library(glmnet)

df2 = df
df2$Y = NULL
df2 = data.matrix(df2)
df2

Resultado = as.matrix(df$Y)
Resultado

vc_glm.lasso = cv.glmnet(df2[split == "Treinamento",], # tem q estar como matriz
                         Resultado[split == "Treinamento"],
                         alpha = 1,
                         family = "binomial")

predito_logistica_penalizacao = predict(vc_glm.lasso, s = vc_glm.lasso$lambda.min, # Penalização s é a que minimiza o erro preditivo 
                                        newx = df2[split == "Teste",],
                                        type = "response")

predito_logistica_penalizacao
classe_log_penalizacao = ifelse(predito_logistica_penalizacao<0.5, "Bom pagador", "Mau pagador") # Classificador plug-in

riscos = list()
riscos$logistica_penalizacao = mean(classe_log_penalizacao != Resultado[split == "Teste"] )
riscos$logistica_penalizacao # 0.1987 praticamente a mesma coisa do sem penalização
riscos

table(classe_log_penalizacao, Resultado[split == "Teste"])




# Logistica simples
treino = df[split == "Treinamento"]
df_treino
M1 = glm(Y ~ ., family = binomial, data = df_treino)
summary(M1)
anova(M1, test = "Chisq")



# predicao
pred = predict(M1, df_teste, type = "response")
pred
classe = ifelse(pred<0.5, "Bom pagador", "Mau pagador") 
classe

riscos = list()
riscos$logistica = mean(classe != df_teste$Y)
riscos

table(classe, df_teste$Y)


# ROC
roc = plot.roc(df_treino$Y, fitted(M1))

plot(roc,
     print.auc=TRUE, 
     auc.polygon=TRUE, 
     grud=c(0.1,0.2),
     grid.col=c("blue","yeelow"), 
     max.auc.polygon=TRUE, 
     auc.polygon.col="lightgreen", 
     print.thres=TRUE)


# odds
library(mfx)

logitor(Y ~ ., data = df_treino)

exp(M1$coefficients)

library(ROCR)
pred <- prediction(pred, df$Y[split == "Teste"])
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)


# Interpretação 
exp(coef(M1))