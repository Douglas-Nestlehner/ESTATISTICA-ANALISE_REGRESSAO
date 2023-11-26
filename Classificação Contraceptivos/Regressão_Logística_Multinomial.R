if(!require('nnet')) install.packages('nnet')
if(!require('rpart')) install.packages('rpart')
if(!require('hnp')) install.packages('hnp')
if(!require('cowplot')) install.packages('cowplot')

library('nnet')
library('ggplot2')
library('hnp')
library('corrplot')
library('xtable')
library('car')


#https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice

cmc = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data')

colnames(cmc) = c("Idade_Mulher", "Educacao_Mulher", "Educacao_Marido",
                  "Num_Filhos", "Religiao_Mulher", "Mulher_Trabalha",
                  "Profissao_Marido", "Qualidade_de_Vida", "Exposicao_Midia",
                  "Metodo")

#Invertendo 2 com 3 pra mais clareza
cmc$Metodo = ifelse(cmc$Metodo == 1, 'N', ifelse(cmc$Metodo == 2, 'CP',
                                                 'LP'))

#Transforma colunas em fatores
colunas_fatores = which(colnames(cmc) %in% c("Idade_Mulher","Num_Filhos") == F)
cmc[,colunas_fatores] = lapply(cmc[,colunas_fatores], as.factor)
cmc$Metodo = relevel(cmc$Metodo, 'N')

sapply(cmc, class)

classes = sapply(cmc, class)

sumario = sapply(cmc[,classes %in% c('numeric', 'integer')], summary)

sumario

dados_hist = cmc[,classes %in% c('numeric', 'integer')]

cor(dados_hist)

grafs = lapply(1:ncol(dados_hist),
               function(col) ggplot2::ggplot(data.frame(dados_hist[[col]]), 
                                             aes(x = dados_hist[[col]] )) + geom_histogram(aes(y=..density..), 
                                                                                           color="grey", fill="springgreen4", 
                                                                                           bins = 20) + 
                 theme_minimal() +
                 theme(plot.title = element_text(hjust = 0.5)) + 
                 labs(x = dimnames(dados_hist)[[2]][col], y = 'densidade'))

obj = cowplot::plot_grid(plotlist = grafs, ncol = 2)

#ggsave('figuras_MLG/histogramas.pdf', obj, units = 'in', width = 8, height = 5)

dados_barras = cmc[,!(classes %in% c('numeric', 'integer'))]

sumario_quali = lapply(dados_barras, summary)

grafs = lapply(1:ncol(dados_barras),
               function(col) ggplot2::ggplot(data.frame(dados_barras[[col]]), 
                                             aes(x = dados_barras[[col]]                )) +
                 geom_bar(aes(fill= dados_barras[[col]])) + 
                 theme_minimal() +
                 theme(plot.title = element_text(hjust = 0.5)) + 
                 labs(x = dimnames(dados_barras)[[2]][col], y = 'Contagem') + 
                 scale_fill_brewer(name = "",palette="Set1")
) 

obj = cowplot::plot_grid(plotlist = grafs, ncol = 2)

#ggsave('figuras_MLG/barplots.pdf', obj, units = 'in', width = 7, height = 5)

#modelo = multinom(Metodo ~ ., data = cmc)
#modelo com interações

cmc_modelo = cmc
#A função nnet funciona melhor quando os preditores numéricos
#estão padronizados (função scale)
cmc_modelo$S_Idade_Mulher = scale(cmc$Idade_Mulher)
cmc_modelo$S_Num_Filhos = scale(cmc$Num_Filhos)

#Declando modelo com todas as interações 2 a 2 como limite pro stepwise

#A referência é o não uso
modelo_completo = paste0('Metodo ~ (',
                         paste0(colnames(cmc_modelo)
                                [!(colnames(cmc_modelo) %in% c('Metodo','Idade_Mulher','Filhos'))], 
                                collapse = ' + '),')^2')

#set.seed(154
#amostra = sample(0.3*nrow(cmc), nrow(cmc))
#cmc_modelo = cmc_modelo[-amostra,]

modelo = step(multinom(Metodo ~ 1, cmc_modelo), modelo_completo)

sumario = summary(modelo)

#Envelope
hnp(modelo, halfnormal = T, pch = 16, cex = 0.5, paint.out = T,
    xlab = "Quantis Half-Normal", ylab = "Resíduos")

vif(modelo)

beta1 = coef(modelo)[1,]
beta2 = coef(modelo)[2,]

termos = modelo$terms
variaveis = attr(termos, "term.labels")

novas_variaveis = 
  paste0("Metodo ~ ",
         paste0(variaveis, 
                collapse = ' + '))


#Teste normal para os coeficientes
z = coef(modelo)/sumario$standard.errors
(1 - pnorm(abs(z), 0, 1)) * 2

#Deviance contra o modelo nulo
anova(multinom(Metodo ~ 1, data = cmc_modelo), modelo)

Anova(modelo, type = 'III')

#Sem retirar nenhuma covariável
modelo_final = modelo
sumario = summary(modelo_final)

#Teste normal para os coeficientes
z = coef(modelo_final)/sumario$standard.errors
(1 - pnorm(abs(z), 0, 1)) * 2

sumario_mf = t(rbind(coef(modelo_final), sumario$standard.errors, (1 - pnorm(abs(z), 0, 1)) * 2))
sumario_mf = sumario_mf[, order(colnames(sumario_mf))]

#Sumario estilo glm com o coef, valor, erro e teste z.
sumario_mf

xtable(sumario_mf, digits = 4)

#Tabela de ajustados vs reais
table(cmc_modelo$Metodo, predict(modelo_final))

#Se der >0.05, a regressão tem poder preditivo muito ruim
chisq.test(cmc_modelo$Metodo, predict(modelo_final))

fit_modelo = fitted(modelo_final)

#Média de probabilidade ajustada versus cada variável
termos = modelo_final$terms
variaveis = attr(termos, "variables")

variaveis = gsub('S_','', variaveis)

grafs = lapply(3:(length(variaveis)),
               function(col) {
                 df1 = aggregate(as.formula(paste0('fit_modelo[,1] ~ ',
                                                   variaveis[col])), cmc_modelo, mean)
                 df2 = aggregate(as.formula(paste0('fit_modelo[,2] ~ ',
                                                   variaveis[col])), cmc_modelo, mean)
                 df3 = aggregate(as.formula(paste0('fit_modelo[,3] ~ ',
                                                   variaveis[col])), cmc_modelo, mean)
                 
                 ggplot2::ggplot() +
                   geom_line(aes(x = df1[,1], y = df1[,2], colour = "a", group = 1)) +
                   geom_line(aes(x = df2[,1], y = df2[,2], colour = "b", group = 1)) +
                   geom_line(aes(x = df3[,1], y = df3[,2], colour = "c", group = 1)) +
                   theme_minimal() +
                   theme(plot.title = element_text(hjust = 0.5), legend.position = 'bottom') + 
                   labs(x = paste0(variaveis[col]), y = 'Fitted Values') +
                   scale_colour_manual(name = "Uso", values = c("tomato",
                                                                "royalblue",
                                                                "springgreen4"),
                                       labels = c("Não Usa", "Usa Longo Prazo", "Usa Curto Prazo"))
               })

obj = cowplot::plot_grid(plotlist = grafs, ncol = 2)

obj

#ggsave('figuras_MLG/medias_var.pdf', plot = obj, units = 'in', width = 8, height =7)

#####Odds ----

t(exp(coef(modelo_final)))

#xtable(t(exp(coef(modelo_final))), digits = 4)

#####Predito ----
cmc_preditos = cmc
cmc_preditos$S_Idade_Mulher = scale(cmc$Idade_Mulher)
cmc_preditos$S_Num_Filhos = scale(cmc$Num_Filhos)

cmc_preditos = cmc_preditos[amostra,]

table(cmc_preditos$Metodo, predict(modelo_final, cmc_preditos))

library('caret')

cm = confusionMatrix(predict(modelo_final, cmc_preditos), 
                     reference = cmc_preditos$Metodo)

cm[["byClass"]]

cm[["overall"]]

f1_score = function(predicted, expected, positive.class="1") {
  predicted = factor(as.character(predicted), levels=unique(as.character(expected)))
  expected  = as.factor(expected)
  cm = as.matrix(table(expected, predicted))
  
  precision = diag(cm) / colSums(cm)
  recall = diag(cm) / rowSums(cm)
  f1 =  ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  
  f1[is.na(f1)] = 0
  
  ifelse(nlevels(expected) == 2, f1[positive.class], mean(f1))
}

f1_score(predict(modelo_final, cmc_preditos), 
         cmc_preditos$Metodo)

#####Comparando com o RPART
modelo_rpart = rpart(Metodo ~., cmc_modelo, method = "class", 
                     control = rpart.control(cp = 0.001, maxdepth = 10))

cm_rpart = confusionMatrix(predict(modelo_rpart, cmc_preditos, type = "class"), 
                           reference = cmc_preditos$Metodo)

cm_rpart[["byClass"]]