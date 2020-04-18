# Load libraries
library(collections)
library(multcomp)

# Load data
d = read.csv('/Users/nknyazev/Documents/Delft/Thesis/temporal/data/results/statistics/RQ1/nonwide.csv')
metrics <- c('Recall', 'MRR', 'UserRecall', 'UserMRR')
metricsRank <- c('RecallRank', 'MRRRank', 'UserRecallRank', 'UserMRRRank')

# Scale each metric for each model
data <- list()
for (dset in unique(d$Dataset)) {
  dataset <- d[which(d$Dataset == dset),]
  dataset[metrics] <- scale(dataset[metrics])
  for (m in metrics) {
    r <- paste(c(m, "Rank"), sep="", collapse = "")
    dataset[,c(r)] <- rank(dataset[,m])
  }
  data <- rbind(data, dataset)
}

m2int <- function(x){as.numeric(substr(x, 2,2))}
data$model <- sapply(data$model, m2int)

summaries <- Dict$new()
for (metric in metrics) {
  
  model_model <- lm(get(metric) ~ model, data)
  equation_model <- lm(get(metric) ~ eq1*eq2*eq3, data)
    
  # check validity of the fits
  model_shapiro <- as.numeric(shapiro.test(residuals(model_model))["p.value"])
  equation_shapiro <- as.numeric(shapiro.test(residuals(model_model))["p.value"])
  model_R2 <- as.numeric(summary(model_model)["adj.r.squared"])
  equation_R2 <- as.numeric(summary(equation_model)["adj.r.squared"])
  
  # Get the p-values
  model_summary <- summary(aov(model_model, data = data))
  equation_summary <- summary(aov(equation_model, data = data))

  # Dunnett's test
  model_dunnett <- summary(glht(model_model, linfct=mcp(model="Dunnett"), alternative="greater"))
  interactions <- interaction(data$eq1, data$eq2, data$eq3)
  interaction_model <- lm(get(metric) ~ interactions, data)
  equation_dunnett <- summary(glht(interaction_model, linfct=mcp(interactions="Dunnett"), alternative="greater"))
  
  # Record all the values
  summaries$set(metric, Dict$new())
  summaries$get(metric)$set("model", Dict$new()); summaries$get(metric)$set("equation", Dict$new())
  summaries$get(metric)$get("model")$set("summary", model_summary); summaries$get(metric)$get("equation")$set("summary", equation_summary)
  summaries$get(metric)$get("model")$set("R2", model_R2); summaries$get(metric)$get("equation")$set("R2", equation_R2)
  summaries$get(metric)$get("model")$set("shapiro", model_shapiro);summaries$get(metric)$get("equation")$set("shapiro", equation_shapiro);
  summaries$get(metric)$get("model")$set("dunnett", model_dunnett);summaries$get(metric)$get("equation")$set("dunnett", equation_dunnett);
  
}

for (metric in metrics){print(metric);print(summaries$get(metric)$get("model")$get("summary"))}
for (metric in metrics){print(metric);print(summaries$get(metric)$get("model")$get("dunnett"))}
for (metric in metrics){print(metric);print(summaries$get(metric)$get("model")$get("R2"))}

p1 <- sapply(c(1,2,3,4,5,6,7), function(x){wilcox.test(RecallRank~model, data=data, alternative="less", subset=model %in% c(0,x))$p.value})
p2 <- sapply(c(1,2,3,4,5,6,7), function(x){wilcox.test(UserRecallRank~model, data=data, alternative="less", subset=model %in% c(0,x))$p.value})
p3 <- sapply(c(1,2,3,4,5,6,7), function(x){wilcox.test(MRRRank~model, data=data, alternative="less", subset=model %in% c(0,x))$p.value})
p4 <- sapply(c(1,2,3,4,5,6,7), function(x){wilcox.test(UserMRRRank~model, data=data, alternative="less", subset=model %in% c(0,x))$p.value})

# Export dunnet's test p-values to csv
get_dunnett <- function(metric) {
  summaries$get(metric)$get("model")$get("dunnett")$test$pvalues[1:7]
}
dunnett_p_values <- sapply(metrics, get_dunnett)
write.csv(dunnett_p_values, "/Users/nknyazev/Documents/Delft/Thesis/temporal/data/results/RQ1/dunnett_p_values.csv", row.names=FALSE)
