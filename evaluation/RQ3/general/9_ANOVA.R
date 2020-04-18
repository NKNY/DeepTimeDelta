# Adapted https://cran.r-project.org/web/packages/multcomp/vignettes/multcomp-examples.pdf

# Load libraries
library(collections)
library(multcomp)

# Load data
d = read.csv('/Users/nknyazev/Documents/Delft/Thesis/temporal/data/results/statistics/RQ3/dt_mean_group_nonwide.csv') #Adjust accordingly
d = read.csv('/Users/nknyazev/Documents/Delft/Thesis/temporal/data/results/statistics/RQ3/Kendalls Tau_nonwide.csv')
metrics <- c('Recall', 'MRR', 'UserRecall', 'UserMRR')

# Scale each metric for each model
data <- list()
# Scale per dataset per dt_group
for (dset in unique(d$Dataset)) {
  dataset <- d[which(d$Dataset == dset),]
  tmp <- list()
  for (dtg in unique(dataset$dt_group)) {
    dataset_dt_group <- dataset[dataset$dt_group==dtg,]
    dataset_dt_group[metrics] <- scale(dataset_dt_group[metrics])
    for (m in metrics) {
      r <- paste(c(m, "Rank"), sep="", collapse = "")
      dataset_dt_group[,c(r)] <- rank(dataset_dt_group[,m])
    }
    tmp <- rbind(tmp, dataset_dt_group)
  }
  data <- rbind(data, tmp)
}

summaries <- Dict$new()
for (metric in metrics) {
  summaries$set(metric, Dict$new())
  mod <- lm(get(metric) ~ model * dt_group, data = data)
  a <- summary(aov(mod))
  tmp <- expand.grid(model = unique(data$model), dt_group = unique(data$dt_group))
  X <- model.matrix(~ model * dt_group , data = tmp)
  Dunnett <- contrMat(table(data$model), "Dunnett")
  K1 <- cbind(Dunnett, matrix(0, nrow=nrow(Dunnett), ncol = ncol(Dunnett)), matrix(0, nrow=nrow(Dunnett), ncol = ncol(Dunnett)))
  rownames(K1) <- paste(levels(data$dt_group)[1], rownames(K1), sep = ":")
  K2 <- cbind(matrix(0, nrow = nrow(Dunnett), ncol = ncol(Dunnett)), Dunnett, matrix(0, nrow=nrow(Dunnett), ncol = ncol(Dunnett)))
  rownames(K2) <- paste(levels(data$dt_group)[2], rownames(K2), sep = ":")
  K3 <- cbind(matrix(0, nrow = nrow(Dunnett), ncol = ncol(Dunnett)), matrix(0, nrow = nrow(Dunnett), ncol = ncol(Dunnett)), Dunnett)
  rownames(K3) <- paste(levels(data$dt_group)[3], rownames(K3), sep = ":")
  K <- rbind(K1, K2,K3)
  colnames(K) <- c(colnames(Dunnett), colnames(Dunnett), colnames(Dunnett))
  s <- summary(glht(mod, linfct = K %*% X, alternative="greater"))
  summaries$get(metric)$set("anova", a)
  summaries$get(metric)$set("dunnett", s)
  
  print(as.numeric(summary(mod)["adj.r.squared"]))
}

for (metric in metrics){print(metric);print(summaries$get(metric)$get("anova"))}
for (metric in metrics){print(metric);print(summaries$get(metric)$get("dunnett"))}
for (metric in metrics){print(summaries$get(metric)$get("dunnett")$test$pvalues[1:21])}