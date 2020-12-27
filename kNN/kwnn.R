euclideanDistance <- function(u, v)
{
  sqrt(sum((u - v)^2))
}

sortObjectsByDist <- function(trainData, z, metricFunction = euclideanDistance)
{
  l <- dim(trainData)[1]
  n <- dim(trainData)[2] - 1
  distances <- matrix(NA, l, 2)
  for (i in 1:l)
  {
    distances[i,] <- c(i, metricFunction(trainData[i, 1:n], z))
  }

  orderedTrainData <- trainData[order(distances[, 2]),]

  return(orderedTrainData)
}

kwNN <- function(trainData, z, q = 0.6, k = 1)
{
  w <- get_weight(q, k)
  orderedData <- sortObjectsByDist(trainData, z)
  n <- dim(orderedData)[2] - 1
  classes <- orderedData[1:k, n + 1]
  counts <- c("setosa" = 0, "versicolor" = 0, "virginica" = 0)
  for (i in 1:k) {
    counts[classes[i]] <- counts[classes[i]] + w[i]
  }
  class <- names(which.max(counts))
  return(class)
}

get_weight <- function(q, k)
{
  w <- c(1, k)
  for (i in 1:k)
    w[i] <- q**i
  return(w)
}

loo <- function(trainData, q_min, q_max, k_min, k_max)
{
  l <- dim(trainData)[1]
  n <- dim(trainData)[2] - 1
  rows <- k_max - k_min + 1
  cols <- (q_max - q_min) / 0.1 + 1
  loo <- matrix(0, rows, cols)
  for (i in 1:(l - 1))
  {
    orderedData <- sortObjectsByDist(trainData[-i,], trainData[i, 1:n])
    for (k in k_min:k_max)
    {
      classes <- orderedData[1:k, n + 1]
      q <- q_min
      for (j in 1:cols) {
        w <- get_weight(q, k)
        counts <- c("setosa" = 0, "versicolor" = 0, "virginica" = 0)
        for (z in 1:k) {
          counts[classes[z]] <- counts[classes[z]] + w[z]
        }
        class <- names(which.max(counts))
        if (trainData[i, 3] != class)
        {
          loo[k, j] = loo[k, j] + 1
        }
        q <- q + 0.1
        #print(loo)
      }
    }
  }
  heatmap.2(loo, xlab = "q", ylab = "k")
  for (i in 1:rows) {
    for (j in 1:cols) {
      if (loo[which.min(loo)] == loo[i, j]) {
        return(c(i, q_min + 0.1 * (j - 1)))
      }
    }
  }
}

colors <- c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species], col = colors[iris$Species], asp = 1)

# test data
z <- cbind(runif(20, min = 0.1, max = 7.1),
           runif(20, min = 0.1, max = 3.0))
trainData <- iris[, 3:5]
for (i in 1:dim(z)[1])
{
  class <- kwNN(trainData, c(z[i, 1], z[i, 2]), q = 0.6, k = 6)
  points(z[i, 1], z[i, 2], pch = 22, bg = colors[class], lwd = 2)
}
Loo <- loo(trainData, 0.1, 1, 1, 150)


