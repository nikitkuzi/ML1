euclideanDistance <- function(u,v)
{
  sqrt(sum((u-v)^2))
}

sortObjectsByDist <- function(trainData, z, metricFunction = euclideanDistance)
{
  l <- dim(trainData)[1]
  n <- dim(trainData)[2] - 1
  distances <- matrix(NA, l, 2)
  for (i in 1:l)
  {
    distances[i, ] <- c(i, metricFunction(trainData[i, 1:n], z))
  }

  orderedTrainData <- trainData[order(distances[, 2]), ]

  return (orderedTrainData)
}

loo <- function(trainData)
{
  l <- dim(trainData)[1]
  n <- dim(trainData)[2]
  loo <- matrix(0, l-1, 1)

  for (i in 1:l)
  {
    orderedTrainData <- sortObjectsByDist(trainData[-i, ], trainData[i, 1:n-1])

    for (j in 1:(l-1))
    {
      classes <- orderedTrainData[1:j, n]
      counts <- table(classes)
      class <- names(which.max(counts))

      if (trainData[i, 3] != class)
      {
        loo[j] = loo[j] +1
      }
    }
  }
  return(loo)
}

colors <-c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
trainData <-iris[, 3:5]
Loo <- loo(trainData)
plot(Loo, type="l")
print(which.min(Loo))




