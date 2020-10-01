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

kNN <- function(trainData, z, k)
{
  # looking for the closest neighbours and return their class
  orderedData <- sortObjectsByDist(trainData, z)
  n <- dim(orderedData)[2] - 1

  classes <- orderedData[1:k, n + 1]

  counts <- table(classes)

  class <- names(which.max(counts))

  return (class)

}

colors <-c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg= colors[iris$Species], col = colors[iris$Species], asp = 1)

# test data
z <-cbind(runif(20, min = 0.1, max = 7.1),
          runif(20, min = 0.1, max = 3.0))

trainData <-iris[, 3:5]

for (i in 1:20)
{
  class <- kNN(trainData, c(z[i, 1], z[i, 2]), k=6)
  points(z[i, 1],z[i, 2], pch = 22, bg = colors[class], lwd = 2)
}



