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

kNN <- function(trainData, z, k)
{
  orderedData <- sortObjectsByDist(trainData, z)
  n <- dim(orderedData)[2] - 1

  classes <- orderedData[1:k, n + 1]

  counts <- table(classes)

  class <- names(which.max(counts))

  return(class)

}

kwNN <- function(trainData, z, k = 1)
{
  w <- get_weight(k)
  orderedData <- sortObjectsByDist(trainData, z)
  n <- dim(orderedData)[2] - 1
  classes <- orderedData[1:k, n + 1]
  counts <- c("a" = 0, "b" = 0)
  for (i in 1:k) {
    counts[classes[i]] <- counts[classes[i]] + w[i]
  }
  class <- names(which.max(counts))
  return(class)
}

get_weight <- function(k)
{
  w <- c(1, k)
  for (i in 1:k)
    w[i] <- ((k + 1 - i) / k)
  return(w)
}

a <- vector()
b <- vector()
classes <- vector()

a[1] <- 1
a[2] <- 1
a[3] <- 4
a[4] <- 5
a[5] <- 4

b[1] <- 1
b[2] <- 1.5
b[3] <- 2
b[4] <- 2
b[5] <- 1.8

classes[1] <- "a"
classes[2] <- "a"
classes[3] <- "b"
classes[4] <- "b"
classes[5] <- "b"

trainData <- data.frame(a,b,classes)


colors <- c("a" = "red", "b" = "blue")
plot(a, b, pch = 21, col = colors[classes], bg = colors[classes])

#class <- kNN(trainData, c(1, 1.2), k = 5)
#points(1, 1.2, pch = 22, bg = colors[class], lwd = 2)

class <- kwNN(trainData, c(1, 1.2), k = 5)
points(1, 1.2, pch = 23, bg = colors[class], lwd = 2)

