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
  # looking for the closest neighbours and return their class
  orderedData <- sortObjectsByDist(trainData, z)
  n <- dim(orderedData)[2] - 1

  classes <- orderedData[1:k, n + 1]

  counts <- table(classes)

  class <- names(which.max(counts))

  return(class)

}

colors <- c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species], col = colors[iris$Species], xlab = "Petal.Length", ylab = "Petal.Width")
trainData <- iris[, 3:5]

deltaX <- 0.1
deltaY <- 0.1
x <- min(iris$Petal.Length)
while (x <= max(iris$Petal.Length)) {
  y <- min(iris$Petal.Width)
  while (y <= max(iris$Petal.Width)) {
    tmp <- cbind(x, y)
    class <- kNN(trainData, tmp, 6)
    points(x, y, pch = 21, col = colors[class], lwd = 1)
    y <- y + deltaY
  }
  x <- x + deltaX
}

x <- max(iris$Petal.Length)
y <- min(iris$Petal.Width)
while (y <= max(iris$Petal.Width)) {
  tmp <- cbind(x, y)
  class <- kNN(trainData, tmp, 6)
  points(x, y, pch = 21, col = colors[class], lwd = 1)
  y <- y + deltaY
  print(y)
}

x <- min(iris$Petal.Lengt)
y <- max(iris$Petal.Width)
while (x <= max(iris$Petal.Length)) {
  tmp <- cbind(x, y)
  class <- kNN(trainData, tmp, 6)
  points(x, y, pch = 21, col = colors[class], lwd = 1)
  x <- x + deltaX
}


