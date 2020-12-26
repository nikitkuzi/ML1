euclideanDistance <- function(u,v)
{
  sqrt(sum((u-v)^2))
}


nn <- function(xl, u)
{
  l <- dim(xl)[1]
  n <- dim(xl)[2]

  min <- Inf
  nearestI <- -1

  # looking for the closest neighbour and return it's class
  for (i in 1:l)
  {
    tempDist <- euclideanDistance(xl[i, 1:n-1],u)
    if (tempDist < min)
    {
      min <- tempDist
      nearestI <- i
    }

  }
  return(xl[nearestI, n])

}

colors <-c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg= colors[iris$Species], col = colors[iris$Species], asp = 1)

# test data
z <-cbind(runif(20, min = 0.1, max = 7.1),
          runif(20, min = 0.1, max = 3.0))

# training data
xl <-iris[, 3:5]

for (i in 1:20)
{
  class <- nn(xl, c(z[i, 1], z[i, 2]))
  points(z[i, 1],z[i, 2], pch = 22, bg = colors[class], asp = 1)
}


