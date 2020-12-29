mat_expect <- function(data, probability) {
  l <- dim(data)[1]
  n <- dim(data)[2]
  expect <- rep(0, n)
  for (i in 1:l) {
    expect <- expect + data[i,] * probability[i]
  }
  return(expect)
}

dispersion <- function(data, probability) {
  l <- dim(data)[1]
  n <- dim(data)[2]
  mat_expect <- mat_expect(data, probability)
  first <- matrix(0, 0, n)
  for (i in 1:l) {
    first <- rbind(first, ((data[i,] - mat_expect)^2))
  }
  return(mat_expect(first, probability))
}

density <- function(x, mat_expect, deviation) {
  return((1 / (deviation * sqrt(2 * pi))) * exp(-((x - mat_expect)^2) / (2 * deviation^2)))
}

naive <- function(data, z, lambda) {
  l <- dim(data)[1]
  n <- dim(data)[2]
  mat_exp = matrix(0, 0, 2)
  dispersion = matrix(0, 0, 2)
  tmp <- table(data[3])
  prior <- tmp / sum(tmp)
  classesNames <- unique(data[, 3])
  mat_exp <- rbind(mat_exp, mat_expect(data[1:50,], rep(1 / tmp[1], tmp[1])))
  mat_exp <- rbind(mat_exp, mat_expect(data[51:100,], rep(1 / tmp[2], tmp[2])))
  mat_exp <- rbind(mat_exp, mat_expect(data[101:150,], rep(1 / tmp[3], tmp[3])))
  dispersion <- rbind(dispersion, dispersion(data[1:50,], rep(1 / tmp[1], tmp[1])))
  dispersion <- rbind(dispersion, dispersion(data[51:100,], rep(1 / tmp[2], tmp[2])))
  dispersion <- rbind(dispersion, dispersion(data[101:150,], rep(1 / tmp[3], tmp[3])))
  classes <- c("setosa" = 0, "versicolor" = 0, "virginca" = 0)
  for (i in 1:3) {
    density <- 0
    for (j in 1:2) {
      density <- density + log(density(z[j], mat_exp[i, j], sqrt(dispersion[i, j])))
    }
    classes[i] <- log(lambda[i]) + log(prior[i]) + density
  }
  return(which.max(classes))
}

colors <- c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species], col = colors[iris$Species], asp = 1)
Data <- iris[, 3:5]
## test data
#z <- cbind(runif(20, min = 0.1, max = 7.1),
#           runif(20, min = 0.1, max = 3.0))
#
#Data <- iris[, 3:5]
#for (i in 1:20)
#{
#  class <- naive(Data, c(z[i, 1], z[i, 2]), c(1, 1, 1))
#  points(z[i, 1], z[i, 2], pch = 22, bg = colors[class], lwd = 2)
#}
deltaX <- 2
  deltaY <- 2

l <- min(iris$Petal.Length)
r <- max(iris$Petal.Length)
b <- min(iris$Petal.Width)
t <- max(iris$Petal.Width)

for (x in seq(l, r, deltaX)) {
  for (y in seq(b, t, deltaY)) {
    z <- c(x, y)
    class <- naive(Data, z, c(10, 1, 5))
    points(x, y, bg = colors[class], col = colors[class])
  }
}