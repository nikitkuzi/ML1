library(MASS)
mat_expec <- function(objects)
{
  n <- dim(objects)[2]
  math_expec <- matrix(NA, 1, n)
  for (col in 1:n)
  {
    math_expec[1, col] = mean(objects[, col])
  }
  return(math_expec)
}

cov_mat <- function(data, math_expec)
{
  l <- dim(data)[1]
  n <- dim(data)[2]
  cov_matrix <- matrix(0, n, n)
  for (i in 1:l)
  {
    cov_matrix <- cov_matrix + (t(data[i,] - math_expec) %*% (data[i,] - math_expec)) / (l - 1)
  }
  return(cov_matrix)
}


plug_in <- function(data)
{
  mat_expec1 <- mat_expec(data[data[, 3] == 1, 1:2])
  mat_expec2 <- mat_expec(data[data[, 3] == 2, 1:2])
  cov_matrix1 <- cov_mat(data[data[, 3] == 1, 1:2], mat_expec1)
  cov_matrix2 <- cov_mat(data[data[, 3] == 2, 1:2], mat_expec2)
  inv_cov_matrix1 <- solve(cov_matrix1)
  inv_cov_matrix2 <- solve(cov_matrix2)
  f <- log(abs(det(cov_matrix1))) - log(abs(det(cov_matrix2))) + mat_expec1 %*% inv_cov_matrix1 %*% t(mat_expec1) - mat_expec2 %*% inv_cov_matrix2 %*% t(mat_expec2)
  alpha <- inv_cov_matrix1 - inv_cov_matrix2
  a <- alpha[1, 1]
  b <- 2 * alpha[1, 2]
  c <- alpha[2, 2]
  beta <- inv_cov_matrix1 %*% t(mat_expec1) - inv_cov_matrix2 %*% t(mat_expec2)
  d <- -2 * beta[1, 1]
  e <- -2 * beta[2, 1]
  return(c("x^2" = a, "xy" = b, "y^2" = c, "x" = d, "y" = e, "1" = f))
}

Sigma1 <- matrix(c(1, 0, 0, 2), 2, 2)
Sigma2 <- matrix(c(10, 0, 0, 15), 2, 2)
Mu1 <- c(5, 5)
Mu2 <- c(15, 2)
set1 <- mvrnorm(250, Mu1, Sigma1)
set2 <- mvrnorm(250, Mu2, Sigma2)
data <- rbind(cbind(set1, 1), cbind(set2, 2))
colors <- c("blue", "green")
plot(data[, 1], data[, 2], pch = 21, bg = colors[data[, 3]], asp = 1)

coeffs <- plug_in(data)
## Рисуем дискриминантую функцию – красная линия
x <- y <- seq(-10, 20, len = 100)
z <- outer(x, y, function(x, y) coeffs["x^2"] * x^2 +
  coeffs["xy"] * x * y
  +
  coeffs["y^2"] * y^2 +
  coeffs["x"] * x
  +
  coeffs["y"] * y +
  coeffs["1"])
contour(x, y, z, levels = 0, drawlabels = FALSE, lwd = 5, col = "red", add = TRUE)