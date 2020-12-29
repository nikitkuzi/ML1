library(MASS)

mat_expec <- function(data)
{
  n <- dim(data)[2]
  math_expec <- matrix(NA, 1, n)
  for (col in 1:n)
  {
    math_expec[1, col] = mean(data[, col])
  }
  return(math_expec)
}

fisher <- function(data)
{
  n <- table(data[, 3])[1]
  mat_expec1 <- mat_expec(data[data[, 3] == 1, 1:2])
  mat_expec2 <- mat_expec(data[data[, 3] == 2, 1:2])
  cov_mat <- matrix(0, 2, 2)
  for (i in 1:n) {
    cov_mat <- cov_mat + (t(data[i, 1:2] - mat_expec1) %*% (data[i, 1:2] - mat_expec1)) / (n - 1)
  }
  for (i in range(n, 2 * n)) {
    cov_mat <- cov_mat + (t(data[i, 1:2] - mat_expec2) %*% (data[i, 1:2] - mat_expec2)) / (n - 1)
  }
  return(cov_mat)
}

Sigma1 <- matrix(c(3, 0, 0, 2), 2, 2)
Sigma2 <- matrix(c(3, 0, 0, 2), 2, 2)
Mu1 <- c(-2, 4)
Mu2 <- c(2, 0)
set1 <- mvrnorm(20, Mu1, Sigma1)
set2 <- mvrnorm(20, Mu2, Sigma2)
data <- rbind(cbind(set1, 1), cbind(set2, 2))

colors <- c("blue", "green")
plot(data[, 1], data[, 2], pch = 21, bg = colors[data[, 3]], asp = 1)

cov_mat <- fisher(data)

inverse_cov_mat <- solve(cov_mat)
mat_expec1 <- mat_expec(data[data[, 3] == 1, 1:2])
mat_expec2 <- mat_expec(data[data[, 3] == 2, 1:2])
alpha <- inverse_cov_mat %*% t(mat_expec1 - mat_expec2)
mat_expec <- (mat_expec1 + mat_expec2) / 2
beta <- mat_expec %*% alpha
abline(beta / alpha[2, 1], -alpha[1, 1] / alpha[2, 1], col = "red", lwd = 3)
