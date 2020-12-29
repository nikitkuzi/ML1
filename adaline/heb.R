lossFunctionHeb <- function(x)
{
  return(if (x < 1) -x else 0)
}


normalize <- function(xl)
{
  n <- dim(xl)[2] - 1
  for (i in 1:n) {
    xl[, i] <- (xl[, i] - mean(xl[, i])) / sd(xl[, i])
  }
  return(xl)
}

addcol <- function(xl)
{
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  xl <- cbind(xl[, 1:n], seq(from = -1, to = -1, length.out = l), xl[, n + 1])
}

sgHeb <- function(xl, eta = 1, lambda = 1 / 6)
{
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  w <- c(1 / 2, 1 / 2, 1 / 2)
  iterCount <- 0
  Q <- 0
  for (i in 1:l) {
    wx <- sum(w * xl[i, 1:n])
    margin <- wx * xl[i, n + 1]
    Q <- Q + lossFunctionAdaline(margin)
  }
  repeat
  {
    margins <- array(dim = l)
    for (i in 1:l)
    {
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      margins[i] <- crossprod(w, xi) * yi
    }
    errorIndexes <- which(margins <= 0)
    if (length(errorIndexes) > 0)
    {
      i <- sample(1:l, 1)
      iterCount <- iterCount + 1
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      wx <- crossprod(w, xi)
      margin <- wx * yi
      ex <- lossFunctionAdaline(margin)
      w <- w - eta * (wx - yi) * xi
      Qprev <- Q
      Q <- (1 - lambda) * Q + lambda * ex
    } else
    {
      break
    }
  }
  return(w)
}

library(MASS)
Sigma1 <- matrix(c(3, 0, 0, 3), 2, 2)
Sigma2 <- matrix(c(3, 0, 0, 3), 2, 2)
Mu1 <- c(3, 0)
Mu2 <- c(8, 7)
set1 <- mvrnorm(111, Mu1, Sigma1)
set2 <- mvrnorm(111, Mu2, Sigma2)
data <- rbind(cbind(set1, 1), cbind(set2, -1))
dataNormalized <- addcol(normalize(data))
colors <- c("1" = "blue", "-1" = "green")
plot(dataNormalized[, 1], dataNormalized[, 2], pch = 21, bg = colors[as.character(data[, 3])], asp = 1)

ww <- sgHeb(dataNormalized)
abline(a = ww[3] / ww[2], b = -ww[1] / ww[2], lwd = 3, col = "purple")