math_expec <- function(objects)
{
  n <- dim(objects)[2]
  mathexpec <- matrix(NA, 1, n)
  for (col in 1:n)
  {
    mathexpec[1, col] = mean(objects[, col])
  }
  return(mathexpec)
}

cov_matrix <- function(objects, mathexpec)
{
  l <- dim(objects)[1]
  n <- dim(objects)[2]
  sigma <- matrix(0, n, n)
  for (i in 1:l)
  {
    sigma <- sigma + (t(objects[i,] - mathexpec) %*%
      (objects[i,] - mathexpec)) / (l - 1)
  }
  return(sigma)
}

## Получение коэффициентов подстановочного алгоритма
getPlugInDiskriminantCoeffs <- function(mathexpec1, sigma1, mathexpec2, sigma2)
{
  ## Line equation: a*x1^2 + b*x1*x2 + c*x2 + d*x1 + e*x+f = 0
  invSigma1 <- solve(sigma1)
  invSigma2 <- solve(sigma2)
  f <- log(abs(det(sigma1))) - log(abs(det(sigma2))) +
    mathexpec1 %*% invSigma1 %*% t(mathexpec1) - mathexpec2 %*% invSigma2 %*%
    t(mathexpec2);
  alpha <- invSigma1 - invSigma2
  a <- alpha[1, 1]
  b <- 2 * alpha[1, 2]
  c <- alpha[2, 2]
  beta <- invSigma1 %*% t(mathexpec1) - invSigma2 %*% t(mathexpec2)
  d <- -2 * beta[1, 1]
  e <- -2 * beta[2, 1]
  return(c("x^2" = a, "xy" = b, "y^2" = c, "x" = d, "y"
    = e, "1" = f))
}

## Количество объектов в каждом классе
ObjectsCountOfEachClass <- 100
## Подключаем библиотеку MASS для генерации многомерного нормальногораспределения
library(MASS)
## Генерируем тестовые данные
Sigma1 <- matrix(c(10, 0, 0, 1), 2, 2)
Sigma2 <- matrix(c(1, 0, 0, 5), 2, 2)
Mu1 <- c(4, 0)
Mu2 <- c(7, 0)
xy1 <- mvrnorm(n = ObjectsCountOfEachClass, Mu1, Sigma1)
xy2 <- mvrnorm(n = ObjectsCountOfEachClass, Mu2, Sigma2)
## Собираем два класса в одну выборку
xl <- rbind(cbind(xy1, 1), cbind(xy2, 2))
## Рисуем обучающую выборку
colors <- c(rgb(0 / 255, 162 / 255, 232 / 255), rgb(0 / 255,
                                                    200 / 255, 0 / 255))
plot(xl[, 1], xl[, 2], pch = 21, bg = colors[xl[, 3]], asp =
  1)
## Оценивание
objectsOfFirstClass <- xl[xl[, 3] == 1, 1:2]
objectsOfSecondClass <- xl[xl[, 3] == 2, 1:2]
me1 <- math_expec(objectsOfFirstClass)
me2 <- math_expec(objectsOfSecondClass)
sigma1 <- cov_matrix(objectsOfFirstClass, me1)
sigma2 <- cov_matrix(objectsOfSecondClass, me2)
coeffs <- getPlugInDiskriminantCoeffs(me1, sigma1, me2, sigma2)
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