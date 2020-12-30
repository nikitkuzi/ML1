<a name="Оглавление"></a>
## Оглавление

1. Метрические алгоритмы классификации

   1.1.[Метод ближайшего соседа](#Метод_ближайшего_соседа)
  
   1.2. [Метод К соседей](#Метод_к_соседей)
   
   1.3. [Метод К взвешенных соседей](#Метод_к_взвешенных)

1. Байесовские алгоритмы классификации

   1.1. [Линии уровня нормального распределения](#Линии_уровня_нормального)
   
   1.2. [Наивный нормальный байесовский классификатор](#Наивный_алгоритм)
   
   1.3. [Подстановочный алгоритм](#Подстановочный_алгоритм)
   
   1.4. [Линейный дискриминант Фишера](#Фишер) 
1. Линейные алгоритмы классификации

   1.1. [Стохастический градиентный спуск](#Стохастический)
   
   1.2. [Адалайн. Правило Хэбба](#Адалайн)
   
   1.3. [Логическая регрессия](#Логическая)
  


<a name="Метод_ближайшего_соседа"></a>
## Метод ближайшего соседа
[К оглавлению](#Оглавление)

Суть даного метода заключается  в том, что классифицируемому объекту присваевается тот класс, к которому принадлежит его ближайший сосед. 

Мы можем достичь результата на основании гипотезы о компактности, находя общие признаки объектов  применив определенные метрики.
```R
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
```
В результате работы программы мы получаем график, на котором круглые точки - выборка данных из датасета _Iris_, а квадратные точки - объекты из тестовой выборки.
![alt text](https://github.com/nikitkuzi/ML1/blob/master/kNN/img/1nn.jpeg?raw=true)

## Метод К соседей

[К оглавлению](#Оглавление) 

<a name="Метод_к_соседей"></a>
При использовании метода ближайших соседей (_kNN_) возникает вопрос: как выбрать оптимальное _k_? 

При _k_ = 1 получаем метод ближайшего соседа
и, соответственно, неустойчивость к шуму, при _k_ = _l_, наоборот, алгоритм
чрезмерно устойчив и вырождается в константу. Таким образом, крайние значения k нежелательны. На практике оптимальное _k_ подбирается по критерию
скользящего контроля _LOO_.

Преимущества _LOO_ в том, что каждый объект ровно один раз участвует в контроле, а длина обучающих подвыборок лишь на единицу меньше длины полной выборки.

```R
loo <- function(trainData)
{
  l <- dim(trainData)[1]
  n <- dim(trainData)[2]
  loo <- matrix(0, l-1, 1)

  for (i in 1:l)
  {
    # sorting data without curent elementh
    orderedTrainData <- sortObjectsByDist(trainData[-i, ], trainData[i, 1:n-1])

    for (j in 1:(l-1))
    {
      # get it's class
      classes <- orderedTrainData[1:j, n]
      counts <- table(classes)
      class <- names(which.max(counts))

      # looking for an error
      if (trainData[i, 3] != class)
      {
        loo[j] = loo[j] +1
      }
    }
  }
  return(loo)
}
```
Для каждого k находится ошибка классификации и в качестве оптимального _k_ выбирается то значение, для которого минимальна ошибка классфикации.

![alt text](https://github.com/nikitkuzi/ML1/blob/master/kNN/img/Loo.jpeg?raw=true)

Карта классификации ирисов Фишера алгоритмом _kNN_, при _k_ = 6:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/kNN/img/knn_Map.jpeg?raw=true)

## Метод К взвешенных соседей

[К оглавлению](#Оглавление) 
<a name="Метод_к_взвешенных"></a>

Еще одной модификацией алгоритма K ближайших соседей является метод ближайших _взвешенных_ соседей.

Суть данного алгоритма является в том, что каждому из соседей присваивается вес, который влияет на конечный класс объекта. Это позволяет избежать некоторых ошибок, которые возникают в результате _kNN_.

Вес очередного соседа задается при помощи функции _w(i)_, строго убывающая последовательность вещественных весов, задающая вклад _i_-го соседа при классификации очередного объекта. 
```R
get_weight <- function(k)
{
  w <- c(1, k)
  for (i in 1:k)
    w[i] <- ((k + 1 - i) / k)
  return(w)
}
```
```R
kwNN <- function(trainData, z, k = 1)
{
  w <- get_weight(k)
  orderedData <- sortObjectsByDist(trainData, z)
  n <- dim(orderedData)[2] - 1
  classes <- orderedData[1:k, n + 1]
  counts <- c("setosa" = 0, "versicolor" = 0, "virginica" = 0)
  for (i in 1:k){
    counts[classes[i]] <- counts[classes[i]] + w[i]
  }
  class <- names(which.max(counts))
  return(class)
}
```

Карта классификации ирисов Фишера алгоритмом _kwNN_, при _k_ = 6, <img src="https://bit.ly/3rvcGRj" align="center" border="0" alt="w(i) =  \frac{k + 1 - i}{k} " width="129" height="43" />
:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/kNN/img/kwNN_Map.jpeg?raw=true)

Можно задать такой же вопрос, как и при использовании метода _k_ ближайших соседей: какое именно _k_ и _w(i)_ выбрать? Можно использовать все тот же критерий скользящего контроля, который использовался при методе _k_ ближаших соседей:

```R
loo <- function(trainData, q_min, q_max, k_min, k_max)
{
  l <- dim(trainData)[1]
  n <- dim(trainData)[2] - 1
  rows <- k_max - k_min + 1
  cols <- (q_max - q_min) / 0.1 + 1
  loo <- matrix(0, rows, cols)
  for (i in 1:(l - 1))
  {
    orderedData <- sortObjectsByDist(trainData[-i,], trainData[i, 1:n])
    for (k in k_min:k_max)
    {
      classes <- orderedData[1:k, n + 1]
      q <- q_min
      for (j in 1:cols) {
        w <- get_weight(q, k)
        counts <- c("setosa" = 0, "versicolor" = 0, "virginica" = 0)
        for (z in 1:k) {
          counts[classes[z]] <- counts[classes[z]] + w[z]
        }
        class <- names(which.max(counts))
        if (trainData[i, 3] != class)
        {
          loo[k, j] = loo[k, j] + 1
        }
        q <- q + 0.1
      }
    }
  }
  for (i in 1:rows) {
    for (j in 1:cols) {
      if (loo[which.min(loo)] == loo[i, j]) {
        return(c(i, q_min + 0.1 * (j - 1)))
      }
    }
  }
}
```

Зависимость ошибки от значения _k_ и _q_

![alt text](https://github.com/nikitkuzi/ML1/blob/master/kNN/img/kwnn_loo_map.jpeg?raw=true)

Рассмотрим пример, показывающий преимущество _kwNN_ над _kNN_:

![alt text](https://github.com/nikitkuzi/ML1/blob/master/kNN/img/compare_kNN.jpeg?raw=true)
![alt text](https://github.com/nikitkuzi/ML1/blob/master/kNN/img/compare_kwNN.jpeg?raw=true)


<a name="Линии_уровня_нормального"></a>
## Линии уровня нормального распределения
[К оглавлению](#Оглавление) 

Байесовский подход является классическим в теории распознавания образов и лежит в основе многих методов. Он опирается на теорему о том, что
если плотности распределения классов известны, то eалгоритм классификации, имеющий минимальную вероятность ошибок, можно выписать
в явном виде.

Однако, на практике зачастую плотности
распределения классов неизвестны и их приходится восстанавливать
по обучающей выборке. В этом случае байесовский алгоритм перестает
быть оптимальным. Поэтому, чем лучше удастся восстановить
функции правдоподобия, тем ближебудет к оптимальному построенный алгоритм.

Одним из способов представления функций многих переменных являются линии уровня: на разных значениях n-мерной функции проводятся гипперплоскости и все точки пересечения этой гипперплоскости с функцией отображаются на (n-1)-мерное пространство. Для функции двух переменных эти точки пересечения отображаются на плоскость XOY.

Плотность _n_-мерного нормального распределения задается формулой:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/plug_in/img/equation1.jpeg?raw=true)

Выведем из нее уравнения для линии уровня. Не сложными математическими операциями придем к уравнению:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/plug_in/img/equation3.jpeg?raw=true)

Программная реализация данного алгоритма:
```R
density <- function(x, Mat_expect, Sigma) {
  n = dim(Sigma)[1]
  det = det(Sigma)
  left <- 1/(sqrt((2 * pi)^ n * det))
  right <- exp(1 / -2 * t(x - Mat_expect) %*% ginv(Sigma) %*% (x - Mat_expect))
  return(left * right)
}
```

Признаки некоррелированы, одинаковые дисперсии:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/levels/img/ne_kor_same_disp.jpeg?raw=true)

Признаки некоррелированы, разные дисперсии:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/levels/img/ne_cor_diff_disp.jpeg?raw=true)

Признаки коррелированы, разные дисперсии:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/levels/img/cor_diff_disp.png?raw=true)

## Наивный нормальный байесовский классификатор
<a name="Наивный_алгоритм"></a>
[К оглавлению](#Оглавление) 

Специальный частный случай байесовского классификатора, основанный на дополнительном предположении, что объекты _x∈X_ описываются _n_ статистически независимыми признаками:

![alt text](https://github.com/nikitkuzi/ML1/blob/master/naive/img/equation1.jpeg?raw=true)

Предположение о независимости означает, что функции правдоподобия классов представимы в виде:

![alt text](https://github.com/nikitkuzi/ML1/blob/master/naive/img/equation2.jpeg?raw=true)

Предположение о независимости существенно упрощает задачу, так как оценить _n_ одномерных плотностей гораздо легче, чем одну _n_-мерную плотность. К сожалению, оно крайне редко выполняется на практике, отсюда и название метода.

Наивный байесовский классификатор может быть как параметрическим, так и непараметрическим, в зависимости от того, каким методом восстанавливаются одномерные плотности.

Функция _density_ считает плотность заданного нормального расрпделения в точке:

```R
density <- function(x, mat_expect, deviation) {
  return((1 / (deviation * sqrt(2 * pi))) * exp(-((x - mat_expect)^2) / (2 * deviation^2)))
}
```

Функция _mat_expect_ находит матожидание случайной величины:
```R
mat_expect <- function(data, probability) {
  l <- dim(data)[1]
  n <- dim(data)[2]
  expect <- rep(0, n)
  for (i in 1:l) {
    expect <- expect + data[i,] * probability
  }
  return(expect)
}
```

Функция _dispersion_ считает дисперсию случайной величины:
```R
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
```

Далее необходимо реализовать классификатор, который будет выбирать класс по максимальной вероятности: 

![alt text](https://github.com/nikitkuzi/ML1/blob/master/naive/img/equation3.jpeg?raw=true)


```R
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
```

Карты классификации классов ирисов для различных _λ_: 

λ(1,1,1):
![alt text](https://github.com/nikitkuzi/ML1/blob/master/naive/img/naive_map1.jpeg?raw=true)

λ(10,1,5):
![alt text](https://github.com/nikitkuzi/ML1/blob/master/naive/img/naive_map2.jpeg?raw=true)

## Подстановочный алгоритм
<a name="Подстановочный_алгоритм"></a>
[К оглавлению](#Оглавление) 

Нормальный _дискриминантный анализ_ — это один из вариантов байесовской классификации,
восстанавливаемых в котором плотностей в качестве рассматривают моделей многомерные нормальные плотности:

![alt text](https://github.com/nikitkuzi/ML1/blob/master/plug_in/img/equation1.jpeg?raw=true)

Восстанавливая параметры нормального распределения _μ(y)_ , Σ*y*
для каждого класса _y ∈ Y_ и подставляя в формулу оптимального
байесовского классификатора восстановленные плотности, получим
подстановочный (_plug-in_) алгоритм классификации либо линейный
дискриминант _Фишера_ (если предположить, что матрицы ковариации
равны для всех классов).
Параметры нормального распределения оценивают согласно
принципа максимума правдоподобия:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/plug_in/img/equation2.jpeg?raw=true)

Функция _cov_mat_ - подсчет матрици ковариации:
```R
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
```

Реализация подстановочного алгоритма:
```R
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
```

В случаях, когда ковариационные матрицы классов не диагональны и не равны, разделяющие плоскости не линейны.

Параболическая разделяющая плоскость:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/plug_in/img/parabola.jpeg?raw=true)

Гиперболическая разделяющая плоскость:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/plug_in/img/hyperparabola.jpeg?raw=true)

Эллипсоидная разделяющая плоскость:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/plug_in/img/ellipse.jpeg?raw=true)

## Линейный дискриминант Фишера
<a name="Фишер"></a>
[К оглавлению](#Оглавление) 

Фишер (1936 г.) предложил простую эвристику «Ковариационные матрицы классов равны», позволяющую увеличить число объектов, по которым
оценивается ковариационная матрица, повысить её устойчивость и заодно упростить алгоритм обучения.
Пусть ковариационные матрицы классов одинаковы и равны _Σ_. Оценим _Σ_ по
всем _ℓ_ объектам обучающей выборки. С учетом поправки на смещённость
![alt text](https://github.com/nikitkuzi/ML1/blob/master/fisher/img/equation1.jpg?raw=true)

В этом случае разделяющая поверхность линейна (кусочно-линейна). Подстановочный алгоритм имеет вид:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/fisher/img/equation2.jpg?raw=true)

Этот алгоритм называется линейным дискриминантом Фишера (ЛДФ).
Он неплохо работает, когда формы классов действительно близки к нормальным и не слишком сильно различаются. В этом случае линейное решающее
правило близко к оптимальному байесовскому, но существенно более устойчиво, чем квадратичное, и часто обладает лучшей обобщающей способностью.
Вероятность ошибки линейного дискриминанта Фишера выражается через
расстояние Махаланобиса между классами, в случае, когда классов два:

![alt text](https://github.com/nikitkuzi/ML1/blob/master/fisher/img/equation3.jpg?raw=true)

Реализация поиска ковариационных матриц:

```R
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
```

Разделяющая плоскость задается уравнением: ![alt text](https://github.com/nikitkuzi/ML1/blob/master/fisher/img/equation5.svg?raw=true)

В результате получаем такие разделяющие поверхности:

![alt text](https://github.com/nikitkuzi/ML1/blob/master/fisher/img/line1.jpeg?raw=true)
![alt text](https://github.com/nikitkuzi/ML1/blob/master/fisher/img/line2.jpeg?raw=true)

Основным преимуществом алгоритма по сравнению с подстановочным алгоритмом является уменьшение эффекта плохой обусловленности ковариационной матрицы при недостаточных данных и простате реализации метода.

![alt text](https://github.com/nikitkuzi/ML1/blob/master/fisher/img/fisher_compare.jpeg?raw=true)
![alt text](https://github.com/nikitkuzi/ML1/blob/master/fisher/img/plugin_compare.jpeg?raw=true)


## Стохастический градиентный спуск
<a name="Стохастический"></a>
[К оглавлению](#Оглавление) 
Пусть _X_ = _R_ и _Y_ = {−1; +1}. Алгоритм _a_*(x,w)* = _sign(w, x)_ , _w ∈ (R)_ является _линейным алгоритмом классификации_.

Для подбора оптимального (минимизирующего эмпирический
риск _Q(w,X(ℓ))_ значения вектора весов w будем пользоваться методом
стохастического градиента — итерационный процесс, на каждом
шаге которого сдвигаемся в сторону противоположную вектору
градиента _Q′(w, X(ℓ))_ до тех пор, пока вектор весов _w_ не перестанет
изменяться, причем вычисления градиента производится не на всех
объектах обучения, а выбирается случайный объект (отсюда и название
метода _стохастический_), на основе которого и происходят
вычисления. В зависимости от функции потерь, которая используется
в функционале эмпирического риска, будем получать различные
линейные алгоритмы классификации.

При использовании метода стохастического градиента необходимо нормализовать данные:
```R
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
```

Реализация алгоритма:

```R
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
      ex <- lossFunction(margin)
      w <- w - eta * (wx - yi) * xi
      Qprev <- Q
      Q <- (1 - lambda) * Q + lambda * ex
    } else
    {
      break
    }
```

## Адалайн. Правило Хэбба
<a name="Адалайн"></a>
[К оглавлению](#Оглавление) 

Adaline или же Адаптивный линейный элемент - частный случай линейного классификатора или искусственной нейронной сети с одним слоем.
Возьмем функцию потерь ![alt text](https://github.com/nikitkuzi/ML1/blob/master/adaline/img/equation1.svg?raw=true),
Продифференцировав эту функцию, получим правило обновления весов ![alt text](https://github.com/nikitkuzi/ML1/blob/master/adaline/img/equation2.svg?raw=true)

![alt text](https://github.com/nikitkuzi/ML1/blob/master/adaline/img/equation3.jpg?raw=true)

Реализация:

```R
sgAdaline <- function(xl, eta = 1, lambda = 1 / 6)
{
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  w <- c(1 / 2, 1 / 2, 1 / 2)
  iterCount <- 0
  Q <- 0
  for (i in 1:l) {
    wx <- sum(w * xl[i, 1:n])
    margin <- wx * xl[i, n + 1]
    Q <- Q + lossFunction(margin)
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
      ex <- lossFunction(margin)
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
```

![alt text](https://github.com/nikitkuzi/ML1/blob/master/adaline/img/adaline.jpeg?raw=true)

## Правило Хэбба
[К оглавлению](#Оглавление) 

Персептрон обучают по правилу Хебба. Предъявляем на вход один объект. Если выходной сигнал персептрона совпадает с правильным ответом, то никаких действий предпринимать не надо. 
В случае ошибки необходимо обучить персептрон правильно решать данный пример.
Ошибки могут быть двух типов.

Первый тип ошибки: на выходе персептрона _a(xi)_ = 0, правильный ответ _yi_=1.

Для того, чтобы персептрон выдавал правильный ответ необходимо, чтобы скалярное произведение стало больше.
Поскольку переменные принимают значения 0 или 1, увеличение суммы может быть достигнуто за счет увеличения весов. 
Однако нет смысла увеличивать веса при переменных , которые равны нулю. 
Увеличиваем веса только при тех, которые равны 1. Для закрепления единичных сигналов с _ω_, следует провести ту же процедуру и на всех остальных слоях.

В этом и заключается первое правило Хэбба.

Второй тип ошибки: _a(xi)_ =1, _yi_=0.

Для уменьшения скалярного произведения в правой части, необходимо уменьшить веса связей при тех переменных , которые равны 1. Необходимо также провести эту процедуру для всех активных нейронов предыдущих слоев.

Формализуем данное правило. Пока будем считать признаки бинарными:
fj (x) ∈ {0, 1}, j = 1, . . . , n, yi ∈ {−1, +1}, i = 1, . . . , n. Тогда при классификации a(xi) объекта xi возможны следующие три случая:
1. Если ответ a(xi) совпадает с истинным yi
, то вектор весов изменять
не надо.
2. Если a(xi) = −1 и yi = 1, то вектор весов w увеличивается (можно
только те wj , для которых fj (xi) 6= 0) w = w + ηxi
, где η > 0 — темп
обучения.
3. Если a(xi) = 1 и yi = −1, то вектор весов w уменьшается: w = w−ηxi
.
Эти три случая объединяются в так называемое правило Хэбба.

Выберем функцию потерь:
```R
lossFunctionHeb <- function(x)
{
  return(if (x < 1) -x else 0)
}
```
Правило обновления весов:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/adaline/img/equation4.png?raw=true)


Получаем классификатор Хэбба:
```R
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
      w <- w + eta * yi * xi
      Qprev <- Q
      Q <- (1 - lambda) * Q + lambda * ex
    } else
    {
      break
    }
  }
  return(w)
}
```

Сравним его с Адаптивным линейным элементом, где крассная линия - Адаптивный, фиолетовая - Хэбба:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/adaline/img/adaline_heb.jpeg?raw=true)

## Логическая регрессия
<a name="Логическая"></a>
[К оглавлению](#Оглавление) 

Логистическая регрессия — линейный байесовский классификатор, использующий логарифмическую функцию потерь,
имеет ряд интересных особенностей, например, алгоритм способен помимо определения принадлежности объекта к классу определять и
степень его принадлежности. Является одним из популярных алгоритмом классификации.

Метод логистической регрессии основан на довольно сильных вероятностных
предположениях, которые имеют несколько интересных последствий:
1. линейный классификатор оказывается оптимальным байесовским;
2. однозначно определяется функция потерь;
3. можно вычислять не только принадлежность объектов классам, но
также получать и численные оценки вероятности их принадлежности.

Логистическое правило обновления весов для градиентного шага
в методе стохастического градиента:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/adaline/img/equation5.png?raw=true)
где ![alt text](https://github.com/nikitkuzi/ML1/blob/master/adaline/img/equation6.png?raw=true)

Функция потерь:
```R
lossFunctionLogical <- function(x)
{
  return(log2(1 + exp(-x)))
}
```

Сигмоидная функция:
```R
sigmoid <- function(z)
{
  return (1 / (1 + exp(-z)))
}
```

Реализация метода:
```R
sgLogical <- function(xl, eta = 1, lambda = 1 / 6)
{
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  w <- c(1 / 2, 1 / 2, 1 / 2)
  iterCount <- 0
  Q <- 0
  for (i in 1:l) {
    wx <- sum(w * xl[i, 1:n])
    margin <- wx * xl[i, n + 1]
    Q <- Q + lossFunctionLogical(margin)
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
      ex <- lossFunctionLogical(margin)
      w <- w + eta * xi * yi * sigmoid(-wx * yi)
      Qprev <- Q
      Q <- (1 - lambda) * Q + lambda * ex
    } else
    {
      break
    }
  }
  return(w)
}
```
![alt text](https://github.com/nikitkuzi/ML1/blob/master/adaline/img/logical.jpeg?raw=true)

Карта классификации:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/adaline/img/map.jpeg?raw=true)

Сравним все вышеупомянутые методы:
![alt text](https://github.com/nikitkuzi/ML1/blob/master/adaline/img/compare.jpeg?raw=true)