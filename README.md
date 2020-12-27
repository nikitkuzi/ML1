<a name="Оглавление"></a>
## Оглавление

1. Метрические алгоритмы классификации

   1.1.[Метод ближайшего соседа](#Метод_ближайшего_соседа)
  
   1.2. [Метод К соседей](#Метод_к_соседей)
   
   1.3. [Метод К взвешенных соседей](#Метод_к_взвешенных)

1. Байесовские алгоритмы классификации
1. Линейные алгоритмы классификации
  


<a name="Метод_ближайшего_соседа"></a>
## Метод ближайшего соседа
[Оглавление](#Оглавление)

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

[Оглавление](#Оглавление) 

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

[Оглавление](#Оглавление) 
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

Рассмотрим пример, показывающий преимущество _kwNN_ над _kNN_:

kNN:
<p float="left">
    <img src="/img/compare_kNN" width="100">
</p>

