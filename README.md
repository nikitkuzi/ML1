<a name="Оглавление"></a>
## Оглавление
1.[Метрические алгоритмы классификации](#Метрические)
  
  1.1.[Метод ближайшего соседа](#Метод_ближайшего_соседа)
<a name="Метод_ближайшего_соседа"></a>

## 1nn
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
В результате работы программы мы получаем график, на котором круглые точки - выборка данных из датасета Iris, а квадратные точки - объекты из тестовой выборки.
![alt text](https://github.com/nikitkuzi/ML1/blob/master/kNN/img/1nn.jpeg?raw=true)

# LOO
При использовании метода ближайших соседей (kNN) возникает вопрос: как выбрать оптимальное k? 

При k = 1 получаем метод ближайшего соседа
и, соответственно, неустойчивость к шуму, при k = l, наоборот, алгоритм
чрезмерно устойчив и вырождается в константу. Таким образом, крайние значения k нежелательны. На практике оптимальное k подбирается по критерию
скользящего контроля LOO.

Преимущества LOO в том, что каждый объект ровно один раз участвует в контроле, а длина обучающих подвыборок лишь на единицу меньше длины полной выборки.

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
Для каждого k находится ошибка классификации и в качестве оптимального k выбирается то значение, для которого минимальна ошибка классфикации.

![alt text](https://github.com/nikitkuzi/ML1/blob/master/img/Loo.jpeg?raw=true)
