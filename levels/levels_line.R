density <- function(x, Mat_expect, Sigma) {
  n = dim(Sigma)[1]
  det = det(Sigma)
  left <- 1/(sqrt((2 * pi)^ n * det))
  right <- exp(1 / -2 * t(x - Mat_expect) %*% ginv(Sigma) %*% (x - Mat_expect))
  return(left * right)
}

draw_levels <- function(Mat_expect, Sigma) {
  left_x <- Mat_expect[1] - (Sigma[1, 1] + 1)
  right_x <- Mat_expect[1] + (Sigma[1, 1] + 1)
  left_y <- Mat_expect[2] - (Sigma[2, 2] + 1)
  right_y <- Mat_expect[2] + (Sigma[2, 2] + 1)
  x = seq(left_x, right_x, 0.05)
  y = seq(left_y, right_y, 0.05)
  z = outer(x,y,function(x,y) {
    unlist(lapply(1:length(x), function(i) density(c(x[i],y[i]), Mat_expect, Sigma)))
  })
  contour(x,y,z)
}

Mat_expect <- matrix(c(3, 3), nrow = 2, ncol = 1)
Sigma <- matrix(c(2, 1, 1, 2), nrow = 2, ncol = 2)
draw_levels(Mat_expect, Sigma)
ne_kor