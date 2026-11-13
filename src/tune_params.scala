// 超参数对比：隐因子维度 rank 与正则化系数 lambda
// 运行：spark-shell --master spark://master:7077 -i /root/tune_params.scala

import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.ml.evaluation.RegressionEvaluator

val HDFS = "hdfs://master:9000/user/root/movielens"

val ratings = sc.textFile(s"$HDFS/ratings.dat")
  .map(_.split("::"))
  .map(a => (a(0).toInt, a(1).toInt, a(2).toFloat))
  .toDF("userId", "movieId", "rating")

val Array(train, test) = ratings.randomSplit(Array(0.8, 0.2), seed = 42L)
train.cache()
test.cache()

def fitEval(rank: Int, iter: Int, reg: Double): (ALSModel, Double) = {
  val als = new ALS()
    .setRank(rank)
    .setMaxIter(iter)
    .setRegParam(reg)
    .setUserCol("userId")
    .setItemCol("movieId")
    .setRatingCol("rating")
    .setColdStartStrategy("drop")
  val m = als.fit(train)
  val e = new RegressionEvaluator()
    .setMetricName("rmse")
    .setLabelCol("rating")
    .setPredictionCol("prediction")
    .evaluate(m.transform(test))
  (m, e)
}

// rank 对比
for (r <- Seq(5, 10, 20, 50)) {
  val (_, e) = fitEval(r, 10, 0.01)
  println(f"RANK_RESULT,$r,$e%.4f")
}

// 正则化对比
for (l <- Seq(0.001, 0.01, 0.1, 1.0)) {
  val (_, e) = fitEval(20, 10, l)
  println(f"LAMBDA_RESULT,$l%.3f,$e%.4f")
}

// 迭代次数对比
for (it <- Seq(10, 20)) {
  val t0 = System.currentTimeMillis()
  val (_, e) = fitEval(10, it, 0.01)
  val ms = System.currentTimeMillis() - t0
  println(f"ITER_RESULT,$it,$ms,$e%.4f")
}
