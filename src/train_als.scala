// 基准模型：训练 ALS 并评估 RMSE
// 运行：spark-shell --master spark://master:7077 -i /root/train_als.scala

import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator

val HDFS = "hdfs://master:9000/user/root/movielens"

// 解析以 :: 分隔的评分文件
val ratings = sc.textFile(s"$HDFS/ratings.dat")
  .map(_.split("::"))
  .map(a => (a(0).toInt, a(1).toInt, a(2).toFloat))
  .toDF("userId", "movieId", "rating")
  .cache()

println("RATING_COUNT=" + ratings.count())
println("USER_COUNT=" + ratings.select("userId").distinct().count())
println("MOVIE_COUNT=" + ratings.select("movieId").distinct().count())
println(f"AVG_RATING=${ratings.agg(avg("rating")).first().getDouble(0)}%.4f")

// 8:2 切分，固定种子保证可复现
val Array(train, test) = ratings.randomSplit(Array(0.8, 0.2), seed = 42L)
println(s"TRAIN=${train.count()} TEST=${test.count()}")

val t0 = System.currentTimeMillis()
val als = new ALS()
  .setRank(10)
  .setMaxIter(10)
  .setRegParam(0.01)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")
  .setColdStartStrategy("drop")

val model = als.fit(train)
val rmse = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("rating")
  .setPredictionCol("prediction")
  .evaluate(model.transform(test))

println("TRAIN_MS=" + (System.currentTimeMillis() - t0))
println(f"RMSE=$rmse%.4f")
