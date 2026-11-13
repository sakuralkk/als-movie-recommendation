import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.evaluation.RegressionMetrics

val raw = sc.textFile("hdfs://master:9000/user/root/movielens/ratings.dat")
  .map(_.split("::")).map(a => Rating(a(0).toInt, a(1).toInt, a(2).toDouble)).cache()
println("DATA_READY " + raw.count())

for (seed <- Seq(42, 7, 123, 2024, 999)) {
  val Array(tr, te) = raw.randomSplit(Array(0.8, 0.2), seed)
  for (it <- Seq(10, 20)) {
    val m = ALS.train(tr, 10, it, 0.01)
    val pred = m.predict(te.map(r => (r.user, r.product))).map(p => ((p.user, p.product), p.rating))
    val truth = te.map(r => ((r.user, r.product), r.rating))
    val met = new RegressionMetrics(truth.join(pred).map(x => (x._2._2, x._2._1)))
    println(f"SEED seed=$seed%d iter=$it%d rmse=${met.rootMeanSquaredError}%.4f")
  }
}
println("SEED_DONE")
