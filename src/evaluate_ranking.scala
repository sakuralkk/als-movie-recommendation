import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.evaluation.RankingMetrics

val raw = sc.textFile("hdfs://master:9000/user/root/movielens/ratings.dat").map(_.split("::")).map(a => Rating(a(0).toInt, a(1).toInt, a(2).toDouble)).cache()
val mv = sc.textFile("hdfs://master:9000/user/root/movielens/movies.dat").map(_.split("::")).map(a => (a(0).toInt, a(1))).collectAsMap()
val Array(train, test) = raw.randomSplit(Array(0.8, 0.2), seed = 42L)
train.cache(); test.cache()
val model = ALS.train(train, 10, 10, 0.01)
println("MODEL_READY")

val predRDD = model.recommendProductsForUsers(10).mapValues(_.map(_.product))
val actRDD = test.groupBy(_.user).mapValues(_.map(_.product).toArray)
val joined = predRDD.join(actRDD).map(x => (x._2._1, x._2._2))
val rm = new RankingMetrics(joined)
println(f"RANK10 precision=${rm.precisionAt(10)}%.4f ndcg=${rm.ndcgAt(10)}%.4f")
val rm5 = new RankingMetrics(joined.map(x => (x._1.take(5), x._2)))
println(f"RANK5 precision=${rm5.precisionAt(5)}%.4f")

val stat = raw.map(r => (r.product, (r.rating, 1L))).reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).collectAsMap()
val targets = Seq("Ulysses (Ulisse)", "Schindler", "Mamma Roma", "Smashing Time", "Saltmen of Tibet", "American Beauty", "Shawshank")
for (t <- targets) {
  mv.filter(_._2.contains(t)).foreach { case (mid, title) =>
    val (sm, c) = stat(mid)
    println(f"MOVIE $title%s id=$mid cnt=$c%d avg=${sm / c}%.2f")
  }
}
println("EXP3_DONE")
