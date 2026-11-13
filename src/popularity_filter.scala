import org.apache.spark.mllib.recommendation.{ALS, Rating}

val raw = sc.textFile("hdfs://master:9000/user/root/movielens/ratings.dat")
  .map(_.split("::")).map(a => Rating(a(0).toInt, a(1).toInt, a(2).toDouble)).cache()
val mv = sc.textFile("hdfs://master:9000/user/root/movielens/movies.dat")
  .map(_.split("::")).map(a => (a(0).toInt, a(1))).collectAsMap()

val Array(train, test) = raw.randomSplit(Array(0.8, 0.2), 42L)
train.cache()
val cnt = train.map(r => (r.product, 1L)).reduceByKey(_ + _).collectAsMap()
println("ITEMS_TOTAL " + cnt.size)
for (th <- Seq(20, 50, 100, 200, 500)) {
  println("ELIGIBLE th=" + th + " n=" + cnt.count(_._2 >= th))
}

val model = ALS.train(train, 10, 10, 0.01)
println("MODEL_READY")

for (u <- Seq(1, 100, 6040)) {
  val recs = model.recommendProducts(u, 200)
  for (th <- Seq(0, 50, 100)) {
    val f = if (th == 0) recs else recs.filter(p => cnt.getOrElse(p.product, 0L) >= th)
    val txt = f.take(5).map(p => mv.getOrElse(p.product, "?") + "[" + cnt.getOrElse(p.product, 0L) + "] " + "%.2f".format(p.rating)).mkString(" | ")
    println("REC u=" + u + " th=" + th + " -> " + txt)
  }
}
println("PARTB_DONE")
