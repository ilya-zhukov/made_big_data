package org.apache.spark.ml.made
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector, Vectors}

case class testRandomHyperLSH() extends AnyFlatSpec with should.Matchers {
  val delta = 0.0001
  val spark = SparkSession.builder
    .appName("Simple Application")
    .master("local[4]")
    .getOrCreate()

  val sqlc = spark.sqlContext

  import sqlc.implicits._

  "model" should "should predict" in {
    val df0 = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0)),
      (1, Vectors.dense(1.0, -1.0)),
      (2, Vectors.dense(-1.0, -1.0)),
      (3, Vectors.dense(-1.0, 1.0)),
      (4, Vectors.dense(2.0, 1.0)),
      (5, Vectors.dense(4.0, -5.0)),
      (6, Vectors.dense(-1.0, -2.0)),
      (7, Vectors.dense(-2.0, 1.0)),
      (8, Vectors.dense(3.0, -3.0)),
      (9, Vectors.dense(1.0, -1.2)),
      (10, Vectors.dense(1.0, 0.0))
    )).toDF("id", "features")

    val rh_t = new RandomHyperLSH().setNumHashTables(5)
      .setInputCol("features").setOutputCol("hashValues")
    val rhModel_t = rh_t.fit(df0)

    val rhModel_t1 =rhModel_t.transform(df0)

    val key = Vectors.dense(0.5, -0.5) // 1 8
    val near = rhModel_t.approxNearestNeighbors(df0, key, 2)


    near.select("distCol").collect()(0).getDouble(0) should be(0.0 +- delta)

  }
}
