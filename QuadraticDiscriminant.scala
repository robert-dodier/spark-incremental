import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors,Vector,Matrix,SingularValueDecomposition,DenseMatrix,DenseVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import breeze.linalg.{DenseVector,DenseMatrix,inv}
    
case class QuadraticDiscrminantSufficientStatistics (
  N_positive : Integer,
  N_negative : Integer,
  X_sum_positive : breeze.linalg.DenseVector,
  X_sum_negative : breeze.linalg.DenseVector,
  X2_sum_positive : breeze.linalg.DenseMatrix,
  X2_sum_negative : breeze.linalg.DenseMatrix)

class QuadraticDiscriminant (val summary: QuadraticDiscrminantSufficientStatistics)
{
  val summary = summary

  def this (val summary: QuadraticDiscrminantSufficientStatistics, val X: RDD [LabeledPoint]) {
    val X_positive = X.filter {case LabeledPoint (l, f) => l == 1.0}.map { case LabeledPoint (l, f) => breeze.linalg.DenseVector (f.toArray) }
    val X_negative = X.filter {case LabeledPoint (l, f) => l != 1.0}.map { case LabeledPoint (l, f) => breeze.linalg.DenseVector (f.toArray) }
  
    val N_positive = X_positive.count
    val N_negative = X_negative.count

    val X_sum_positive = X_positive.reduce (_ + _)
    val X_sum_negative = X_negative.reduce (_ + _)

    val m = X_positive.take (1)(0).size
    val z = breeze.linalg.DenseMatrix.zeros[Double] (m, m)
    val X2_sum_positive = X_positive.aggregate (z) ((S, v) => S + v * v.t, (S, T) => S + T)
    val X2_sum_negative = X_negative.aggregate (z) ((S, v) => S + v * v.t, (S, T) => S + T)

    this.summary = QuadraticDiscrminantSufficientStatistics (
                     N_positive + summary.N_positive,
                     N_negative + summary.N_negative,
                     X_sum_positive + summary.X_sum_positive,
                     X_sum_negative + summary.X_sum_negative,
                     X2_sum_positive + summary.X2_sum_positive,
                     X2_sum_negative + summary.X2_sum_negative
                   )
  }

  val N = summary.N_positive + summary.N_negative
  // HEY, IS THIS NEXT LINE STILL NEEDED ??
  val n = max (summary.X_sum_positive.size, summary.X_sum_negative.size) // WHAT IF ONE OR THE OTHER IS NULL ??
  
  val mean_positive = (1/summary.N_positive) * summary.X_sum_positive
  val mean_negative = (1/summary.N_negative) * summary.X_sum_negative

  val cov_positive = (1/summary.N_positive) * summary.X2_sum_positive - mean_positive * mean_positive.t
  val cov_negative = (1/summary.N_negative) * summary.X2_sum_negative - mean_positive * mean_positive.t
  
  val cov_positive_inverse = inv (cov_positive)
  val cov_negative_inverse = inv (cov_negative)

  val p_positive = summary.N_positive / summary.N.toDouble
  val p_negative = summary.N_negative / summary.N.toDouble
  
  def score (x: Vector): Double =
  {
    val X_minus_mean_positive = subtract (x, colstats_positive.mean)
    val log_pxc_positive = -1/2.0 * dot (X_minus_mean_positive, cov_positive_inverse.multiply (X_minus_mean_positive))

    val X_minus_mean_negative = subtract (x, colstats_negative.mean)
    val log_pxc_negative = -1/2.0 * dot (X_minus_mean_negative, cov_negative_inverse.multiply (X_minus_mean_negative))

    log_pxc_positive - log_pxc_negative
  }
}
