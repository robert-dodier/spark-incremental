import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors,Vector,Matrix,SingularValueDecomposition,DenseMatrix,DenseVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
    
case class QuadraticDiscrminantSufficientStatistics {
  val N_positive
  val N_negative
  val X_sum_positive
  val X_sum_negative
  val X2_sum_positive
  val X2_sum_negative
}

class QuadraticDiscriminant (val sc: SparkContext, val summary: QuadraticDiscrminantSufficientStatistics)
{
  val summary = summary

  def this (val sc: SparkContext, val summary: QuadraticDiscrminantSufficientStatistics, val x: RDD [LabeledPoint]) {
    val X_positive = x.filter {case LabeledPoint (l, f) => l == 1.0}
    val X_negative = x.filter {case LabeledPoint (l, f) => l != 1.0}
  
    val colstats_positive = Statistics.colStats (X_positive.map {case LabeledPoint (l, f) => f})
    val colstats_negative = Statistics.colStats (X_negative.map {case LabeledPoint (l, f) => f})
  
    val x0_positive = X_positive.map {case LabeledPoint (l, f) => (f.toArray zip colstats_positive.mean.toArray).map {case (a, b) => a - b}}
    val x0_negative = X_negative.map {case LabeledPoint (l, f) => (f.toArray zip colstats_negative.mean.toArray).map {case (a, b) => a - b}}
  
    val N_positive = X_positive.count
    val N_negative = X_negative.count

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
  val n = max (summary.X_sum_positive.size, summary.X_sum_negative.size) // WHAT IF ONE OR THE OTHER IS NULL ??
  
  val mean_positive = (1/summary.N_positive) * summary.X_sum_positive
  val mean_negative = (1/summary.N_negative) * summary.X_sum_negative

  val cov_positive = (1/summary.N_positive) * summary.X2_sum_positive - (mean_positive)^^2
  val cov_negative = (1/summary.N_negative) * summary.X2_sum_negative - (mean_negative)^^2
  
  val cov_positive_inverse = computeInverse (sc, cov_positive)
  val cov_negative_inverse = computeInverse (sc, cov_negative)

  val p_positive = summary.N_positive / summary.N.toDouble
  val p_negative = summary.N_negative / summary.N.toDouble
  
  def computeInverse (sc: SparkContext, a: Array [Array [Double]]): DenseMatrix =
  {
    val rows = sc.parallelize (a.map (r => Vectors.dense (r)))
    computeInverse (new RowMatrix (rows))
  }

  // nicked from: http://stackoverflow.com/questions/29969521/how-to-compute-the-inverse-of-a-rowmatrix-in-apache-spark
  def computeInverse(X: RowMatrix): DenseMatrix = {
    val nCoef = X.numCols.toInt
    val svd = X.computeSVD(nCoef, computeU = true)
    if (svd.s.size < nCoef) {
      sys.error(s"RowMatrix.computeInverse called on singular matrix.")
    }
  
    // Create the inv diagonal matrix from S 
    val invS = DenseMatrix.diag(new DenseVector(svd.s.toArray.map(x => math.pow(x,-1))))
  
    // U cannot be a RowMatrix
    val U = new DenseMatrix(svd.U.numRows().toInt,svd.U.numCols().toInt,svd.U.rows.collect.flatMap(x => x.toArray))
  
    // If you could make V distributed, then this may be better. However its alreadly local...so maybe this is fine.
    val V = svd.V
    // inv(X) = V*inv(S)*transpose(U)  --- the U is already transposed.
    (V.multiply(invS)).multiply(U)
  }

  def score (x: Vector): Double =
  {
    val X_minus_mean_positive = subtract (x, colstats_positive.mean)
    val log_pxc_positive = -1/2.0 * dot (X_minus_mean_positive, cov_positive_inverse.multiply (X_minus_mean_positive))

    val X_minus_mean_negative = subtract (x, colstats_negative.mean)
    val log_pxc_negative = -1/2.0 * dot (X_minus_mean_negative, cov_negative_inverse.multiply (X_minus_mean_negative))

    log_pxc_positive - log_pxc_negative
  }

  def dot (x: Vector, y: Vector): Double =
  {
    (x.toArray zip y.toArray).map {case (a, b) => a*b}.reduce (_ + _)
  }
}
