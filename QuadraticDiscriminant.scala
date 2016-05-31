import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors,Vector,Matrix,SingularValueDecomposition,DenseMatrix,DenseVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import breeze.linalg.{DenseVector,DenseMatrix,inv,det}
    
case class QuadraticDiscrminantSufficientStatistics (
  N1 : Integer,
  N0 : Integer,
  X_sum1 : breeze.linalg.DenseVector,
  X_sum0 : breeze.linalg.DenseVector,
  X2_sum1 : breeze.linalg.DenseMatrix,
  X2_sum0 : breeze.linalg.DenseMatrix)

class QuadraticDiscriminant (val summary: QuadraticDiscrminantSufficientStatistics)
{
  val summary = summary

  def this (val summary: QuadraticDiscrminantSufficientStatistics, val X: RDD [LabeledPoint]) {
    val X1 = X.filter {case LabeledPoint (l, f) => l == 1.0}.map { case LabeledPoint (l, f) => breeze.linalg.DenseVector (f.toArray) }
    val X0 = X.filter {case LabeledPoint (l, f) => l != 1.0}.map { case LabeledPoint (l, f) => breeze.linalg.DenseVector (f.toArray) }
  
    val N1 = X1.count
    val N0 = X0.count

    val X_sum1 = X1.reduce (_ + _)
    val X_sum0 = X0.reduce (_ + _)

    val m = X1.take (1)(0).size
    val z = breeze.linalg.DenseMatrix.zeros[Double] (m, m)
    val X2_sum1 = X1.aggregate (z) ((S, v) => S + v * v.t, (S, T) => S + T)
    val X2_sum0 = X0.aggregate (z) ((S, v) => S + v * v.t, (S, T) => S + T)

    this.summary = QuadraticDiscrminantSufficientStatistics (
                     N1 + summary.N1,
                     N0 + summary.N0,
                     X_sum1 + summary.X_sum1,
                     X_sum0 + summary.X_sum0,
                     X2_sum1 + summary.X2_sum1,
                     X2_sum0 + summary.X2_sum0
                   )
  }

  val N = summary.N1 + summary.N0
  // HEY, IS THIS NEXT LINE STILL NEEDED ??
  val n = max (summary.X_sum1.size, summary.X_sum0.size) // WHAT IF ONE OR THE OTHER IS NULL ??
  
  val mean1 = (1/summary.N1) * summary.X_sum1
  val mean0 = (1/summary.N0) * summary.X_sum0

  val cov1 = (1/summary.N1) * summary.X2_sum1 - mean1 * mean1.t
  val cov0 = (1/summary.N0) * summary.X2_sum0 - mean1 * mean1.t

  val cov1_det = cov1.det
  val cov0_det = cov0.det
  
  val cov1_inverse = inv (cov1)
  val cov0_inverse = inv (cov0)

  val p1 = summary.N1 / summary.N.toDouble
  val p0 = summary.N0 / summary.N.toDouble
  
  def score (x: org.apache.spark.mllib.linalg.Vector): Double =
  {
    val v = breeze.linalg.DenseVector (x.toArray)
    val m = v.size

    val u1 = v - summary.mean1
    val qf1 = u1.t * (summary.cov1_inverse * u1);
    val log_pxc1 = -1/2.0 * qf1 - m/2.0 * Math.log (2*Math.PI) - 1/2.0 * Math.log (cov1_det)

    val u0 = v - summary.mean0
    val qf0 = u0.t * (summary.cov0_inverse * u0);
    val log_pxc0 = -1/2.0 * qf0 - m/2.0 * Math.log (2*Math.PI) - 1/2.0 * Math.log (cov0_det)

    val log_pc1 = Math.log (summary.p1)
    val log_pc0 = Math.log (summary.p0)

    // Note that p(c=1|x) = p(x|c=1) p(c=1) / p(x), and p(c=0|x) = p(x|c=0) p(c=0) / p(x).
    // Therefore the posterior odds p(c=1|x) / p(c=0|x) = p(x|c=1) p(c=1) / (p(c=0|x) p(c=0))
    // and therefore the posterior log-odds
    // log(p(c=1|x) / p(c=0|x)) = log(p(x|c=1)) + log(p(c=1)) - log(p(x|c=0)) - log(p(c=0))
    //                          = (log(p(x|c=1)) - log(p(x|c=0))) + (log(p(c=1)) - log(p(c=0))).

    (log_pxc1 - log_pxc0) + (log_pc1 - log_pc0)
  }
}
