import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{inv,det}
    
case class QuadraticDiscrminantSufficientStatistics (
  N1 : Integer,
  N0 : Integer,
  X1_sum : breeze.linalg.DenseVector,
  X0_sum : breeze.linalg.DenseVector,
  X1_sum2 : breeze.linalg.DenseMatrix,
  X0_sum2 : breeze.linalg.DenseMatrix)

class QuadraticDiscriminant (val summary: QuadraticDiscrminantSufficientStatistics)
{
  val summary = summary

  def this (val summary: QuadraticDiscrminantSufficientStatistics, val X: RDD [LabeledPoint]) {
    val X1 = X.filter {case LabeledPoint (l, f) => l == 1.0}.map { case LabeledPoint (l, f) => breeze.linalg.DenseVector (f.toArray) }
    val X0 = X.filter {case LabeledPoint (l, f) => l != 1.0}.map { case LabeledPoint (l, f) => breeze.linalg.DenseVector (f.toArray) }
  
    val N1 = X1.count
    val N0 = X0.count

    val X1_sum = X1.reduce (_ + _)
    val X0_sum = X0.reduce (_ + _)

    val m = X1.take (1)(0).size
    val z = breeze.linalg.DenseMatrix.zeros[Double] (m, m)
    val X1_sum2 = X1.aggregate (z) ((S, v) => S + v * v.t, (S, T) => S + T)
    val X0_sum2 = X0.aggregate (z) ((S, v) => S + v * v.t, (S, T) => S + T)

    this.summary = QuadraticDiscrminantSufficientStatistics (
                     N1 + summary.N1,
                     N0 + summary.N0,
                     X1_sum + summary.X1_sum,
                     X0_sum + summary.X0_sum,
                     X1_sum2 + summary.X1_sum2,
                     X0_sum2 + summary.X0_sum2
                   )
  }

  val N = summary.N1 + summary.N0
  
  val mean1 = (1.0/summary.N1) * summary.X1_sum
  val mean0 = (1.0/summary.N0) * summary.X0_sum

  val cov1 = (1.0/summary.N1) * summary.X1_sum2 - mean1 * mean1.t
  val cov0 = (1.0/summary.N0) * summary.X0_sum2 - mean0 * mean0.t

  val cov1_det = cov1.det
  val cov0_det = cov0.det
  
  val cov1_inv = inv (cov1)
  val cov0_inv = inv (cov0)

  val pc1 = summary.N1 / summary.N.toDouble
  val pc0 = summary.N0 / summary.N.toDouble
  
  def score (x: org.apache.spark.mllib.linalg.Vector): Double =
  {
    val v = breeze.linalg.DenseVector (x.toArray)
    val m = v.size

    val u1 = v - summary.mean1
    val qf1 = u1.t * (summary.cov1_inv * u1);
    val log_pxc1 = -1/2.0 * qf1 - m/2.0 * Math.log (2*Math.PI) - 1/2.0 * Math.log (cov1_det)

    val u0 = v - summary.mean0
    val qf0 = u0.t * (summary.cov0_inv * u0);
    val log_pxc0 = -1/2.0 * qf0 - m/2.0 * Math.log (2*Math.PI) - 1/2.0 * Math.log (cov0_det)

    val log_pc1 = Math.log (summary.pc1)
    val log_pc0 = Math.log (summary.pc0)

    // Note that p(c=1|x) = p(x|c=1) p(c=1) / p(x), and p(c=0|x) = p(x|c=0) p(c=0) / p(x).
    // Therefore the posterior odds p(c=1|x) / p(c=0|x) = p(x|c=1) p(c=1) / (p(c=0|x) p(c=0))
    // and therefore the posterior log-odds
    // log(p(c=1|x) / p(c=0|x)) = log(p(x|c=1)) + log(p(c=1)) - log(p(x|c=0)) - log(p(c=0))
    //                          = (log(p(x|c=1)) - log(p(x|c=0))) + (log(p(c=1)) - log(p(c=0))).

    (log_pxc1 - log_pxc0) + (log_pc1 - log_pc0)
  }
}
