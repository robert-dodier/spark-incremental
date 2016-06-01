import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{inv,det}
    
case class QuadraticDiscrminantSufficientStatistics (
  N : Long,
  N1 : Long,
  N0 : Long,
  X1_sum : breeze.linalg.DenseVector [Double],
  X0_sum : breeze.linalg.DenseVector [Double],
  X1_sum2 : breeze.linalg.DenseMatrix [Double],
  X0_sum2 : breeze.linalg.DenseMatrix [Double])

object QuadraticDiscrminantSufficientStatistics {
  def zeros (m: Integer) = {
    val z = breeze.linalg.DenseVector.zeros [Double] (m)
    val zz = breeze.linalg.DenseMatrix.zeros [Double] (m, m)
    new QuadraticDiscrminantSufficientStatistics (0L, 0L, 0L, z, z, zz, zz)
  }
}

class QuadraticDiscriminant (previousSummary: QuadraticDiscrminantSufficientStatistics, X: RDD [LabeledPoint]) {

  def this (X: RDD [LabeledPoint]) = this (QuadraticDiscrminantSufficientStatistics.zeros (X.take (1)(0).features.size), X)

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

  val N = N1 + N0
  
  val summary = new QuadraticDiscrminantSufficientStatistics (
                   N + previousSummary.N,
                   N1 + previousSummary.N1,
                   N0 + previousSummary.N0,
                   X1_sum + previousSummary.X1_sum,
                   X0_sum + previousSummary.X0_sum,
                   X1_sum2 + previousSummary.X1_sum2,
                   X0_sum2 + previousSummary.X0_sum2
                 )

  val mean1 = (1.0/summary.N1) * summary.X1_sum
  val mean0 = (1.0/summary.N0) * summary.X0_sum

  val cov1 = (1.0/summary.N1) * summary.X1_sum2 - mean1 * mean1.t
  val cov0 = (1.0/summary.N0) * summary.X0_sum2 - mean0 * mean0.t

  val cov1_det = det (cov1)
  val cov0_det = det (cov0)
  
  val cov1_inv = inv (cov1)
  val cov0_inv = inv (cov0)

  val pc1 = summary.N1 / summary.N.toDouble
  val pc0 = summary.N0 / summary.N.toDouble
  
  def score (x: org.apache.spark.mllib.linalg.Vector): Double = {

    val v = breeze.linalg.DenseVector (x.toArray)
    val m = v.size

    val u1 = v - mean1
    val qf1 = u1.t * (cov1_inv * u1);
    val log_pxc1 = -1/2.0 * qf1 - m/2.0 * Math.log (2*Math.PI) - 1/2.0 * Math.log (cov1_det)

    val u0 = v - mean0
    val qf0 = u0.t * (cov0_inv * u0);
    val log_pxc0 = -1/2.0 * qf0 - m/2.0 * Math.log (2*Math.PI) - 1/2.0 * Math.log (cov0_det)

    val log_pc1 = Math.log (pc1)
    val log_pc0 = Math.log (pc0)

    // Note that p(c=1|x) = p(x|c=1) p(c=1) / p(x), and p(c=0|x) = p(x|c=0) p(c=0) / p(x).
    // Therefore the posterior odds p(c=1|x) / p(c=0|x) = p(x|c=1) p(c=1) / (p(c=0|x) p(c=0))
    // and therefore the posterior log-odds
    // log(p(c=1|x) / p(c=0|x)) = log(p(x|c=1)) + log(p(c=1)) - log(p(x|c=0)) - log(p(c=0))
    //                          = (log(p(x|c=1)) - log(p(x|c=0))) + (log(p(c=1)) - log(p(c=0))).

    (log_pxc1 - log_pxc0) + (log_pc1 - log_pc0)
  }
}

object QuadraticDiscriminant {

  def example (sc: org.apache.spark.SparkContext) = {
    import org.apache.spark.mllib.util.MLUtils
    val X_all = MLUtils.loadLibSVMFile (sc, "xyc.data-all-libsvm")
    val qd_all = new QuadraticDiscriminant (X_all)

    val X_99pct = MLUtils.loadLibSVMFile (sc, "xyc.data-99%-libsvm")
    val qd_99pct = new QuadraticDiscriminant (X_99pct)
    val X_1pct = MLUtils.loadLibSVMFile (sc, "xyc.data-1%-libsvm")
    val qd_1pct_only = new QuadraticDiscriminant (X_1pct)
    val qd_1pct_plus_99pct = new QuadraticDiscriminant (qd_99pct.summary, X_1pct)

    println ("example: should find summaries of all and 1% + 99% are the same.")
    println ("example: qd_all.summary = ")
    println (qd_all.summary)
    println ("example: qd_1pct_plus_99pct.summary = ")
    println (qd_1pct_plus_99pct.summary)
    println ("example: ... while summaries of 99% and 1% are different from each other and from the preceding.")
    println ("example: qd_99pct.summary = ")
    println (qd_99pct.summary)
    println ("example: qd_1pct_only.summary = ")
    println (qd_1pct_only.summary)

    println ("example: generate data for contour plots.")

    val ((x_ll_all, y_ll_all), (x_ur_all, y_ur_all)) = extract_corners (qd_all)
    val ((x_ll_1pct_plus_99pct, y_ll_1pct_plus_99pct), (x_ur_1pct_plus_99pct, y_ur_1pct_plus_99pct)) = extract_corners (qd_1pct_plus_99pct)
    val ((x_ll_99pct, y_ll_99pct), (x_ur_99pct, y_ur_99pct)) = extract_corners (qd_99pct)
    val ((x_ll_1pct_only, y_ll_1pct_only), (x_ur_1pct_only, y_ur_1pct_only)) = extract_corners (qd_1pct_only)

    // put all plots on same (x, y) grid
    // ll = lower left, ur = upper right

    val x_ll = Math.min (Math.min (Math.min (x_ll_all, x_ll_1pct_plus_99pct), x_ll_99pct), x_ll_1pct_only)
    val y_ll = Math.min (Math.min (Math.min (y_ll_all, y_ll_1pct_plus_99pct), y_ll_99pct), y_ll_1pct_only)
    val x_ur = Math.max (Math.max (Math.max (x_ur_all, x_ur_1pct_plus_99pct), x_ur_99pct), x_ur_1pct_only)
    val y_ur = Math.max (Math.max (Math.max (y_ur_all, y_ur_1pct_plus_99pct), y_ur_99pct), y_ur_1pct_only)

    generate_data (qd_all, x_ll, y_ll, x_ur, y_ur, 50, "qd_all.contour-data")
    generate_data (qd_1pct_plus_99pct, x_ll, y_ll, x_ur, y_ur, 50, "qd_1pct_plus_99pct.contour-data")
    generate_data (qd_99pct, x_ll, y_ll, x_ur, y_ur, 50, "qd_99pct.contour-data")
    generate_data (qd_1pct_only, x_ll, y_ll, x_ur, y_ur, 50, "qd_1pct_only.contour-data")
  }

  def extract_corners (qd: QuadraticDiscriminant): ((Double, Double), (Double, Double)) = {
    val x_ll = Math.min (qd.mean0(0) - 3*Math.sqrt(qd.cov0(0,0)), qd.mean1(0) - 3*Math.sqrt(qd.cov1(0,0)))
    val y_ll = Math.min (qd.mean0(1) - 3*Math.sqrt(qd.cov0(1,1)), qd.mean1(1) - 3*Math.sqrt(qd.cov1(1,1)))
    val x_ur = Math.max (qd.mean0(0) + 3*Math.sqrt(qd.cov0(0,0)), qd.mean1(0) + 3*Math.sqrt(qd.cov1(0,0)))
    val y_ur = Math.max (qd.mean0(1) + 3*Math.sqrt(qd.cov0(1,1)), qd.mean1(1) + 3*Math.sqrt(qd.cov1(1,1)))
    ((x_ll, y_ll), (x_ur, y_ur))
  }

  def generate_data (qd: QuadraticDiscriminant, x_ll: Double, y_ll: Double, x_ur: Double, y_ur: Double,  n: Integer, filename: String) = {
    val n = 50
    val dx = (x_ur - x_ll)/n
    val dy = (y_ur - y_ll)/n
    val out = new java.io.PrintStream (new java.io.FileOutputStream (filename))

    out.print ("0")
    for (i <- 0 to n)
      out.print (", " + (x_ll + i*dx))
    out.println

    for (j <- 0 to n) {
      val y = y_ll + j*dy
      out.print (y)
      for (i <- 0 to n) {
        val x = x_ll + i*dx
        val p = qd.score (org.apache.spark.mllib.linalg.Vectors.dense (x, y))
        out.print (", " + p)
      }
      out.println
    }
    out.close
  }
}
