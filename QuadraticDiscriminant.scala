import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors,Vector,Matrix,SingularValueDecomposition,DenseMatrix,DenseVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
    
class QuadraticDiscriminant (val sc: SparkContext, val x: RDD [LabeledPoint])
{
  val x_positive = x.filter {case LabeledPoint (l, f) => l == 1.0}
  val x_negative = x.filter {case LabeledPoint (l, f) => l != 1.0}
  
  val summary_positive = Statistics.colStats (x_positive.map {case LabeledPoint (l, f) => f})
  val summary_negative = Statistics.colStats (x_negative.map {case LabeledPoint (l, f) => f})
  
  val x0_positive = x_positive.map {case LabeledPoint (l, f) => (f.toArray zip summary_positive.mean.toArray).map {case (a, b) => a - b}}
  val x0_negative = x_negative.map {case LabeledPoint (l, f) => (f.toArray zip summary_negative.mean.toArray).map {case (a, b) => a - b}}
  
  val N_positive = x_positive.count
  val N_negative = x_negative.count
  val N = N_positive + N_negative
  
  val n = x.collect()(0).features.size
  
  val cov_positive = Array.ofDim [Double] (n, n)
  val cov_negative = Array.ofDim [Double] (n, n)
  
  for (i <- 0 until n)
  {
    cov_positive(i)(i) = (x0_positive.map (a => a(i)*a(i))).reduce (_ + _) / N
    cov_negative(i)(i) = (x0_negative.map (a => a(i)*a(i))).reduce (_ + _) / N
  
    for (j <- 0 until i)
    {
      cov_positive(i)(j) = (x0_positive.map (a => a(i)*a(j))).reduce (_ + _) / N
      cov_negative(i)(j) = (x0_negative.map (a => a(i)*a(j))).reduce (_ + _) / N
      cov_positive(j)(i) = cov_positive(i)(j)
      cov_negative(j)(i) = cov_negative(i)(j)
    }
  }
  
  val p_positive = N_positive / N.toDouble
  val p_negative = N_negative / N.toDouble
  
  val cov_positive_inverse = computeInverse (sc, cov_positive)
  val cov_negative_inverse = computeInverse (sc, cov_negative)

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
    val x_minus_mean_positive = subtract (x, summary_positive.mean)
    val log_pxc_positive = -1/2.0 * dot (x_minus_mean_positive, cov_positive_inverse.multiply (x_minus_mean_positive))

    val x_minus_mean_negative = subtract (x, summary_negative.mean)
    val log_pxc_negative = -1/2.0 * dot (x_minus_mean_negative, cov_negative_inverse.multiply (x_minus_mean_negative))

    log_pxc_positive - log_pxc_negative
  }

  def dot (x: Vector, y: Vector): Double =
  {
    (x.toArray zip y.toArray).map {case (a, b) => a*b}.reduce (_ + _)
  }
}
