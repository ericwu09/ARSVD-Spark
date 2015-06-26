import java.io.IOException;
import java.util.ArrayList;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.apache.spark.storage.StorageLevel;

import scala.Serializable;
import scala.Tuple2;



public class BlockARSVD implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3507492113920095993L;
	private IndexedRowMatrix U;
	private Vector s;
	private Matrix V;
	private final int numPartitions = 8;
	private final int numSlices = 3;
	//private final StorageLevel storage = StorageLevel.MEMORY_ONLY_SER(); // Serialized
	private final StorageLevel storage = StorageLevel.MEMORY_ONLY(); // Raw

	public BlockARSVD(String filename, int dstar, int power_iters) {
		//JavaSparkContext sc = new JavaSparkContext("local[4]", "ARSVD",
	    //	      "/srv/spark", new String[]{"target/arsvd.jar"});
		SparkConf conf = new SparkConf();
		conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
		conf.setAppName("ARSVD");
		conf.setMaster("local[4]");
		conf.set("spark.executor.memory", "6g");
		conf.set("spark.driver.maxResultSize", "1g");
		conf.set("spark.shuffle.spill", "false");
		conf.set("spark.storage.memoryFraction", "0.3");
		conf.set("spark.eventLog.enabled", "false");
		conf.set("driver-memory", "4g");
		JavaSparkContext sc = new JavaSparkContext(conf);
		
	    JavaRDD<String> csv = sc.textFile(filename).persist(storage).coalesce(numPartitions);
	    
	    JavaRDD<IndexedRow> entries = csv.zipWithIndex().map(
	    		  new  Function<scala.Tuple2<String, Long>, IndexedRow>() {
					private static final long serialVersionUID = 4795273163954440089L;
					@Override
					public IndexedRow call(Tuple2<String, Long> tuple)
							throws Exception {
						String line = tuple._1;
						long index = tuple._2;
						String[] strings = line.split(",");
						double[] doubles = new double[strings.length];
	    		         for (int i = 0; i < strings.length; i++) {
	    		        	 doubles[i] = Double.parseDouble(strings[i]);
	    		         }
	    		         return new IndexedRow(index, new DenseVector(doubles));
					}
	    		});
	    	    
	    BlockMatrix x = new IndexedRowMatrix(entries.rdd()).toBlockMatrix();
		BlockMatrix xT = x.transpose();
	    
	    // Calculate A^T A.
	    if (x.numRows() < x.numCols()) {
	    	x = xT;
	    }
	    
		if (power_iters < 1) {
			power_iters = 1;
		}
		
		BlockMatrix p = RandomMatrix.getCoordinateMatrix((int) x.numCols(), dstar, sc).toBlockMatrix();
		for (int i = 0; i < power_iters; i++) {
			p = xT.multiply(p);
			p = x.multiply(p);
		}

		BlockQR qr = new BlockQR(p);
		//DistributedQR qr = new DistributedQR(p, sc);
		BlockMatrix q = matrixToBlockMatrix(qr.getQ(), sc);
		//BlockMatrix q = qr.getQ().toBlockMatrix();
		BlockMatrix b = q.transpose().multiply(x);
		
	    SingularValueDecomposition<IndexedRowMatrix, Matrix> svd = b.toIndexedRowMatrix().computeSVD(dstar, true, 1.0E-9d);
	    U = svd.U();
	    s = svd.s();
	    V = svd.V();
	}
	
	private BlockMatrix matrixToBlockMatrix(Matrix m, JavaSparkContext sc) {
		int rows = m.numRows();
		int cols = m.numCols();
		ArrayList<MatrixEntry> entries = new ArrayList<MatrixEntry>();
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				entries.add(new MatrixEntry(i, j, m.apply(i, j)));
			}
		}
		JavaRDD<MatrixEntry> list = sc.parallelize(entries, numSlices).persist(storage).coalesce(numPartitions);
		CoordinateMatrix c = new CoordinateMatrix(list.rdd());
	    return c.toBlockMatrix();
	}
	

	public IndexedRowMatrix getU() {
		return U;
	}
	
	public Vector getS() {
		return s;
	}
	
	public Matrix getV() {
		return V;
	}
	

	public static void main(String[] args) throws NumberFormatException, IOException {
		//String filename = "/Users/Eric/Dropbox/workspace/ARSVD/src/data/matrix.csv";
		String filename = args[0];
		int dimensions = Integer.parseInt(args[1]);
		int powerIterations = 2;
	    //long startTime = System.nanoTime();
	    BlockARSVD solver = new BlockARSVD(filename, dimensions, 2);
	    
	  	//long endTime = System.nanoTime();
	    //long duration = (endTime - startTime)/1000000000;
	    //System.out.println(duration);
	    
		
	    double[] s = solver.getS().toArray();
	    for (int i = 0; i < s.length; i++) {
	    	System.out.println(s[i]);
	    }
	    
		/*
	    Matrix p = new DenseMatrix(3, 3, new double[]{1,2,3,5,6,7,3,5,4});
		//BlockQR qr  = new BlockQR(p);
	    //DistributedQR qr = new DistributedQR(p);
		System.out.println(p);
		System.out.println(qr.getQ());

		System.out.println(qr.getR());
		System.out.println(qr.getQ().multiply((DenseMatrix) qr.getR()));
		*/
	    
	    
	}
	
}