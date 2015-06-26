import java.util.ArrayList;
import java.util.Random;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;

import scala.Tuple2;

public class RandomMatrix {

	static ArrayList<ArrayList<Double>> m;
	static ArrayList<MatrixEntry> me;


	public static ArrayList<ArrayList<Double>> getMatrix(int rows, int cols) {
		Random r = new Random();
		m = new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < rows; i++) {
			ArrayList<Double> s = new ArrayList<Double>();
			for (int j = 0; j < cols; j++) {
				s.add(r.nextGaussian());
			}
			m.add(s);
		}
		return m;
	}
	
	public static ArrayList<MatrixEntry> getMatrixEntries(int rows, int cols) {
		Random r = new Random(1);
		me = new ArrayList<MatrixEntry>();
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				me.add(new MatrixEntry((long) i, (long) j, r.nextGaussian()));
			}
		}
		return me;
	}

	public static CoordinateMatrix getCoordinateMatrix(int rows, int cols, JavaSparkContext sc) {
		JavaRDD<MatrixEntry> l = sc.parallelize(getMatrixEntries(rows, cols));
		return new CoordinateMatrix(l.rdd());
	}
	
	public static BlockMatrix getBlockMatrix(int rows, int cols,
			JavaSparkContext sc) {
		JavaRDD<ArrayList<Double>> l = sc.parallelize(getMatrix(rows, cols));
		JavaRDD<IndexedRow> entries = l
				.zipWithIndex()
				.map(new Function<scala.Tuple2<ArrayList<Double>, Long>, IndexedRow>() {
					private static final long serialVersionUID = 1L;

					@Override
					public IndexedRow call(Tuple2<ArrayList<Double>, Long> tuple)
							throws Exception {
						ArrayList<Double> array = tuple._1;
						long index = tuple._2;
						double[] doubles = new double[array.size()];
						for (int i = 0; i < array.size(); i++) {
							doubles[i] = array.get(i);
						}
						return new IndexedRow(index, new DenseVector(doubles));
					}
				});

		return new IndexedRowMatrix(entries.rdd()).toBlockMatrix();
	}
}
