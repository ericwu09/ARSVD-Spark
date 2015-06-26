import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;

import scala.Serializable;

public class BlockQR implements Serializable {
	private static final long serialVersionUID = -4065478147040847159L;
	private Matrix q;
	private Matrix r;
	private int rows;
	private int columns;

	public BlockQR(BlockMatrix m) {
		this(m.toLocalMatrix());
	}
	
	public BlockQR(Matrix a) {
		rows = (int) a.numRows();
		columns = (int) a.numCols();
		int min = Math.min(rows, columns);
		DenseMatrix qTmp = (DenseMatrix) a.copy();
		double[] arr = qTmp.toArray();

		r = new DenseMatrix(min, columns, new double[min*columns]);
		
		for (int i = 0; i < min; i++) {
			double[] qi = getColumn(arr, i);
			double alpha = l2norm(qi);
			if (Math.abs(alpha) > Double.MIN_VALUE) {
				for (int k = 0; k < qi.length; k++) {
					qi[k] = qi[k]/alpha;
					//qTmp.update(k, i, qTmp.apply(k, i)/alpha);
				}
				setColumn(arr, qi, i);
			} else {
				if (Double.isInfinite(alpha) || Double.isNaN(alpha)) {
					throw new ArithmeticException("Invalid intermediate result");
				}
			}
			r.update(i, i, alpha);

			for (int j = i + 1; j < columns; j++) {
				double[] qj = getColumn(arr, j);
				double norm = l2norm(qj);
				if (Math.abs(norm) > Double.MIN_VALUE) {
					double beta = dot(qi, qj);
					r.update(i, j, beta);
					if (j < min) {
						for (int k = 0; k < qj.length; k++) {
							qj[k] = qj[k]+qi[k]*-beta;
							//qTmp.update(k, j, qTmp.apply(k, j)+qTmp.apply(k, i)*-beta);

						}
						setColumn(arr, qj, j);
					}
				} else {
					if (Double.isInfinite(norm) || Double.isNaN(norm)) {
						throw new ArithmeticException("Invalid intermediate result");
					}
				}
			}
		}
		if (columns > min) {
			q = viewPart(arr, rows, min);
		} else {
			q = new DenseMatrix(rows, columns, arr);
		}
		
	}
	
	private double[] getColumn(double[] a, int i) {
		double[] dest = new double[rows];
		System.arraycopy(a, i*rows, dest, 0, rows);
		return dest;
	}
	
	private double[] setColumn(double[] full, double[] col, int i) {
		System.arraycopy(col, 0, full, i*rows, col.length);
		return full;
	}
	
	private DenseMatrix viewPart(double[] arr, int r, int c) {
		double[] tmp = new double[r*c];
		int k = 0;
		for (int i = 0; i < arr.length; i++) {
			if (((double) i)/((double) rows) > c) {
				break;
			}
			if (i%rows < r) {
				tmp[k] = arr[i];
				k++;
			}
		}
		return new DenseMatrix(r, c, tmp);
		
	}

	private double dot(double[] a, double[] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			sum += a[i]*b[i];
		}
		return sum;
	}
	
	private double l2norm(double[] arr) {
		double sum = 0D;
		for (int i = 0; i < arr.length; i++) {
			sum += arr[i]*arr[i];
		}
		return Math.sqrt(sum);
	}

	public Matrix getQ() {
		return q;
	}
	
	public Matrix getR() {
		return r;
	}

	
}
