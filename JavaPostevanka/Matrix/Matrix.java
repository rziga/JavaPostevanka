package JavaPostevanka.Matrix;

import java.lang.Math;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;
import java.util.function.BiFunction;

public class Matrix {
    
    private float[] data; // storage for matrix elements
    private int[] strides; // [stride for rows, stride for cols]
    private int[] shape; // [n_rows, n_cols]

    public Matrix(int rows, int cols) {
        data = new float[rows * cols];
        strides = new int[] {cols, 1};
        shape = new int[] {rows, cols};
    }

    public Matrix(float[][] buffer) {
        int rows = buffer.length;
        int cols = buffer[0].length; 
        strides = new int[] {cols, 1};
        shape = new int[] {rows, cols};
        data = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i * strides[0] + j * strides[1]] = buffer[i][j];
            }
        }
    }

    public static Matrix fill(int rows, int cols, float value) {
        float[][] buffer = new float[rows][cols];
        for (float[] row: buffer) {
            Arrays.fill(row, value);
        }
        return new Matrix(buffer);
    }

    public static Matrix zeros(int rows, int cols) {
        return fill(rows, cols, 0);
    }

    public static Matrix zerosLike(Matrix other) {
        return zeros(other.rows(), other.cols());
    }

    public static Matrix ones(int rows, int cols) {
        return fill(rows, cols, 1);
    }

    public static Matrix onesLike(Matrix other) {
        return ones(other.rows(), other.cols());
    }

    public static Matrix random(int rows, int cols, Random rng) {
        float[][] buffer = new float[rows][cols];
        for (int i = 0; i < buffer.length; i++) {
            for (int j = 0; j < buffer[0].length; j++) {
                buffer[i][j] = (rng.nextFloat() - 0.5F);
            }
        }
        return new Matrix(buffer);
    }

    public static Matrix randomLike(Matrix other, Random rng) {
        return random(other.rows(), other.cols(), rng);
    }

    public static Matrix catRows(Matrix[] mtxs) {
        int cols = mtxs[0].cols();
        int rows = 0;
        int[] rowCum = new int[mtxs.length];
        for (int i = 0; i < mtxs.length; i++) {
            int r = mtxs[i].rows();
            rows += r;
            rowCum[i] = i == 0 ? r : rowCum[i-1] + r;
        }

        Matrix out = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            int idx = currentMtx(rowCum, i);
            Matrix curr = mtxs[idx];
            int currRow = idx > 0 ? i - rowCum[idx - 1] : i;
            out.setRow(i, curr.getRow(currRow));
        }
        return out;
    }

    private static int currentMtx(int[] rowCum, int idx) {
        int current = rowCum.length - 1;
        for (int i = rowCum.length - 1; i >= 0; i--) {
            if (idx < rowCum[i]) {
                current = i;
            }
        }
        return current;
    }

    public static Matrix catCols(Matrix[] mtxs) {
        Matrix[] mtxsT = new Matrix[mtxs.length];
        for (int i = 0; i < mtxs.length; i++) {
            mtxsT[i] = mtxs[i].T();
        }
        return catRows(mtxsT).T();
    }

    public Matrix[] splitRows(int n) {
        if (rows() % n != 0) {
            throw new IndexOutOfBoundsException();
        }
        int rowsPerMtx = rows() / n;

        Matrix[] out = new Matrix[n];
        for (int i = 0; i < n; i++) {
            // @ i -> rows: (i * rowsPerMtx) + [0, 1, 2, 3, ..., rowsPerMtx]
            out[i] = getRows(range(i * rowsPerMtx, (i + 1) * rowsPerMtx));
        }
        return out;
    }

    public Matrix[] splitCols(int n) {
        Matrix[] out = this.T().splitRows(n);
        for (int i = 0; i < out.length; i++) {
            out[i] = out[i].T();
        }
        return out;
    }

    private int[] range(int start, int stop) {
        int[] out = new int[stop - start];
        for (int i = 0; i < stop - start; i++) {
            out[i] = start + i;
        }
        return out;
    }

    public int rows() {
        return shape[0];
    }

    public int cols() {
        return shape[1];
    }

    public int numel() {
        return rows() * cols();
    }

    public int[] shape() {
        return shape;
    }

    public boolean isRowVector() {
        return rows() == 1;
    }

    public boolean isColVector() {
        return cols() == 1;
    }

    public Matrix T() {
        Matrix mtx = new Matrix(cols(), rows());
        mtx.data = this.data;
        mtx.strides = new int[] {strides[1], strides[0]};
        mtx.shape = new int[] {shape[1], shape[0]};
        return mtx;
    }

    public float get(int r, int c) {
        return data[r * strides[0] + c * strides[1]];
    }

    public Matrix getRow(int r) {
        float[][] buffer = new float[1][cols()];
        for (int i = 0; i < cols(); i++) {
            buffer[0][i] = get(r, i);
        }
        return new Matrix(buffer);
    }

    public Matrix getRows(int[] rs) {
        Matrix out = new Matrix(rs.length, cols());
        for (int i = 0; i < rs.length; i++) {
            out.setRow(i, getRow(rs[i]));
        }
        return out;
    }

    public Matrix getCol(int c) {
        return T().getRow(c).T();
    }

    public void set(int r, int c, float buffer) {
        data[r * strides[0] + c * strides[1]] = buffer;
    }

    public void setRow(int r, Matrix buffer) {
        if (!buffer.isRowVector()) {
            throw new IndexOutOfBoundsException();
        }
        for (int i = 0; i < cols(); i++) {
            set(r, i, buffer.getB(0, i));
        }
    }

    public void setCol(int c, Matrix buffer) {
        if (!buffer.isColVector()) {
            throw new IndexOutOfBoundsException();
        }
        for (int i = 0; i < rows(); i++) {
            set(i, c, buffer.get(i, 0));
        }
    }

    private static boolean broadcastable(int n, int m) {
        return n == m || n == 1 || m == 1;
    }

    private static int broadcastShape(int sh1, int sh2) {
        if (!broadcastable(sh1, sh2)) {
            throw new IndexOutOfBoundsException();
        }
        return sh1 > sh2 ? sh1 : sh2;
    }

    private static int broadcastIndex(int idx, int max) {
        if (idx < max) {
            return idx;
        } else if(max == 1) {
            return 0;
        } else {
            throw new IndexOutOfBoundsException();
        }
    }

    private float getB(int r, int c) {
        r = broadcastIndex(r, rows());
        c = broadcastIndex(c, cols());
        return get(r, c);
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < rows(); i++) {
            if (i != 0) {
                sb.append(",\n ");
            }
            sb.append("[");
            for (int j = 0; j < cols(); j++) {
                if (j != 0) {
                    sb.append(", ");
                }
                sb.append(get(i, j));
            }
            sb.append("]");
        }
        sb.append("]");
        return sb.toString();
    }

    /* OPS right here */

    public Matrix applyBinary(BiFunction<Float, Float, Float> fcn, Matrix other) {
        int r = broadcastShape(this.rows(), other.rows());
        int c = broadcastShape(this.cols(), other.cols());
        Matrix out = new Matrix(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                out.set(i, j, fcn.apply(this.getB(i, j), other.getB(i, j)));
            }
        }
        return out;
    }

    public Matrix applyUnary(Function<Float, Float> fcn) {
        int r = rows();
        int c = cols();
        Matrix out = new Matrix(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                out.set(i, j, fcn.apply(this.get(i, j)));
            }
        }
        return out;
    }

    public Matrix add(Matrix other) {
        return applyBinary((x, y) -> x + y, other);
    }

    public Matrix add(float other) {
        return applyUnary((x) -> x + other);
    }

    public Matrix sub(Matrix other) {
        return applyBinary((x, y) -> x - y, other);
    }

    public Matrix sub(float other) {
        return applyUnary((x) -> x - other);
    }

    public Matrix mul(Matrix other) {
        return applyBinary((x, y) -> x * y, other);
    }

    public Matrix mul(float other) {
        return mul(new Matrix(new float[][] {{other}}));
    }

    public Matrix div(Matrix other) {
        return applyBinary((x, y) -> x/y, other);
    }

    public Matrix div(float other) {
        return div(new Matrix(new float[][] {{other}}));
    }

    public Matrix reciprocal(float other) {
        return applyUnary((x) -> 1 / x);
    }

    public Matrix exp() {
        return applyUnary((x) -> (float) Math.exp((double) x));
    }

    public Matrix log() {
        return applyUnary((x) -> (float) Math.log((double) x));
    }

    public float sum() {
        float sum = 0;
        for (float el: data) {
            sum += el;
        }
        return sum;
    }

    public Matrix rowSum() {
        Matrix out = new Matrix(rows(), 1);
        for (int i = 0; i < rows(); i++) {
            out.set(i, 0, this.getRow(i).sum());
        }
        return out;
    }

    public Matrix colSum() {
        return this.T().rowSum().T();
    }

    public float max() {
        float max = -Float.MAX_VALUE;
        for (float d: data) {
            if (d > max) {
                max = d;
            }
        }
        return max;
    }

    public Matrix rowMax() {
        Matrix out = new Matrix(rows(), 1);
        for (int i = 0; i < rows(); i++) {
            out.set(i, 0, getRow(i).max());
        }
        return out;
    }

    public Matrix colMax() {
        return this.T().rowMax().T();
    }

    public int[] rowArgMax() {
        int[] out = new int[rows()];
        Matrix maxes = rowMax();
        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                if (this.get(i, j) == maxes.get(i, 0)) {
                    out[i] = j;
                }
            }
        }
        return out;
    }

    public int[] colArgMax() {
        return this.T().rowArgMax();
    }

    public float dot(Matrix other) {
        if (!(this.isRowVector() && other.isColVector())) {
            throw new IndexOutOfBoundsException();
        }
        return this.T().mul(other).sum();
    }

    public Matrix matmul(Matrix other) {
        if (this.cols() != other.rows()) {
            throw new IndexOutOfBoundsException();
        }

        Matrix out = new Matrix(this.rows(), other.cols());
        for (int i = 0; i < out.rows(); i++) {
            for (int j = 0; j < out.cols(); j++) {
                out.set(i, j, this.getRow(i).dot(other.getCol(j)));
            }
        }
        return out;
    }

}
