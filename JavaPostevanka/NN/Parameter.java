package JavaPostevanka.NN;
import JavaPostevanka.Matrix.Matrix;

public class Parameter {

    public Matrix data;
    public Matrix grad;

    public Parameter(Matrix data) {
        this.data = data;
        this.grad = Matrix.zeros(data.rows(), data.cols());
    }

    public void zeroGrad() {
        grad.mul(0);
    }
}
