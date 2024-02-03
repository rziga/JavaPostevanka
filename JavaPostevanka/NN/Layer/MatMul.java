package JavaPostevanka.NN.Layer;
import JavaPostevanka.NN.Module;
import JavaPostevanka.Matrix.Matrix;

public class MatMul extends Module{

    private Matrix a;
    private Matrix b;
    
    @Override
    public Matrix[] forward(Matrix[] inputs) {
        a = inputs[0];
        b = inputs[1];
        return new Matrix[] {a.matmul(b)};
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        Matrix p = partials[0];
        return new Matrix[] {p.matmul(b.T()), a.T().matmul(p)};
    }

    @Override
    public void clearContext() {
        a = null;
        b = null;
    }

}
