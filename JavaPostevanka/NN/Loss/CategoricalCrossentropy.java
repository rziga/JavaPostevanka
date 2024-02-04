package JavaPostevanka.NN.Loss;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.NN.Module;

public class CategoricalCrossentropy extends Module {

    private int[] yTrue;
    private int[] shape;
    
    @Override
    public Matrix[] forward(Matrix[] inputs) {
        Matrix y = inputs[0];
        Matrix pred = inputs[1].add(1E-8F).log();

        yTrue = new int[y.cols()];
        shape = pred.shape();

        Matrix out = Matrix.zerosLike(y);
        for (int i = 0; i < out.cols(); i++) {
            yTrue[i] = (int) y.get(0, i);
            out.set(0, i, pred.get(i, yTrue[i]));
        }
        out = Matrix.ones(1, 1).mul(-out.sum());
        return new Matrix[] {out};
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        float p = -partials[0].get(0, 0);
        Matrix out = new Matrix(shape[0], shape[1]);
        for (int i = 0; i < yTrue.length; i++) {
            out.set(i, yTrue[i], 1 / p);
        }
        return new Matrix[] {out};
    }

    @Override
    public void clearContext() {
        yTrue = null;
    }
}
