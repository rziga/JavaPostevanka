package JavaPostevanka.NN.Loss;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.NN.Module;

public class CategoricalCrossentropy extends Module {

    private int[] yTrue;
    private Matrix yPred;
    
    @Override
    public Matrix[] forward(Matrix[] inputs) {
        Matrix y = inputs[1];
        Matrix pred = inputs[0];

        yTrue = new int[y.cols()];
        yPred = pred.add(1E-8F);

        Matrix out = Matrix.zerosLike(y);
        for (int i = 0; i < out.cols(); i++) {
            yTrue[i] = (int) y.get(0, i);
            out.set(0, i, yPred.log().get(i, yTrue[i]));
        }
        out = Matrix.ones(1, 1).mul(-out.sum() / yPred.rows());
        return new Matrix[] {out};
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        float p = partials[0].get(0, 0);
        Matrix out = Matrix.zerosLike(yPred);
        for (int i = 0; i < yTrue.length; i++) {
            out.set(i, yTrue[i], -p / yPred.get(i, yTrue[i]) / yPred.rows());
        }
        return new Matrix[] {out};
    }

    @Override
    public void clearContext() {
        yTrue = null;
    }
}
