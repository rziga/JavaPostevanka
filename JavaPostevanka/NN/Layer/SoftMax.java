package JavaPostevanka.NN.Layer;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.NN.Module;

public class SoftMax extends Module {

    private Matrix activation; 
    
    @Override
    public Matrix[] forward(Matrix[] inputs) {
        Matrix expX = inputs[0].exp();
        activation = expX.div(expX.rowSum());
        return new Matrix[] {activation};
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        Matrix partial = partials[0];
        Matrix out = Matrix.zerosLike(partial);
        for (int i = 0; i < partial.rows(); i++) {
            out.setRow(i, backwardVec(activation.getRow(i), partial.getRow(i)));
        }
        return new Matrix[] {out};
    }

    private Matrix backwardVec(Matrix activationVec, Matrix partial) {
        Matrix jac = activationVec.mul(activationVec.T()).mul(-1);
        for (int i = 0; i < activationVec.cols(); i++) {
            jac.set(i, i, jac.get(i, i) + activationVec.get(0, i));
        }
        return partial.matmul(jac);
    }

    @Override
    public void clearContext() {
        activation = null;
    }
}
