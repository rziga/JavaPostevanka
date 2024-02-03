package JavaPostevanka.NN.Layer;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.NN.Module;

public class ReLU extends Module{
    
    private Matrix mask;

    @Override
    public Matrix[] forward(Matrix[] inputs) {
        Matrix x = inputs[0];
        mask = x.applyUnary((e) -> (e >= 0 ? 1F : 0F));
        return new Matrix[] {x.mul(mask)};
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        return new Matrix[] {partials[0].mul(mask)};
    }

    @Override
    public void clearContext() {
        mask = null;
    }
}
