package JavaPostevanka.NN.Layer;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.NN.Module;

public class SkipBlock extends Module {
    
    private Module subBlock;

    public SkipBlock(Module subBlock) {
        this.subBlock = subBlock;
    }

    @Override
    public Matrix[] forward(Matrix[] inputs) {
        Matrix skip = inputs[0];
        Matrix x = subBlock.forward(inputs)[0];
        return new Matrix[] {x.add(skip)};
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        Matrix skip = partials[0];
        Matrix grad = subBlock.backward(partials)[0];
        return new Matrix[] {skip.add(grad)};
    }

    @Override
    public void clearContext() {
        subBlock.clearContext();
    }
}
