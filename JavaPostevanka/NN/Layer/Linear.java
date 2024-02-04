package JavaPostevanka.NN.Layer;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.NN.Parameter;
import JavaPostevanka.NN.Module;

import java.util.Random;

public class Linear extends Module{

    private Parameter W;
    private Parameter b;
    private MatMul mm;
    private boolean bias;

    public Linear(int inChan, int outChan, boolean bias, Random rng) {
        this.W = new Parameter(Matrix.random(inChan, outChan, rng));
        this.b = new Parameter(Matrix.random(1, outChan, rng));
        this.mm = new MatMul();
        this.bias = bias;
    }

    public Linear(int inChan, int outChan, boolean bias) {
        this(inChan, outChan, bias, new Random());
    }

    @Override
    public Matrix[] forward(Matrix[] inputs) {
        Matrix out = mm.forward(new Matrix[] {inputs[0], W.data})[0];
        if (bias) {
            out = out.add(b.data);
        }
        return new Matrix[] {out};
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        if (bias) {
            b.grad = b.grad.add(partials[0].colSum());
        }
        partials = mm.backward(partials);
        W.grad = W.grad.add(partials[1]);
        return new Matrix[] {partials[0]};
    }

    @Override
    public Parameter[] parameters() {
        return new Parameter[] {W, b};
    }

    @Override
    public void clearContext() {
        mm.clearContext();
    }
}
