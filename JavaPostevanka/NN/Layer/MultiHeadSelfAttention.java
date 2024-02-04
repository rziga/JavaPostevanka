package JavaPostevanka.NN.Layer;

import java.util.Random;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.NN.Module;

public class MultiHeadSelfAttention extends Module{

    private SelfAttentionHead[] heads;
    private Linear linOut;

    public MultiHeadSelfAttention(int numHeads, int dModel, int dHidden, Random rng) {
        this.heads = new SelfAttentionHead[numHeads];
        for (int i = 0; i < numHeads; i++) {
            this.heads[i] = new SelfAttentionHead(dModel, dHidden, rng);
        }
        this.linOut = new Linear(numHeads * dHidden, dModel, false);
    }

    public MultiHeadSelfAttention(int numHeads, int dModel, int dHidden) {
        this(numHeads, dModel, dHidden, new Random());
    }

    @Override
    public Matrix[] forward(Matrix[] inputs) {
        Matrix[] headOuts = new Matrix[heads.length];
        for (int i = 0; i < heads.length; i++) {
            headOuts[i] = heads[i].forward(inputs)[0];
        }
        Matrix out = Matrix.catCols(headOuts);
        return linOut.forward(new Matrix[] {out});
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        partials = linOut.backward(partials);
        Matrix[] dHeads = partials[0].splitCols(heads.length);
        Matrix out = heads[0].backward(new Matrix[] {dHeads[0]})[0];
        for (int i = 1; i < heads.length; i++) {
            out = out.add(heads[i].backward(new Matrix[] {dHeads[i]})[0]);
        }
        return new Matrix[] {out};
    }

    @Override
    public Module[] subModules() {
        Module[] subModules = new Module[heads.length + 1];
        for (int i = 0; i < heads.length; i++) {
            subModules[i] = heads[i];
        }
        subModules[subModules.length-1] = linOut;
        return subModules;
    }

    @Override
    public void clearContext() {
        for (SelfAttentionHead h: heads) {
            h.clearContext();
        }
        linOut.clearContext();
    }
}
