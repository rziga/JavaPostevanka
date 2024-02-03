package JavaPostevanka.NN.Layer;

import java.util.Random;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.NN.Module;
import JavaPostevanka.NN.Parameter;

public class MultiHeadSelfAttention extends Module{

    private SelfAttentionHead[] heads;
    private Linear outLin;

    public MultiHeadSelfAttention(int numHeads, int dModel, int dHidden, Random rng) {
        this.heads = new SelfAttentionHead[numHeads];
        for (int i = 0; i < numHeads; i++) {
            this.heads[i] = new SelfAttentionHead(dModel, dHidden, rng);
        }
        this.outLin = new Linear(numHeads * dHidden, dModel, false);
    }

    public MultiHeadSelfAttention(int numHeads, int dModel, int dHidden) {
        this(numHeads, dModel, dHidden, new Random());
    }

    @Override
    public Matrix[] forward(Matrix[] inputs) {
        return inputs;
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        return partials;
    }

    @Override
    public Parameter[] parameters() {
        return new Parameter[] {};
    }

    @Override
    public void clearContext() {
        for (SelfAttentionHead h: heads) {
            h.clearContext();
        }
        outLin.clearContext();
    }
}
