package JavaPostevanka.NN.Layer;

import java.util.Random;
import java.lang.Math;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.NN.Module;

public class SelfAttentionHead extends Module {

    private Linear linQ;
    private Linear linK;
    private Linear linV;
    private MatMul mm1;
    private MatMul mm2;
    private SoftMax sm;
    private float d;

    public SelfAttentionHead(int inChan, int embedChan, Random rng) {
        this.linQ = new Linear(inChan, embedChan, false, rng);
        this.linK = new Linear(inChan, embedChan, false, rng);
        this.linV = new Linear(inChan, embedChan, false, rng);
        this.mm1  = new MatMul();
        this.mm2  = new MatMul();
        this.sm = new SoftMax();
        this.d = (float) Math.sqrt((double) embedChan);
    }

    public SelfAttentionHead(int inChan, int embedChan) {
        this(inChan, embedChan, new Random());
    }

    @Override
    public Matrix[] forward(Matrix[] inputs) {
        Matrix Q = linQ.forward(inputs)[0];
        Matrix K = linK.forward(inputs)[0];
        Matrix V = linV.forward(inputs)[0];

        Matrix QK = mm1.forward(new Matrix[] {Q, K.T()})[0].div(d);
        fillTriu(QK, Float.NEGATIVE_INFINITY);
        Matrix S = sm.forward(new Matrix[] {QK})[0];
        return mm2.forward(new Matrix[] {S, V});
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        partials = mm2.backward(partials);
        Matrix dS = partials[0];
        Matrix dV = partials[1];

        Matrix dQK = sm.backward(new Matrix[] {dS})[0];
        fillTriu(dQK, 0);
        partials = mm1.backward(new Matrix[] {dQK.div(d)});

        Matrix dQ = linQ.backward(new Matrix[] {partials[0]})[0];
        Matrix dK = linK.backward(new Matrix[] {partials[1].T()})[0];
        dV = linV.backward(new Matrix[] {dV})[0];

        return new Matrix[] {dQ.add(dK).add(dV)};
    }

    @Override
    public void clearContext() {
        linQ.clearContext();
        linK.clearContext();
        linV.clearContext();
        mm1.clearContext();
        mm2.clearContext();
        sm.clearContext();
    }

    @Override
    public Module[] subModules() {
        return new Module[] {linQ, linK, linV, mm1, mm2, sm};
    }

    private static void fillTriu(Matrix m, float fill) {
        int d = m.rows();
        for (int i = 0; i < d - 1; i++) {
            for (int j = i + 1; j < d; j++) {
                m.set(i, j, fill);
            }
        }
    }

}