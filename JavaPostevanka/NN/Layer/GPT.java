package JavaPostevanka.NN.Layer;

import java.util.Random;
import java.lang.Math;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.NN.Sequential;
import JavaPostevanka.NN.Module;

public class GPT extends Module {
    
    private Embedding emb;
    private Sequential decoder;
    private Sequential linOut;

    public GPT(int vocabSize, int nLayers, int nHeads, int dModel, int dHidden, int dFF, Random rng) {
        // input embedding
        this.emb = new Embedding(vocabSize, dModel, rng);
        
        // decoder stack
        Module[] decoderLayers = new Module[nLayers];
        for (int i = 0; i < nLayers; i++) {
            decoderLayers[i] = new DecoderLayer(nHeads, dModel, dHidden, dFF, rng);
        }
        decoder = new Sequential(decoderLayers);

        // output linear net
        linOut = new Sequential(new Module[] {
            new Linear(dModel, vocabSize, false, rng),
            new SoftMax()
        });
    }

    @Override
    public Matrix[] forward(Matrix[] inputs) {
        Matrix x = emb.forward(inputs)[0];
        x = x.add(encodePositions(x));
        inputs = decoder.forward(new Matrix[] {x});
        return linOut.forward(inputs);
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        partials = linOut.backward(partials);
        partials = decoder.backward(partials);
        return emb.backward(partials);
    }

    @Override
    public Module[] subModules() {
        return new Module[] {emb, decoder, linOut};
    }

    @Override
    public void clearContext() {
        emb.clearContext();
        decoder.clearContext();
        linOut.clearContext();
    }

    private static Matrix encodePositions(Matrix x) {
        Matrix PE = Matrix.zerosLike(x);
        float d = (float) PE.cols();
        for (int pos = 0; pos < PE.rows(); pos++) {
            for (int i = 0; i < PE.cols(); i++) {
                float pe;
                if (i % 2 == 0) {
                    pe = (float) Math.sin(pos / Math.pow(10000D, 2*i / d));
                } else {
                    pe = (float) Math.cos(pos / Math.pow(10000D, 2*i / d));
                }
                PE.set(pos, i, pe);
            }
        }
        return PE;
    }

}
