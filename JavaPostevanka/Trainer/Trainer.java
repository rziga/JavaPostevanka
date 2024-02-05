package JavaPostevanka.Trainer;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.Data.Dataset;
import JavaPostevanka.NN.Module;
import JavaPostevanka.NN.Optimizer;

public class Trainer {
    
    private Module model;
    private Module loss;
    private Optimizer optim;
    private Dataset dataset;

    public Trainer(Module model, Module loss, Optimizer optim, Dataset dataset) {
        this.model = model;
        this.loss = loss;
        this.optim = optim;
        this.dataset = dataset;
    }

    public void fit(int batchSize, int maxEpochs, boolean verbose) {
        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            float epochLoss = 0;
            for (int i = 1; i <= dataset.length(); i++) {

                // gradient accumulation to handle batches
                Matrix[] batch = dataset.getItem(i);
                Matrix x = batch[0];
                Matrix y = batch[1];
                Matrix pred = model.forward(new Matrix[] {x})[0];
                Matrix l = loss.forward(new Matrix[] {pred, y})[0];
                model.backward(loss.backward(new Matrix[] {Matrix.onesLike(l)}));
                epochLoss += l.get(0, 0) / dataset.length();
                
                if (i % batchSize == 0) {
                    optim.step();
                    optim.zeroGrad();
                }
                
            }

            // empirically proven that it prints out
            // correct multiplication table if true
            if (epochLoss < 0.5) return;

            if (verbose) {
                System.out.printf("epoch: %4d, ", epoch);
                System.out.printf("loss %.3f\n", epochLoss);
                epochLoss = 0;
            }

        }   
    }
}
