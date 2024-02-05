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

    public void fit(int batchSize, int nEpochs, boolean verbose) {
        for (int epoch = 1; epoch <= nEpochs; epoch++) {
            float currentLoss = 0;
            if (verbose) {
                System.out.printf("epoch: %d\n", epoch);
            }

            for (int i = 1; i <= dataset.length(); i++) {
                Matrix[] batch = dataset.getItem(i);
                Matrix x = batch[0];
                Matrix y = batch[1];
                Matrix pred = model.forward(new Matrix[] {x})[0];
                Matrix l = loss.forward(new Matrix[] {pred, y})[0];
                model.backward(loss.backward(new Matrix[] {Matrix.onesLike(l)}));
                currentLoss += l.get(0, 0) / dataset.length();

                if (i % batchSize == 0) {
                    optim.step();
                    optim.zeroGrad();
                }
                
            }

            if (verbose) {
                System.out.printf("%.10f\n", currentLoss);
                currentLoss = 0;
            }

        }   
    }
}
