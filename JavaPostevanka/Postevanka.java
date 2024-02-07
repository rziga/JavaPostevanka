package JavaPostevanka;

import java.util.Random;

import JavaPostevanka.Data.Datasets.MultiplicationDataset;
import JavaPostevanka.GPTMul.GPTMul;
import JavaPostevanka.NN.Layer.GPT;
import JavaPostevanka.NN.Loss.CategoricalCrossentropy;
import JavaPostevanka.NN.Optim.SGD;
import JavaPostevanka.Trainer.Trainer;

public class Postevanka {
    
    public static void main(String[] args) {
        Random rng = new Random(1337);
        
        // hyperparameters
        int nLayers, nHeads, dModel, dHidden, dFF; // for GPT
        float learningRate; // for optimizer
        int batchSize, maxEpochs; // for trainer
        nLayers = 3; nHeads = 2; dModel = 8; dHidden = 4; dFF = 16;
        learningRate = 1E-3F;
        batchSize = 50; maxEpochs = 1000; boolean verbose = true;

        // train the GPT
        MultiplicationDataset dataset = new MultiplicationDataset(rng);
        GPT model = new GPT(
            dataset.vocabSize(), 
            nLayers, nHeads, dModel, dHidden, dFF, rng);
        SGD optim = new SGD(model.parameters(), learningRate);
        CategoricalCrossentropy loss = new CategoricalCrossentropy();
        Trainer trainer = new Trainer(model, loss, optim, dataset);
        trainer.fit(batchSize, maxEpochs, verbose);

        // print out the multiplication table
        GPTMul muller = new GPTMul(model, dataset.tokenizer);
        for (int i = 0; i < 10; i++) {
            for(int j = 0; j < 10; j++) {
                System.out.printf("%2d ", muller.mul(i, j));
            }
            System.out.println();
        }
        
    }
}
