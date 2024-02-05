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
        
        // train the GPT
        MultiplicationDataset dataset = new MultiplicationDataset(rng);
        GPT model = new GPT(
            dataset.vocabSize(), 
            3, 2, 8, 4, 16, rng);
        SGD optim = new SGD(model.parameters(), 1E-3F);
        CategoricalCrossentropy loss = new CategoricalCrossentropy();
        Trainer trainer = new Trainer(model, loss, optim, dataset);
        trainer.fit(50, 1000, true);

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
