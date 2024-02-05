package JavaPostevanka;

import java.util.Random;
import java.util.Base64.Decoder;

import javax.sound.midi.Sequence;

import JavaPostevanka.Data.Datasets.MultiplicationDataset;
import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.NN.Module;
import JavaPostevanka.NN.Sequential;
import JavaPostevanka.NN.Layer.DecoderLayer;
import JavaPostevanka.NN.Layer.FFSkipBlock;
import JavaPostevanka.NN.Layer.GPT;
import JavaPostevanka.NN.Layer.Linear;
import JavaPostevanka.NN.Layer.MatMul;
import JavaPostevanka.NN.Layer.MultiHeadSelfAttention;
import JavaPostevanka.NN.Layer.ReLU;
import JavaPostevanka.NN.Layer.SoftMax;
import JavaPostevanka.NN.Loss.CategoricalCrossentropy;
import JavaPostevanka.NN.Optim.SGD;
import JavaPostevanka.Trainer.Trainer;
import JavaPostevanka.NN.Layer.SelfAttentionHead;

public class Postevanka {
    public static void main(String[] args) {
        Random rng = new Random(1337);
        
        //Matrix x = new Matrix(new float[][] {{1, 2}, {3, 4}, {5, 6}});
        //Module module = new SoftMax(); // OK
        //Module module = new Linear(3, 2, true, rng); // OK
        //Module module = new SelfAttentionHead(2, 3, rng); // OK
        //Module module = new FFSkipBlock(2, 3); // OK
        //Module module = new ReLU(); // OK
        //Module module = new MultiHeadSelfAttention(2, 2, 3); // OK
        //Module module = new DecoderLayer(2, 2, 1, 1); // OK
        //Matrix y = new Matrix(new float[][] {{0, 1, 1, 0, 1}});
        //Matrix pred = new Matrix(new float[][] {{0.5F, 0.5F}, {0.1F, 0.9F}, {0.4F, 0.6F}, {0.7F, 0.3F}, {0.8F, 0.2F}});
        //Module module = new CategoricalCrossentropy();
        //module.checkGrad(new Matrix[] {pred, y}, 1E-3F);
        //module.checkGrad(new Matrix[] {x}, 1E-3F);

        
        MultiplicationDataset dataset = new MultiplicationDataset(rng);
        GPT model = new GPT(
            dataset.vocabSize(), 
            3, 2, 8, 4, 16, rng);
        //System.out.println(model.parameters()[0].data);
        SGD optim = new SGD(model.parameters(), 1E-3F);
        CategoricalCrossentropy loss = new CategoricalCrossentropy();
        Trainer trainer = new Trainer(model, loss, optim, dataset);
        trainer.fit(50, 1000, true);

        System.out.println("h");
        

        /*
        Matrix x = new Matrix(new float[][] {{1, 2, 3}, {4, 5, 6}});
        Matrix y = new Matrix(new float[][] {{0, 1, 2}});
        System.out.println(x.add(y));
        System.out.println(x.add(1));
        System.out.println(x.sub(y));
        System.out.println(x.sub(1));
        */

        /*
        GPT model = new GPT(10, 1, 1, 4, 2, 8, rng);
        Matrix x = new Matrix(new float[][]{{0, 1, 1, 9, 3, 4}});
        Matrix out = model.forward(new Matrix[] {x})[0];
        //System.out.println(out);
        model.backward(new Matrix[] {Matrix.onesLike(out)});
        System.out.println(model.parameters().length);
        */

        //Matrix x = Matrix.random(1, 2, rng);
        //Linear lin = new Linear(2, 3, true, rng);
        //SoftMax sm = new SoftMax();

        //Matrix out = lin.forward(new Matrix[] {x})[0];
        //Matrix out = sm.forward(new Matrix[] {x})[0];
        //Matrix partial = Matrix.onesLike(out);
        //System.out.println(out);
        //System.out.println(sm.backward(new Matrix[] {partial})[0]);
        //System.out.println(lin.parameters()[0].grad);

        //System.out.println(Matrix.ones(3, 3).exp());
        //Matrix a = new Matrix(new float[][] {{1, 2, 3}});
        //System.out.println(a.mul(a.T()));

        /*
        Matrix a = new Matrix(new float[][] {{1, 2}, {3, 4}, {5, 6}, {7, 8}});
        Matrix b = new Matrix(new float[][] {{7, 8}, {10, 11}});
        System.out.println(Matrix.catRows(new Matrix[] {a, b, b}));

        Matrix[] s = a.splitCols(2);
        for (Matrix m: s) {
            System.out.println(m);
        }
        */

        /*
        Matrix x = Matrix.ones(2, 3);
        SelfAttentionHead att = new SelfAttentionHead(3, 4);
        Matrix out = att.forward(new Matrix[] {x})[0];
        Matrix grad = att.backward(new Matrix[] {Matrix.onesLike(out)})[0];
        System.out.println(out);
        System.out.println(grad);
        */
    }
}
