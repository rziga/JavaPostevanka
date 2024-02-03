package JavaPostevanka;

import java.util.Random;

import javax.sound.midi.Sequence;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.NN.Module;
import JavaPostevanka.NN.Sequential;
import JavaPostevanka.NN.Layer.Linear;
import JavaPostevanka.NN.Layer.SoftMax;
import JavaPostevanka.NN.Layer.SelfAttentionHead;

public class Postevanka {
    public static void main(String[] args) {
        Random rng = new Random(1337);

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

        Sequential model = new Sequential(new Module[] {
            new Linear(2, 3, false)}
            );
    }
}
