package JavaPostevanka.NN.Layer;

import JavaPostevanka.NN.Module;
import JavaPostevanka.NN.Parameter;
import JavaPostevanka.Matrix.Matrix;
import java.util.Random;

public class Embedding extends Module {
    
    private Parameter table;
    private int[] rows;

    public Embedding(int vocabSize, int embedChan, Random rng) {
        this.table = new Parameter(Matrix.random(vocabSize, embedChan, rng));
    }

    public Embedding(int vocabSize, int embedChan) {
        this(vocabSize, embedChan, new Random());
    }

    @Override
    public Matrix[] forward(Matrix[] inputs) {
        rows = convertToInt(inputs[0]);
        return new Matrix[] {table.data.getRows(rows)};
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        for (int i = 0; i < rows.length; i++) {
            int r = rows[i];
            table.grad.setRow(r, table.grad.getRow(r).add(partials[0].getRow(i)));
        }
        return null;
    }

    @Override
    public Parameter[] parameters() {
        return new Parameter[] {table};
    }

    @Override
    public void clearContext() {
        rows = null;
    }

    private int[] convertToInt(Matrix x) {
        if (!x.isRowVector()) {
            throw new IndexOutOfBoundsException();
        }
        int[] out = new int[x.cols()];
        for (int i = 0; i < x.cols(); i++) {
            out[i] = (int) x.get(0, i);
        }
        return out;
    }
}
