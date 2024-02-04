package JavaPostevanka.Data.Datasets;

import java.util.Random;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.Data.Dataset;
import JavaPostevanka.Data.Tokenizers.CharacterTokenizer;

public class MultiplicationDataset extends Dataset{
    
    private Random rng;
    private CharacterTokenizer tokenizer = new CharacterTokenizer(
        "0123456789*=".toCharArray()
        );

    public MultiplicationDataset(Random rng) {
        this.rng = rng;
    }

    public MultiplicationDataset() {
        this(new Random());
    }

    @Override
    public Matrix[] getItem(int index) {
        int num1 = rng.nextInt(10);
        int num2 = rng.nextInt(10);
        String trainText = String.format("%d*%d=%02d", num1, num2, num1*num2);
        int l = trainText.length();
        Matrix x = tokenizer.transform(trainText.substring(0, l-2));
        Matrix y = tokenizer.transform(trainText.substring(1, l-1));
        return new Matrix[] {x, y};
    }

    @Override
    public int length() {
        return 123;
    }

    public int vocabSize() {
        return tokenizer.vocabSize();
    }

}
