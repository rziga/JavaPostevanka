package JavaPostevanka.GPTMul;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.Data.Tokenizer;
import JavaPostevanka.NN.Module;

public class GPTMul {
    
    private Module model;
    private Tokenizer tokenizer;

    public GPTMul(Module model, Tokenizer tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
    }

    public int mul(int a, int b) {
        String prompt1 = String.format("%d*%d=0", a, b);
        Matrix input1 = tokenizer.transform(prompt1);
        Matrix output1 = model.forward(new Matrix[] {input1})[0];
        int new_token1 = output1.rowArgMax()[3];

        String prompt2 = String.format("%d*%d=%d", a, b, new_token1);
        Matrix input2 = tokenizer.transform(prompt2);
        Matrix output2 = model.forward(new Matrix[] {input2})[0];
        int new_token2 = output2.rowArgMax()[4];

        Matrix pred = new Matrix(new float[][] {{new_token1, new_token2}});
        String out = tokenizer.transformInverse(pred);
        return Integer.parseInt(out);
    }


}
