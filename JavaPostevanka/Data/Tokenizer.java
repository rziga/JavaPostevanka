package JavaPostevanka.Data;

import JavaPostevanka.Matrix.Matrix;

public abstract class Tokenizer {
    
    public abstract Matrix transform(String str); 

    public abstract String transformInverse(Matrix mtx);

    public abstract int vocabSize();
    
}
