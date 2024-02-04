package JavaPostevanka.Data.Tokenizers;

import java.util.HashMap;

import JavaPostevanka.Matrix.Matrix;
import JavaPostevanka.Data.Tokenizer;

public class CharacterTokenizer extends Tokenizer {
    
    private HashMap<Character, Integer> vocab;
    private HashMap<Integer, Character> vocabInverse; 

    public CharacterTokenizer(char[] vocabulary) {
        this.vocab = new HashMap<>();
        this.vocabInverse = new HashMap<>();
        for (int i = 0; i < vocabulary.length; i++) {
            this.vocab.put(vocabulary[i], i);
            this.vocabInverse.put(i, vocabulary[i]);
        }
    }

    @Override
    public Matrix transform(String str) {
        char[] chars = str.toCharArray();
        Matrix out = new Matrix(1, chars.length);
        for (int i = 0; i < chars.length; i++) {
            out.set(0, i, (float) vocab.get(chars[i]));
        }
        return out;
    }

    @Override
    public String transformInverse(Matrix mtx) {
        char[] chars = new char[mtx.cols()];
        for (int i = 0; i < chars.length; i++) {
            chars[i] = vocabInverse.get((int) mtx.get(0, i));
        }
        return new String(chars);
    }

    @Override
    public int vocabSize() {
        return vocab.keySet().size();
    }
}
