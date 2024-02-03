package JavaPostevanka.NN.Layer;

import java.util.Random;

public class AttentionSkipBlock extends SkipBlock{
    
    public AttentionSkipBlock(int numHeads, int dModel, int dHidden, Random rng) {
        super(new MultiHeadSelfAttention(numHeads, dModel, dHidden, rng));
    }

    public AttentionSkipBlock(int numHeads, int dModel, int dHidden) {
        this(numHeads, dModel, dHidden, new Random());
    }
}
