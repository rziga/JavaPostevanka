package JavaPostevanka.NN.Layer;

import java.util.Random;
import JavaPostevanka.NN.Sequential;
import JavaPostevanka.NN.Module;

public class DecoderLayer extends Sequential {
    
    public DecoderLayer(int numHeads, int dModel, int dHidden, int dFF, Random rng) {
        super(new Module[] {
            new AttentionSkipBlock(numHeads, dModel, dHidden, rng),
            new FFSkipBlock(dModel, dFF, rng)
        });
    }

    public DecoderLayer(int numHeads, int dModel, int dHidden, int dFF) {
        this(numHeads, dModel, dHidden, dFF, new Random());
    }

}
