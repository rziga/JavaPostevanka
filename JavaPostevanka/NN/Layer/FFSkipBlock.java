package JavaPostevanka.NN.Layer;

import java.util.Random;

import JavaPostevanka.NN.Module;
import JavaPostevanka.NN.Sequential;

public class FFSkipBlock extends SkipBlock {
    
    public FFSkipBlock(int dModel, int dFF, Random rng) {
        super(new Sequential(new Module[] {
            new Linear(dModel, dFF, true, rng),
            new ReLU(),
            new Linear(dFF, dModel, true, rng)
        }));
    }

    public FFSkipBlock(int dModel, int dFF) {
        this(dModel, dFF, new Random());
    }
}
