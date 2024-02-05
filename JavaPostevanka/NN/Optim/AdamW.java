package JavaPostevanka.NN.Optim;

import JavaPostevanka.NN.Optimizer;
import JavaPostevanka.NN.Parameter;

public class AdamW extends Optimizer{
    
    private float lr;

    public AdamW(Parameter[] parameters, float lr) {
        super(parameters);
        this.lr = lr;
    }

    @Override
    public void step() {
        for (Parameter p: parameters) {
            p.data = p.data.sub(p.grad.mul(lr));
        }
    }

}
