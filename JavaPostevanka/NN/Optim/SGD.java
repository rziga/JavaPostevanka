package JavaPostevanka.NN.Optim;

import JavaPostevanka.NN.Optimizer;
import JavaPostevanka.NN.Parameter;

public class SGD extends Optimizer {
    
    private float lr;

    public SGD(Parameter[] parameters, float lr) {
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
