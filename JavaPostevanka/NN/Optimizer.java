package JavaPostevanka.NN;

public abstract class Optimizer {
    
    protected Parameter[] parameters;

    public Optimizer(Parameter[] parameters) {
        this.parameters = parameters;
    }

    public abstract void step();

    public void zeroGrad() {
        for (Parameter p: parameters) {
            p.zeroGrad();
        }
    }

}
