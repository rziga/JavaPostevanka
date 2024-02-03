package JavaPostevanka.NN;
import JavaPostevanka.Matrix.Matrix;;

public abstract class Module {
    
    public abstract Matrix[] forward(Matrix[] inputs);

    public abstract Matrix[] backward(Matrix[] partials);

    public abstract void clearContext();

    public Parameter[] parameters() {
        return new Parameter[] {};
    };

}
