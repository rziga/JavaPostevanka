package JavaPostevanka.NN;

import JavaPostevanka.Matrix.Matrix;

public class Sequential extends Module{

    private Module[] modules;
    
    public Sequential(Module[] modules) {
        this.modules = modules;
    }

    @Override
    public Matrix[] forward(Matrix[] inputs) {
        Matrix[] out = inputs;
        for (Module m: modules) {
            out = m.forward(out);
        }
        return out;
    }

    @Override
    public Matrix[] backward(Matrix[] partials) {
        Matrix[] out = partials;
        for (int i = modules.length-1; i >= 0; i--) {
            Module m = modules[i];
            out = m.backward(out);
        }
        return out;
    }

    @Override
    public Module[] subModules() {
        return modules;
    }

    @Override
    public void clearContext() {
        for (Module m: modules) {
            m.clearContext();
        }
    }

}
