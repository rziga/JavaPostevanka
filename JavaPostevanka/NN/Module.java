package JavaPostevanka.NN;

import java.util.ArrayList;
import java.util.List;

import JavaPostevanka.Matrix.Matrix;

public abstract class Module {
    
    public abstract Matrix[] forward(Matrix[] inputs);

    public abstract Matrix[] backward(Matrix[] partials);

    public abstract void clearContext();

    public Module[] subModules() {
        return new Module[] {};
    }

    public Parameter[] parameters() {
        return parameters(subModules());
    };

    public static Parameter[] parameters(Module[] subModules) {
        List<Parameter> x = new ArrayList<Parameter>();
        for (Module m: subModules) {
            for (Parameter el: m.parameters()) x.add(el);
        }
        return x.toArray(new Parameter[0]);
    }

    public void checkGrad(Matrix[] inputs, float h) {

        // pass through module 1 time and record backwards
        Matrix out = forward(inputs)[0];
        Matrix partial = Matrix.zerosLike(out);
        partial.set(0, 0, 1F);
        Matrix[] inOutGrads = backward(new Matrix[] {partial});

        // check parameter gradient
        for (Parameter p: parameters()) {
            for (int i = 0; i < p.data.rows(); i++) {
                for (int j = 0; j < p.data.cols(); j++) {
                    
                    // from backward pass
                    float backGrad = p.grad.get(i, j);

                    // toggle param
                    float pre = out.get(0, 0);
                    p.data.set(i, j, p.data.get(i, j) + h);
                    float post = forward(inputs)[0].get(0, 0);
                    p.data.set(i, j, p.data.get(i, j) - h);

                    // calculate differences
                    float diffGrad = (post - pre) / h;
                    float d = (backGrad-diffGrad);
                    d = d > 0 ? d : -d;
                    System.out.printf("backward: %.3f, finite: %.3f, pass: %b\n", backGrad, diffGrad, d < 0.001);
                }
            }
            System.out.println();
        }

        // check input gradients
        for (int k = 0; k < inputs.length; k++) {
            Matrix inOutGrad = inOutGrads[k];
            Matrix input = inputs[k];

            for (int i = 0; i < inOutGrad.rows(); i++) {
                for (int j = 0; j < inOutGrad.cols(); j++) {

                    // from backward pass
                    float backGrad = inOutGrad.get(i, j);

                    // toggle param
                    float pre = out.get(0, 0);
                    input.set(i, j, input.get(i, j) + h);
                    float post = forward(inputs)[0].get(0, 0);
                    input.set(i, j, input.get(i, j) - h);

                    // calculate differences
                    float diffGrad = (post - pre) / h;
                    float d = (backGrad-diffGrad);
                    d = d > 0 ? d : -d;
                    System.out.printf("backward: %2.3f, finite: %2.3f, pass: %b\n", backGrad, diffGrad, d < 0.001);
                }
            }
            System.out.println();
        }
    }

}
