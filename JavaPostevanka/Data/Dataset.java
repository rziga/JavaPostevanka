package JavaPostevanka.Data;

import JavaPostevanka.Matrix.Matrix;

public abstract class Dataset {
    
    public abstract Matrix[] getItem(int index);

    public abstract int length();

}
