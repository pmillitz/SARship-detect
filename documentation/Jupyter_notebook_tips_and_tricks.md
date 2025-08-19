# How to run a Jupyter-notebook file without the front-end interface

Suppose that you want to run your Python code in a file called `myfile.ipynb` on your computer, you can run the following command in a terminal window (assuming that you have activated the appropriate Python environment):

```bash
jupyter nbconvert --execute --to notebook myfile.ipynb
```

When finished, the file `myfile.nbconvert.ipynb`, which contains all the output cells (if your code didn't crash), would be created in the same directory. 

If you want to free the terminal window to work on something else, you can put an ampersand at the end of the command so that it would be run as a background job:

```bash
jupyter nbconvert --execute --to notebook myfile.ipynb &
```

If you have a Kaya account and if you want to run `myfile.ipynb` on Kaya as a batch job, then just put the first command (without the "&" symbol) in your slurm file.  (Note: You shouldn't put the & symbol, as your job would be queued and run as a background job anyway).

You can replace `myfile.ipynb` by `myfile.nbconvert.ipynb` using the **mv** command in Linux.

It is often useful to convert a Jupyter-notebook file to html format so that the contents of the file can be viewed in a web browser. The command below would do the job:

```bash
jupyter nbconvert --to html myfile.ipynb
```

It would generate `myfile.html` for you.

# How to reference a directory at a different level when importing modules

For example, I want to import a function from a module two directory levels up from my current working directory:

```python
import sys
sys.path.append('../../working')

from utilities import preview_image_shapes

# Examine the shape png files
preview_image_shapes(str(PROJECT_ROOT / 'data' / 'val_alt' / 'images'), 'png', limit=5)
```

The output:

```
f298dbd78ef977d5v_058.00846400999999730175_006.22928041299999968317_swath3_proc.png: shape=(96, 96, 3), mode=RGB, dtype=uint8
f298dbd78ef977d5v_056.93961363999999747421_007.31962743500000012631_swath2_proc.png: shape=(96, 96, 3), mode=RGB, dtype=uint8
758991708403f218v_004.10213002600000020692_008.17413784700000078942_swath2_proc.png: shape=(96, 96, 3), mode=RGB, dtype=uint8
758991708403f218v_003.96133040000000002934_009.09650325500000001000_swath3_proc.png: shape=(96, 96, 3), mode=RGB, dtype=uint8
f298dbd78ef977d5v_057.59236422000000032995_007.53724813699999973693_swath2_proc.png: shape=(96, 96, 3), mode=RGB, dtype=uint8
```
