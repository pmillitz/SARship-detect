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
