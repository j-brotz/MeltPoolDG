# Create Doxygen documentation

Go to the root path of the repo e.g.
```sh
cd MeltPoolDG-dev
```
and execute 
```sh
doxygen doc/doxygen/Doxyfile
``` 
to generate the documentation. This creates two directories, i.e., `html` and `latex`. You may open `index.html` to view the Doxygen documentation in a web browser. For the LaTeX documentation you need to execute `make` inside the `latex` directory.
