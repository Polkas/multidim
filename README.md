# multidim

<span>
<a href="https://github.com/polkas/multidim/actions">
<img src="https://github.com/polkas/multidim/workflows/ci/badge.svg" alt="Build Status">
</a>
<a href="https://codecov.io/gh/Polkas/multidim">
<img src="https://codecov.io/gh/Polkas/multidim/branch/main/graph/badge.svg" alt="codecov">
</a>
</span>

Multidimensional Analysis WNE University of Warsaw

## Installation

```bash
$ git clone https://github.com/Polkas/multidim
$ # (optional) open multidim directory in VScode
$ # (optional) open (zsh or bash) terminal in VScode
```

optional part start - virtual env

```bash
$ cd multidim
$ python -m venv .venv
$ # activate is only a one click in VScode
$ # on Mac/Linux
$ source .venv/bin/activate
$ # Windows
$ # useful for Windows https://docs.python.org/3/library/venv.html
$ # PowerShell command POSSIBLY needed - 
$ # Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
$ .venv\Scripts\activate
```

optional part end - virtual env

```bash
$ pip install ".[all]"
$ # now open any notebook from src/multidim/notebooks in VScode
$ # or use 
$ jupyter notebook
$ # when ready to end Ctrl-C
$ deactivate
```

or simply open the github codespaces and go to src/multidim/notebooks

When you want to contribute, then fork https://github.com/Polkas/multidim

```bash
$ git clone https://github.com/YOURUSER/multidim
...
```

Please, `git pull` regulary and `pip install ".[all]"` might be needed too].

## Setup Videos

[Simple Setup](https://drive.google.com/file/d/1ZMStipXFeXl81CFcJ7k-BXWHsA_K9kZG/view?usp=sharing)

[Advanced Setup](https://drive.google.com/file/d/1jImDrznuluIZ400JRVfpsyBu80xQHMEK/view?usp=sharing)

[Update Repo](https://drive.google.com/file/d/1kMOHSrXUL7fHnefdxOw_t_8vqz5RsRpM/view?usp=sharing)

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`multidim` was created by Maciej Nasinski and Pawel Strawinski. It is licensed under the terms of the Apache 2.0 license.
