# pyCHomP

CHomP (Computational Homology Project) with Python bindings

## Installation

To get the latest tagged version from the PyPi repository:

```bash
pip install pychomp
```

To uninstall:

```bash
pip uninstall pychomp
```


## Installing from source

To get the bleeding edge version (i.e., this repo):

```bash
git clone https://github.com/shaunharker/pyCHomP.git
cd pyCHomP
git submodule update --init --recursive
pip install . --ignore-installed --no-cache-dir -v -v -v --user
```

## Troubleshooting

### Can't get it to work with your version of python

Try Anaconda3 <https://www.anaconda.com>.

This will install into a folder named `anaconda3` and add a line in `~/.bash_profile`:

```bash
#added by Anaconda3 5.0.1 installer
#export PATH="/PATH/TO/anaconda3/bin:$PATH"
```

This will redirect command line python and pip. Note you may have to start a new `bash` session for the path changes to take effect (i.e. close and reopen the terminal program). This has the effect of plastering over any problems you might be having with multiple installations/permissions problems/jupyter not seeing the package/etc.

After installing this,

```
pip install jupyter
```

and try the installation above again.

### For macOS users who don't want anaconda but instead want to use `homebrew` python:

If the installation gives permissions issues, oftentimes the culprit is broken permissions on the subfolders of the homebrew folder `/usr/local`. 

First, see what

```
brew doctor
```

says. A common fix is:

```
sudo chown -R $(whoami) $(brew --prefix)/*
```

If it still doesn't work after this, then you might try uninstalling and reinstalling homebrew.

To uninstall homebrew:

```
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall)"
```

Or `sudo` if it gives issues:

```
sudo ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall)"
```

To install homebrew (don't use `sudo` here!):

```
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

### Python/Jupyter Integration issues:

If the package installs but it is not visible in jupyter, the likely problem is that the jupyter python kernel is not the same python for which pychomp was installed. That is, you may have multiple pythons on your system.

You can try to confirm this by typing

```
which python
which pip
which jupyter
```

Possible fixes include steps such as 

1. checking/changing your environmental variable `PATH` in `~/.bash_profile` or `.bashrc`
2. uninstalling python and jupyter, then reinstalling python then jupyter


