## Set environment
- Go to folder /jmspack
- Get the latest version of the develop branch

  - `$ git checkout develop`
  - `$ git pull origin develop`

- activate conda environment jmspack-dev
  - `$ conda activate jmspack-dev`
  - if the environment is not installed execute: `$ conda env create --file requirements.txt`
  - make sure pre-commit is installed

## Set Release Branch & Update Documentation
- Switch to a new branch `release-#.#.#`
  - `$ git checkout -b release-#.#.# develop`
  - where `#,#,#`, is the new version e.g. `1.4.2`
- bump the version number:
  - `./bump-version.sh 1.4.2`
  - if permission is denied execute `chmod u+x bump-version.sh`
- Update the version number in `/jmspack/__init__.py`.
- run `$ python setup.py develop` in the terminal of the directory which includes `setup.py`.
- Test new functions by importing and running examples.
- Build Documentation
  - `pdoc --html jmspack --force --output-dir documentation`
  - If you do not have pdoc installed run `conda install pdoc3`
- Once you have built the documentation you should find an “index.html” file in the “documentation“
- Commit changes
  - `git commit -a -m "Bumped version number to #.#.#"`
  - `git push origin release-#.#.#`

## Build package
- Build new package _(BE AWARE: when running the following command conda builds everything in the directory
into the package - hence if you have large files that are not a part of what you wish to be included
consider removing them prior to the build)_
  - `$ conda build --python meta.yaml # for macOS`
  - `$ conda build meta.yaml # for Linux`
  - currently `{PYTHON_VERSION}` is set to `3.8`
- For MacOS case the builds are found in `/Users/{USER}/opt/anaconda3/conda-bld/osx-64/`
- For linux systems it is suggested to specify the render folder
you can specify the render folder using `--croot`.
- The package name is `jmspack-#.#.#-py{PYTHON_VERSION}.tar.bz2`

## Push to pypi
- `python3 -m build`
  - If build is not installed do this: `python3 -m pip install --upgrade build`
- `python3 -m pip install --upgrade twine`
- `python3 -m twine upload dist/*`
  - Fill in credentials for pypi.org

## Push to anaconda
- `anaconda login`
  - If anaconda is not installed do this: `conda install anaconda-client`
- `anaconda upload ~/path/to/package/{NEW_VERSION}.tar.bz2`

## Release update on GitHub
- Go to Compare & pull request  pull request for `release-#.#.#` branch
- Update __Release__ and __Tag__
- Create a documentation.zip folder
  - `zip -r documentation.zip documentation`
- Please add the `documentation zip`, the `**package name**.tar.bz2` to the assets
