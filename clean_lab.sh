jupyter nbconvert --clear-output --inplace ./Lab.ipynb
find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
find . -depth -name '.ipynb_checkpoints' -execdir rm -rf {} +
rm -rf ./classroom.egg-info
rm -rf ./build
