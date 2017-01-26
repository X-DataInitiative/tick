#!/bin/sh

git clone git@github.com:X-DataInitiative/mlpp.git tick

cur_dir=$(pwd)

(
  cd tick  && ./build_test.sh
  cd doc && make clean html
)

cp -R tick/doc/_build/html .

git add -A
git commit -am "Website update"
git push

cd ${cur_dir}
