#!/bin/sh

git clone git@github.com:X-DataInitiative/mlpp.git tick

cur_dir=$(pwd)

(
  cd tick  && MLPP_NO_CPPLINT=true ./build_test.sh

  export PYTHONPATH=${PYTHONPATH}:${cur_dir}/tick

  cd DOC && make clean html
)

cp -Rv tick/DOC/_build/html/* .

git add -A
git commit -am "Website update"
git push

#cd ${cur_dir}
