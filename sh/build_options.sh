#!/usr/bin/env bash

set -e

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd $CWD/.. 2>&1 > /dev/null
ROOT=$PWD
popd 2>&1 > /dev/null

## This is expected to be called before this file
# source $ROOT/sh/configure_env.sh

BIN_TRAY_USER="dekken"
BIN_TRAY_REPO="tick"
BINTRAY_BASE_URL="https://bintray.com/${BIN_TRAY_USER}/${BIN_TRAY_REPO}"
BINTRAY_BASE_BIN_URL="${BINTRAY_BASE_URL}/download_file?file_path="
BINTRAY_NIX_INFO_URL="${BINTRAY_BASE_BIN_URL}nix%2Flatest%2F" #${PYVER}.info
BINTRAY_MAC_INFO_URL="${BINTRAY_BASE_BIN_URL}mac%2Flatest%2F" #${PYVER}.info

APPVEYOR_BASE_BIN_URL="https://ci.appveyor.com/api/projects/dekken/tick/artifacts/dist/"

GIT_TO_CHECK_FILES=(
  "setup.py"
)

IFS=$'\n'
function top_git_files_changed(){
  git log -2 --pretty="format:" --name-only
}

function check_grep_exit_code(){
  set +e
  echo $1 | grep -E "$2" 2>&1 > /dev/null
  WIN=$?
  set -e
  echo "$WIN"
}

function should_build_profile(){
  [ -z "$1" ] && "FAILURE - function expects one parameter \$PROFILE"
  for FILE in $(top_git_files_changed); do
    EC=$(check_grep_exit_code "$FILE" "^lib/include/tick/$1|^lib/cpp$1")
    (( $EC == 0 )) && echo "1" && return 0
  done
  echo "0"
}

function should_build_any_CI(){
  RET=0
  for P in "${PROFILES[@]}"; do
    RET="$(should_build_profile $P)"
    (( $RET == 1 )) && break
  done
  if (( $RET == 0 )); then
    for F in "${GIT_TO_CHECK_FILES[@]}"; do
      EC=$(check_grep_exit_code "$(top_git_files_changed)" "^$F")
      (( $EC == 0 )) && RET="1" && break
    done
  fi
  if (( $RET == 0 )); then
    EC=$(check_grep_exit_code $(git log --pretty=oneline --abbrev-commit -1) "FORCE_CI")
    (( $EC == 0 )) && RET="1" && break
  fi
  echo "$RET"
}

function process_wheel(){
  [ -z "$1" ] && "FAILURE - function expects one parameter \$WHEEL_FILE"
  IFS=$'\n'
  unzip $1
  for dir in $(find tick -type d -name build); do
    cp -r $dir $ROOT/$(dirname $dir)
  done
}

function can_configure_bintray_travis(){
  [ -z "$1" ] || [ -z "$2" ] && "FAILURE - function expects two parameters"
  if curl -o /dev/null -L  --silent --fail -r 0-0 "${2}${1}.info"; then
    echo "1" # "URL exists: $url"
  else
    echo "0" # "URL does not exist: $url"
  fi
}
function travis_bintray_configure(){
  [ -z "$1" ] || [ -z "$2" ] && "FAILURE - function expects two parameters"
  CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  rm -rf $CWD/tmp && mkdir -p $CWD/tmp
  pushd $CWD/tmp
  curl -o $1.info -L ${2}${1}.info
  curl -o $1.whl  -L ${2}$(cat $1.info)
  process_wheel $1.whl
  popd
}

function can_appveyor_configure(){
  [ -z "$1" ] || [ -z "$2" ] && "FAILURE - function expects two parameters"
  if curl -o /dev/null -L  --silent --fail -r 0-0 "${2}${1}.info"; then
    echo "1" # "URL exists: $url"
  else
    echo "0" # "URL does not exist: $url"
  fi
}
function appveyor_configure(){
  [ -z "$1" ] || [ -z "$2" ] && "FAILURE - function expects two parameters"
  mkdir -p dist
  cd dist
  curl -o ${1}.info -L "${2}${1}.info"
  WHL=$(cat $1.info)
  curl -o $WHL -L ${2}$(cat $1.info)
  unzip $WHL 2>&1 > /dev/null
  for dir in $(find tick -type d -name build); do
    cp -r $dir $ROOT/$(dirname $dir)
  done
  rm -rf tick *dist-info
}
