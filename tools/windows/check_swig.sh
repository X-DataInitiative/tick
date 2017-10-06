


echo "swig check"
[ -f "$CWD/.spath" ] && [ -f $(cat "$CWD/.spath") ] && SPATH=$(cat "$CWD/.spath")
[ $SPATH == 0 ] && which swig.exe &> /dev/null && SPATH=$(which swig.exe)
if [ $SPATH == 0 ]; then
    echo "WARNING: swig.exe not on path"
    read -p "Do you have swig installed? yes/no " GET
    if [ "$GET" == "yes" ]; then
        read -p "Please enter directory path containing swig.exe (/c/path/to/swig) " SPATH
        [ -d "$SPATH" ] && SPATH="$SPATH"/swig.exe
        [ ! -f "$SPATH" ] && echo "ERROR: swig.exe not found at given path: $SPATH" && exit 1
        echo "$SPATH" > $CWD/.spath
    elif [ "$GET" == "no" ]; then
        if [ -f "$CURL" ]; then
            read -p "Would you like me to download swig for you? yes/no " GET
            if [ "$GET" == "yes" ]; then
                mkdir -p downloaded
                curl -Lo downloaded/swigwin-3.0.12.zip http://downloads.sourceforge.net/project/swig/swigwin/swigwin-3.0.12/swigwin-3.0.12.zip
                echo "Please extract the downloaded zip and rerun this script" && exit 0
            elif [ "$GET" == "no" ]; then
                echo "Please download swigwin and install it" && exit 1
            else
                echo "ERROR: invalid entry" && exit 1
            fi
        fi    
    else
        echo "ERROR: invalid entry"
    fi
fi
