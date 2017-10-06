


echo "python check"
[ -f "$CWD/.ppath" ] && [ -f $(cat "$CWD/.ppath") ] && PPATH=$(cat "$CWD/.ppath")
[ $PPATH == 0 ] && which python.exe &> /dev/null && PPATH=$(which python.exe)
if [ $PPATH == 0 ]; then
    echo "WARNING: python.exe not on path"
    read -p "Do you have python installed? yes/no " GET
    if [ "$GET" == "yes" ]; then
        read -p "Please enter directory path containing python.exe (/c/path/to/python) " PPATH
        [ -d "$PPATH" ] && PPATH="$PPATH"/python.exe
        [ ! -f "$PPATH" ] && echo "ERROR: python.exe not found at given path: $PPATH" && exit 1
        echo "$PPATH" > $CWD/.ppath
    elif [ "$GET" == "no" ]; then
        if [ -f "$CURL" ]; then
            read -p "Would you like me to download python for you? yes/no " GET
            if [ "$GET" == "yes" ]; then
                mkdir -p downloaded
                curl -o downloaded/python-$P_VER-amd64.exe https://www.python.org/ftp/python/$P_VER/python-$P_VER-amd64.exe
                echo "Executing python installer (make sure to install pip): "
                ./downloaded/python-$P_VER-amd64.exe
                echo "Please rerun this script now and enter the python path if required" && exit 0
            elif [ "$GET" == "no" ]; then
                echo "Please download python, install it and rerun this script" && exit 0
            else
                echo "ERROR: invalid entry" && exit 1
            fi
        fi    
    else
        echo "ERROR: invalid entry"
    fi
fi
