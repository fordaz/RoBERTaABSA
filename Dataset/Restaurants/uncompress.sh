for df in `ls *.tgz`
    do
        echo "Uncompressing file $df"
        tar -xzvf $df
    done