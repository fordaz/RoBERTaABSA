for df in `ls | grep -v compress | grep -v tgz`
    do
        echo "Compressing file $df"
        tar -czvf $df.tgz $df
        echo `ls -la $df.tgz`
    done