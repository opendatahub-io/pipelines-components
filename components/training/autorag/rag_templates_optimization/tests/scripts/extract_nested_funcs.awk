

BEGIN {
    read_code_block=0
    indentation_lvl=0
    dedent_lvl=0
}

!read_code_block && /__name__.*==.*main/ {
    # omitting locally executable part of the module. Nothing of interest there
    print("Module definition ended before EOF (if __name__ == \"__main__\"). Exiting early...")
    exit 0
}

# catch top-level imports (those inside pipeline/component function)
!read_code_block && /^[[:blank:]]+(from.+)?import/ {
    sub(/^[[:blank:]]+/, "", $0)

    import_line=$0

    # Multi-line import: everything between `(` and `)`
    if (import_line ~ /[[:alnum:][:blank:]_,]+\([[:alnum:][:blank:]_,]*?$/) {
        while ((getline) > 0) {
            sub(/^[[:blank:]]+/, "  ", $0)
            import_line = import_line"\n"$0
            if ($ 0~ /\)/) {
                break
            }
        }
    }
    print import_line >>nested_names_file_path
}

read_code_block && !/^[[:space:]]*$/ {

    padding=substr($0, 1, indentation_lvl)
    if (padding !~ /^[[:blank:]]+$/) {  # end of current code block
        read_code_block=0
    }
    else{
        print substr($0, dedent_lvl) >>nested_names_file_path
    }

}

!read_code_block && /[[:blank:]]+(def|class)[[:blank:]]+[[:alpha:]_]/ {
    
    match($0, /^[[:blank:]]+/)
    dedent_lvl=length(substr($0,RSTART,RLENGTH))+1

    # Print complete function signature
    # due to linter formatting rules multi line signature's indentation level does not need to be consistent (same across all lines)
    sig = substr($0, dedent_lvl)
    while ($0 !~ /:$/ && (getline) > 0) {
        sig = sig"\n"substr($0, dedent_lvl)
    }
    print "\n\n" sig >>nested_names_file_path


    while ((getline) > 0 && $0 ~ /^[[:space:]]*$/) ;
    match($0, /^[[:blank:]]+/)
    indentation_lvl=length(substr($0,RSTART,RLENGTH))

    read_code_block=1
    
    print substr($0, dedent_lvl) >>nested_names_file_path

    next
}


END {
    print "Cleaning opened file descriptors... Exiting..."
    close(nested_names_file_path)
}



