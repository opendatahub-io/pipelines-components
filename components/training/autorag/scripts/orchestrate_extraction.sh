#!/usr/bin/env bash


#TODO create logger prefix and lvl to add to all logs


if [[ -z "$(command -v awk)" ]]; then

  echo 'Could not find an `awk` executable (almost impossible it is not installed). Make sure it is findable in the PATH env variable.'
  exit 1

fi

echo "Parsing file ($2) in order to extract nested functions to a separate file..."



awk -v nested_names_file_path="$1" -f components/training/autorag/scripts/extract_nested_funcs.awk $2
rc=$?

if (( $rc == 0)); then 
  echo "Created a file ($1) with nested functions extracted for further unit tests suite execution..."

else
  echo 'Problems encountered during file processing using `awk`. Please refer to the script output above.'
  echo "Exiting with error code ($rc)..."
fi

exit $rc