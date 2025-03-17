run-clang-tidy -quiet -p $1 -j $2 -use-color  2>clang-tidy-error.txt >clang-tidy-output.txt

# grep interesting errors and make sure we remove duplicates:
grep -E '(warning|error): ' clang-tidy-output.txt | sort | uniq >clang-tidy.log
