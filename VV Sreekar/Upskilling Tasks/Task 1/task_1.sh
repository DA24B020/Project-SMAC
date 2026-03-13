#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Error: Please provide exactly two arguments - 'pi' or 'e' and a number between 1-200"
    exit 1
fi

x=$1
n=$2

if [ "$n" -lt 1 ] || [ "$n" -gt 200 ]; then
    echo "Error: The second argument must be between 1 and 200"
    exit 1
fi

if [ "$x" = "pi" ]; then
    num=$(echo "scale=$((n+5)); 4*a(1)" | bc -l)
elif [ "$x" = "e" ]; then
    num=$(awk -v scale=$((n+5)) '
    BEGIN {
        e = 1.0;
        fact = 1.0;
        for (i = 1; i < 100; i++) {
            fact *= i;
            e += 1.0 / fact;
        }
        printf "%.*f\n", scale, e;
    }')
else
    echo "Error: First argument must be either 'pi' or 'e'"
    exit 1
fi

digit=$(echo "$num" | cut -d '.' -f 2 | cut -c "$n")

echo "$digit"
