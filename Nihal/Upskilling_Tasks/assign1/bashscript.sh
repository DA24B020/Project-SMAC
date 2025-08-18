#!/bin/bash
read input
if [ "$input" = "pi" ]; then
    num="141592653589793"
elif [ "$input" = "e" ]; then
    num="718281828459045"
fi
read index
str=${num:$((index-1)):1}
if [ "$str" = "" ]; then
    echo "Digit not available in memory"
else
    echo "$str"
fi