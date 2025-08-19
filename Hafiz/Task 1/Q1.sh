#!/usr/bin/env bash

# assn2a.sh — display the nth digit after the decimal of π or e

# 1) Check argument count
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 {pi|e} n"
  echo "  where n is an integer between 1 and 200"
  exit 1
fi

mode="$1"
n="$2"

# 2) Validate n is an integer 1–200
if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -lt 1 ] || [ "$n" -gt 200 ]; then
  echo "Error: n must be an integer between 1 and 200."
  exit 1
fi

# 3) Compute (mode) to scale n+1 so we have at least n digits after decimal
#    We use 'bc -l' for π (4*a(1)) or e (e(1)).
scale=$((n+1))
case "$mode" in
  pi)
    # π = 4*arctan(1)
    value=$(echo "scale=$scale; 4*a(1)" | bc -l)
    ;;
  e)
    # e = exp(1)
    value=$(echo "scale=$scale; e(1)" | bc -l)
    ;;
  *)
    echo "Error: first argument must be 'pi' or 'e'."
    exit 1
    ;;
esac

# 4) Extract the part after the decimal point
#    Bash parameter‐expansion: remove up to the dot, then take nth character
fractional=${value#*.}
digit=${fractional:$((n-1)):1}

# 5) If something went wrong (e.g. bc output shorter than expected), guard
if [ -z "$digit" ]; then
  echo "Error: could not compute the $n-th digit."
  exit 1
fi

# 6) Output the digit
echo "$digit"
exit 0
