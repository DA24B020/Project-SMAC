import sys
from decimal import Decimal, getcontext

def arctan(x: Decimal, terms: int) -> Decimal:
    total = term = x
    x2 = x * x
    for k in range(1, terms):
        term *= -x2
        total += term / (2*k + 1)
    return total

def compute_pi(n: int) -> Decimal:
    terms = n//3 + 3
    return (Decimal(16) * arctan(Decimal(1)/Decimal(5), terms)
            - Decimal(4) * arctan(Decimal(1)/Decimal(239), terms))

def nth_digit_after_decimal(constant: str, n: int) -> str:
    getcontext().prec = n + 10

    if constant.lower() == 'pi':
        val = compute_pi(n)
    elif constant.lower() == 'e':
        val = Decimal(1).exp()
    else:
        raise ValueError("First argument must be 'pi' or 'e'")

    s = format(val, 'f')
    if '.' not in s:
        return ''
    frac = s.split('.', 1)[1]
    return frac[n-1] if len(frac) >= n else ''

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} {{pi|e}} n", file=sys.stderr)
        sys.exit(1)

    const, n_str = sys.argv[1], sys.argv[2]
    if not n_str.isdigit() or not (1 <= int(n_str) <= 200):
        print("Error: n must be a positive integer between 1 and 1000", file=sys.stderr)
        sys.exit(1)
    n = int(n_str)

    try:
        digit = nth_digit_after_decimal(const, n)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if digit == '':
        print(f"Could not compute digit #{n}", file=sys.stderr)
        sys.exit(1)

    print(digit)

if __name__ == "__main__":
    main()
