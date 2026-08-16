"""RNAfold command adapter backed by the ViennaRNA Python package."""

import sys

import RNA


def main():
    lines = [line.strip() for line in sys.stdin if line.strip()]
    identifier = lines[0] if lines[0].startswith(">") else None
    sequence = lines[1] if identifier else lines[0]
    structure, energy = RNA.fold(sequence)
    if identifier:
        print(identifier)
    print(sequence)
    print(f"{structure} ({energy:.2f})")


if __name__ == "__main__":
    main()
