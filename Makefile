# Default target
.PHONY: all
all: run

# Install dependencies
.PHONY: install
install:
	julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run specific script (example)
.PHONY: run
run:
	julia --project=. main.jl

