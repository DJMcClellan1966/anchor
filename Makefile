# Run the full test suite (anchor + corpus graph + anchor_math + all others)
test:
	pytest tests/ -v --tb=short

# Same as test (convenience)
check: test

.PHONY: test check
