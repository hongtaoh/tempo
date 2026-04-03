# Generate all synthetic data (lowdim + highdim, train + test).
# Skips any split that already exists.
# To regenerate from scratch, delete the relevant directories first:
#   rm -rf train test
#   rm -rf train_highdim test_highdim
#
# To generate only one mode:
#   python3 gen.py --mode lowdim
#   python3 gen.py --mode highdim

python3 gen.py --mode all
