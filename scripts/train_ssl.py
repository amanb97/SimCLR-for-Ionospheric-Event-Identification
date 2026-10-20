"""Train the CNN with multi-head attention using the SimCLR objective."""
from ionospheric_simclr.train import main, parse_args

if __name__ == '__main__':
    main(parse_args())
