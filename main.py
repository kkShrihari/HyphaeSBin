"""HyphaeSBin entry point. The pipeline lives in hyphaesbin/pipeline.py (so an installed wheel contains it);
this file keeps `python main.py ...` working from a source checkout."""
from hyphaesbin.pipeline import main

if __name__ == "__main__":
    main()
