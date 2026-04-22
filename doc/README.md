
# Documentation for tick

This contains the full documentation of tick, as displayed on the
[https://x-datainitiative.github.io/tick](https://x-datainitiative.github.io/tick).
To reproduce the published documentation locally, use Python 3.11+ and run

    python -m pip install .[docs]
    make SPHINXBUILD="python -m sphinx" html

This build executes the example gallery and the inline module plots, so if your
dataset cache is empty it may trigger downloads through `fetch_tick_dataset`.

For a faster build that reuses the checked-in gallery assets instead of
re-running the full example gallery, you can run

    make SPHINXBUILD="python -m sphinx" html-noplot

This shortcut still executes a few inline module plots, but it does not provide
full parity with the published example pages.
