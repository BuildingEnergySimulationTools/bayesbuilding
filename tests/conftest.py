import matplotlib

# The matplotlib-backend tests only need a figure object and a saved PNG, not
# an actual GUI -- force the non-interactive Agg backend before anything
# imports pyplot, so the suite doesn't depend on a working local Tk/Tcl
# install (which can be broken/inconsistent across shells on Windows).
matplotlib.use("Agg")
