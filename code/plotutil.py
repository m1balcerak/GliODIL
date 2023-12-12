import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

g_extlist = None

def set_extlist(extlist=None):
    global g_extlist
    if extlist is None:
        g_extlist = ['pdf']
    else:
        g_extlist = extlist


set_extlist()


def savefig(fig,
            path,
            extlist=None,
            skip_existing=False,
            printf=print,
            **kwargs):
    if extlist is None:
        extlist = g_extlist
    for ext in extlist:
        metadata = {
            'Date': None,
        } if ext == 'svg' else {
            'DateModified': None,
            'CreationDate': None,
        } if ext == 'pdf' else {}
        p = path + '.' + ext
        if skip_existing and os.path.isfile(p):
            printf("skip existing '{}'".format(p))
            continue
        printf(p)
        fig.savefig(p, metadata=metadata, **kwargs)


def savelegend(fig, ax, path, **kwargs):
    figleg, axleg = plt.subplots()
    handles, labels = ax.get_legend_handles_labels()
    legend = axleg.legend(handles, labels, loc='center', frameon=False)
    axleg.set_axis_off()
    figleg.canvas.draw()
    bbox = legend.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted())
    savefig(figleg, path, bbox_inches=bbox, **kwargs)


def set_log_ticks(xaxis):
    locmin = matplotlib.ticker.LogLocator(base=10.,
                                          subs=np.arange(0.1, 0.99, 0.1),
                                          numticks=12)
    xaxis.set_minor_locator(locmin)
    xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
