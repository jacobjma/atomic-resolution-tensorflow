import numpy as np
import matplotlib.pyplot as plt

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):

    if not exponent:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = np.round(num / float(10**exponent), decimal_digits)
    if not precision:
        precision = decimal_digits
    
    if num < 100:
        return "${0:.{1}f}$".format(num, precision-1)
    
    #if num%10==0:
    #    return "$10^{{{1:d}}}$".format(coeff, exponent, precision)
    
    return "${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)

def set_lines_bw(ax,marker_size=3,styles=None):
    
    if styles is None:
        styles = {
            'b': {'marker': None, 'dash': (None,None)},
            'g': {'marker': None, 'dash': [5,5]},
            'r': {'marker': None, 'dash': [5,3,1,3]},
            'c': {'marker': None, 'dash': [1,3]},
            'm': {'marker': None, 'dash': [5,2,5,2,5,10]},
            'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
            'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
            }
    
    lines_to_adjust = ax.get_lines()
    try:
        lines_to_adjust += ax.get_legend().get_lines()
    except AttributeError:
        pass

    for line in lines_to_adjust:
        color = line.get_color()
        line.set_color('black')
        line.set_dashes(styles[color]['dash'])
        line.set_marker(styles[color]['marker'])
        line.set_markersize(marker_size)
        

def discrete_cmap(N, base_cmap='Paired'):
    """Create an N-bin discrete colormap from the specified input map"""

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)