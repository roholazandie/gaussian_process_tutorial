import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_multi_plots(data, colors=None, interval=100):
    plotlays = list(range(data.shape[1]))

    fig = plt.figure()
    ax = plt.axes(xlim=(0, np.shape(data)[0]), ylim=(np.min(data), np.max(data)))
    #timetext = ax.text(0.5, 50, '')

    lines = []
    for index, lay in enumerate(plotlays):
        if colors:
            lobj = ax.plot([], [], lw=2, color=colors[index])[0]
        else:
            lobj = ax.plot([], [], lw=2)[0]
        lines.append(lobj)

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(i):
        #timetext.set_text(i)
        x = np.array(range(1, data.shape[0] + 1))
        for lnum, line in enumerate(lines):
            line.set_data(x, data[:, plotlays[lnum] - 1, i])
        return tuple(lines)# + (timetext,)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=np.shape(data)[2], interval=interval, blit=True)

    plt.show()


def pred_range(x, portion=0.2, points=200, randomize=False):
    """Return a one dimensional range for prediction across given a data set, x

    :param x: input data from which to create a range.
    :param portion: portion of the range to extend (default 0.2)
    :param points: number of points in the range.
    :param randomize: whether tho randomize the points slightly (add small Gaussian noise to each input location)."""
    span = x.max()-x.min()
    xt = np.linspace(x.min()-portion*span, x.max()+portion*span, points)[:, np.newaxis]
    if not randomize:
        return xt
    else:
        return xt + np.random.randn(points, 1)*span/float(points)


def model_output(model, samples=np.array([]), output_dim=0, scale=1.0, offset=0.0, title="", xlim=None, portion=0.2):
    """Plot the output of a GP.
    :param model: the model for the output plotting.
    :param output_dim: the output dimension to plot.
    :param scale: how to scale the output.
    :param offset: how to offset the output.
    :param ax: axis to plot on.
    :param xlabel: label for the x axis (default: '$x$').
    :param ylabel: label for the y axis (default: '$y$').
    :param xlim: limits of the x axis
    :param ylim: limits of the y axis
    :param fontsize: fontsize (default 20)
    :param portion: What proportion of the input range to put outside the data."""

    xt = pred_range(model.X, portion=portion)
    if xlim is None:
        xlim = [xt.min(), xt.max()]

    yt_mean, yt_var = model.predict(xt)
    yt_mean = yt_mean*scale + offset
    yt_var *= scale*scale
    yt_sd = np.sqrt(yt_var)
    if yt_sd.shape[1]>1:
        yt_sd = yt_sd[:, output_dim]

    plot_gp(yt_mean, yt_var, xt, model.X.flatten(), model.Y.flatten(), samples=samples, title=title)


def plot_gp(mu, var, X, X_train=None, Y_train=None, samples=np.array([]), title=""):
    X = X.ravel()
    mu = mu.ravel()
    var = var.ravel()
    uncertainty = 1.96 * np.sqrt(var)
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')

    #for i, sample in enumerate(samples):
    if samples.any():
        plt.plot(X, samples, 'o', alpha=0.4, markersize=5)

    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    plt.title(title)
    plt.show()


def visualize_pinball(model, ax=None, scale=1.0, offset=0.0, xlabel='input', ylabel='output',
                      xlim=None, ylim=None, fontsize=20, portion=0.2, points=50,
                      vertical=True, fig_name="pinball.svg"):
    """Visualize the layers in a deep GP with one-d input and output."""

    def scale_data(x, portion):
        scale = (x.max() - x.min()) / (1 - 2 * portion)
        offset = x.min() - portion * scale
        return (x - offset) / scale, scale, offset

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    depth = len(model.layers)

    last_name = xlabel
    last_x = model.X

    # Recover input and output scales from output plot
    model_output(model, scale=scale, offset=offset, ax=ax,
                 xlabel=xlabel, ylabel=ylabel,
                 fontsize=fontsize, portion=portion)
    xlim = ax.get_xlim()
    xticks = ax.get_xticks()
    xtick_labels = ax.get_xticklabels().copy()
    ylim = ax.get_ylim()
    yticks = ax.get_yticks()
    ytick_labels = ax.get_yticklabels().copy()

    # Clear axes and start again
    ax.cla()
    if vertical:
        ax.set_xlim((0, 1))
        ax.invert_yaxis()

        ax.set_ylim((depth, 0))
    else:
        ax.set_ylim((0, 1))
        ax.set_xlim((0, depth))

    ax.set_axis_off()  # frame_on(False)

    def pinball(x, y, y_std, color_scale=None,
                layer=0, depth=1, ax=None,
                alpha=1.0, portion=0.0, vertical=True):

        scaledx, xscale, xoffset = scale_data(x, portion=portion)
        scaledy, yscale, yoffset = scale_data(y, portion=portion)
        y_std /= yscale

        # Check whether data is anti-correlated on output
        if np.dot((scaledx - 0.5).T, (scaledy - 0.5)) < 0:
            scaledy = 1 - scaledy
            flip = -1
        else:
            flip = 1

        if color_scale is not None:
            color_scale, _, _ = scale_data(color_scale, portion=0)
        scaledy = (1 - alpha) * scaledx + alpha * scaledy

        def color_value(x, cmap=None, width=None, centers=None):
            """Return color as a function of position along x axis"""
            if cmap is None:
                cmap = np.asarray([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
            ncenters = cmap.shape[0]
            if centers is None:
                centers = np.linspace(0 + 0.5 / ncenters, 1 - 0.5 / ncenters, ncenters)
            if width is None:
                width = 0.25 / ncenters

            r = (x - centers) / width
            weights = np.exp(-0.5 * r * r).flatten()
            weights /= weights.sum()
            weights = weights[:, np.newaxis]
            return np.dot(cmap.T, weights).flatten()

        for i in range(x.shape[0]):
            if color_scale is not None:
                color = color_value(color_scale[i])
            else:
                color = (1, 0, 0)
            x_plot = np.asarray((scaledx[i], scaledy[i])).flatten()
            y_plot = np.asarray((layer, layer + alpha)).flatten()
            if vertical:
                ax.plot(x_plot, y_plot, color=color, alpha=0.5, linewidth=3)
                ax.plot(x_plot, y_plot, color='k', alpha=0.5, linewidth=0.5)
            else:
                ax.plot(y_plot, x_plot, color=color, alpha=0.5, linewidth=3)
                ax.plot(y_plot, x_plot, color='k', alpha=0.5, linewidth=0.5)

            # Plot error bars that increase as sqrt of distance from start.
            std_points = 50
            stdy = np.linspace(0, alpha, std_points)
            stdx = np.sqrt(stdy) * y_std[i]
            stdy += layer
            mean_vals = np.linspace(scaledx[i], scaledy[i], std_points)
            upper = mean_vals + stdx
            lower = mean_vals - stdx
            fillcolor = color
            x_errorbars = np.hstack((upper, lower[::-1]))
            y_errorbars = np.hstack((stdy, stdy[::-1]))
            if vertical:
                ax.fill(x_errorbars.T, y_errorbars,
                        color=fillcolor, alpha=0.01)
                ax.plot(scaledy[i], layer + alpha, '.', markersize=10, color=color, alpha=0.5)
            else:
                ax.fill(y_errorbars, x_errorbars,
                        color=fillcolor, alpha=0.01)
                ax.plot(layer + alpha, scaledy[i], '.', markersize=10, color=color, alpha=0.5)
            # Marker to show end point
        return flip

    # Whether final axis is flipped
    flip = 1
    first_x = last_x
    for i, layer in enumerate(reversed(model.layers)):
        if i == 0:
            xt = pred_range(last_x, portion=portion, points=points)
            color_scale = xt
        yt_mean, yt_var = layer.predict(xt)
        if layer == model.obslayer:
            yt_mean = yt_mean * scale + offset
            yt_var *= scale * scale
        yt_sd = np.sqrt(yt_var)
        flip = flip * pinball(xt, yt_mean, yt_sd, color_scale, portion=portion,
                              layer=i, ax=ax, depth=depth, vertical=vertical)  # yt_mean-2*yt_sd,yt_mean+2*yt_sd)
        xt = yt_mean
    # Make room for axis labels
    if vertical:
        ax.set_ylim((2.1, -0.1))
        ax.set_xlim((-0.02, 1.02))
        ax.set_yticks(range(depth, 0, -1))
    else:
        ax.set_xlim((-0.1, 2.1))
        ax.set_ylim((-0.02, 1.02))
        ax.set_xticks(range(0, depth))

    def draw_axis(ax, scale=1.0, offset=0.0, level=0.0, flip=1,
                  label=None, up=False, nticks=10, ticklength=0.05,
                  vertical=True,
                  fontsize=20):
        def clean_gap(gap):
            nsf = np.log10(gap)
            if nsf > 0:
                nsf = np.ceil(nsf)
            else:
                nsf = np.floor(nsf)
            lower_gaps = np.asarray([0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
                                     0.1, 0.25, 0.5,
                                     1, 1.5, 2, 2.5, 3, 4, 5, 10, 25, 50, 100])
            upper_gaps = np.asarray([1, 2, 3, 4, 5, 10])
            if nsf > 2 or nsf < -2:
                d = np.abs(gap - upper_gaps * 10 ** nsf)
                ind = np.argmin(d)
                return upper_gaps[ind] * 10 ** nsf
            else:
                d = np.abs(gap - lower_gaps)
                ind = np.argmin(d)
                return lower_gaps[ind]

        tickgap = clean_gap(scale / (nticks - 1))
        nticks = int(scale / tickgap) + 1
        tickstart = np.round(offset / tickgap) * tickgap
        ticklabels = np.asarray(range(0, nticks)) * tickgap + tickstart
        ticks = (ticklabels - offset) / scale
        axargs = {'color': 'k', 'linewidth': 1}

        if not up:
            ticklength = -ticklength
        tickspot = np.linspace(0, 1, nticks)
        if flip < 0:
            ticks = 1 - ticks
        for tick, ticklabel in zip(ticks, ticklabels):
            if vertical:
                ax.plot([tick, tick], [level, level - ticklength], **axargs)
                ax.text(tick, level - ticklength * 2, ticklabel, horizontalalignment='center',
                        fontsize=fontsize / 2)
                ax.text(0.5, level - 5 * ticklength, label, horizontalalignment='center', fontsize=fontsize)
            else:
                ax.plot([level, level - ticklength], [tick, tick], **axargs)
                ax.text(level - ticklength * 2, tick, ticklabel, horizontalalignment='center',
                        fontsize=fontsize / 2)
                ax.text(level - 5 * ticklength, 0.5, label, horizontalalignment='center', fontsize=fontsize)

        if vertical:
            xlim = list(ax.get_xlim())
            if ticks.min() < xlim[0]:
                xlim[0] = ticks.min()
            if ticks.max() > xlim[1]:
                xlim[1] = ticks.max()
            ax.set_xlim(xlim)

            ax.plot([ticks.min(), ticks.max()], [level, level], **axargs)
        else:
            ylim = list(ax.get_ylim())
            if ticks.min() < ylim[0]:
                ylim[0] = ticks.min()
            if ticks.max() > ylim[1]:
                ylim[1] = ticks.max()
            ax.set_ylim(ylim)
            ax.plot([level, level], [ticks.min(), ticks.max()], **axargs)

    _, xscale, xoffset = scale_data(first_x, portion=portion)
    _, yscale, yoffset = scale_data(yt_mean, portion=portion)
    draw_axis(ax=ax, scale=xscale, offset=xoffset, level=0.0, label=xlabel,
              up=True, vertical=vertical)
    draw_axis(ax=ax, scale=yscale, offset=yoffset,
              flip=flip, level=depth, label=ylabel, up=False, vertical=vertical)

    fig.savefig(fig_name)

if __name__ == "__main__":
    npdata = np.random.randint(100, size=(10, 5, 100))  # (length, n_layers, n_frame)
    animate_multi_plots(npdata)