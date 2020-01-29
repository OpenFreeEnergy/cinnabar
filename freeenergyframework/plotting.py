import numpy as np



def plot_DDGs(results,method_name='',target_name='',symmetrise=False):
    import matplotlib.pylab as plt
    import stats
    # data
    if not symmetrise:
        x_data = np.asarray([x.exp_DDG for x in results])
        y_data = np.asarray([x.calc_DDG for x in results])
        xerr = np.asarray([x.dexp_DDG for x in results])
        yerr = np.asarray([x.dcalc_DDG for x in results])
    else:
        x_data = np.asarray([x.exp_DDG for x in results]+[-x.exp_DDG for x in results])
        y_data = np.asarray([x.calc_DDG for x in results]+[-x.calc_DDG for x in results])
        xerr = np.asarray([x.dexp_DDG for x in results]+[x.dexp_DDG for x in results])
        yerr = np.asarray([x.dcalc_DDG for x in results]+[x.dcalc_DDG for x in results])
    nsamples = len(x_data)
    # aesthetics
    plt.figure(figsize=(10,10))
    plt.xlabel(f'Experimental / kcal mol$^{-1}$')
    plt.ylabel(f'Calculated {method_name} / kcal mol$^{-1}$')

    ax_min = min(min(x_data),min(y_data)) - 0.5
    ax_max = max(max(x_data),max(y_data)) + 0.5
    scale = [ax_min,ax_max]

    plt.xlim(scale)
    plt.ylim(scale)
    plt.plot([0, 0],scale, 'gray')
    plt.plot(scale, [0, 0], 'gray')
    plt.plot(scale,scale,'k:')
    small_dist = 0.5
    plt.fill_between(scale,[ax_min-small_dist,ax_max-small_dist],[ax_min+small_dist,ax_max+small_dist],color='grey',alpha=0.2)
    plt.fill_between(scale,[ax_min-small_dist*2,ax_max-small_dist*2],[ax_min+small_dist*2,ax_max+small_dist*2],color='grey',alpha=0.2)

    # actual plotting
    plt.scatter(x_data, y_data,color='hotpink')
    plt.errorbar(x_data, y_data, xerr=xerr, yerr=yerr,color='hotpink',linewidth=0.,elinewidth=2.)

    # stats and title

    stats = {
        statistic : stats.bootstrap_statistic(x_data, y_data, statistic=statistic)
        for statistic in ('RMSE', 'MUE') # DONT WANT TO RUN CORRELATION STATS ON DDGs
    }

    title = """{} (N = {})
    RMSE:  {:5.2f} [95%: {:5.2f}, {:5.2f}] kcal/mol
    MUE :  {:5.2f} [95%: {:5.2f}, {:5.2f}] kcal/mol
    """.format(target_name, nsamples,
               stats[('RMSE')]['mle'], stats[('RMSE')]['low'], stats[('RMSE')]['high'],

               stats[('MUE')]['mle'], stats[('MUE')]['low'], stats[('MUE')]['high']
              )
    plt.title(title, fontsize=11, loc='right', horizontalalignment='right', family='monospace');
    plt.show()


def plot_DGs(results,method_name='',target_name=''):
    import absolute
    import matplotlib.pylab as plt
    import stats
    # data
    x_data, y_data, xerr, yerr = absolute.generate_absolute_values(results)
    nsamples = len(x_data)
    # aesthetics
    plt.figure(figsize=(10,10))
    plt.xlabel(f'Experimental / kcal mol$^{-1}$')
    plt.ylabel(f'Calculated {method_name} / kcal mol$^{-1}$')

    ax_min = min(min(x_data),min(y_data)) - 0.5
    ax_max = max(max(x_data),max(y_data)) + 0.5
    scale = [ax_min,ax_max]

    plt.xlim(scale)
    plt.ylim(scale)
    plt.plot(scale,scale,'k:')
    small_dist = 0.5
    plt.fill_between(scale,[ax_min-small_dist,ax_max-small_dist],[ax_min+small_dist,ax_max+small_dist],color='grey',alpha=0.2)
    plt.fill_between(scale,[ax_min-small_dist*2,ax_max-small_dist*2],[ax_min+small_dist*2,ax_max+small_dist*2],color='grey',alpha=0.2)

    # actual plotting
    plt.scatter(x_data, y_data,color='hotpink')
    plt.errorbar(x_data, y_data, xerr=xerr, yerr=yerr,color='hotpink',linewidth=0.,elinewidth=2.)

    # stats and title

    stats = {
        statistic : stats.bootstrap_statistic(x_data, y_data, statistic=statistic)
        for statistic in ('RMSE', 'MUE', 'R2', 'rho')
    }

    title = """{} (N = {})
    RMSE:  {:5.2f} [95%: {:5.2f}, {:5.2f}] kcal/mol
    MUE :  {:5.2f} [95%: {:5.2f}, {:5.2f}] kcal/mol
    R2 :  {:5.2f} [95%: {:5.2f}, {:5.2f}] kcal/mol
    rho :  {:5.2f} [95%: {:5.2f}, {:5.2f}] kcal/mol
    """.format(target_name, nsamples,
               stats[('RMSE')]['mle'], stats[('RMSE')]['low'], stats[('RMSE')]['high'],
               stats[('MUE')]['mle'], stats[('MUE')]['low'], stats[('MUE')]['high'],
               stats[('R2')]['mle'], stats[('R2')]['low'], stats[('R2')]['high'],
               stats[('rho')]['mle'], stats[('rho')]['low'], stats[('rho')]['high']
              )
    plt.title(title, fontsize=11, loc='right', horizontalalignment='right', family='monospace')
    plt.show()


def plot_all_DGs(results,method_name='',target_name=''):
    import absolute
    import matplotlib.pylab as plt
    import stats
    import itertools
    # data
    x_abs, y_abs, xabserr, yabserr = absolute.generate_absolute_values(results)

    # do all to plot_all
    x_data = []
    y_data = []
    xerr = []
    yerr = []
    for a, b in itertools.combinations(range(len(x_abs)),2):
        x = x_abs[a] - x_abs[b]
        x_data.append(x)
        x_data.append(-x)
        err = (xabserr[a]**2 + xabserr[b]**2)**0.5
        xerr.append(err)
        xerr.append(err)
        y = y_abs[a] - y_abs[b]
        y_data.append(y)
        y_data.append(-y)
        err = (yabserr[a]**2 + yabserr[b]**2)**0.5
        yerr.append(err)
        yerr.append(err)
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
#     x_data = np.asarray(x_data)
#     y_data = np.asarray(y_data)
    nsamples = len(x_data)
    # aesthetics
    plt.figure(figsize=(10,10))
    plt.xlabel(f'Experimental / kcal mol$^{-1}$')
    plt.ylabel(f'Calculated {method_name} / kcal mol$^{-1}$')

    ax_min = min(min(x_data),min(y_data)) - 0.5
    ax_max = max(max(x_data),max(y_data)) + 0.5
    scale = [ax_min,ax_max]

    plt.xlim(scale)
    plt.ylim(scale)
    plt.plot([0, 0],scale, 'gray')
    plt.plot(scale, [0, 0], 'gray')
    plt.plot(scale,scale,'k:')
    small_dist = 0.5
    plt.fill_between(scale,[ax_min-small_dist,ax_max-small_dist],[ax_min+small_dist,ax_max+small_dist],color='grey',alpha=0.2)
    plt.fill_between(scale,[ax_min-small_dist*2,ax_max-small_dist*2],[ax_min+small_dist*2,ax_max+small_dist*2],color='grey',alpha=0.2)

    # actual plotting
    plt.scatter(x_data, y_data, color='hotpink')
    plt.errorbar(x_data, y_data, xerr=xerr, yerr=yerr,color='hotpink',linewidth=0.,elinewidth=2.)

    # stats and title

    stats = {
        statistic : stats.bootstrap_statistic(x_data, y_data, statistic=statistic)
        for statistic in ('RMSE', 'MUE')
    }

    title = """{} (N = {})
    RMSE:  {:5.2f} [95%: {:5.2f}, {:5.2f}] kcal/mol
    MUE :  {:5.2f} [95%: {:5.2f}, {:5.2f}] kcal/mol
    """.format(target_name, nsamples,
               stats[('RMSE')]['mle'], stats[('RMSE')]['low'], stats[('RMSE')]['high'],
               stats[('MUE')]['mle'], stats[('MUE')]['low'], stats[('MUE')]['high']
              )
    plt.title(title, fontsize=11, loc='right', horizontalalignment='right', family='monospace')
    plt.show()
