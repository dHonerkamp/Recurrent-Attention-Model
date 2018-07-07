import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


class Visualization:
    def __init__(self, model, FLAGS):
        self.num_glimpses = FLAGS.num_glimpses
        self.num_scales   = len(FLAGS.scale_sizes)
        self.scale_sizes  = FLAGS.scale_sizes
        self.img_shape    = FLAGS.img_shape
        self.path         = FLAGS.path
        self.img_shape_squeezed = (self.img_shape[:2] if self.img_shape[2] == 1 else self.img_shape)
        os.makedirs(os.path.join(FLAGS.path, 'glimpses'), exist_ok=True)

        self.model = model
        self.vars = ['step', 'batch_xs', 'batch_ys', 'locs', 'gl_composed', 'preds', 'step_preds']
        self.fetch = [model.global_step, model.x, model.y, model.locs, model.glimpses_composed,
                      model.prediction, model.intermed_preds]

    def __call__(self, sess, prefix, handle, nr_obs_overview=8, nr_obs_reconstr=6):
        feed =   {self.model.is_training: False,
                  self.model.handle: handle}
        out = sess.run(self.fetch, feed_dict=feed)
        d = dict(zip(self.vars, out))

        # extract_glimpses: (-1,-1) is top left. 0 is y-axis, 1 is x-axis. Scale of imshow shifted by 1.
        d['locs'] = np.clip(d['locs'], -1, 1)
        d['locs'] = (d['locs'] / 2 + 0.5) * self.img_shape[1::-1] - 1  # in img_shape y comes first, then x

        self._plot_overview(d, prefix, nr_obs_overview)

    def _plot_img_plus_locs(self, d, ax, n, nr_examples):
        ax.imshow(d['batch_xs'][n].reshape(self.img_shape_squeezed), cmap='gray')
        ax.set_title('Label: {} Prediction: {}'.format(d['batch_ys'][n], d['preds'][n]))

        for i in range(0, self.num_glimpses):
            c = ('green' if d['preds'][n] == d['batch_ys'][n] else 'red')
            if i == 0:
                m = 'x'; fc = c
            else:
                m = 'o'; fc = 'none'
            # plot glimpse location
            ax.scatter(d['locs'][n, i, 1], d['locs'][n, i, 0], marker=m, facecolors=fc, edgecolors=c,
                       linewidth=2.5, s=0.25 * (5 * nr_examples * 24))
            # connecting line
            ax.plot(d['locs'][n, i - 1:i + 1, 1], d['locs'][n, i - 1:i + 1, 0], linewidth=2.5, color=c)
            # rectangle around location?
            # ax.add_patch(Rectangle(locs[n,i][::-1] - FLAGS.scale_sizes[0] / 2, width=FLAGS.scale_sizes[0], height=FLAGS.scale_sizes[0], edgecolor=c, facecolor='none'))

        ax.set_ylim([self.img_shape[0] - 1, 0])
        ax.set_xlim([0, self.img_shape[1] - 1])
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_composed_glimpse(self, d, ax, n, t):
        ax.imshow(d['gl_composed'][t][n].reshape(2 * [np.max(self.scale_sizes)] + [self.img_shape[-1]]).squeeze(), cmap='gray')
        ax.set_title('class {}: {:.3f}'.format(d['step_preds'][t][0][n], d['step_preds'][t][1][n]))
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_overview(self, d, prefix, nr_examples):
        f, axes = plt.subplots(nr_examples, self.num_glimpses + 1, figsize=(6 * self.num_glimpses, 5 * nr_examples))
        axes = axes.reshape([nr_examples, self.num_glimpses + 1])

        for n in range(nr_examples):
            self._plot_img_plus_locs(d, axes[n, 0], n, nr_examples)
            [self._plot_composed_glimpse(d, axes[n, t + 1], n, t) for t in range(self.num_glimpses)]

        f.tight_layout()
        f.savefig('{}/glimpses/{}_{}.png'.format(self.path, d['step'], prefix), bbox_inches='tight')
        plt.close(f)
