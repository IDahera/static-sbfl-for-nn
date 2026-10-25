# SPDX-License-Identifier: GPL-3.0-or-later
# DeepFault (c) 2019 Bogazici University -- https://github.com/hfeniser/DeepFault (commit 7a068e7).
# MODIFIED 2026-10-20 by Daniel Rashedi for Python 3.14 / TensorFlow 2.22 eager with Keras 3; every change is marked
# `# PORT:` and listed in README.md. Unmodified original: ../deepfault_upstream/.
from utils import normalize
import keras  # PORT: Keras 3 has no K.gradients / K.function
import tensorflow as tf  # PORT: gradients via tf.GradientTape (TF2 eager)
from keras import ops  # PORT: K.mean -> keras.ops.mean
import numpy as np
import random


def synthesize(model, x_original, suspicious_indices, step_size, d):

    input_tensor = model.inputs  # PORT: Sequential.input is undefined before a call in Keras 3

    perturbed_set_x = []
    perturbed_set_y = []
    original_set_x  = []

    for x in x_original:
        all_grads = []
        for s_ind in suspicious_indices:
            # PORT: symbolic K.gradients + K.function -> GradientTape on a
            # sub-model up to the suspicious layer. Same loss (mean activation
            # of the suspicious neuron), same gradient w.r.t. the input.
            sub_model = keras.Model(input_tensor, model.layers[s_ind[0]].output)
            x_in = tf.convert_to_tensor(np.expand_dims(x, axis=0))
            with tf.GradientTape() as tape:
                tape.watch(x_in)
                loss = ops.mean(sub_model([x_in])[..., s_ind[1]])
            grad_vals = tape.gradient(loss, x_in).numpy()
            all_grads.append(grad_vals[0])

        perturbed_x = x.copy()

        for i in range(x.shape[1]):
            for k in range(x.shape[2]):
                sum_grad = 0
                for j in range(len(all_grads)):
                    sum_grad += all_grads[j][0][i][k]
                avg_grad = float(sum_grad) / len(suspicious_indices)
                avg_grad = avg_grad * step_size

                # Clipping gradients.
                if avg_grad > d:
                    avg_grad = d
                elif avg_grad < -d:
                    avg_grad = -d

                perturbed_x[0][i][k] = max(min(x[0][i][k] + avg_grad, 1), 0)
                # perturbed_x.append(max(min(x[0][i][k] + avg_grad, 1), 0))

        '''
        for i in range(len(flatX)):
            sum_grad = 0
            for j in range(len(all_grads)):
                sum_grad += all_grads[j][0][i]

            avg_grad = float(sum_grad) / len(suspicious_indices)
            avg_grad = avg_grad * step_size

            if avg_grad > d:
                avg_grad = d
            elif avg_grad < -d:
                avg_grad = -d

            perturbed_x.append(max(min(flatX[i] + avg_grad, 1), 0))
        '''

        perturbed_set_x.append(perturbed_x)
        # perturbed_set_y.append(y)
        # original_set_x.append(x)

    return perturbed_set_x
