# SPDX-License-Identifier: GPL-3.0-or-later
# Part of a modified version of DeepFault (c) 2019 Bogazici University; written 2026-10-20 by
# Daniel Rashedi for the diss-code runtime comparison. Distributed under the GPL-3.0-or-later.
"""PORT (DF@tf1): restore the TensorFlow 1 execution model on TensorFlow 2.

DeepFault was written for Keras 2.3.1 on TensorFlow 1.13, where every Keras call
builds a static graph that a ``tf.Session`` executes. TensorFlow 1.x has no
build for Python 3.14 (or for arm64 macOS), so this port runs on TensorFlow 2
with eager execution switched off *before* any layer is created. Together with
the legacy Keras 2 package ``tf_keras`` this keeps the original semantics:
``K.function`` compiles a graph function that runs in a session, ``K.gradients``
is symbolic differentiation, and ``layer.output`` is a graph tensor.

Every DeepFault module imports this file first; it has no other effect.
"""

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
