"""StaticSBFL: static prediction of spectrum-based suspiciousness for neural networks.

Package layout:

    staticsbfl.variation       the architecture populations of the four phases
    staticsbfl.training        train / evaluate loops
    staticsbfl.suspiciousness  hit-spectrum accumulation and the SBFL metrics
    staticsbfl.dataset         the neuron table: static features + measured targets
    staticsbfl.evaluation      the final evaluation and the runtime comparison, as data
    staticsbfl.benchmark       the memory and time probes of the runtime comparison
    staticsbfl.log, .viz       console, log and HTML rendering
    staticsbfl.cli             the command-line entry points the Makefile calls
    staticsbfl.data            CIFAR-10 and Fashion-MNIST loaders
    staticsbfl.models          two fixed dense reference models, parameter counting
    staticsbfl.utils           device selection, seeding, paths
"""

__version__ = "2.0.0"
