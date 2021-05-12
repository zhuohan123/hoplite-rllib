from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import ray
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.memory import ray_get_and_free

import ray.rllib.utils.hoplite as hoplite

class AsyncGradientsOptimizer(PolicyOptimizer):
    """An asynchronous RL optimizer, e.g. for implementing A3C.

    This optimizer asynchronously pulls and applies gradients from remote
    workers, sending updated weights back as needed. This pipelines the
    gradient computations on the remote workers.
    """

    def __init__(self, workers, grads_per_step=100, broadcast_interval=4, use_hoplite=False):
        """Initialize an async gradients optimizer.

        Arguments:
            grads_per_step (int): The number of gradients to collect and apply
                per each call to step(). This number should be sufficiently
                high to amortize the overhead of calling step().
        """
        PolicyOptimizer.__init__(self, workers)

        self.apply_timer = TimerStat()
        self.wait_timer = TimerStat()
        self.dispatch_timer = TimerStat()
        self.grads_per_step = grads_per_step
        self.learner_stats = {}
        self.broadcast_interval = broadcast_interval
        self.use_hoplite = use_hoplite
        if not self.workers.remote_workers():
            raise ValueError(
                "Async optimizer requires at least 1 remote workers")

    @override(PolicyOptimizer)
    def step(self):
        weights = ray.put(self.workers.local_worker().get_weights())
        pending_gradients = {}
        object_ids = {}
        num_gradients = 0

        # Kick off the first wave of async tasks
        for e in self.workers.remote_workers():
            e.set_weights.remote(weights)
            object_id = hoplite.random_object_id()
            future = e.compute_gradients.remote(e.sample.remote(), object_id)
            pending_gradients[future] = e
            object_ids[future] = object_id
            num_gradients += 1

        while pending_gradients:
            with self.wait_timer:
                wait_results = ray.wait(
                    list(pending_gradients.keys()),
                    num_returns=min(self.broadcast_interval, len(pending_gradients)))
                ready_list = wait_results[0]
                all_gradients = []
                reduce_object_ids = []
                finished_workers = []
                for future in ready_list:
                    gradient, info = ray_get_and_free(future)
                    all_gradients.append(gradient)
                    e = pending_gradients.pop(future)
                    object_id = object_ids.pop(future)
                    finished_workers.append(e)
                    reduce_object_ids.append(object_id)
                    self.learner_stats = get_learner_stats(info)
                if self.use_hoplite:
                    gradient = all_gradients[0]  # meta data
                else:
                    gradient = [np.mean(ws, axis=0) for ws in zip(*all_gradients)]

            if gradient is not None:
                with self.apply_timer:
                    self.workers.local_worker().apply_gradients(gradient, object_ids=reduce_object_ids)
                self.num_steps_sampled += info["batch_count"] * len(all_gradients)
                self.num_steps_trained += info["batch_count"] * len(all_gradients)

            with self.dispatch_timer:
                weights = None
                for e in finished_workers:
                    if num_gradients < self.grads_per_step:
                        if weights is None:
                            weights = self.workers.local_worker().get_weights()
                        e.set_weights.remote(weights)
                        object_id = hoplite.random_object_id()
                        future = e.compute_gradients.remote(e.sample.remote(), object_id)
                        pending_gradients[future] = e
                        object_ids[future] = object_id
                        num_gradients += 1

    @override(PolicyOptimizer)
    def stats(self):
        return dict(
            PolicyOptimizer.stats(self), **{
                "wait_time_ms": round(1000 * self.wait_timer.mean, 3),
                "apply_time_ms": round(1000 * self.apply_timer.mean, 3),
                "dispatch_time_ms": round(1000 * self.dispatch_timer.mean, 3),
                "learner": self.learner_stats,
            })
