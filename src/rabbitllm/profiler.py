import logging

import torch

logger = logging.getLogger(__name__)


class LayeredProfiler:
    def __init__(self, print_memory=False):
        self.profiling_time_dict = {}
        self.print_memory = print_memory
        self.min_free_mem = 1024 * 1024 * 1024 * 1024

    def add_profiling_time(self, item, time):
        if item not in self.profiling_time_dict:
            self.profiling_time_dict[item] = []

        self.profiling_time_dict[item].append(time)

        if self.print_memory:
            free_mem = torch.cuda.mem_get_info()[0]
            self.min_free_mem = min(self.min_free_mem, free_mem)
            logger.debug(
                "free vmem @%s: %.02fGB, min free: %.02fGB",
                item,
                free_mem / 1024 / 1024 / 1024,
                self.min_free_mem / 1024 / 1024 / 1024,
            )

    def clear_profiling_time(self):
        for item in self.profiling_time_dict.keys():
            self.profiling_time_dict[item] = []

    def print_profiling_time(self):
        for item in self.profiling_time_dict.keys():
            logger.info("total time for %s: %s", item, sum(self.profiling_time_dict[item]))
