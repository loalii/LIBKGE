import os
from kge.job import AutoSearchJob, Job
from kge import Config
import itertools


class SFSearchJob(Job):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

        if self.__class__ == SFSearchJob:
            for f in Job.job_created_hooks:
                f(self)

    def _run(self):

        sf_configs = self.config.get("sf_search.parameters")
        for k, v in sorted(Config.flatten(sf_configs).items()):
            
