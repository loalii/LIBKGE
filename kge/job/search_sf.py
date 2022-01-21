import os
from kge.job import  Job
from kge import Config
import itertools
import random
import numpy as np

class SFSearchJob(Job):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.num_trials = self.config.get("sf_search.num_trials")
        self.structure_genarator = self.config.get("sf_search.structure_genarator")
        self.prob = self.config.get("sf_search.prob")
        self.K = self.config.get("sf_search.K")

        if self.__class__ == SFSearchJob:
            for f in Job.job_created_hooks:
                f(self)

    def _run(self):
        if(self.structure_genarator == 'random'): 
            structures = np.random.randint(0, self.K+1, (self.num_trials, self.K**2))
            As = structures.tolist()
            search_configs = []
            for index in range(0,self.num_trials):
                search_config = {'folder': 'struct'+str(index), 'autosf': {'A': As[index], 'K': self.K}}
                search_configs.append(search_config)

            self.config.set("search.type", "manual_search")
            self.config.set("manual_search.configurations", search_configs)
            self.config.save(os.path.join(self.config.folder, "config.yaml"))    

            if self.config.get("grid_search.run"):
                job = Job.create(self.config, self.dataset, parent_job=self)
                job.run()
            else:
                self.config.log("Skipping running of search job as requested by user...")
