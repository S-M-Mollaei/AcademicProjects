# %% [markdown]
# # Description
# Add to the queueing simulator you have already developed (lab 1) some routines to:
# 
# a) detect the end of transient in an automated way (write a short report to describe the algorithm you have employed)
# 
# b) evaluate the accuracy of results.
# 
# Your code should employ a "batch means" technique that adaptively chooses  the number of batches so to achieve outputs with a desired degree of accuracy. 
# 
# Define properly the accuracy metric, which should be related to the width of confidence intervals.
# 
# Plot of the average delay in function of the utilisation, where the utilisation is: 0.1, 0.2, 0.4, 0.7, 0.8, 0.9, 0.95, 0.99. Show also the 95%-level confidence intervals.
# 
# Consider three scenarios for the service time:
# 
# EXP: exponentially distributed with mean=1
# 
# DET: deterministic =1
# 
# HYP:  distributed according to a hyper-exponential distribution with mean=1 standard deviation=10

# %%
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
import math
import pandas as pd
import matplotlib.pylab as plt
import ast

from typing import Callable, List

# %% [markdown]
# # Definition of functions and classes
# There are two main classes for clients and simulation with different useful attributes. Main funtions are related to arrival and departure of the clients, finding warm-up points, batch means delay and so on.  

# %%
TOTAL_SIMULATION_TIME = 100000
PLOT_DELAYS = False


class Customer:
    def __init__(self, arrival_data, service_start_date, service_time):
        """
        a class that virtualizes a job. contains all the information for a job
        :param arrival_data: the time at which the job arrives to the queue
        :param service_start_date: the time at which processing of job starts
        :param service_time: time spent to process the job
        """
        self.arrival_data = arrival_data
        self.service_start_data = service_start_date
        self.service_time = service_time
        self.service_end_time = self.service_start_data + self.service_time
        self.wait = self.service_end_time - self.arrival_data


class Simulation:
    def __init__(self, seed: int,
                 simulation_time: int,
                 utilization: float,
                 service_time_distribution: str,
                 accuracy: float,
                 confidence_level: float = 0.95):

        np.random.seed(seed)

        self.simulation_time = simulation_time
        self.utilization = utilization
        self.accuracy = accuracy
        self.confidence_level = confidence_level

        # stores the type of distribution used to generate random service times. it can be: EXP, DET and HYP
        self.service_time_distribution_name = service_time_distribution

        self.total_simulation_time = TOTAL_SIMULATION_TIME
        self.current_time = 0
        self.delays = []
        self.average_delay_list = []
        self.customers: List[Customer] = []
        self.warmup_ending_point: int = None
        self.find_transient_end: bool = True
        self.finish_simulation: bool = False

        self.batch_size: int = 0
        self.batches_means_list = []
        self.CI = ()
        self.CI_up = None
        self.CI_low = None

        # setting distribution parameters for both inter-arrival and service time
        # then defining the function which we should every time we want a random number
        # each time we invoke these function, they return a different random number
        self.inter_arrival_time_random_generator: Callable = lambda: np.random.exponential(1 / self.utilization)
        self._make_service_random_num_generator(service_time_distribution)

    def start_simulation(self):
        """
        Overall simulation for a defined setting is done in here
        """
        while self.current_time < self.total_simulation_time:

            if len(self.customers) == 0:
                arrival_date = self.inter_arrival_time_random_generator()
                service_start_date = arrival_date

            else:
                inter_arrival = self.inter_arrival_time_random_generator()
                arrival_date += inter_arrival

                # if the last customer has been processed then the current customer can be processed upon arrival
                # otherwise it should wait for the previous customer to end
                service_start_date = max(arrival_date, self.customers[-1].service_end_time)

            # each time service_time_random_generator is called, new random number based on the dist will be drawn
            service_time = self.service_time_random_generator()

            new_customer = Customer(arrival_date, service_start_date, service_time)
            self.customers.append(new_customer)
            self.delays.append(new_customer.wait)
            self.average_delay_list.append(sum(self.delays) / len(self.delays))

            # updating current time
            self.current_time = arrival_date

            if self.current_time > self.simulation_time:
                # after going to almost middle of simulation, we go back and find the end of the warmup period

                if self.find_transient_end:
                    self.warmup_ending_point = self._calculate_warmup_ending_point()
                    if PLOT_DELAYS:
                        return
                    # removing warm-up data
                    self.average_delay_list = self.average_delay_list[self.warmup_ending_point:]

                    self._calculate_batch_size()

                    # end of transient have been calculated
                    self.find_transient_end = False

                # divide into batched
                self.apply_batch_means()

                # this boolean variable is modified in apply_batch_means
                # if confidence interval of the batch means is as small as out expectation, we can halt the simulation
                if self.finish_simulation:
                    return
                else:
                    # if C.I is not small enough, then we have to go on with simulation
                    self.simulation_time += self.batch_size

    def _make_service_random_num_generator(self, service_time_distribution: str):
        """

        :param service_time_distribution: name of distribution
        :return: a callable function which returns a new random variable each time it is called
        """
        if service_time_distribution == "DET":
            self.service_time_random_generator = lambda: 1

        elif service_time_distribution == "EXP":
            self.service_time_random_generator = lambda: np.random.exponential(1)

        elif service_time_distribution == "HYP":
            self.service_time_random_generator = lambda: self._draw_hyper_exponential(0.6, 1.4, 0.5)

    @staticmethod
    def _draw_hyper_exponential(mu1: float, mu2: float, p: float) -> float:
        """

        :param mu1: rate for the fist exponential
        :param mu2: rate for the second exponential
        :param p: probability of selecting the first exponential
        :return: random number based on the hyper-exponential distribution
        """
        u = np.random.random()
        v = np.random.random()
        if u <= p:
            service = (-1 * mu1) * math.log(1 - v)
        else:
            service = (-1 * mu2) * math.log(1 - v)

        return service

    def plot_delay_average(self):
        """
        this function is for plotting the point at which the transient period ends
        """
        plt.plot(self.average_delay_list)
        plt.xlabel("step")
        plt.ylabel("average delay")
        plt.axvline(x=self.warmup_ending_point, color="red")
        plt.savefig("det_trans.png")
        plt.show()

    def _calculate_warmup_ending_point(self):
        """
        function which computes the point at which the transient period ends
        """
        # computing the average of the average delays
        mean = np.mean(self.average_delay_list)
        std = np.std(self.average_delay_list)

        # calculating the interval in which the values are acceptable
        lower_bound = mean - std
        upper_bound = mean + std

        starting_point = int(len(self.average_delay_list) * 0.2)

        # finding the point at which warm-up period ends
        for counter in range(starting_point, len(self.average_delay_list)):
            if (self.average_delay_list[counter] > lower_bound) and (self.average_delay_list[counter] < upper_bound):
                return counter

        # if no point was found
        return None

    def apply_batch_means(self):
        """
        In this function, the remaining data after the warmup period is divided into equal-sized batches
        Then their statistics are computed. If the CI is small enough, then simulation will be halted
        """

        idx = 0
        self.batches_means_list = []

        while idx < len(self.average_delay_list):
            # this line to avoid the out-of-range error
            batch_end_idx = idx + self.batch_size if (idx + self.batch_size) < len(self.average_delay_list) else \
                len(self.average_delay_list)

            self.batches_means_list.append(np.mean(self.average_delay_list[idx: batch_end_idx]))

            idx += self.batch_size

        self.finish_simulation, self.CI_low, self.CI_up = self._compute_batch_means_ci()

    def _calculate_batch_size(self):
        """
        Assuming that the initial number of batches should be 10, the number of instances in each batch is calculated
        """
        self.batch_size = int(len(self.average_delay_list) / 10)

    def _compute_batch_means_ci(self):
        """
        This function is in charge of calculating the statistics of batch means.
        """
        mu = np.mean(self.batches_means_list)  # mean of batch means
        std = np.std(self.batches_means_list)  # standard deviation of batch means
        n = len(self.batches_means_list)
        t = np.abs(ss.t.ppf((1 - self.confidence_level) / 2, n - 1))

        z = t * std / np.sqrt(n)

        interval = 2 * z / mu

        # if interval is small enough, it would be counted as acceptable to finish the simulation
        if interval > self.accuracy:
            is_acceptable = False
        else:
            is_acceptable = True

        if len(self.batches_means_list) == 30:
            is_acceptable = True

        return is_acceptable, mu - z, mu + z


if __name__ == "__main__":
    stat_df: pd.DataFrame = pd.DataFrame(
        columns=["utilization", "distribution", "batch_count", "batch_ci_low", "batch_ci_up", "batch_mean"])
    utilization_list = [0.1, 0.2, 0.4, 0.7, 0.8, 0.9, 0.95, 0.99]
    distribution_list = ["DET", "EXP", "HYP"]

    for utilization in utilization_list:
        for distribution in distribution_list:
            sim = Simulation(seed=32, simulation_time=10000, utilization=utilization,
                             service_time_distribution=distribution,
                             accuracy=0.1)
            sim.start_simulation()

            df_row = {"utilization": utilization, "distribution": distribution,
                      "batch_count": len(sim.batches_means_list),
                      "batch_ci_low": sim.CI_low, "batch_ci_up": sim.CI_up,
                      "batch_mean": np.mean(sim.batches_means_list)}

            stat_df = pd.concat([stat_df, pd.DataFrame([df_row])], ignore_index=True)

    print(stat_df.sort_values(["distribution", "utilization"]))

    plt.figure(figsize=(10, 5))

    distribution_dict = {"DET": "red", "EXP": "purple", "HYP": "yellow"}

    for distribution in distribution_dict.keys():
        df = stat_df.loc[stat_df.distribution == distribution]
        plt.plot(df["utilization"], df["batch_mean"])
        plt.xlabel("Utilization")
        plt.ylabel("Mean of batch means")
        plt.fill_between([i for i in df["utilization"]], df["batch_ci_low"], df["batch_ci_up"], alpha=.5,
                         color=distribution_dict[distribution], label=f"{distribution} CI")
    plt.legend(loc = 'upper left')
    plt.savefig("utilization_vs_delay.png")
    plt.show()


# %% [markdown]
# # Plotting all average delay vs simulation time considering transient point

# %%
df_plot = pd.read_csv('simulation_results.csv', header=None)
df_plot_drop = df_plot.drop(0, axis=1)

for key in df_plot_drop.keys():
    delays_plot = ast.literal_eval(df_plot_drop[key][7])
    k_point = ast.literal_eval(df_plot_drop[key][2])
    plt.plot(delays_plot)
    plt.vlines(k_point, min(delays_plot), max(delays_plot), colors='r')
    plt.xlabel('Simulation Time')
    plt.ylabel('Average Delay')
    plt.title(f'util. {df_plot_drop[key][0]}, dist. {df_plot_drop[key][1]}')
    plt.savefig(f'util. {df_plot_drop[key][0]}, dist. {df_plot_drop[key][1]}.png')
    plt.show()

    



