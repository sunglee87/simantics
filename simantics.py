# importing the module
import json
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics


def beta_random_generator(k1, k2, display_plot, type):
    """
    :param k1: first parameter to create beta distribution
    :param k2: second parameter to create beta distribution
    :param display_plot: a flag to indicate the last run for the monte carlo to create a plot.
    :param type: the type of plot being created: normal, pessimistic, or optimistic
    :return: a random point from the beta distribution created chosen uniformly
    """
    mean, var, skew, kurt = beta.stats(k1, k2, moments='mvsk')

    x = np.linspace(beta.ppf(0.01, k1, k2),
                    beta.ppf(0.99, k1, k2), 100)

    rv = beta(k1, k2)

    # calculate the mean
    mean = beta.mean(k1, k2, loc=0, scale=1)
    vals = beta.ppf([0.001, 0.5, 0.999], k1, k2)
    np.allclose([0.001, 0.5, 0.999], beta.cdf(vals, k1, k2))
    r = beta.rvs(k1, k2, size=1000)

    # if plot is set to true, create plots
    if display_plot==True:
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, beta.pdf(x, k1, k2),'r-', lw=5, alpha=0.6, label='beta pdf')
        ax.plot(mean, .05, color='green', marker='|', markersize=20)
        ax.set_title(type, fontsize=20)
        ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
        ax.legend(loc='best', frameon=False)
        plt.show()

    # pick values based on uniform distribution from the r array output
    std = beta.std(k1, k2, loc=0, scale=1)
    cdf = beta.cdf(x, k1, k2, loc=0, scale=1)
    u = np.random.choice(r)

    return u

def import_json(source_file):
    """
    This method imports the json file and parses out the jira information to extract info to feed into creating project
    estimate.
    :param source_file: the full source path for the json file to be imported
    :return: original_project_duration: the combined hours from jira divided by the number of developers
    """
    # Opening JSON file
    with open(source_file) as json_file:
        data = json.load(json_file)
        estimate_array = []
        total_man_days = 0.0
        num_developers = 3
        hours = 0.0
        num_issues = 0

        # Print the data of dictionary
        for i in range(0,len(data)):
            print("\nissue_name:", data[i]['issue_name'])
            print(" timeSpent:", data[i]['timeSpent'])
            print(" due_date:", data[i]['due_date'])
            print(" assigned_to:", data[i]['assigned_to'])
            print(" status:", data[i]['status'])
            print(" originalEstimate:", str(int(data[i]['originalEstimate']) / 60 / 60))

            # if the original estimate is populated, convert it to hours and add it to the array
            if data[i]['originalEstimate'] != None:
                estimate_array.append(data[i]['originalEstimate'] / (60 * 60))
                days = data[i]['originalEstimate']/(60*60*8)
                total_man_days = total_man_days + days

            #increment number of stories
            num_issues = num_issues + 1

    avg_time_per_issue = total_man_days/num_issues
    original_project_duration = total_man_days/num_developers

    #print initial JIRA information
    print("\n--++ Initial summary from JIRA:")
    print("    estimates for each issue in hours:", estimate_array)
    print("    total_man_days:", total_man_days)
    print("    Number of Issues:", num_issues)
    print("    Number of Developers:", num_developers)
    print("    Average time to implement each feature:", "{:.2f}".format(avg_time_per_issue), "days")

    return original_project_duration

def process_monte_carlo(original_project_duration):
    """
    Run the monte carlo simulation a set number of times
    :param original_project_duration: the combined hours from jira divided by the number of developers
    """

    # initialize variables
    display_plot = False
    max_run = 100 #number of simulations
    std_dev = 0
    average_array = []
    pessimistic_array = []
    optimistic_array = []

    print("\n--+ Beta Distribution:")
    for i in range(0,max_run):

        # only show plots for the very last run
        if i==max_run-1:
            display_plot = True

        # average case: chance it is over time is 50%
        mid_average = beta_random_generator(4, 4, display_plot, 'normal')
        average_array.append(mid_average)

        # pessimistic case: chance it is over time is high
        mid_pessimistic = beta_random_generator(3+math.sqrt(2), 3-math.sqrt(2), display_plot, 'pessimistic beta')
        pessimistic_array.append(mid_pessimistic)

        # optimistic case: chance it is over time is low
        mid_optimistic = beta_random_generator(5-math.sqrt(2), 5+math.sqrt(2), display_plot, 'optimistic beta')
        optimistic_array.append(mid_optimistic)


    print("    average array:",average_array)
    print("    pessimistic array:",pessimistic_array)
    print("    optimistic array:", optimistic_array)

    # get the averages
    mid_average = statistics.mean(average_array)
    mid_pessimistic = statistics.mean(pessimistic_array)
    mid_optimistic = statistics.mean(optimistic_array)

    print("    monte carlo average for average array", "{:.2f}".format(mid_average))
    print("    monte carlo average for pessimistic array", "{:.2f}".format(mid_pessimistic))
    print("    monte carlo average for optimistic array", "{:.2f}".format(mid_optimistic))

    # calculate estimated project duration based on completion percentage and std deviation
    pessimistic_project_duration = (mid_pessimistic/mid_average) * original_project_duration
    optimistic_project_duration = (mid_optimistic/mid_average) * original_project_duration
    pert_project_duration = (pessimistic_project_duration
                             + (4 * original_project_duration)
                             + optimistic_project_duration) / 6

    # Print out final statistics and project duration
    print("\n--+ Final Statistics and Project Estimates:")
    print("    Original Estimate:", "{:.2f}".format(original_project_duration), "days")
    print("    Pessimistic Project Estimate:", "{:.2f}".format(pessimistic_project_duration) , "days")
    print("    Optimistic Project Estimate:", "{:.2f}".format(optimistic_project_duration) , "days")
    print("    Pert Project Estimate:", "{:.2f}".format(pert_project_duration) , "days")

""" ------------------- MAIN ------------------------------------------- """
# import json and parse jira information
original_project_duration = import_json('C:/Users/sungl/Downloads/results (10).json')

# run the monte carlo simulation to produce project estimates
process_monte_carlo(original_project_duration)
