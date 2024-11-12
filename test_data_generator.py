from scipy.stats import norm
import numpy as np
import pickle
from random import sample
from scipy.stats import truncnorm

def syntetic_data(
        batch_size=10000,
        n_batches=100,
        n_effects=(1, 4, 9, 16),
        n_sizes=[100, 200, 400, 800],
        n_null=[0.2, 0.4, 0.6, 0.8]
        ):
    
    dataset = []
    for n in n_effects:
        for _ in range(n_batches):
            e_s = []
            n_e = []
            b = batch_size
            for k in range(n):
                e_s.append(np.random.uniform(0.1, 4))

                if k + 1 == n:
                    n_e.append(b)
                else:
                    num = np.random.randint(1, int(b / 2))
                    n_e.append(num)
                    b -= num

            data = np.concatenate([truncnorm.rvs(a=-j, b=np.inf, loc=j, scale=1, size=n_e[i]) for i, j in enumerate(e_s)])
            dataset.append(data)

    h0_effects = []
    for n_size in n_sizes:
        for n_nu in n_null:
            for d in dataset:
                sampled_data = np.random.choice(d, int((1 - n_nu) * n_size), replace=False)
                h0_null_data = truncnorm.rvs(a=0, b=np.inf, loc=0, scale=1, size=int(n_nu * n_size))
                data = np.concatenate([h0_null_data, sampled_data])
                h0_effects.append({'data':data})


    h1_effects = []
    for bias in range(5, 100, 5):

        for n_nu in n_null:
            for d in dataset:
                # Create the source batch
                source_batch = np.concatenate([
                    np.random.choice(d, int(batch_size*(1-n_nu))),
                    truncnorm.rvs(a=0, b=np.inf, loc=0, scale=1, size=int(n_nu * batch_size))
                ])

                # Find the indices of values below 1.95
                below_threshold_indices = np.where(source_batch < 1.95)[0]

                # Determine how many values to remove based on the bias percentage
                num_to_remove = int(len(below_threshold_indices) * (bias / 100))

                # Randomly choose indices to remove
                if num_to_remove > 0:
                    remove_indices = np.random.choice(below_threshold_indices, num_to_remove, replace=False)

                    # Remove the chosen indices from the source_batch
                    source_batch = np.delete(source_batch, remove_indices)
            
                missing = num_to_remove/10000

                for n_size in n_sizes:
                    data = np.random.choice(source_batch,n_size,replace=False)

                    h1_effects.append({"bias":bias,"size":n_size, "missing":missing,"data":data})

    return h0_effects, h1_effects

np.random.seed(2137)
h0, h1 = syntetic_data()

with open('h0.pkl', 'wb') as file:
     pickle.dump(h0, file)

with open('h1.pkl', 'wb') as file:
     pickle.dump(h1, file)


filtered_h1 = [effect for effect in h1 if np.sum(effect['data'] > 1.96) >= 40]
sample_to_tests = sample(filtered_h1,2000)

with open('sample_to_tests.pkl', 'wb') as file:
     pickle.dump(sample_to_tests, file)


i = 1
for bias in range(5, 100, 5):
    print(i)
    i+=1