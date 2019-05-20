import numpy as np
import numexpr as ne

def PDF(pos, nb_samples,bins):
    samples = np.random.choice(range(len(pos)),nb_samples)
    dists = []
    for s in samples:
        sample = pos[s,:]
        #print(sample)
        #print(np.sqrt(np.sum((pos-sample)**2,axis=1)))
        dists += [a for a in np.sqrt(ne.evaluate("sum((pos-sample)**2,axis=1)")) if a]
    hist, edges = np.histogram(dists,bins=bins,weights=1/np.array(dists))
    return edges[:-1], hist/nb_samples
