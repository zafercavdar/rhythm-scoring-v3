import numpy as np
from rhythm import Rhythm

class RhythmScorer(object):
    def __init__(self):
        pass

    def compare(self, ref, rec, stretch):
        difflist = []

        if stretch == 1:
            #print("Time align: ", stretch)
            rec.final_onset_signal = rec.stretched_onset_signal
            rec.final_onset_times = rec.stretched_onset_times
        else:
            print("Time align: ", stretch)
            rec.final_onset_signal = rec.shifted_onset_signal
            rec.final_onset_times = rec.shifted_onset_times

        #print("Ref, rec lengths:", len(ref.onset_times), len(rec.final_onset_times))

        min_len = min(len(ref.onset_times), len(rec.final_onset_times))
        for i in range(0, min_len):
            difflist.append(ref.onset_times[i] - rec.final_onset_times[i])  #old version
        	#difflist.append(ref.onset_times[i] - rec.onset_times[i])
        #print("Difflist:", difflist)

        return (self.timing_similarity(difflist, ref.bpm)[0], self.hamming_similarity(difflist,ref.bpm)[0])

    def timing_similarity(self, difflist, bpm):

        #print("BPM:", bpm)
        diffrange = 60. / bpm    #time for 1/4th beat
        #print("diffrange:", diffrange)
        scorelist = []
        for diff_ in difflist:
            diff = abs(diff_)
            if diff < diffrange / 32.:       #128th
                scorelist.append(1)
            elif diff < diffrange / 16.:     #64th
                scorelist.append(0.9)
            elif diff < diffrange / 8.:      #32th
                scorelist.append(0.8)
            elif diff < diffrange / 4.:      #16th
                scorelist.append(0.3)
            elif diff < diffrange / 2.:      #4th
                scorelist.append(0.2)
            else:
                scorelist.append(0.1)
        #print(scorelist)
        if len(scorelist) <= 2:
            score = 0
        else:
            score = float(sum(scorelist[1:-1])) / (len(scorelist) - 2)
        #print("Timing similarity:", score)
        return score, scorelist

    def hamming_similarity(self, difflist, bpm):

        #print("BPM:", bpm)
        diffrange = 60. / bpm    #time for 1/4th beat
        #print("diffrange:", diffrange)
        scorelist = []
        for diff_ in difflist:
            diff = abs(diff_)
            if diff < diffrange / 32.:       #128th
                scorelist.append(1)
            elif diff < diffrange / 16.:     #64th
                scorelist.append(1)
            elif diff < diffrange / 8.:      #32th
                scorelist.append(1)
            elif diff < diffrange / 4.:      #16th
                scorelist.append(0)
            elif diff < diffrange / 2.:      #4th
                scorelist.append(0)
            else:
                scorelist.append(0)
        #print(scorelist)
        if len(scorelist) <= 2:
            score = 0
        else:
            score = float(sum(scorelist[1:-1])) / (len(scorelist) - 2)
        #print("Timing similarity:", score)
        return score, scorelist

    def accent_similarity(self, ref, rec):
        accent_sims = list()
        cnt = 0
        for eng in ref.onset_energies:
            print("velocities:", eng, rec.onset_energies[cnt])
            if eng == rec.onset_energies[cnt]:
                accent_sims.append(1.0)
            else:
                accent_sims.append(0.0)
            cnt += 1
        accent_sim = sum(accent_sims) / len(accent_sims)
        print(accent_sim)
        return accent_sim
