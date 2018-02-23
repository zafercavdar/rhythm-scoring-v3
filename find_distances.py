import rhythm
import glob
import os
import rhythm_similarity as rs
from distance import calc_distance
from io_ops import save

scorer = rs.RhythmScorer()


class RefOnset(object):

    def __init__(self, name, onsets):
        self.name = name
        self.onset_times = []
        for element in onsets:
            self.onset_times.append(element)


class Bundle(object):

    def __init__(self, identity):
        self.identity = identity
        self.refs = []
        self.pers = []

    def addref(self, ref):
        self.refs.append(ref)

    def addper(self, per):
        self.pers.append(per)

    def __str__(self):
        return "{}: refs: {} - pers: {}".format(self.identity, len(self.refs), len(self.pers))


def findBundle(iden):
    if bundles.get(iden, None) is not None:
        return bundles[iden]
    else:
        newBundle = Bundle(iden)
        bundles[iden] = newBundle
        return newBundle


def createBundles(directory):
    os.chdir(directory)
    for file in glob.glob("*.wav"):
        name = file.split(".", 1)[0]
        yearset = name.split("_", 4)[0]
        ritimno = name.split("_", 4)[1]
        refper = (name.split("_", 4)[2])[0:3]
        iden = yearset + "_" + ritimno
        bundle = findBundle(iden)
        if refper == "ref":
            bundle.addref(name)
        elif refper == "per":
            bundle.addper(name)
    print("Bundles are created.")


def saveDistances(ref, per, distanceSet, jury, timing):
    write_file = "results/"
    iden = per.split("_")[0] + "_" + per.split("_")[1]
    write_file += iden + ".txt"

    result = dict(distanceSet)
    result["timing"] = timing
    result["jury"] = jury
    result["ref"] = ref
    result["per"] = per

    save(write_file, result)


def main():
    sample_path = "samples/"
    createBundles(sample_path)

    # for key, value in bundles.iteritems():
    #    print(value)

    os.chdir("..")
    bundleNo = 1
    start = 31
    count = 0
    for key, bundle in bundles.iteritems():
        refs = bundle.refs
        pers = bundle.pers
        ref_objs = []
        print("Creating ref objects for {} ... Please wait".format(bundleNo))

        if count < start:
            count += 1
            bundleNo += 1
            continue

        for ref_id in refs:
            ref_path = sample_path + ref_id + ".wav"
            full_path = "{}/{}".format(os.getcwd(), ref_path)
            ref = rhythm.Rhythm(full_path)
            ref_objs.append(ref)

        for per_id in pers:
            print("processing {}".format(per_id))
            per_path = sample_path + per_id + ".wav"
            full_path = "{}/{}".format(os.getcwd(), per_path)
            result = per_id.split("_", 4)[3]  # pass or fail
            per = rhythm.Rhythm(full_path)
            for i in range(0, len(ref_objs)):
                print("\twith {}".format(refs[i]))
                ref = ref_objs[i]
                per.reset()
                per.set_rec(ref, 1)				# 1:adaptive stretch, 0:cumulative stretch

                score = scorer.compare(ref, per, stretch=1)
                timing = score[0]

                min_len = min(len(ref.onset_times), len(per.stretched_onset_times))
                if min_len < 3:
                    continue
                ref_onset_times = ref.onset_times[0:min_len]
                per_onset_times = per.stretched_onset_times[0:min_len]

                distanceSet = calc_distance(ref_onset_times[1:-1], per_onset_times[1:-1])
                saveDistances(refs[i], per_id, distanceSet, result, timing)

        print("completed processing {}".format(bundleNo))
        bundleNo += 1


bundles = {}
main()
