import numpy as np
import essentia.standard as e


class Rhythm(object):

    def __init__(self, fname):
        self.fpath = fname

        self.signal = None              #__get_signal__
        self.signal_length = None       #__get_signal__
        self.is_rec = False

        self.duration = None            #__get_duration__
        self.bpm = None                 #__get_bpm__

        self.onset_candidates = []      #onset_candidates
        self.frames = list()            #onset_candidates
        self.frame_count = None         #onset_candidates

        self.onset_times = None         #onset_decider
        self.onset_frames = list()      #onset_decider
        self.onset_count = None         #onset_decider

        self.onset_energies = list()    #__get_onset_energies__
        ## end of fetching data from original signal

        ##creating useful data for calculations
        self.onset_signal = None
        self.shifted_onset_times = []
        self.shifted_onset_signal = []

        self.stretched_onset_times = []
        self.stretched_onset_signal = []

        self.final_onset_times = []
        self.final_onset_signal = []

        self.__get_signal__()
        self.__get_duration__()
        self.__get_bpm__()
        self.__onset_candidate_detection__()
        self.__onset_decider__(0.20)

    def reset(self):
        self.shifted_onset_times = []
        self.shifted_onset_signal = []

        self.stretched_onset_times = []
        self.stretched_onset_signal = []

        self.final_onset_times = []
        self.final_onset_signal = []


    def set_rec(self, ref, mode):
        self.is_rec = True
        try:
            self.__rec_onset_decider__(ref)
            self.__shift_times__(ref)
            if mode == 0:
                self.__stretch_cumulative__(ref)
                #print("Stretching cumulatively.")
            elif mode == 1:
                self.__stretch_adaptive__(ref)
                #print("Stretching adaptively.")
            else:
                pass
        except Exception:
            print("ERROR: Recording signal is not good.", self.fpath)
            return -1

    def __get_signal__(self):
        """
        :rtype: returns the audio signal by reading the file
        """
        e_monoloader = e.MonoLoader(filename=self.fpath)
        self.signal = e_monoloader()
        self.signal_length = len(self.signal)

    def __get_duration__(self):
        e_duration = e.Duration()
        self.duration = e_duration(self.signal)

    def __get_bpm__(self):
        e_rhythmextractor2013 = e.RhythmExtractor2013(maxTempo=120, minTempo=40)
        bpm, ticks, confidence, estimates, bpmintervals = e_rhythmextractor2013(self.signal)
        #print("bpm:", bpm)
        assert isinstance(bpm, object)
        self.bpm = bpm

    def __onset_candidate_detection__(self):
        spectrum = e.Spectrum()
        e_onsetdetection = e.OnsetDetection(method="flux")

        onsetspecs = []
        for frame in e.FrameGenerator(self.signal, 1024, 512):
            self.frames.append(frame)
            onsetspecs.append(spectrum(frame))
            self.onset_candidates.append(e_onsetdetection(onsetspecs[-1], [0]*len(onsetspecs[-1])))

        self.frame_count = len(self.frames)

    def __get_onset_frames__(self):
        self.onset_frames = [int((len(self.frames) - 1) * t / self.duration) for t in self.onset_times]
        #print(self.onset_frames)
        #return onsetframes

    def __get_onset_energies__(self):
        self.onset_energies = []

        e_energy = e.Energy()
        englist = list()

        for frame in self.onset_frames:
            #print(frame, len(self.frames))
            eng = e_energy(self.frames[frame])
            englist.append(eng)
        #print("onset energies initial:", englist)
        maxeng = max(englist)
        for eng in englist:
            if eng > maxeng/2:
                self.onset_energies.append(1)
            else:
                self.onset_energies.append(0.5)
        #englist = [eng / maxeng for eng in englist]
        #print("onset energies normalized:", self.onset_energies)

        #self.onset_energies = englist

    def __time2signal__(self, timelist):
        SR = 44100
        #tempbeatsignal = [0] * self.length
        tempbeatsignal = [0] * int(1 + max(timelist) * SR)
        cnt = 0
        for x in timelist:
            #print(int(x * SR), len(tempbeatsignal))
            tempbeatsignal[int(x * SR)] = self.onset_energies[cnt]  #temporarily disabled for generating figures
            #tempbeatsignal[int(x * SR)] = 1
            cnt += 1
        #self.onset_signal = tempbeatsignal
        return tempbeatsignal

    def __onset_decider__(self, noise_threshold):

        e_onsets = e.Onsets(silenceThreshold=noise_threshold, frameRate=44100 / 512.0)
        onsetdetectsM = [self.onset_candidates]
        onsetresults = e_onsets(onsetdetectsM, [1])
        self.onset_times = onsetresults
        self.onset_count = len(self.onset_times)

        self.__get_onset_frames__()
        self.__get_onset_energies__()
        self.onset_signal = self.__time2signal__(self.onset_times)

    def __rec_onset_decider__(self, ref):

        nof_tries = 0
        threshold = 0.15
        while self.onset_count != ref.onset_count:
            if self.onset_count < ref.onset_count:
                #print("Decrease threshold.", threshold)
                if threshold - 0.02 <= 0:
                    print("Signal not good.")
                    break
                else:
                    threshold -= 0.02
                #self.performance = audionset.AudioPiece(WAVE_OUTPUT_FILENAME, threshold)
            else:
                threshold += 0.035
                #print("Increase threshold.", threshold)

            self.__onset_decider__(threshold)
            nof_tries += 1
            if nof_tries == 50:
                print("Cannot detect good number of onsets from recording.")
                break
        #print("Threshold:", threshold)

    def __shift_times__(self, ref):
        #CALL JUST AFTER ONSET DETECTION
        amount = float(self.onset_times[0])
        #print(amount)
        #print("Rhythm onset times length:", len(self.onset_times))
        for e in self.onset_times:
            self.shifted_onset_times.append(e - amount + ref.onset_times[0])
        #print("Rhythm shifted onset times:", self.shifted_onset_times)
        self.shifted_onset_signal = self.__time2signal__(self.shifted_onset_times)
        self.shifted_onset_signal = self.shifted_onset_signal[:1 + max([i for i, j in enumerate(self.shifted_onset_signal) if j == 1])]

    def __stretch_cumulative__(self, ref):
        #self.shifttimes(ref)
        ratio = (ref.onset_times[-1] - ref.onset_times[0]) / (self.shifted_onset_times[-1] - self.shifted_onset_times[0])
        #print("Stretch ratio=", ratio)
        for i in self.shifted_onset_times:
                self.stretched_onset_times.append((i - self.shifted_onset_times[0]) * ratio + self.shifted_onset_times[0])

        self.stretched_onset_signal = self.__time2signal__(self.stretched_onset_times)
        self.stretched_onset_signal = self.stretched_onset_signal[:ref.signal_length]

    def __stretch_adaptive__(self, ref):

        distances = list()
        distancesums = list()
        mindistance = 100
        ratio = 0.5
        rate = 0.01
        x = ref.onset_times
        y = self.shifted_onset_times

        for i in range(0, 301):

            scaled_y = ratio * (y - y[0]) + y[0]
            #print("Scaler:", i, "Scaled Y:", scaled_y)
            distances.append(x - scaled_y)
            distancesums.append(sum(abs(distances[-1])))
            ratio += rate

        mindistance = min(distancesums)
        index_of_mindistance = distancesums.index(mindistance)
        closest_scaled_y = distances[index_of_mindistance]
        ratio = 0.5 + (index_of_mindistance * rate)
        #print(mindistance, index_of_mindistance, ratio)
        #print("Mind difference:", closest_scaled_y)

        y = ratio * (y - y[0]) + y[0]
        #print("Final Y:", y)
        self.stretched_onset_times = y
        self.stretched_onset_signal = self.__time2signal__(self.stretched_onset_times)
        self.stretched_onset_signal = self.stretched_onset_signal[:ref.signal_length]
