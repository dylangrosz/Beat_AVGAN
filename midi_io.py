import pickle as pk
import glob
import os
import sys

import magenta.music as mm
import numpy as np
import pandas as pd
import tensorflow as tf
from midi import NOTE_NAME_MAP_SHARP
from music21 import converter
import midi_reader as mr

import pprint as pp

class MIDI_IO():
    def __init__(self):
        self.note_info_path = 'C:/Users/dylan/PycharmProjects/firstone/note_mapping_dict.pkl'
        self.midi_training_path_trans = ['C:/Users/dylan/PycharmProjects/firstone/jazz_midi_trans.pkl',
                                         'C:/Users/dylan/PycharmProjects/firstone/folk_midi_trans.pkl']

        if not os.path.exists(self.midi_training_path_trans[0]) or not os.path.exists(self.midi_training_path_trans[1]):
            self.load_all_midi_data(self.midi_training_path_trans)
        else:
            with open(self.note_info_path, "rb") as openfile:
                self.note_info_dict = pk.load(openfile)
                self.note_info_dict_swap = dict((y, x) for x, y in self.note_info_dict.iteritems())

                # print len(self.note_info_dict)

    def midi_file_to_seq(self, midi_file):
        seq = np.array([])
        try:
            # encode A0 with 33, while stop with -2, long press with successive -2
            melody = mm.midi_file_to_melody(midi_file, steps_per_quarter=8)
            print("MELODY")
            print(melody.to_sequence())
            print(melody._events)
            print(np.asarray(melody._events))
            # print melody.steps_per_bar
            # if melody.steps_per_bar % 3 != 0:
            seq = np.array(melody._events)
            print("right after")
            print(seq)

            # get difference for shifting, midi.NOTE_NAME_MAP_SHARP['C_4']=48
            pitch = converter.parse(midi_file).analyze('key').pitchFromDegree(1)
            pitch_str = '{}_{}'.format(pitch._step, pitch._octave)
            pitch_shift = NOTE_NAME_MAP_SHARP[pitch_str] - 48

            def shift_note(x):
                if x not in [-1, -2]:
                    return x - pitch_shift
                else:
                    return x

            seq = list(map(shift_note, seq))
            print(seq)

            tf.logging.info('Extract melody events from {} file'.format(midi_file))
            return seq
        except mm.MultipleTempoException as e:
            tf.logging.warning('Melody of {} file has multiple tempos'.format(midi_file))
        except mm.MultipleTimeSignatureException as e:
            tf.logging.warning('Melody of {} file has multiple signature'.format(midi_file))
        except mm.NegativeTimeException as e:
            tf.logging.warning('Melody of {} file has negative time'.format(midi_file))
        except mm.midi_io.MIDIConversionError as e:
            tf.logging.warning('Melody of {} file has wrong format'.format(midi_file))

        return list(seq)

    def seq_to_midi_file(self, seq, output_file):
        melody = mm.Melody(events=seq.tolist())
        note_sequence = melody.to_sequence(qpm=80.0)
        mm.sequence_proto_to_midi_file(note_sequence, output_file)
        return seq

    def check_note_mapping_exist(self):
        if not os.path.exists(self.note_info_path):
            self.load_all_midi_data()

    def load_all_midi_data(self, domain_name, seq_len=36, stride_len=16):
        # midi_dir = os.path.expanduser("midi/")
        # filenames = os.listdir(midi_dir)
        # filenames = glob.glob('midi_' + domain_name[0]) + glob.glob('midi_' + domain_name[1])

        cnt = 0
        uniques = []
        longest = 0
        midis_dir_path = "C:/Users/dylan/Music/tosample/aq-Hip-Hop-Beats-60-110-bpm/(aq) Hip Hop Beats 60-110 bpm/"
        result = [[]]
        files = mr.find_files(midis_dir_path)
        genre_dist = []  # for stat
        for f in files:
            print(f)
            seq = self.midi_file_to_seq(f)
            genre_dist.extend(seq)
            print("returned is:")
            print(seq)
            print(list(seq))
            self.seq_to_midi_file(np.array(seq), 'outputEx.mid')
            if len(list(seq)) > 0:
                print("we made it")
                cnt += 1

                if len(seq) > longest:
                    longest = len(seq)

                for i in seq:
                    if i not in uniques:
                        uniques.append(i)

                result.append(seq)

            # for stat
            pk.dump(genre_dist, open(f[:-4] + '_dist', 'wb'))
        print("uniques now")
        print(uniques)
        sorted_vals = list(sorted(uniques, key=abs))
        print(sorted_vals)
        sorted_vals = list(map(int, sorted_vals))
        print(sorted_vals)
        sorted_vals = np.asarray(sorted_vals)
        print(sorted_vals)
        note_info = pd.DataFrame(data=sorted_vals, columns=['note'])

        self.note_info_dict = note_info['note'].to_dict()
        self.note_info_dict_swap = dict((y, x) for x, y in self.note_info_dict.items())
        print(result)

        for _rid, r in enumerate(result):
            print(r)
            trans_list = self.trans_raw_songs_to_trans(r)
            # without transfer matrix
            # trans_list = result

            windowed_trans_list = []
            length = seq_len
            stride = stride_len

            for midi in trans_list:
                if len(midi) > length:
                    last_index = 0

                    while last_index + length < len(midi):
                        windowed_trans_list.append(midi[last_index:last_index + length])
                        last_index += stride

            # print len(windowed_trans_list)
            # print("{} melodies extracted from {} mid files in {}"
            #       .format(cnt, len(filenames), midi_dir))

            with open(self.midi_training_path_trans[_rid], "wb") as output_file:
                pk.dump(windowed_trans_list, output_file)

        with open(self.note_info_path, "wb") as openfile:
            pk.dump(self.note_info_dict, openfile)

    def raw_note_to_trans(self, raw_note):

        result = []

        for entry in raw_note:
            result.append(self.note_info_dict_swap.get(entry))

        return result

    def trans_raw_songs_to_trans(self, raw_list):

        trans_list = []
        for midi in raw_list:
            trans_list.append(np.asarray(self.raw_note_to_trans(midi)))

        return trans_list

    def trans_to_raw_note(self, trans_note):

        result = []

        for entry in trans_note:
            result.append(self.note_info_dict.get(entry))

        return result

    def trans_trans_songs_to_raw(self, trans_list):

        raw_list = []
        for midi in trans_list:
            raw_list.append(np.asarray(self.trans_to_raw_note(midi)))
            # without transfer matrix
            # raw_list.append(np.asarray(midi+20))

        return raw_list

    def trans_generated_to_midi(self, file_name, trans=True):

        # with open(file_name + ".pkl", 'rb') as files:
        #     res = pk.load(files)
        #     print res

        with open(file_name, 'rb') as files:
            res = files.readlines()
            res = np.array([[int(_) for _ in x.strip().split()] for x in res])

        raws = self.trans_trans_songs_to_raw(res) if trans else res

        # remove the old files
        output_dir = 'outputs_' + file_name.split(os.sep)[-1]
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        index = 0
        for raw in raws:
            print(index)
            path = output_dir + os.sep + '{}.mid'.format(index)
            try:
                self.seq_to_midi_file(raw, path)
            except:
                print("Unexpected error:" + str(sys.exc_info()[0]))
            index += 1

    def trans_ndarray_to_midi(self, arr, name):
        raws = arr

        # remove the old files
        output_dir = 'outputs_' + name
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        index = 0
        for raw in raws:
            print(index)
            path = output_dir + os.sep + '{}.mid'.format(index)
            try:
                self.seq_to_midi_file(np.array(raw), path)
            except:
                print("Unexpected error:" + str(sys.exc_info()[0]))
            index += 1


if __name__ == "__main__":
    # convert midi to matrix
    io = MIDI_IO()

    # GAN training
    # seqgan('jazz_midi_trans.pkl', 'folk_midi_trans.pkl')

    # convert matrix to midi
    # io.trans_generated_to_midi('SeqGAN/jazz_midi_trans_gan.out0')
    # io.trans_generated_to_midi('SeqGAN/jazz_midi_trans_gan.out3')
    # io.trans_generated_to_midi('SeqGAN/jazz_midi_trans_gan.out6')
    # io.trans_generated_to_midi('SeqGAN/jazz_midi_trans_gan.out9')
    # io.trans_generated_to_midi('SeqGAN/folk_midi_trans_mle.out')
    # io.trans_generated_to_midi('SeqGAN/folk_midi_trans_gan.out')
    # io.trans_generated_to_midi('SeqGAN/jazz_midi_trans_gan.out')
    # io.trans_generated_to_midi('SeqGAN/mix.out')

    # io.trans_generated_to_midi('/home/danny/PycharmProjects/hybridmusic/fusion_0')
    # io.trans_generated_to_midi('/home/danny/PycharmProjects/hybridmusic/output_music_fusion_2')
    # io.trans_generated_to_midi('/home/danny/PycharmProjects/hybridmusic/output_music_fusion_3')
